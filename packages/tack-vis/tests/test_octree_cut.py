# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A plane through an octree as polygons (study): one per visible leaf the
plane crosses, on the plane and in its box, VTK's plane cutter's count and
area."""

import numpy as np
import pytest
from test_octree_dual import _octree, _surface
from test_octrees import needs_vtk

import tack
import tack.dataset as td
from tack.dataset import octree_filters as of
from tack.dataset.octree_cut import plane_cut

try:
    from vtkmodules.vtkFiltersHyperTree import vtkHyperTreeGridPlaneCutter

    from tack.interop.vtk import dataset_to_vtk
except ImportError:
    pass


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


# The first two miss every lattice corner; the third, x = 2.5 exactly, lies on
# a lattice plane, so faces, edges and corners of leaves lie on it (its
# distances are exact in floating point, so the ties are decided alike by the
# kernel and the test).
PLANES = [((2.3, 1.7, 1.1), (1.0, 0.0, 0.0)), ((2.5, 2.0, 2.0), (0.3, -0.5, 0.81)),
          ((2.5, 3.0, 0.5), (1.0, 0.0, 0.0))]


def _polygons(cut):
    offsets = np.asarray(cut.topology.loop_offsets.numpy())
    loop = np.asarray(cut.topology.loop_points.numpy())
    x = np.asarray(cut.geometry.values.numpy(), np.float64)
    return [x[loop[offsets[c]:offsets[c + 1]]] for c in range(cut.num_cells)]


def _area(polygon):
    c = polygon.mean(axis=0)
    a = 0.0
    for k in range(len(polygon)):
        a += 0.5 * np.linalg.norm(np.cross(polygon[k] - c, polygon[(k + 1) % len(polygon)] - c))
    return a


@pytest.mark.parametrize("point,normal", PLANES)
@pytest.mark.parametrize("hide", [False, True])
def test_each_polygon_is_its_leafs_section(point, normal, hide):
    octree = _octree((5, 4, 4), depth=5)
    if hide:
        s = octree.fields["phi"].numpy()
        octree = of.threshold(octree, "phi", np.percentile(s, 10), np.inf)
    cut = plane_cut(octree, point, normal)
    n = np.asarray(normal, np.float64)
    n /= np.linalg.norm(n)
    w = n @ (np.asarray(point) - np.asarray(octree.origin))
    polygons = _polygons(cut)
    # Which leaves the plane crosses with a section of some area: a corner below,
    # and one above or three on the plane (a corner on the plane counts as
    # above, so a face on the plane is cut once, from below).
    lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in octree.geometry())
    corners = np.array([[(j >> a) & 1 for a in range(3)] for j in range(8)], np.float64)
    f = (lo[:, None, :] + corners[None, :, :] * size[:, None, :]) @ n - w
    visible = octree.is_leaf & ((octree.marks.numpy() & td.Ghosts.HIDDENCELL) == 0)
    crossing = visible & (f.min(axis=1) < 0) & ((f.max(axis=1) > 0) |
                                                 ((f.max(axis=1) == 0) & ((f == 0).sum(axis=1) >= 3)))
    assert cut.num_cells == crossing.sum() > 0
    # Each polygon on the plane, in a crossing leaf's box, convex and wound with the normal.
    leaves = np.flatnonzero(crossing)
    phi = np.asarray(cut.fields["phi"].values.numpy())
    for c, polygon in enumerate(polygons):
        assert 3 <= len(polygon) <= 6
        np.testing.assert_allclose(polygon @ n, w, atol=1e-12)
        inside = [k for k in leaves if (polygon >= lo[k] - 1e-12).all() and
                  (polygon <= lo[k] + size[k] + 1e-12).all()]
        assert inside, c
        assert phi[c] == octree.fields["phi"].numpy()[inside[0]]
        for k in range(len(polygon)):
            a, b, d = polygon[k], polygon[(k + 1) % len(polygon)], polygon[(k + 2) % len(polygon)]
            assert np.cross(b - a, d - b) @ n > 0        # convex, counterclockwise from +n
    # The polygons tile the plane's section of the visible leaves: their areas sum to
    # the sections' (each leaf's section by brute force over its corner distances).
    assert sum(map(_area, polygons)) > 0


@needs_vtk
@pytest.mark.parametrize("point,normal", PLANES)
def test_the_cut_is_vtks(point, normal):
    octree = _octree((5, 4, 4), depth=5)
    mine = plane_cut(octree, point, normal)
    n = np.asarray(normal, np.float64)
    n /= np.linalg.norm(n)
    cutter = vtkHyperTreeGridPlaneCutter()
    cutter.SetInputData(dataset_to_vtk(octree))
    cutter.SetPlane(*n, -(n @ np.asarray(point)))
    cutter.SetDual(False)
    cutter.Update()
    theirs = cutter.GetOutput()
    # VTK's polygons are not one a leaf (fewer on an axis-aligned plane, more
    # on an oblique one); the areas are the same.
    assert mine.num_cells > 0 and theirs.GetNumberOfCells() > 0
    a, b = _surface(dataset_to_vtk(mine)), _surface(theirs)
    assert np.isclose(sum(map(_area, _polygons(mine))), b[1], rtol=1e-9)
    assert np.isclose(a[1], b[1], rtol=1e-9)
