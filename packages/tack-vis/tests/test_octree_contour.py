# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Contours of an octree from the dual cells the surface crosses alone (study):
the whole dual grid's contour, polygon for polygon, and VTK's HyperTreeGrid
contour."""

import numpy as np
import pytest
from test_octree_dual import _octree, _surface
from test_octrees import needs_vtk

import tack
import tack.dataset as td
from tack.dataset import octree_contour as oc
from tack.dataset import octree_filters as of
from tack.dataset.octree_dual import dual_grid

try:
    from vtkmodules.vtkFiltersHyperTree import vtkHyperTreeGridContour

    from tack.interop.vtk import dataset_to_vtk
except ImportError:
    pass


@pytest.fixture(autouse=True, params=["chained", "split", "split from the roots"])
def cpu(request, monkeypatch):
    # The CPU's searches in turn, and the GPUs' a thread each, forced here:
    # starting deep as GPUs do (at the first level of 4k nodes), and at the
    # roots, reading no counts.
    if request.param != "chained":
        monkeypatch.setattr(oc, "_split_searches", lambda device: True)
    if request.param == "split from the roots":
        monkeypatch.setattr(oc, "_START_NODES", 0)
        monkeypatch.setattr(oc, "_READ_ABOVE", 1 << 60)
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


@needs_vtk
@pytest.mark.parametrize("dims", [(5, 4, 4), (5, 4, 1)])
@pytest.mark.parametrize("hide", [False, True])
def test_the_crossing_cells_are_the_dual_grids_that_cross(dims, hide):
    octree = _octree(dims, depth=5)
    if hide:
        s = octree.fields["phi"].numpy()
        octree = of.threshold(octree, "phi", np.percentile(s, 3), np.inf)
    whole = dual_grid(octree)
    corners = 8 if dims[2] > 1 else 4
    conn = np.asarray(whole.topology.connectivity.numpy()).reshape(-1, corners)
    phi = whole.fields["phi"].values.numpy()
    r = oc.ranges(octree, "phi")
    visible = octree.is_leaf & ((octree.marks.numpy() & td.Ghosts.HIDDENCELL) == 0)
    for value in np.percentile(octree.fields["phi"].numpy()[visible], [5, 30, 50, 85]):
        crossing = (phi[conn].min(axis=1) <= value) & (phi[conn].max(axis=1) >= value)
        expected = whole.geometry.values.numpy()[conn[crossing]].reshape(int(crossing.sum()), -1)
        expected = expected[np.lexsort(expected.T[::-1])]
        mine = oc.crossing_cells(octree, r, value)
        assert mine.num_cells == crossing.sum() > 0
        got = mine.geometry.values.numpy()[np.asarray(mine.topology.connectivity.numpy())]
        got = got.reshape(mine.num_cells, -1)
        np.testing.assert_array_equal(got[np.lexsort(got.T[::-1])], expected)


@needs_vtk
@pytest.mark.parametrize("depth", [4, 5])
def test_the_contour_is_vtks(depth):
    octree = _octree((5, 4, 4), depth)
    htg = dataset_to_vtk(octree)
    r = oc.ranges(octree, "phi")
    for value in (3.0, 6.0, 9.0):
        mine = _surface(dataset_to_vtk(oc.contour(octree, "phi", value, r)))
        reference = vtkHyperTreeGridContour()
        reference.SetInputData(htg)
        reference.SetInputArrayToProcess(0, 0, 0, 1, "phi")
        reference.SetNumberOfContours(1)
        reference.SetValue(0, value)
        reference.Update()
        theirs = _surface(reference.GetOutput())
        assert mine[0] == theirs[0] > 0
        assert np.isclose(mine[1], theirs[1], rtol=1e-7)
        assert mine[2] == theirs[2]          # open only where the surface leaves the domain
        assert mine[3] == theirs[3]
        if value < 9.0:
            assert mine[2] == 0
