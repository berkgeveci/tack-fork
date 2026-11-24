# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A 3D octree's leaves as conforming polyhedra (study): each cell the leaf's box,
closed, every face shared by the two leaves it lies between, and regions
through shared points the leaves' that touch."""

import numpy as np
import pytest
from test_octree_filters import _with_field
from test_octree_topology import CASES

import tack
import tack.dataset as td
from tack.dataset import octree_filters as of
from tack.dataset import octree_topology as ot
from tack.dataset.octree_polyhedra import leaves_as_polyhedra

CASES_3D = [c for c in CASES if "3D" in c[0]]
IDS = [c[0] for c in CASES_3D]


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


@pytest.mark.parametrize("name,make", CASES_3D, ids=IDS)
def test_each_cell_is_its_leafs_box(name, make):
    octree = make()
    mesh = leaves_as_polyhedra(octree)
    leaf = octree.is_leaf
    assert mesh.num_cells == leaf.sum()
    geometry = td.cell_geometry(mesh)
    eps = 1e-12 if octree.roots.dtype == tack.f64 else 1e-6
    np.testing.assert_allclose(geometry.measure.values.numpy(), octree.measures()[leaf],
                               rtol=eps)
    np.testing.assert_allclose(geometry.centroid.values.numpy(), octree.centers()[leaf],
                               atol=eps)


@pytest.mark.parametrize("name,make", CASES_3D, ids=IDS)
def test_the_cells_are_closed_and_faces_shared(name, make):
    octree = make()
    mesh = leaves_as_polyhedra(octree)
    # Each cell's area vectors, signed by side, sum to zero: closed, wound out.
    normal, area = td.face_geometry(mesh)
    vectors = normal.values.numpy() * area.values.numpy()[:, None]
    t = mesh.topology
    offsets, cell_faces, sides = (np.asarray(a.numpy()) for a in
                                  (t.cell_offsets, t.cell_faces, t.cell_face_sides))
    sign = np.where(sides == 0, 1.0, -1.0)[:, None]
    totals = np.add.reduceat(vectors[cell_faces] * sign, offsets[:-1], axis=0)
    scale = np.abs(vectors).max()
    eps = 1e-12 if octree.roots.dtype == tack.f64 else 1e-6
    np.testing.assert_allclose(totals, 0.0, atol=eps * scale)
    # The boundary faces are the domain's surface.
    boundary = td.boundary_faces(mesh)
    axes = [np.asarray(a) for a in td.octree._root_axes(octree.roots)]
    e = [a[-1] - a[0] for a in axes]
    surface = 2 * (e[0] * e[1] + e[1] * e[2] + e[0] * e[2])
    picked = np.asarray(boundary.ids.numpy()) if hasattr(boundary, "ids") else \
        np.asarray(boundary.numpy())
    assert np.isclose(area.values.numpy()[picked].sum(), surface, rtol=1e-6)
    found = ot.faces(octree)
    assert len(picked) == int((found.large.numpy() < 0).sum())


def _touching_regions(octree, kept):
    """Regions of the kept leaves through shared points, by brute force: leaves
    whose closed boxes meet are joined."""
    lo, hi = ot.lattice_boxes(octree)
    leaves = np.flatnonzero(octree.is_leaf)
    lo, hi = lo[leaves], hi[leaves]
    parent = np.arange(len(leaves))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    meet = (np.minimum(hi[:, None], hi[None]) >= np.maximum(lo[:, None], lo[None])).all(axis=2)
    for a, b in zip(*np.nonzero(meet)):
        if kept[a] and kept[b]:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(i) for i in range(len(leaves))])
    labels = np.full(len(leaves), -1)
    _, labels[kept] = np.unique(roots[kept], return_inverse=True)
    return labels


@pytest.mark.parametrize("name,make", CASES_3D, ids=IDS)
def test_regions_of_a_threshold_are_the_touching_leaves(name, make):
    octree = _with_field(make())
    s = octree.fields["s"].numpy()
    lower, upper = np.nanpercentile(s[octree.is_leaf], [60, 100])
    mesh = leaves_as_polyhedra(of.threshold(octree, "s", lower, upper))
    got = td.connected_components(mesh).fields["region"].values.numpy()
    kept = (s[octree.is_leaf] >= lower) & (s[octree.is_leaf] <= upper)
    expected = _touching_regions(octree, kept)
    assert (got[~kept] == -1).all()
    # The same partition: one label each way.
    pairs = set(zip(got[kept].tolist(), expected[kept].tolist()))
    assert len(pairs) == len(set(got[kept])) == len(set(expected[kept]))
