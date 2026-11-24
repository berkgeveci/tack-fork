# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An octree's topology from its keys (study): parents, subtree ends and the
faces between leaves, held to brute force over the nodes' integer boxes on the
finest lattice."""

import numpy as np
import pytest
from test_octrees import ramses_like

import tack
import tack.dataset as td
from tack.dataset import octree_topology as ot

try:
    from vtkmodules.vtkFiltersSources import vtkRandomHyperTreeGridSource

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtkRandomHyperTreeGridSource = None


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _ramses(d):
    roots, levels, centres, rho, _ = ramses_like(d=d, trees=(3, 2, 2))
    return td.OctreeDataSet.from_cells(roots, levels, centres, fields={"rho": rho})


def _random(dims, seed=7):
    source = vtkRandomHyperTreeGridSource(dimensions=dims, max_depth=5, seed=seed,
                                          split_fraction=0.45)
    source.Update()
    return vtk_to_dataset(source.GetOutput(), dtype=tack.f64)


def _octrees():
    out = [("ramses 2D", lambda: _ramses(2)), ("ramses 3D", lambda: _ramses(3))]
    if vtkRandomHyperTreeGridSource is not None:
        out += [("random 3D", lambda: _random((4, 3, 3))),
                ("random 2D", lambda: _random((4, 3, 1)))]
    return out


CASES = _octrees()


def _boxes(octree):
    return ot.lattice_boxes(octree)


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_parents_and_subtrees_are_the_keys(name, make):
    octree = make()
    d = octree.dimension
    tree, level, key = octree._tree, octree._level, octree._key
    where = {(int(t), int(lv), int(k)): n for n, (t, lv, k) in enumerate(zip(tree, level, key))}
    expected = np.array([where[(int(t), int(lv) - 1, int(k) >> d)] if lv else -1
                         for t, lv, k in zip(tree, level, key)])
    np.testing.assert_array_equal(ot.parents(octree).numpy(), expected)

    end = ot.subtree_ends(octree).numpy()
    lo, hi = _boxes(octree)
    for n in range(octree.num_nodes):
        inside = (tree == tree[n]) & (level > level[n]) & (lo >= lo[n]).all(axis=1) & \
            (hi <= hi[n]).all(axis=1)
        assert np.array_equal(np.flatnonzero(inside), np.arange(n + 1, end[n])), n


def _brute_faces(octree):
    """Every face once, as (small, large, direction): between two leaves sharing a
    face of positive measure, the smaller one's square (the lower one's between
    equals); on the domain's boundary, large -1."""
    d = octree.dimension
    leaves = np.flatnonzero(octree.is_leaf)
    lo, hi = (b[leaves] for b in _boxes(octree))
    level = octree._level[leaves]
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    extent = np.array([nx, ny, nz]) << octree.finest
    found = set()
    for axis in range(d):
        others = [a for a in range(d) if a != axis]
        touch = hi[:, None, axis] == lo[None, :, axis]          # a's upper face on b's lower
        for a in others:
            touch &= (np.minimum(hi[:, None, a], hi[None, :, a]) >
                      np.maximum(lo[:, None, a], lo[None, :, a]))
        for a, b in zip(*np.nonzero(touch)):
            if level[a] >= level[b]:                            # a is the smaller (or equal)
                found.add((leaves[a], leaves[b], 2 * axis + 1))
            else:
                found.add((leaves[b], leaves[a], 2 * axis))
        for c in np.flatnonzero(lo[:, axis] == 0):
            found.add((leaves[c], -1, 2 * axis))
        for c in np.flatnonzero(hi[:, axis] == extent[axis]):
            found.add((leaves[c], -1, 2 * axis + 1))
    return found


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_the_faces_are_every_shared_face_once(name, make):
    octree = make()
    got = ot.faces(octree)
    rows = list(zip(got.small.numpy().tolist(), got.large.numpy().tolist(),
                    got.direction.numpy().tolist()))
    assert len(rows) == len(set(rows))
    assert set(rows) == _brute_faces(octree)


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_the_faces_tile_every_leafs_surface(name, make):
    # Each leaf's faces in each direction (as the small side, or the large side
    # of smaller ones) add up to its own face's measure.
    octree = make()
    d = octree.dimension
    lo, hi = _boxes(octree)
    size = (hi - lo)[:, :d].astype(np.float64)
    got = ot.faces(octree)
    small, large, direction = (a.numpy() for a in (got.small, got.large, got.direction))
    covered = np.zeros((octree.num_nodes, 2 * d))
    area = np.prod(size, axis=1)
    for s, g, f in zip(small, large, direction):
        face = area[s] / size[s, f // 2]
        covered[s, f] += face
        if g >= 0:
            covered[g, f ^ 1] += face
    leaves = np.flatnonzero(octree.is_leaf)
    expected = area[leaves, None] / np.repeat(size[leaves], 2, axis=1)
    np.testing.assert_allclose(covered[leaves], expected)
