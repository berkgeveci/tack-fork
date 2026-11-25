# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Regions of an octree (study): a box extracted, clumps found, a covering
grid sampled -- each held to brute force over the nodes' boxes."""

import numpy as np
import pytest
from test_octree_filters import _with_field
from test_octree_topology import CASES, _boxes

import tack
import tack.dataset as td
from tack.dataset import octree_filters as of
from tack.dataset.octree import _check_ancestors
from tack.dataset.octree_clumps import clumps
from tack.dataset.octree_topology import faces


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _keys(octree, which):
    return {(int(t), int(lv), int(k)) for t, lv, k in
            zip(octree._tree[which], octree._level[which], octree._key[which])}


def _visible(octree):
    return octree.is_leaf & ((octree.marks.numpy() & td.Ghosts.HIDDENCELL) == 0)


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_a_box_keeps_the_leaves_meeting_it(name, make):
    octree = _with_field(make())
    d = octree.dimension
    lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in octree.geometry())
    origin = np.asarray(octree.origin)
    whole = (lo.min(axis=0), (lo + size).max(axis=0))
    lower = whole[0] + 0.3 * (whole[1] - whole[0])
    upper = whole[0] + 0.7 * (whole[1] - whole[0])
    region = of.extract_box(octree, lower + origin, upper + origin)
    _check_ancestors(region._tree, region._level, region._key, d)
    meets = (lo[:, :d] < upper[:d]).all(axis=1) & ((lo + size)[:, :d] > lower[:d]).all(axis=1)
    assert _keys(region, _visible(region)) == _keys(octree, _visible(octree) & meets)
    assert region.num_nodes < octree.num_nodes
    # The values are the nodes' own.
    before = dict(zip(zip(octree._tree, octree._level, octree._key), octree.fields["s"].numpy()))
    after = region.fields["s"].numpy()
    for k, (t, lv, key) in enumerate(zip(region._tree, region._level, region._key)):
        assert np.array_equal(after[k], before[(t, lv, key)], equal_nan=True)


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_clumps_are_the_components_through_faces(name, make):
    octree = _with_field(make())
    s = octree.fields["s"].numpy()
    lower, upper = np.nanpercentile(s[octree.is_leaf], 40), np.nanpercentile(s[octree.is_leaf], 80)
    octree = of.threshold(octree, "s", -np.inf, np.nanpercentile(s[octree.is_leaf], 75))
    got = clumps(octree, "s", lower, upper).fields["clump"].numpy()
    inside = _visible(octree) & (s >= lower) & (s <= upper)
    # Brute force: union-find over the faces between leaves in the range.
    found = faces(octree)
    parent = np.arange(octree.num_nodes)

    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for a, b in zip(found.small.numpy(), found.large.numpy()):
        if b >= 0 and inside[a] and inside[b]:
            ra, rb = root(a), root(b)
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(n) if inside[n] else -1 for n in range(octree.num_nodes)])
    expected = np.full(octree.num_nodes, -1)
    distinct = np.unique(roots[roots >= 0])
    expected[roots >= 0] = np.searchsorted(distinct, roots[roots >= 0])
    np.testing.assert_array_equal(got, expected)
    assert expected.max() >= 1


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_the_covering_grid_samples_the_nodes(name, make):
    octree = _with_field(make())
    s = octree.fields["s"].numpy()
    octree = of.threshold(octree, "s", np.nanpercentile(s[octree.is_leaf], 5), np.inf)
    octree = of.average_up(octree, ["s"])
    d = octree.dimension
    at = max(octree.finest - 1, 1)
    grid = of.covering_grid(octree, at, ["s"])
    got = np.asarray(grid.fields["s"].values.numpy())
    lo, hi = _boxes(octree)                        # lattice boxes at the finest level
    step = 1 << (octree.finest - at)
    dims = np.array(([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]) << at
    if d == 2:
        dims[2] = 1
    assert grid.num_cells == len(got) == dims.prod()
    values = np.asarray(octree.fields["s"].numpy())
    hidden = (octree.marks.numpy() & td.Ghosts.HIDDENCELL) != 0
    level = octree._level
    rng = np.random.default_rng(1)
    for cell in rng.choice(len(got), size=min(200, len(got)), replace=False):
        i, j, k = cell % dims[0], (cell // dims[0]) % dims[1], cell // (dims[0] * dims[1])
        probe = np.array([i, j, k]) * step + (0 if d == 3 else [0, 0, 0])
        if d == 2:
            probe[2] = 0
        holds = (lo <= probe).all(axis=1) & (probe < hi).all(axis=1) & (level <= at)
        nodes = np.flatnonzero(holds)
        n = nodes[np.argmax(level[nodes])]
        expected = np.nan if hidden[n] else values[n]
        assert np.array_equal(got[cell], expected, equal_nan=True), cell
