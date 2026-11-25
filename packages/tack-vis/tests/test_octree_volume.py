# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The octree as the volume march's sampler (study): its samples probe's, its
tents continuous, its hidden subtrees and macrocell ranges right."""

import numpy as np
import pytest
from test_octree_dual import _octree

import tack
import tack.dataset as td
from tack.dataset import octree_filters as of
from tack.dataset import octree_volume as ov


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


@tack.kernel
def _sample_points(tents, ps, values, sizes):
    for n in range(ps.shape[0]):
        v, s = tents.sample(ps[n], 0, 1)
        values[n] = v
        sizes[n] = s


@tack.kernel
def _hidden_at(tents, ps, hides, boxes):
    for n in range(ps.shape[0]):
        h, lx, ly, lz, hx, hy, hz = tents.hidden(ps[n], 0, 1)
        hides[n] = h
        boxes[n] = [lx, ly, lz, hx, hy, hz]


def _smooth(octree, c):
    lo, hi = c.min(axis=0), c.max(axis=0)
    u = (c - lo) / (hi - lo)
    return (np.exp(-((u - 0.4) ** 2).sum(axis=1) / 0.05) + 0.3 * np.sin(5 * u[:, 0])
            * np.cos(4 * u[:, 1])).astype(np.float32)


def _points(octree, count, seed=0):
    rng = np.random.default_rng(seed)
    lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in octree.geometry())
    low, high = lo.min(axis=0), (lo + size).max(axis=0)
    return low + rng.random((count, 3)) * (high - low) + np.asarray(octree.origin)


def test_leaf_values_are_probes_and_sizes_the_leaves(seed=1):
    octree = _octree((5, 4, 4), depth=5)
    c = octree.centers() + np.asarray(octree.origin)
    octree.fields = {"phi": tack.array(_smooth(octree, c), tack.f32)}
    s = octree.fields["phi"].numpy()
    octree = of.threshold(octree, "phi", np.percentile(s, 5), np.inf)
    x = _points(octree, 20_000)
    tents = ov.octree_tents(octree, "phi", blend=0)
    values, sizes = tack.empty((len(x),), tack.f32), tack.empty((len(x),), tack.f32)
    _sample_points(tents, tack.array(x.astype(np.float32), tack.vec3(tack.f32)), values, sizes)
    # probe finds no leaf in a hidden one (-1); the sampler gives 0 there.
    found = np.asarray(octree._found(x - np.asarray(octree.origin)).numpy())
    held = found >= 0
    assert held.any() and not held.all()
    phi = octree.fields["phi"].numpy()
    np.testing.assert_array_equal(values.numpy()[held], phi[found[held]])
    assert (values.numpy()[~held] == 0).all()
    _, size = octree.geometry()
    leaf_size = np.asarray(size.numpy(), np.float64).reshape(-1, 3).min(axis=1)
    np.testing.assert_allclose(sizes.numpy()[held], leaf_size[found[held]], rtol=1e-6)


def test_tents_are_continuous_and_at_least_as_close():
    octree = _octree((5, 4, 4), depth=5)
    c = octree.centers() + np.asarray(octree.origin)
    octree.fields = {"phi": tack.array(_smooth(octree, c), tack.f32)}
    x = _points(octree, 50_000)
    rng = np.random.default_rng(2)
    y = x + 1e-5 * rng.standard_normal(x.shape)
    exact = _smooth(octree, x)
    out = {}
    for blend in (0, 1):
        tents = ov.octree_tents(octree, "phi", blend=blend)
        a, b = tack.empty((len(x),), tack.f32), tack.empty((len(x),), tack.f32)
        sizes = tack.empty((len(x),), tack.f32)
        _sample_points(tents, tack.array(x.astype(np.float32), tack.vec3(tack.f32)), a, sizes)
        _sample_points(tents, tack.array(y.astype(np.float32), tack.vec3(tack.f32)), b, sizes)
        jumps = np.abs(a.numpy() - b.numpy())
        out[blend] = (np.abs(a.numpy() - exact).mean(), np.percentile(jumps, 99.9), jumps.max())
    assert out[1][0] <= out[0][0]                       # the tents are no farther from the field
    assert out[1][1] < 1e-3 < out[0][2]                 # and continuous almost everywhere


def test_hidden_subtrees_are_the_boxes_to_skip():
    octree = _octree((5, 4, 4), depth=5)
    c = octree.centers() + np.asarray(octree.origin)
    octree.fields = {"phi": tack.array(_smooth(octree, c), tack.f32)}
    x = _points(octree, 5_000)
    ps = tack.array(x.astype(np.float32), tack.vec3(tack.f32))
    tents = ov.octree_tents(octree, "phi")
    hides = tack.empty((len(x),), tack.i32)
    boxes = tack.empty((len(x),), tack.vec(6, tack.f32))
    lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) + (np.asarray(octree.origin)
                if k == 0 else 0) for k, a in enumerate(octree.geometry()))
    found = np.asarray(octree._found(x - np.asarray(octree.origin)).numpy())
    # Everything can show: nothing hides, the box is the leaf's.
    _hidden_at(tents, ps, hides, boxes)
    assert not hides.numpy().any()
    np.testing.assert_allclose(np.asarray(boxes.numpy())[:, :3], lo[found], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(np.asarray(boxes.numpy())[:, 3:], size[found], rtol=1e-6)
    # Subtrees of level 2 cannot show: a point in one hides, with the level-2
    # ancestor's box; a point in a coarser leaf does not.
    seen = (octree._level < 2).astype(np.int32)
    tents.seen = tack.array(seen, tack.i32)
    _hidden_at(tents, ps, hides, boxes)
    deep = octree._level[found] >= 2
    np.testing.assert_array_equal(hides.numpy() == 1, deep)
    assert deep.any() and not deep.all()
    b = np.asarray(boxes.numpy())
    np.testing.assert_allclose(b[deep, 3:], np.broadcast_to(size[octree._level == 2][0],
                                                            (deep.sum(), 3)), rtol=1e-6)
    assert ((b[:, :3] <= x + 1e-5) & (x < b[:, :3] + b[:, 3:] + 1e-5)).all()
    # Nothing can show: the root cell.
    tents.seen = tack.zeros((octree.num_nodes,), tack.i32)
    _hidden_at(tents, ps, hides, boxes)
    assert hides.numpy().all()
    np.testing.assert_allclose(np.asarray(boxes.numpy())[:, 3:],
                               np.broadcast_to(size[octree._level == 0][0], (len(x), 3)), rtol=1e-6)


def test_macrocell_ranges_bound_the_leaves_reaching_them():
    octree = _octree((5, 4, 4), depth=4)
    c = octree.centers() + np.asarray(octree.origin)
    octree.fields = {"phi": tack.array(_smooth(octree, c), tack.f32)}
    s = octree.fields["phi"].numpy()
    octree = of.threshold(octree, "phi", np.percentile(s, 5), np.inf)
    r = ov.ranges(octree, "phi")
    grid, lows, highs = ov.macro_ranges(octree, r, most=16)
    lows, highs = lows.numpy(), highs.numpy()
    dims, origin, size = (np.array(grid[k]) for k in ("dims", "origin", "size"))
    lo, side = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in octree.geometry())
    lo = lo + np.asarray(octree.origin)
    visible = octree.is_leaf & ((octree.marks.numpy() & td.Ghosts.HIDDENCELL) == 0)
    phi = octree.fields["phi"].numpy()
    rng = np.random.default_rng(3)
    for m in rng.choice(len(lows), 300, replace=False):
        i, j, k = m % dims[0], (m // dims[0]) % dims[1], m // (dims[0] * dims[1])
        a = origin + np.array([i, j, k]) * size
        b = a + size
        # The visible leaves whose tent (half a leaf beyond the box, zero at its
        # end) meets the cell's inside.
        eps = 1e-6 * size
        reach = (lo - 0.5 * side < b - eps).all(axis=1) & (lo + 1.5 * side > a + eps).all(axis=1) \
            & visible
        if reach.any():
            assert lows[m] <= phi[reach].min() + 1e-6 and highs[m] >= phi[reach].max() - 1e-6
        inside = (lo <= a).all(axis=1) & (lo + side >= b).all(axis=1) & visible
        if inside.any():                             # the leaf holding the cell is counted
            assert lows[m] <= phi[inside].min() + 1e-6
