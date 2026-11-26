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
def _sample_points(tents, ps, stop, values, sizes):
    for n in range(ps.shape[0]):
        v, s = tents.sample(ps[n], 0, stop)
        values[n] = v
        sizes[n] = s


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
    _sample_points(tents, tack.array(x.astype(np.float32), tack.vec3(tack.f32)),
                   octree.finest + 1, values, sizes)
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
        stop = octree.finest + 1
        _sample_points(tents, tack.array(x.astype(np.float32), tack.vec3(tack.f32)), stop, a, sizes)
        _sample_points(tents, tack.array(y.astype(np.float32), tack.vec3(tack.f32)), stop, b, sizes)
        jumps = np.abs(a.numpy() - b.numpy())
        out[blend] = (np.abs(a.numpy() - exact).mean(), np.percentile(jumps, 99.9), jumps.max())
    assert out[1][0] <= out[0][0]                       # the tents are no farther from the field
    assert out[1][1] < 1e-3 < out[0][2]                 # and continuous almost everywhere


def test_tents_are_the_weighted_mean_of_every_leaf_reaching_the_point():
    # Brute force: every visible leaf's tent (its own size wide about its centre)
    # at each point, over all leaves.
    octree = _octree((5, 4, 4), depth=4)
    c = octree.centers() + np.asarray(octree.origin)
    octree.fields = {"phi": tack.array(_smooth(octree, c), tack.f32)}
    s = octree.fields["phi"].numpy()
    octree = of.threshold(octree, "phi", np.percentile(s, 5), np.inf)
    x = _points(octree, 3_000, seed=4)
    tents = ov.octree_tents(octree, "phi", blend=1)
    values, sizes = tack.empty((len(x),), tack.f32), tack.empty((len(x),), tack.f32)
    _sample_points(tents, tack.array(x.astype(np.float32), tack.vec3(tack.f32)),
                   octree.finest + 1, values, sizes)
    lo, side = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in octree.geometry())
    visible = octree.is_leaf & ((octree.marks.numpy() & td.Ghosts.HIDDENCELL) == 0)
    centre = (lo + 0.5 * side)[visible] + np.asarray(octree.origin)
    size = side[visible]
    phi = s[visible].astype(np.float64)
    expected = np.zeros(len(x))
    for k, p in enumerate(x):
        w = np.prod(np.maximum(1.0 - np.abs(p - centre) / size, 0.0), axis=1)
        if w.sum() > 0:
            expected[k] = (w * phi).sum() / w.sum()
        else:                                        # no tent: the leaf's value (hidden: 0)
            found = int(np.asarray(octree._found(p[None] - np.asarray(octree.origin)).numpy())[0])
            expected[k] = s[found] if found >= 0 else 0.0
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-4, atol=1e-5)


def test_macrocell_ranges_bound_the_leaves_reaching_them():
    octree = _octree((5, 4, 4), depth=4)
    c = octree.centers() + np.asarray(octree.origin)
    octree.fields = {"phi": tack.array(_smooth(octree, c), tack.f32)}
    s = octree.fields["phi"].numpy()
    octree = of.threshold(octree, "phi", np.percentile(s, 5), np.inf)
    r = ov.ranges(octree, "phi")
    grid, lows, highs, firsts, lasts = ov.macro_ranges(octree, r, most=16)
    lows, highs, firsts, lasts = (a.numpy() for a in (lows, highs, firsts, lasts))
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
            levels = octree._level[reach]
            assert firsts[m] <= levels.min() and lasts[m] >= levels.max()
        inside = (lo <= a).all(axis=1) & (lo + side >= b).all(axis=1) & visible
        if inside.any():                             # the leaf holding the cell is counted
            assert lows[m] <= phi[inside].min() + 1e-6


def test_corner_interpolation_is_the_tents_at_the_corners():
    octree = _octree((5, 4, 4), depth=4)
    c = octree.centers() + np.asarray(octree.origin)
    octree.fields = {"phi": tack.array(_smooth(octree, c), tack.f32)}
    fast = ov.octree_tents(octree, "phi", blend=2)
    exact = ov.octree_tents(octree, "phi", blend=1)
    # At each leaf's centre the trilinear value is the mean of its corners, which
    # are the exact tents a hair inside each corner.
    leaves = np.flatnonzero(octree.is_leaf)[:2000]
    centres = (c[leaves]).astype(np.float32)
    ps = tack.array(centres, tack.vec3(tack.f32))
    values, sizes = tack.empty((len(leaves),), tack.f32), tack.empty((len(leaves),), tack.f32)
    _sample_points(fast, ps, octree.finest + 1, values, sizes)
    corners = np.asarray(fast.corners.numpy()).reshape(-1, 8)[leaves]
    np.testing.assert_allclose(values.numpy(), corners.mean(axis=1), rtol=1e-5, atol=1e-6)
    lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in octree.geometry())
    q = lo[leaves] + size[leaves] * [0.001, 0.001, 0.999] + np.asarray(octree.origin)
    _sample_points(exact, tack.array(q.astype(np.float32), tack.vec3(tack.f32)), octree.finest + 1,
                   values, sizes)
    np.testing.assert_allclose(values.numpy(), corners[:, 4], rtol=1e-4, atol=1e-5)


def test_the_balanced_corner_kernel_is_the_tents_at_the_corners():
    # A tree refined to a blob is 2:1 balanced where _octree's random one is not:
    # a uniform grid refined once in a box.
    from test_octrees import ramses_like

    import tack.dataset as td
    roots, levels, centres, rho, _ = ramses_like(d=3, trees=(3, 2, 2), levelmin=2, levelmax=3)
    octree = td.OctreeDataSet.from_cells(roots, levels, centres, fields={"rho": rho})
    from tack.dataset.octree_topology import face_neighbours
    faces = face_neighbours(octree).numpy().reshape(-1, 6)
    leaves = np.flatnonzero(octree.is_leaf)
    across = faces[leaves]
    held = across >= 0
    assert (np.abs(octree._level[across[held]] - np.repeat(octree._level[leaves], 6)[held.reshape(-1)]) <= 1).all()
    fast = ov.OctreeTents(octree, "rho", blend=2, balanced=1)
    slow = ov.OctreeTents(octree, "rho", blend=2, balanced=0)
    a = np.asarray(fast.corners.numpy()).reshape(-1, 8)[leaves]
    b = np.asarray(slow.corners.numpy()).reshape(-1, 8)[leaves]
    np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)       # the slow form samples a hair inside
