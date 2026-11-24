# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Filters from an octree to an octree (study): averages up, depth limits,
thresholds and axis slices, held to brute force over the nodes' boxes and to
probes of the octree they came from."""

import numpy as np
import pytest
from test_octree_topology import CASES, _boxes

import tack
import tack.dataset as td
from tack.dataset import octree_filters as of
from tack.dataset.ghosts import Ghosts


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _with_field(octree, seed=0):
    """``octree`` with a field ``s`` on every node, NaN on refined ones (a
    reader's parents), a smooth function of the leaf centres plus noise."""
    rng = np.random.default_rng(seed)
    c = octree.centers()
    s = np.sin(2 * c[:, 0]) + np.cos(3 * c[:, 1]) + 0.5 * c[:, 2] + 0.1 * rng.random(len(c))
    s[~octree.is_leaf] = np.nan
    octree.fields["s"] = tack.array(s, tack.f64)
    return octree


def _descendant_leaves(octree):
    lo, hi = _boxes(octree)
    leaf = octree.is_leaf & ((octree.marks.numpy() & Ghosts.HIDDENCELL) == 0)
    for n in range(octree.num_nodes):
        inside = (octree._tree == octree._tree[n]) & (lo >= lo[n]).all(axis=1) & \
            (hi <= hi[n]).all(axis=1) & leaf
        yield n, np.flatnonzero(inside)


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_average_up_is_the_leaves_mean_by_measure(name, make):
    octree = _with_field(make())
    up = of.average_up(octree, ["s"])
    got = up.fields["s"].numpy()
    s, m = octree.fields["s"].numpy(), octree.measures()
    for n, leaves in _descendant_leaves(octree):
        expected = np.sum(s[leaves] * m[leaves]) / np.sum(m[leaves])
        assert np.isclose(got[n], expected, rtol=1e-12), n


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_a_depth_limit_keeps_the_integral(name, make):
    octree = _with_field(make())
    whole = td.statistics(octree, "s", weights="volume")
    for depth in range(octree.finest + 1):
        limited = of.limit_depth(octree, depth)
        assert limited.finest <= depth
        assert limited.num_nodes == int((octree._level <= depth).sum())
        got = td.statistics(limited, "s", weights="volume")
        assert np.isclose(got.sum, whole.sum, rtol=1e-12), depth
        assert np.isclose(got.weight, whole.weight, rtol=1e-12), depth


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_a_threshold_hides_the_leaves_outside(name, make):
    octree = _with_field(make())
    s = octree.fields["s"].numpy()
    lower, upper = np.nanpercentile(s, [30, 70])
    out = of.threshold(octree, "s", lower, upper)
    hidden = (out.marks.numpy() & Ghosts.HIDDENCELL) != 0
    was = (octree.marks.numpy() & Ghosts.HIDDENCELL) != 0
    outside = octree.is_leaf & ((s < lower) | (s > upper))
    np.testing.assert_array_equal(hidden, was | outside)
    stats = td.statistics(out, "s")
    kept = octree.is_leaf & ~(was | outside)
    assert stats.n == kept.sum()
    assert np.isclose(stats.mean, s[kept].mean())


def _plane_points(octree, axis, position, n=400, seed=1):
    rng = np.random.default_rng(seed)
    lo, hi = np.zeros(3), np.zeros(3)
    axes = [np.asarray(a) for a in td.octree._root_axes(octree.roots)]
    for a in range(3):
        lo[a], hi[a] = axes[a][0], axes[a][-1]
    p = lo + (hi - lo) * rng.random((n, 3))
    p[:, axis] = position
    return p + np.asarray(octree.origin)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("name,make", [c for c in CASES if "3D" in c[0]],
                         ids=[c[0] for c in CASES if "3D" in c[0]])
def test_an_axis_slice_is_the_octree_on_the_plane(name, make, axis):
    octree = _with_field(make())
    axes = td.octree._root_axes(octree.roots)
    for position in (axes[axis][0] + 0.37 * (axes[axis][-1] - axes[axis][0]),
                     float(axes[axis][1])):                   # on a root face
        cut = of.axis_slice(octree, axis, position + octree.origin[axis])
        u, v = [a for a in range(3) if a != axis]
        p = _plane_points(octree, axis, position)
        q = np.zeros_like(p)
        q[:, 0], q[:, 1], q[:, 2] = p[:, u], p[:, v], position + octree.origin[axis]
        here = td.probe(octree, p)
        there = td.probe(cut, q)
        np.testing.assert_array_equal(here.fields["valid"].values.numpy(),
                                      there.fields["valid"].values.numpy())
        np.testing.assert_array_equal(here.fields["s"].values.numpy(),
                                      there.fields["s"].values.numpy())
        # The 2D leaves tile the plane.
        area = (axes[u][-1] - axes[u][0]) * (axes[v][-1] - axes[v][0])
        assert np.isclose(cut.measures()[cut.is_leaf].sum(), area)


@pytest.mark.parametrize("name,make", [c for c in CASES if "3D" in c[0]],
                         ids=[c[0] for c in CASES if "3D" in c[0]])
def test_cell_gradients_are_exact_for_a_linear_field(name, make):
    octree = make()
    slope = np.array([0.7, -1.3, 2.1])
    octree.fields["lin"] = tack.array(octree.centers() @ slope + 0.25, tack.f64)
    got = of.cell_gradients(octree, "lin").numpy()
    np.testing.assert_allclose(got, np.broadcast_to(slope, got.shape), atol=1e-9)
