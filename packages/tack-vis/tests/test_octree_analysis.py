# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Analysis of an octree (study): profiles and phase plots against NumPy's
histograms, vector gradients exact on linear fields, particles deposited where
brute force puts them."""

import numpy as np
import pytest
from test_octree_filters import _with_field
from test_octree_topology import CASES

import tack
import tack.dataset as td
from tack.dataset import octree_filters as of
from tack.dataset.octree_binning import histogram
from tack.dataset.octree_particles import deposit

CASES_3D = [c for c in CASES if "3D" in c[0]]


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _visible(octree):
    return octree.is_leaf & ((octree.marks.numpy() & td.Ghosts.HIDDENCELL) == 0)


def _radius(octree):
    c = octree.centers() + np.asarray(octree.origin)
    return np.linalg.norm(c - c.mean(axis=0), axis=1)


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_a_profile_is_numpys_weighted_histogram(name, make):
    octree = _with_field(make())
    s = np.asarray(octree.fields["s"].numpy())
    octree.fields["r"] = tack.array(_radius(octree), tack.f64)
    octree.fields["w"] = tack.array(np.abs(s) + 0.5, tack.f64)
    octree = of.threshold(octree, "s", np.nanpercentile(s[octree.is_leaf], 5), np.inf)
    visible = _visible(octree)
    r, w, m = (np.asarray(a)[visible] for a in
               (octree.fields["r"].numpy(), octree.fields["w"].numpy(), octree.measures()))
    sv = s[visible]
    edges = np.linspace(r.min(), r.max(), 9)
    edges[-1] = r.max() * 0.98                     # some leaves fall outside the last edge
    got = histogram(octree, "r", edges, weight="w", values=["s"])
    inside = (r >= edges[0]) & (r <= edges[-1])
    count, _ = np.histogram(r, edges)
    weight, _ = np.histogram(r, edges, weights=m * w)
    sums, _ = np.histogram(r, edges, weights=m * w * sv)
    np.testing.assert_array_equal(got["count"], count)
    assert count.sum() == inside.sum() < visible.sum()
    np.testing.assert_allclose(got["weight"], weight, rtol=1e-12)
    np.testing.assert_allclose(got["s"], np.divide(sums, weight, out=np.full(8, np.nan),
                                                   where=weight > 0), rtol=1e-12)
    # Unweighted: by measure; a 2D phase plot against NumPy's.
    got = histogram(octree, ("r", "s"), (edges, np.linspace(np.nanmin(sv), np.nanmax(sv), 6)),
                    values=["w"])
    count2, _, _ = np.histogram2d(r, sv, (edges, np.linspace(np.nanmin(sv), np.nanmax(sv), 6)))
    np.testing.assert_array_equal(got["count"], count2)
    weight2, _, _ = np.histogram2d(r, sv, (edges, np.linspace(np.nanmin(sv), np.nanmax(sv), 6)),
                                   weights=m)
    np.testing.assert_allclose(got["weight"], weight2, rtol=1e-12)
    assert got["w"].shape == (8, 5)


@pytest.mark.parametrize("name,make", CASES_3D, ids=[c[0] for c in CASES_3D])
def test_vector_gradients_are_exact_for_a_linear_field(name, make):
    octree = make()
    a = np.array([[0.3, -1.2, 0.7], [2.0, 0.1, -0.4], [-0.9, 0.5, 1.5]])
    c = octree.centers()
    octree.fields = {"v": tack.array(c @ a.T + [1.0, -2.0, 0.5], tack.vec3(tack.f64)),
                     "s": tack.array(c[:, 0], tack.f64)}
    octree = of.threshold(octree, "s", np.percentile(c[octree.is_leaf, 0], 10), np.inf)
    visible = _visible(octree)[octree.is_leaf]
    g = np.asarray(of.vector_gradients(octree, "v").numpy()).reshape(-1, 3, 3)
    np.testing.assert_allclose(g[visible], np.broadcast_to(a, g[visible].shape), atol=1e-9)
    assert np.isnan(g[~visible]).all() and (~visible).any()
    div = np.asarray(of.divergence(octree, "v").numpy())
    np.testing.assert_allclose(div[visible], np.trace(a), atol=1e-9)
    curl = np.asarray(of.curl(octree, "v").numpy())
    expected = [a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]]
    np.testing.assert_allclose(curl[visible], np.broadcast_to(expected, curl[visible].shape),
                               atol=1e-9)


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_particles_land_in_the_leaves_holding_them(name, make):
    octree = make()
    d = octree.dimension
    lo, size = (np.asarray(x.numpy(), np.float64).reshape(-1, 3) for x in octree.geometry())
    origin = np.asarray(octree.origin)
    rng = np.random.default_rng(5)
    low, high = lo.min(axis=0), (lo + size).max(axis=0)
    low[:d] -= 0.1 * (high - low)[:d]                          # some outside
    x = low + rng.random((3000, 3)) * (high - low)
    if d == 2:
        x[:, 2] = lo[0, 2] + 0.5 * size[0, 2]
    w = rng.random(3000)
    got = np.asarray(deposit(octree, x + origin, w).numpy())
    counts = np.asarray(deposit(octree, x + origin).numpy())
    # Where probe puts each particle (it accepts a point within a tolerance of
    # a boundary), -1 outside.
    found = np.asarray(octree._found(x).numpy())
    expected = np.bincount(found[found >= 0], w[found >= 0], minlength=octree.num_nodes)
    expected_counts = np.bincount(found[found >= 0], minlength=octree.num_nodes)
    held = int((found >= 0).sum())
    assert octree.is_leaf[found[found >= 0]].all()
    assert 0 < held < 3000
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(counts, expected_counts)
    density = np.asarray(deposit(octree, x + origin, w, density=True).numpy())
    np.testing.assert_allclose(density, expected / octree.measures(), rtol=1e-12)
