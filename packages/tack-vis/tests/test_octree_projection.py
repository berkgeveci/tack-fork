# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A 3D octree's field integrated along an axis, as a 2D octree (study): each
column's integral by brute force over the leaves it crosses, and the whole's."""

import numpy as np
import pytest
from test_octree_filters import _with_field
from test_octree_topology import CASES

import tack
import tack.dataset as td
from tack.dataset import octree_filters as of
from tack.dataset.octree_projection import project_along

CASES_3D = [c for c in CASES if "3D" in c[0]]
IDS = [c[0] for c in CASES_3D]


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _columns(octree, axis, p, weight=None):
    """Each point's column integral (or weighted mean), by brute force: every
    visible leaf whose box holds the point on the plane."""
    lo, size = (np.asarray(a.numpy(), np.float64) for a in octree.geometry())
    u, v = [a for a in range(3) if a != axis]
    from tack.dataset.ghosts import Ghosts
    leaf = octree.is_leaf & ((octree.marks.numpy() & Ghosts.HIDDENCELL) == 0)
    s = octree.fields["s"].numpy()
    w = np.ones(len(s)) if weight is None else octree.fields[weight].numpy()
    out = []
    for q in p - np.asarray(octree.origin):
        hit = leaf & (lo[:, u] <= q[u]) & (q[u] < lo[:, u] + size[:, u]) & \
            (lo[:, v] <= q[v]) & (q[v] < lo[:, v] + size[:, v])
        total = np.sum(s[hit] * w[hit] * size[hit, axis])
        with np.errstate(invalid="ignore"):              # a column wholly hidden: NaN
            out.append(total if weight is None else total / np.sum(w[hit] * size[hit, axis]))
    return np.array(out)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("name,make", CASES_3D, ids=IDS)
def test_each_column_is_its_leaves_integral(name, make, axis):
    octree = _with_field(make())
    s = octree.fields["s"].numpy()
    lower, upper = np.nanpercentile(s, [10, 90])
    octree = of.threshold(octree, "s", lower, upper)           # hidden leaves add nothing
    octree.fields["w"] = tack.array(1.0 + octree.centers()[:, 0] ** 2, tack.f64)
    rng = np.random.default_rng(axis)
    axes = td.octree._root_axes(octree.roots)
    lo = np.array([a[0] for a in axes]) + np.asarray(octree.origin)
    hi = np.array([a[-1] for a in axes]) + np.asarray(octree.origin)
    p = lo + (hi - lo) * rng.random((300, 3))
    u, v = [a for a in range(3) if a != axis]
    q = np.stack([p[:, u], p[:, v], np.full(len(p), lo[axis])], axis=1)
    for weight in (None, "w"):
        projected = project_along(octree, "s", axis, weight=weight)
        got = td.probe(projected, q).fields["s"].values.numpy()
        np.testing.assert_allclose(got, _columns(octree, axis, p, weight), rtol=1e-12,
                                   atol=1e-12)
    # The column integrals over the plane are the volume integral.
    projected = project_along(octree, "s", axis)
    area = projected.measures()[projected.is_leaf]
    total = np.sum(projected.fields["s"].numpy()[projected.is_leaf] * area)
    assert np.isclose(total, td.statistics(octree, "s", weights="volume").sum, rtol=1e-12)
