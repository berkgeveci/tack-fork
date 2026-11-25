# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Particles deposited onto a cell-centred octree (study,
``study/octree-filters``): each particle's weight (its mass, or 1) added to the
leaf holding it, as yt's nearest-grid-point deposition (RAMSES's stars and
dark matter onto the gas grid). The leaves are found as ``probe`` finds them;
the particles are sorted by leaf and each leaf's sum taken in that order
(``reduce_by_key``), so the bits do not depend on the launch.
"""

import numpy as np

import tack
from tack.dataset.octree import _compute_type

__all__ = ["deposit"]


@tack.kernel
def _scatter(keys, sums, out):
    for k in range(keys.shape[0]):
        if keys[k] >= 0:
            out[keys[k]] = sums[k]


def deposit(octree, positions, weights=None, density=False):
    """Each node's sum of the weights of the particles at ``positions`` (an
    ``(n, 3)`` array, world coordinates) that its leaf holds: a device array,
    one entry a node, 0 on refined nodes and where no particle is; the count
    without ``weights``; divided by the leaf's measure with ``density``.
    Particles outside every tree are dropped."""
    from tack.algorithms import argsort, gather, reduce_by_key, unique

    x = np.asarray(positions, np.float64).reshape(-1, 3) - np.asarray(octree.origin, np.float64)
    dtype = _compute_type(octree.device, tack.f32)
    found = octree._found(x)
    n = octree.num_nodes
    with octree.device:
        w = tack.ones((len(x),), dtype) if weights is None else \
            tack.array(np.asarray(weights, dtype.numpy_dtype).reshape(-1), dtype)
        if len(w) != len(x):
            raise ValueError("a weight a particle")
        out = tack.zeros((n,), dtype)
        if len(x):
            order = argsort(found, index_dtype=tack.i64)
            sorted_keys = gather(found, order)
            distinct, _ = unique(sorted_keys)
            _, sums = reduce_by_key(sorted_keys, gather(w, order))
            _scatter(distinct, sums, out)
        if density:
            _, size = octree.geometry()
            measure = tack.empty((n,), dtype)
            _measures(size, octree.dimension, measure)
            out = _divided(out, measure)
    return out


@tack.kernel
def _measures(size, dimension, out):
    for n in range(out.shape[0]):
        m = size[n][0] * size[n][1]
        if dimension == 3:
            m = m * size[n][2]
        out[n] = m


@tack.kernel
def _divide(values, measure, out):
    for n in range(out.shape[0]):
        out[n] = values[n] / measure[n]


def _divided(values, measure):
    out = tack.empty((len(values),), values.dtype)
    if len(values):
        _divide(values, measure, out)
    return out
