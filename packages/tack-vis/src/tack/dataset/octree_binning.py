# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Binned reductions of a cell-centred octree (study, ``study/octree-filters``):
profiles and phase plots, as yt makes them. The visible leaves are binned by
one field (a profile) or two (a phase plot), each weighted by its measure
times a weight field (mass, say: density times volume), and each named
field's weighted mean is taken per bin.

On the device: a bin a leaf by a binary search of the edges, the leaves
sorted by bin (a radix sort), each bin's sums by ``reduce_by_key`` in the
sorted order, so the bits do not depend on the launch; the few bins come back
to the host.
"""

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import _compute_type

__all__ = ["histogram"]


@tack.func
def _bin(edges, v):
    # The bin of v among the edges, -1 outside; the last edge's value is in
    # the last bin, as NumPy's histogram has it.
    n = edges.shape[0]
    b = -1
    if v >= edges[0] and v <= edges[n - 1]:
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if v < edges[mid]:
                hi = mid
            else:
                lo = mid
        b = lo
    return b


@tack.kernel
def _bins(first, second, edges1, edges2, two, size, dimension, weight, marks, refined_or_hidden,
          keys, weights):
    # Each visible leaf's bin (-1 elsewhere, or outside the edges), and its
    # weight: its measure times the weight field.
    for n in range(keys.shape[0]):
        keys[n] = tack.i64(-1)
        weights[n] = 0.0 * weight[n]
        if (marks[n] & refined_or_hidden) == 0:
            b = _bin(edges1, first[n])
            if two == 1 and b >= 0:
                c = _bin(edges2, second[n])
                b = b * (edges2.shape[0] - 1) + c if c >= 0 else -1
            if b >= 0:
                measure = size[n][0] * size[n][1]
                if dimension == 3:
                    measure = measure * size[n][2]
                keys[n] = tack.i64(b)
                weights[n] = measure * weight[n]


@tack.kernel
def _weighted(order, values, weights, out):
    for s in range(out.shape[0]):
        n = order[s]
        out[s] = values[n] * weights[n]


def histogram(octree, by, edges, weight=None, values=()):
    """The visible leaves binned by field ``by`` with ``edges`` (ascending; a
    value on the last edge is in the last bin), or by two fields with two
    sets of edges (``by=(a, b)``, ``edges=(ea, eb)``), each leaf weighted by its
    measure times ``weight`` (a field's name; 1 without: volume weighting).
    Returns host arrays, one entry a bin (``(len(ea) - 1, len(eb) - 1)`` for
    two fields): ``count``, the leaves; ``weight``, their weights' sum; and
    each named field's weighted mean (NaN in an empty bin). A profile is one
    field's means over bins of radius, say; a phase plot two fields' bins of
    a weight."""
    from tack.algorithms import argsort, gather, reduce_by_key, unique

    two = isinstance(by, (tuple, list))
    names = list(by) if two else [by]
    if two and len(names) != 2:
        raise ValueError("by is one field's name or two")
    edge_arrays = [np.asarray(e, np.float64) for e in (edges if two else (edges,))]
    for e in edge_arrays:
        if e.ndim != 1 or len(e) < 2 or (np.diff(e) <= 0).any():
            raise ValueError("edges are ascending, two or more")
    shape = tuple(len(e) - 1 for e in edge_arrays)
    bins = int(np.prod(shape))
    n = octree.num_nodes
    dtype = _compute_type(octree.device, tack.f32)
    with octree.device:
        first = octree._field(names[0]).astype(dtype)
        second = octree._field(names[1]).astype(dtype) if two else first
        e1 = tack.array(edge_arrays[0].astype(dtype.numpy_dtype), dtype)
        e2 = tack.array(edge_arrays[1].astype(dtype.numpy_dtype), dtype) if two else e1
        w = tack.ones((n,), dtype) if weight is None else octree._field(weight).astype(dtype)
        _, size = octree.geometry()
        keys = tack.empty((n,), tack.i64)
        weights = tack.empty((n,), dtype)
        count = np.zeros(bins, np.int64)
        total = np.zeros(bins, np.float64)
        sums = {name: np.zeros(bins, np.float64) for name in values}
        if n:
            _bins(first, second, e1, e2, int(two), size, octree.dimension, w, octree.marks,
                  int(Ghosts.REFINEDCELL | Ghosts.HIDDENCELL), keys, weights)
            order = argsort(keys, index_dtype=tack.i64)
            sorted_keys = gather(keys, order)
            distinct, counts = unique(sorted_keys)
            distinct, counts = np.asarray(distinct.numpy()), np.asarray(counts.numpy())
            inside = distinct >= 0
            count[distinct[inside]] = counts[inside]
            total[distinct[inside]] = np.asarray(
                reduce_by_key(sorted_keys, gather(weights, order))[1].numpy())[inside]
            for name in values:
                vw = tack.empty((n,), dtype)
                _weighted(order, octree._field(name).astype(dtype), weights, vw)
                sums[name][distinct[inside]] = np.asarray(
                    reduce_by_key(sorted_keys, vw)[1].numpy())[inside]
    out = {"count": count.reshape(shape), "weight": total.reshape(shape)}
    for name in values:
        out[name] = np.divide(sums[name], total, out=np.full(bins, np.nan),
                              where=total > 0).reshape(shape)
    return out
