# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pairs of integer boxes that meet, found on the device: what an AMR hierarchy's
marks and checks are made from.

The boxes are host metadata (an ``(n, 2, 3)`` array of ``[lo, hi)`` boxes, which
every rank gathers); everything per pair is on the current device. Both sets of
boxes are put in the bins of a lattice whose cells are the median box of the
second set; the second set's (bin, box) entries are sorted by bin, and each
entry of the first set tests the boxes in its bin. A pair meeting in several
bins is kept in one: the bin of the low corner of where they meet, which both
boxes cover. So no sort or ``unique`` of the pairs is needed, and the pairs come
out grouped by the first set's box (in its bins' order within a box)."""

import numpy as np

import tack
from tack.algorithms import exclusive_scan, sort_by_key

__all__ = ["Pairs", "pairs"]


@tack.func
def _bins_along(lo, hi, k, a, d, origin, size, last):
    """Box ``k``'s first and last bin along axis ``a``; one bin on axes past ``d``."""
    first = tack.i64(0)
    end = tack.i64(0)
    if a < d:
        first = (lo[k, a] - origin) // size
        end = max((hi[k, a] - last - origin) // size, first)
    return first, end


@tack.kernel
def _bin_counts(lo, hi, d, o0, o1, o2, s0, s1, s2, last, counts):
    for k in range(lo.shape[0]):
        f0, e0 = _bins_along(lo, hi, k, 0, d, o0, s0, last)
        f1, e1 = _bins_along(lo, hi, k, 1, d, o1, s1, last)
        f2, e2 = _bins_along(lo, hi, k, 2, d, o2, s2, last)
        counts[k] = (e0 - f0 + 1) * (e1 - f1 + 1) * (e2 - f2 + 1)


@tack.kernel
def _bin_fill(lo, hi, d, o0, o1, o2, s0, s1, s2, last, n0, n1, offsets, bins, owners):
    for k in range(lo.shape[0]):
        f0, e0 = _bins_along(lo, hi, k, 0, d, o0, s0, last)
        f1, e1 = _bins_along(lo, hi, k, 1, d, o1, s1, last)
        f2, e2 = _bins_along(lo, hi, k, 2, d, o2, s2, last)
        at = offsets[k]
        for z in range(f2, e2 + 1):
            for y in range(f1, e1 + 1):
                for x in range(f0, e0 + 1):
                    bins[at] = (z * n1 + y) * n0 + x
                    owners[at] = k
                    at += 1


@tack.func
def _low_bin(alo, blo, i, j, a, d, origin, size):
    b = tack.i64(0)
    if a < d:
        b = (max(alo[i, a], blo[j, a]) - origin) // size
    return b


@tack.func
def _kept(alo, ahi, blo, bhi, i, j, d, closed, distinct, place, o0, o1, o2, s0, s1, s2, n0, n1):
    """Whether boxes ``i`` and ``j`` meet, counted in bin ``place``: the bin of the low
    corner of where they meet."""
    meet = 1
    if distinct != 0 and i == j:
        meet = 0
    for a in range(3):
        if a < d:
            low = max(alo[i, a], blo[j, a])
            high = min(ahi[i, a], bhi[j, a])
            if closed != 0:
                if low > high:
                    meet = 0
            elif low >= high:
                meet = 0
    if meet != 0:
        corner = (_low_bin(alo, blo, i, j, 2, d, o2, s2) * n1
                  + _low_bin(alo, blo, i, j, 1, d, o1, s1)) * n0 \
            + _low_bin(alo, blo, i, j, 0, d, o0, s0)
        if corner != place:
            meet = 0
    return meet


@tack.func
def _first_at_least(sorted_bins, n, place):
    a = tack.i64(0)
    b = n
    while a < b:
        m = (a + b) // 2
        if sorted_bins[m] < place:
            a = m + 1
        else:
            b = m
    return a


@tack.kernel
def _pair_counts(abins, aowners, sbins, sowners, nb, alo, ahi, blo, bhi, d, closed, distinct,
                 o0, o1, o2, s0, s1, s2, n0, n1, counts):
    for e in range(abins.shape[0]):
        place = abins[e]
        i = aowners[e]
        c = tack.i64(0)
        k = _first_at_least(sbins, nb, place)
        while k < nb and sbins[k] == place:
            c += _kept(alo, ahi, blo, bhi, i, sowners[k], d, closed, distinct, place,
                       o0, o1, o2, s0, s1, s2, n0, n1)
            k += 1
        counts[e] = c


@tack.kernel
def _pair_fill(abins, aowners, sbins, sowners, nb, alo, ahi, blo, bhi, d, closed, distinct,
               o0, o1, o2, s0, s1, s2, n0, n1, offsets, first, second):
    for e in range(abins.shape[0]):
        place = abins[e]
        i = aowners[e]
        at = offsets[e]
        k = _first_at_least(sbins, nb, place)
        while k < nb and sbins[k] == place:
            j = sowners[k]
            if _kept(alo, ahi, blo, bhi, i, j, d, closed, distinct, place,
                     o0, o1, o2, s0, s1, s2, n0, n1) != 0:
                first[at] = i
                second[at] = j
                at += 1
            k += 1


@tack.kernel
def _box_offsets(entry_offsets, pair_offsets, out):
    # Box k's pairs start where its first bin's do.
    for k in range(out.shape[0]):
        out[k] = pair_offsets[entry_offsets[k]]


class Pairs:
    """The pairs found: ``first[p]`` meets ``second[p]`` (``count`` of them, device
    arrays of indices into the two sets of boxes), grouped by ``first``:
    ``offsets[k]`` to ``offsets[k + 1]`` are box ``k``'s. ``lo``/``hi`` (``a``'s)
    and ``blo``/``bhi`` are the boxes on the device, ``(n, 3)``."""

    def __init__(self, first, second, count, offsets, lo, hi, blo, bhi):
        self.first, self.second, self.count, self.offsets = first, second, count, offsets
        self.lo, self.hi, self.blo, self.bhi = lo, hi, blo, bhi


def pairs(a, b, d, closed, distinct=False):
    """Every pair of a box of ``a`` and a box of ``b`` (host ``(n, 2, 3)`` integer
    arrays of ``[lo, hi)`` boxes, ``d`` axes used) that share a cell, or with
    ``closed`` a point (``[lo, hi]``); with ``distinct`` not a box with itself (``a``
    and ``b`` the same boxes). On the current device; returns ``Pairs``."""
    a = np.asarray(a, np.int64).reshape(-1, 2, 3)
    b = np.asarray(b, np.int64).reshape(-1, 2, 3)
    lo, hi = tack.array(np.ascontiguousarray(a[:, 0])), tack.array(np.ascontiguousarray(a[:, 1]))
    blo, bhi = tack.array(np.ascontiguousarray(b[:, 0])), tack.array(np.ascontiguousarray(b[:, 1]))
    empty = tack.empty(0, tack.i64)
    if not len(a) or not len(b):
        return Pairs(empty, empty, 0, tack.zeros(len(a) + 1, tack.i64), lo, hi, blo, bhi)
    # The lattice, from the boxes' metadata.
    last = 0 if closed else 1
    size = np.ones(3, np.int64)
    size[:d] = np.maximum(np.median(b[:, 1, :d] - b[:, 0, :d], axis=0).astype(np.int64), 1)
    origin = np.zeros(3, np.int64)
    origin[:d] = np.minimum(a[:, 0, :d].min(axis=0), b[:, 0, :d].min(axis=0))
    top = np.maximum(a[:, 1, :d].max(axis=0), b[:, 1, :d].max(axis=0))
    shape = np.ones(3, np.int64)
    shape[:d] = np.maximum((top - last - origin[:d]) // size[:d], 0) + 1
    lattice = (d, *(int(v) for v in origin), *(int(v) for v in size), last)
    n0, n1 = int(shape[0]), int(shape[1])

    def entries(lo, hi, n):
        offsets = tack.empty(n + 1, tack.i64)
        _bin_counts(lo, hi, *lattice, offsets)
        total = exclusive_scan(offsets, offsets, n, close=True)
        bins = tack.empty(max(total, 1), tack.i64)
        owners = tack.empty(max(total, 1), tack.i64)
        _bin_fill(lo, hi, *lattice, n0, n1, offsets, bins, owners)
        return offsets, bins[:total] if total else empty, owners[:total] if total else empty

    entry_offsets, abins, aowners = entries(lo, hi, len(a))
    _, bbins, bowners = entries(blo, bhi, len(b))
    nb = bbins.shape[0]
    sbins, sowners = sort_by_key(bbins, bowners)
    ne = abins.shape[0]
    tests = (sbins, sowners, nb, lo, hi, blo, bhi, d, int(bool(closed)), int(bool(distinct)),
             *lattice[1:7], n0, n1)
    offsets = tack.empty(ne + 1, tack.i64)
    _pair_counts(abins, aowners, *tests, offsets)
    count = exclusive_scan(offsets, offsets, ne, close=True)
    first = tack.empty(max(count, 1), tack.i64)
    second = tack.empty(max(count, 1), tack.i64)
    _pair_fill(abins, aowners, *tests, offsets, first, second)
    by_box = tack.empty(len(a) + 1, tack.i64)
    _box_offsets(entry_offsets, offsets, by_box)
    return Pairs(first[:count] if count else empty, second[:count] if count else empty, count,
                 by_box, lo, hi, blo, bhi)
