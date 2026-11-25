# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Contours of a cell-centred octree without its whole dual grid (study,
``study/octree-filters``), as ``vtkHyperTreeGridContour`` makes them: only the
nodes near the surface are visited, and each dual cell is made where it is
needed -- with no search of the keys: each node's neighbourhood is carried down
the tree from its parent's, as VTK's Moore supercursor carries it.

1. **Ranges.** Each node's least and greatest visible leaf value in its
   subtree, a level at a time from the finest (each refined node over its
   children). They depend on the field alone, so every contour value shares
   them.
2. **Descent.** From the roots, a level at a time. The frontier is rows of
   3^d entries: a node's neighbours at its level (x fastest, the node in the
   middle), each the node there, or the coarser leaf holding the place, or -1
   beyond the roots. A node is kept when the range over its row holds the
   value. A kept leaf's row goes to the active rows; a kept refined node's
   children get their rows from its own: a child's neighbour at an offset is
   in the parent's neighbour at the offset's half, that node itself when it is
   a leaf, else its child at the octant the offset lands in (two reads of the
   children table). A dual cell's leaves all lie in its finest leaf's row, and
   an ancestor's row holds its descendants', so every leaf of a cell the
   surface crosses is kept with its ancestors.
3. **Cells.** Each active leaf looks at its corners: the leaves around a
   corner are entries of its row, and the corner's cell is made by its owner,
   the finest leaf around it (a corner is always a corner of the finest leaves
   around it), the one in the lowest slot on a tie. A cell is kept when every
   leaf around is inside and visible and its values hold the contour value.
4. **The surface.** In 3D, marching cubes on those cells alone, in one pass:
   each crossing lies on a dual edge, between two leaves across a face, and is
   named by that face (the finer leaf's, or the lower one's between equals),
   so every cell meeting it writes the same point to the same slot. In 2D,
   ``contour`` of the crossing cells.

On Metal (f32): 8.8 ms at 1M leaves given the ranges, the descent 6.5 of it,
with a thread a child row (``_advance_split``; a thread a node, its eight rows
in turn, was 10.8: a GPU thread's serial chain is the cost at the small
levels, not the launches -- folding the kept test into the advance saved one
launch a level and 0.5 ms, leaving the counts on the device saved nothing).

The first form searched the keys for each neighbour (27 a kept node, 64 an
active leaf, each a binary search of 20 dependent reads): 13 ms of 8 threads
for the descent at 1M leaves, 5 ms more for the cells, 4 ms for the general
contour of the cells, 23.5 ms in all (VTK, one thread: 82 ms). This form, on
yavin's M1 Max with the f64 field: 6.4 ms on 8 threads and 12.8 on one, of
which the ranges 1.8 / 4.2, the descent 2.5 / 3.6 (mostly the launches and
scans of its nine levels: the frontier holds 24k nodes in all) and the
surface 1.6 / 5.0.
"""

import numpy as np

import tack
from tack.dataset import _contour_tables as _ct
from tack.dataset.algorithms._common import _crossing
from tack.dataset.ghosts import Ghosts, _before
from tack.dataset.octree import _compute_type
from tack.dataset.octree_topology import (
    children,
    nodes_per_level,
    refined_by_level,
    root_rows,
    subtree_ends,
)
from tack.dataset.shapes import _HEXAHEDRON_EDGES

__all__ = ["OctreeRanges", "contour", "crossing_cells", "ranges"]


@tack.kernel
def _leaf_ranges(values, marks, refined_or_hidden, big, lo, hi):
    for n in range(lo.shape[0]):
        if (marks[n] & refined_or_hidden) == 0:
            lo[n] = values[n]
            hi[n] = values[n]
        else:
            lo[n] = big
            hi[n] = -big


@tack.kernel
def _child_ranges(nodes, end, lo, hi):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        a = lo[n]
        b = hi[n]
        c = n + 1
        while c < end[n]:
            a = min(a, lo[c])
            b = max(b, hi[c])
            c = end[c]
        lo[n] = a
        hi[n] = b


class OctreeRanges:
    """Each node's least and greatest visible leaf value of a field over its
    subtree (``lo``, ``hi``: device arrays), and the subtree ends, for any
    number of contours of that field."""

    def __init__(self, octree, name):
        self.octree, self.name = octree, name
        n = octree.num_nodes
        self.ends = subtree_ends(octree)
        # f64 where the device has it, else f32 (Metal).
        dtype = _compute_type(octree.device, tack.f32)
        with octree.device:
            values = octree._field(name).astype(dtype)
            self.lo, self.hi = tack.empty((n,), dtype), tack.empty((n,), dtype)
            if n:
                _leaf_ranges(values, octree.marks, int(Ghosts.REFINEDCELL | Ghosts.HIDDENCELL),
                             float(np.finfo(dtype.numpy_dtype).max), self.lo, self.hi)
                for nodes in refined_by_level(octree):
                    _child_ranges(nodes, self.ends, self.lo, self.hi)
            self.values = values


def ranges(octree, name):
    """``OctreeRanges`` of field ``name``: computed once, shared by contours."""
    return OctreeRanges(octree, name)


# -- the descent ---------------------------------------------------------------------


@tack.func
def _parent_offset(j, o, dimension):
    # Child j's (its octant's bits, x lowest) neighbour at offset o is in its
    # parent's neighbour at this offset of the parent's row: the child lattice
    # position (u) of the neighbour is -1..2 along each axis, in the parent's
    # neighbour at its half (-1, 0, 0, 1).
    ux = (j & 1) + o % 3 - 1
    uy = ((j >> 1) & 1) + (o // 3) % 3 - 1
    po = (ux + 2) // 2 + 3 * ((uy + 2) // 2)
    if dimension == 3:
        uz = ((j >> 2) & 1) + o // 9 - 1
        po = po + 9 * ((uz + 2) // 2)
    return po


@tack.func
def _octant(j, o, dimension):
    # ...and, when that neighbour is refined, at this octant of it: the child
    # lattice position's parity.
    ux = (j & 1) + o % 3 - 1
    uy = ((j >> 1) & 1) + (o // 3) % 3 - 1
    c = ((ux + 2) & 1) + 2 * ((uy + 2) & 1)
    if dimension == 3:
        uz = ((j >> 2) & 1) + o // 9 - 1
        c = c + 4 * ((uz + 2) & 1)
    return c


@tack.func
def _descend(p, c, dimension, marks, refined, number, table):
    # Entry p of a parent's row, at a child's: p itself (a leaf, or -1), or its
    # child at octant c.
    if p >= 0:
        if (marks[p] & refined) != 0:
            p = table[(number[p] - 1) * (1 << dimension) + c]
    return p


@tack.kernel
def _kept_rows(rows, fcount, bound, span, lo, hi, marks, refined, value, packed):
    # A node kept when the range over its row holds ``value``: 1 for a leaf,
    # 1 << 32 for a refined node, 0 otherwise (and past the count), so one scan
    # counts both. Launched over a bound; the frontier's count is on the device.
    for k in range(bound):
        packed[k] = tack.i64(0)
        if k < fcount[0]:
            n = rows[k * span + span // 2]
            a = lo[n]
            b = hi[n]
            for o in range(span):
                q = rows[k * span + o]
                if q >= 0:
                    a = min(a, lo[q])
                    b = max(b, hi[q])
            if a <= value and value <= b:
                packed[k] = tack.i64(1) << tack.i64(32) if (marks[n] & refined) != 0 \
                    else tack.i64(1)


@tack.kernel
def _advance_rows(rows, fcount, bound, packed, numbers, span, dimension, marks, refined, number,
                  table, lo, hi, value, fold, state, active, next_rows, next_packed):
    # From one scan of the packed counts: a kept leaf's row goes to ``active``
    # after the ones before (``state[0]``), a kept refined node's children's
    # rows to the next frontier -- each tested as it is written when ``fold``
    # is set (its packed code to ``next_packed``, in place of a kernel a level).
    for k in range(bound):
        if k < fcount[0]:
            p = packed[k]
            if p != 0:
                low = numbers[k] & ((tack.i64(1) << tack.i64(32)) - 1)
                high = numbers[k] >> tack.i64(32)
                if p == 1:
                    base = (state[0] + low - 1) * span
                    for o in range(span):
                        active[base + o] = rows[k * span + o]
                else:
                    corners = 1 << dimension
                    for j in range(corners):
                        at = (high - 1) * corners + j
                        base = at * span
                        for o in range(span):
                            next_rows[base + o] = _descend(
                                rows[k * span + _parent_offset(j, o, dimension)],
                                _octant(j, o, dimension), dimension, marks, refined, number,
                                table)
                        if fold == 1:
                            c = next_rows[base + span // 2]
                            a = lo[c]
                            b = hi[c]
                            for o in range(span):
                                q = next_rows[base + o]
                                if q >= 0:
                                    a = min(a, lo[q])
                                    b = max(b, hi[q])
                            code = tack.i64(0)
                            if a <= value and value <= b:
                                code = tack.i64(1) << tack.i64(32) \
                                    if (marks[c] & refined) != 0 else tack.i64(1)
                            next_packed[at] = code


@tack.kernel
def _advance_split(rows, fcount, bound, packed, numbers, span, dimension, marks, refined, number,
                   table, lo, hi, value, fold, state, active, next_rows, next_packed):
    # _advance_rows with a thread a child: a GPU thread's chain of 2^d rows was
    # the descent's cost (its latency, at the small levels), so each writes and
    # tests one row; a kept leaf's row is copied by its first thread. Launched
    # over bound * 2^d.
    for e in range(bound):
        corners = 1 << dimension
        k = e // corners
        j = e % corners
        if k < fcount[0]:
            p = packed[k]
            if p == 1:
                if j == 0:
                    low = numbers[k] & ((tack.i64(1) << tack.i64(32)) - 1)
                    base = (state[0] + low - 1) * span
                    for o in range(span):
                        active[base + o] = rows[k * span + o]
            elif p != 0:
                high = numbers[k] >> tack.i64(32)
                at = (high - 1) * corners + j
                base = at * span
                for o in range(span):
                    next_rows[base + o] = _descend(
                        rows[k * span + _parent_offset(j, o, dimension)],
                        _octant(j, o, dimension), dimension, marks, refined, number, table)
                if fold == 1:
                    c = next_rows[base + span // 2]
                    a = lo[c]
                    b = hi[c]
                    for o in range(span):
                        q = next_rows[base + o]
                        if q >= 0:
                            a = min(a, lo[q])
                            b = max(b, hi[q])
                    code = tack.i64(0)
                    if a <= value and value <= b:
                        code = tack.i64(1) << tack.i64(32) \
                            if (marks[c] & refined) != 0 else tack.i64(1)
                    next_packed[at] = code


@tack.kernel
def _advance_entries(rows, fcount, bound, packed, numbers, span, dimension, marks, refined,
                     number, table, state, active, next_rows):
    # _advance_rows with a thread an entry of a child's row (and of a kept
    # leaf's copy): no chain at all; the rows are then tested by _kept_rows.
    # Launched over bound * 2^d * span.
    for e in range(bound):
        corners = 1 << dimension
        k = e // (span * corners)
        j = (e // span) % corners
        o = e % span
        if k < fcount[0]:
            p = packed[k]
            if p == 1:
                if j == 0:
                    low = numbers[k] & ((tack.i64(1) << tack.i64(32)) - 1)
                    active[(state[0] + low - 1) * span + o] = rows[k * span + o]
            elif p != 0:
                high = numbers[k] >> tack.i64(32)
                next_rows[((high - 1) * corners + j) * span + o] = _descend(
                    rows[k * span + _parent_offset(j, o, dimension)], _octant(j, o, dimension),
                    dimension, marks, refined, number, table)


@tack.kernel
def _compact_kept(rows, fcount, bound, packed, numbers, span, state, active, next_rows):
    # The two-level descent's start: kept leaves' rows to ``active``, kept
    # refined nodes' own rows to the frontier.
    for k in range(bound):
        if k < fcount[0]:
            p = packed[k]
            if p != 0:
                low = numbers[k] & ((tack.i64(1) << tack.i64(32)) - 1)
                high = numbers[k] >> tack.i64(32)
                base = (state[0] + low - 1) * span if p == 1 else (high - 1) * span
                for o in range(span):
                    if p == 1:
                        active[base + o] = rows[k * span + o]
                    else:
                        next_rows[base + o] = rows[k * span + o]


@tack.kernel
def _step_two(rows, fcount, bound, totals, numbers, span, dimension, marks, refined, number,
              table, lo, hi, value, state, active, next_rows, write):
    # Two levels a step: the frontier's nodes are kept refined nodes; each
    # makes and tests its children's rows (in registers) and its kept refined
    # children's children's; kept leaves go to ``active``, kept refined
    # grandchildren's rows to the next frontier. Twice: counting (``write``
    # 0: a node's counts to ``totals``, scanned between) and writing.
    for k in range(bound):
        if k < fcount[0]:
            na = tack.i64(0)
            nf = tack.i64(0)
            a_at = tack.i64(0)
            f_at = tack.i64(0)
            if write == 1:
                mask = (tack.i64(1) << tack.i64(32)) - 1
                a_at = state[0] + (numbers[k] & mask) - (totals[k] & mask)
                f_at = (numbers[k] >> tack.i64(32)) - (totals[k] >> tack.i64(32))
            corners = 1 << dimension
            crow = tack.local_array(tack.i32, 27)
            grow = tack.local_array(tack.i32, 27)
            for j in range(corners):
                for o in range(span):
                    crow[o] = _descend(rows[k * span + _parent_offset(j, o, dimension)],
                                       _octant(j, o, dimension), dimension, marks, refined,
                                       number, table)
                c = crow[span // 2]
                a = lo[c]
                b = hi[c]
                for o in range(span):
                    q = crow[o]
                    if q >= 0:
                        a = min(a, lo[q])
                        b = max(b, hi[q])
                if a <= value and value <= b:
                    if (marks[c] & refined) == 0:
                        if write == 1:
                            for o in range(span):
                                active[(a_at + na) * span + o] = crow[o]
                        na += 1
                    else:
                        for g in range(corners):
                            for o in range(span):
                                grow[o] = _descend(crow[_parent_offset(g, o, dimension)],
                                                   _octant(g, o, dimension), dimension, marks,
                                                   refined, number, table)
                            gc = grow[span // 2]
                            ga = lo[gc]
                            gb = hi[gc]
                            for o in range(span):
                                q = grow[o]
                                if q >= 0:
                                    ga = min(ga, lo[q])
                                    gb = max(gb, hi[q])
                            if ga <= value and value <= gb:
                                if (marks[gc] & refined) == 0:
                                    if write == 1:
                                        for o in range(span):
                                            active[(a_at + na) * span + o] = grow[o]
                                    na += 1
                                else:
                                    if write == 1:
                                        for o in range(span):
                                            next_rows[(f_at + nf) * span + o] = grow[o]
                                    nf += 1
            if write == 0:
                totals[k] = na | (nf << tack.i64(32))


@tack.kernel
def _counts(numbers, fcount, shift, state, next_fcount):
    # One thread: the level's kept leaves added to ``state[0]``, and the next
    # frontier's count, from the scan's last entry (the kept refined nodes'
    # children, ``shift`` the dimension; or the kept nodes themselves, 0).
    for i in range(1):
        f = fcount[0]
        total = tack.i64(0)
        if f > 0:
            total = numbers[f - 1]
        state[0] = state[0] + (total & ((tack.i64(1) << tack.i64(32)) - 1))
        next_fcount[0] = (total >> tack.i64(32)) << tack.i64(shift)


#: On a GPU, a level whose bound (its nodes) is at most this is launched over the
#: bound with its count left on the device; a larger one reads its count first,
#: so as not to launch, and size its rows for, every node of the level (never
#: reading cost Metal 2.5 ms at 1M leaves; reading below this, nothing). The
#: CPU reads every count (a read costs it nothing).
_READ_ABOVE = 1 << 17

#: Whether a child's row is tested as its parent writes it (its packed code with
#: it), saving the kept-test launch a level (0.3 ms a contour on the CPU, 0.5
#: on Metal); else a kernel a level tests the frontier's rows.
_FOLD_TEST = True

#: Levels a step of the descent: 1, a frontier of rows to test; or 2, a
#: frontier of kept refined nodes, each making and testing its children's and
#: grandchildren's rows in one kernel (run twice, to count and to write), so a
#: GPU pays half the launches and scans. Tried and lost: a thread's chain of
#: 72 rows is what a GPU is slow at (3-4x slower on Metal), and the second pass
#: costs the CPU 1 ms at 1M leaves. Kept as the measurement.
_LEVELS_PER_STEP = 1


def _reads_counts(device, bound):
    return device.type == "cpu" or bound > _READ_ABOVE


def _threads(device):
    """What a thread of the advance does: a node's 2^d children's rows in turn
    (``"node"``, the CPU), one child's row (``"child"``, GPUs) or one entry of
    one (``"entry"``, which forgoes the folded test). Measured on Metal, the
    descent at 1M leaves: a node 10.8 ms, a child 6.5, an entry 19.7 (its 216
    threads a node each divide a 64-bit index twice, emulated, and the kept
    kernel comes back); on the CPU a child costs nothing, an entry 3 ms."""
    return "node" if device.type == "cpu" else "child"


def _active_leaves(octree, r, value):
    """The leaves the descent keeps for ``value``: their rows (a device i32
    array of 3^d entries a leaf, at least as long as their count) and their
    count. Each level's arrays are sized by a bound the host knows (that
    level's nodes, or the count read back), its count kept on the device."""
    from tack.algorithms import inclusive_scan

    d = octree.dimension
    span = 3 ** d
    finest = octree.finest
    per_level = nodes_per_level(octree)
    refined = int(Ghosts.REFINEDCELL)
    number, table = children(octree)
    leaves = octree.num_nodes - (len(table) >> d)
    value = float(value)
    threads = _threads(octree.device)
    fold = _FOLD_TEST and threads != "entry"
    with octree.device:
        rows = root_rows(octree)
        bound = int(per_level[0])
        fcount = tack.array(np.array([bound], np.int64))
        state = tack.zeros((1,), tack.i64)
        active = tack.empty((max(leaves, 1) * span,), tack.i32)
        packed = tack.empty((max(bound, 1),), tack.i64)
        if bound:
            _kept_rows(rows, fcount, bound, span, r.lo, r.hi, octree.marks, refined, value,
                       packed)
        if _LEVELS_PER_STEP == 1:
            for lv in range(finest + 1):
                if bound == 0:
                    break
                numbers = tack.empty((bound,), tack.i64)
                inclusive_scan(packed, numbers, bound, total=False)
                next_bound = int(per_level[lv + 1])
                next_rows = tack.empty((max(next_bound, 1) * span,), tack.i32)
                next_packed = tack.empty((max(next_bound, 1),), tack.i64)
                if threads == "entry":
                    _advance_entries(rows, fcount, (bound << d) * span, packed, numbers, span, d,
                                     octree.marks, refined, number, table, state, active,
                                     next_rows)
                elif threads == "child":
                    _advance_split(rows, fcount, bound << d, packed, numbers, span, d,
                                   octree.marks, refined, number, table, r.lo, r.hi, value,
                                   int(fold), state, active, next_rows, next_packed)
                else:
                    _advance_rows(rows, fcount, bound, packed, numbers, span, d, octree.marks,
                                  refined, number, table, r.lo, r.hi, value, int(fold),
                                  state, active, next_rows, next_packed)
                next_fcount = tack.empty((1,), tack.i64)
                _counts(numbers, fcount, d, state, next_fcount)
                rows, fcount, packed = next_rows, next_fcount, next_packed
                if _reads_counts(octree.device, next_bound):
                    next_bound = int(next_fcount.numpy()[0])
                bound = next_bound
                if bound and not fold:
                    _kept_rows(rows, fcount, bound, span, r.lo, r.hi, octree.marks, refined,
                               value, packed)
        else:
            # The frontier becomes the kept refined roots, then two levels a step.
            if bound:
                numbers = tack.empty((bound,), tack.i64)
                inclusive_scan(packed, numbers, bound, total=False)
                next_rows = tack.empty((bound * span,), tack.i32)
                _compact_kept(rows, fcount, bound, packed, numbers, span, state, active,
                              next_rows)
                next_fcount = tack.empty((1,), tack.i64)
                _counts(numbers, fcount, 0, state, next_fcount)
                rows, fcount = next_rows, next_fcount
                if _reads_counts(octree.device, bound):
                    bound = int(next_fcount.numpy()[0])
            lv = 0
            while bound and lv < finest:
                totals = tack.empty((bound,), tack.i64)
                _step_two(rows, fcount, bound, totals, totals, span, d, octree.marks, refined,
                          number, table, r.lo, r.hi, value, state, active, rows, 0)
                numbers = tack.empty((bound,), tack.i64)
                inclusive_scan(totals, numbers, bound, total=False)
                next_bound = int(per_level[lv + 2]) if lv + 2 <= finest else 0
                next_rows = tack.empty((max(next_bound, 1) * span,), tack.i32)
                _step_two(rows, fcount, bound, totals, numbers, span, d, octree.marks, refined,
                          number, table, r.lo, r.hi, value, state, active, next_rows, 1)
                next_fcount = tack.empty((1,), tack.i64)
                _counts(numbers, fcount, 0, state, next_fcount)
                rows, fcount = next_rows, next_fcount
                if _reads_counts(octree.device, next_bound):
                    next_bound = int(next_fcount.numpy()[0])
                bound = next_bound
                lv += 2
        n_active = int(state.numpy()[0])
    return active, n_active


# -- the cells -----------------------------------------------------------------------


@tack.func
def _slot_offset(v, s, dimension):
    # Corner v's slot s (a hexahedron's order: x as (s ^ s >> 1) & 1, y as
    # s >> 1 & 1, z as s >> 2 & 1) in the leaf's row.
    o = (v & 1) + ((s ^ (s >> 1)) & 1) + 3 * (((v >> 1) & 1) + ((s >> 1) & 1))
    if dimension == 3:
        o = o + 9 * (((v >> 2) & 1) + ((s >> 2) & 1))
    return o


@tack.kernel
def _owned_cells(active, span, dimension, level, marks, refined, hidden, values, value, cells,
                 keep):
    # Corner e (leaf e // corners, corner e % corners): its cell's leaves in a
    # hexahedron's order, kept when its leaf owns it (the finest leaf around:
    # none refined, none as fine in a lower slot), every leaf around is inside
    # and visible, and their values hold ``value``.
    for e in range(keep.shape[0]):
        corners = 1 << dimension
        base = (e // corners) * span
        v = e % corners
        n = active[base + span // 2]
        lv = level[n]
        ok = 1
        seen = 0
        a_min = values[n]
        a_max = values[n]
        for s in range(corners):
            q = active[base + _slot_offset(v, s, dimension)]
            cells[e * corners + s] = tack.i64(q)
            if q < 0:
                ok = 0
            elif (marks[q] & refined) != 0:
                ok = 0
            elif (marks[q] & hidden) != 0:
                ok = 0
            else:
                if level[q] == lv:
                    if q == n:
                        seen = 1
                    elif seen == 0:
                        ok = 0
                a_min = min(a_min, values[q])
                a_max = max(a_max, values[q])
        keep[e] = 1 if ok == 1 and a_min <= value and value <= a_max else 0


@tack.kernel
def _cell_rows(cells, keep, numbers, corners, out):
    for e in range(keep.shape[0]):
        if keep[e] == 1:
            k = numbers[e] - 1
            for s in range(corners):
                out[k * corners + s] = cells[e * corners + s]


@tack.kernel
def _mark_used(rows, used):
    for k in range(rows.shape[0]):
        used[rows[k]] = 1


@tack.kernel
def _used_ids(used, number, out):
    for n in range(used.shape[0]):
        if used[n] == 1:
            out[number[n] - 1] = n


@tack.kernel
def _renumber(rows, number, out):
    for k in range(rows.shape[0]):
        out[k] = number[rows[k]] - 1


@tack.kernel
def _node_centres(nodes, lo, size, out):
    for c in range(nodes.shape[0]):
        n = nodes[c]
        out[c] = lo[n] + size[n] * 0.5


def crossing_cells(octree, r, value):
    """The dual cells the contour at ``value`` crosses, as a ``DataSet`` of
    hexahedra (quads in 2D) on the centres of the leaves they use, the field as
    point data."""
    from tack.algorithms import gather, inclusive_scan
    from tack.dataset import ids, shapes
    from tack.dataset.dataset import DataSet, Field
    from tack.dataset.spaces import H1
    from tack.dataset.topology import UnstructuredTopology, uniform_cells

    d = octree.dimension
    corners = 1 << d
    dtype = octree.roots.dtype
    active, n_active = _active_leaves(octree, r, value)
    with octree.device:
        entries = n_active * corners
        cells = tack.empty((entries * corners,), tack.i64)
        keep = tack.empty((entries,), tack.i32)
        kept = 0
        if entries:
            _owned_cells(active, 3 ** d, d, octree.level, octree.marks, int(Ghosts.REFINEDCELL),
                         int(Ghosts.HIDDENCELL), r.values, float(value), cells, keep)
            numbers = tack.empty((entries,), tack.i64)
            kept = int(inclusive_scan(keep, numbers, entries))
        rows = tack.empty((kept * corners,), tack.i64)
        if kept:
            _cell_rows(cells, keep, numbers, corners, rows)
        # The nodes the cells use, numbered in the nodes' order.
        used = tack.zeros((octree.num_nodes,), tack.i32)
        if kept:
            _mark_used(rows, used)
        number = tack.empty((octree.num_nodes,), tack.i64)
        points = int(inclusive_scan(used, number, octree.num_nodes)) if octree.num_nodes else 0
        node_ids = tack.empty((points,), tack.i64)
        if points:
            _used_ids(used, number, node_ids)
        idt = ids.choose(None, max(points, kept * corners))
        connectivity = tack.empty((kept * corners,), idt)
        if kept:
            _renumber(rows, number, connectivity)
        types, offsets = uniform_cells(shapes.HEXAHEDRON if d == 3 else shapes.QUAD, kept, idt)
        topology = UnstructuredTopology(types, offsets, connectivity, num_points=points)
        lo, size = octree.geometry()
        positions = tack.empty((points,), tack.vec3(dtype))
        if points:
            _node_centres(node_ids, lo, size, positions)
        data = DataSet(topology, positions, dtype=dtype, **octree._frame())
        data.fields[r.name] = Field(H1(data), gather(octree.fields[r.name], node_ids),
                                    octree.units.get(r.name))
    return data


# -- the surface, in 3D ----------------------------------------------------------------

#: Entries a case of VTK's hexahedron table: three corners of up to five triangles.
_HEX_CASE = tack.constant(3 * _ct.HEXAHEDRON_MAX_TRIANGLES, tack.i32)


@tack.kernel
def _active_numbers(active, span, count, out):
    for a in range(count):
        out[active[a * span + span // 2]] = tack.i32(a)


@tack.func
def _edge_ends(active, base, v, e):
    # The leaves at the ends of hexahedron edge e of corner v's cell, the lower
    # along the edge's axis first, and that axis (0, 1, 2).
    ja = _HEXAHEDRON_EDGES[2 * e]
    jb = _HEXAHEDRON_EDGES[2 * e + 1]
    ax = (ja ^ (ja >> 1)) & 1
    ay = (ja >> 1) & 1
    az = (ja >> 2) & 1
    bx = (jb ^ (jb >> 1)) & 1
    by = (jb >> 1) & 1
    bz = (jb >> 2) & 1
    axis = 0 if ax != bx else (1 if ay != by else 2)
    p = active[base + _slot_offset(v, ja, 3)]
    q = active[base + _slot_offset(v, jb, 3)]
    # The two corners differ along the axis alone, so the sums tell the lower.
    if ax + ay + az > bx + by + bz:
        p, q = q, p
    return p, q, axis


@tack.func
def _edge_slot(p, q, axis, level, number):
    # The dual edge between p and q (p the lower along the axis), as the face it
    # crosses: the finer leaf's face toward the other, or the lower's upper face
    # between equals -- one slot a dual edge, among 6 an active leaf.
    return number[p] * 6 + 2 * axis + 1 if level[p] >= level[q] else number[q] * 6 + 2 * axis


@tack.func
def _corner_edge(active, base, v, case, t, i):
    # Corner i of triangle t of the case: its dual edge's ends and axis.
    return _edge_ends(active, base, v, _ct.HEXAHEDRON_EDGES[case * _HEX_CASE + 3 * t + i])


@tack.func
def _triangle_slots(active, base, v, case, t, level, number):
    p0, q0, a0 = _corner_edge(active, base, v, case, t, 0)
    p1, q1, a1 = _corner_edge(active, base, v, case, t, 1)
    p2, q2, a2 = _corner_edge(active, base, v, case, t, 2)
    return _edge_slot(p0, q0, a0, level, number), _edge_slot(p1, q1, a1, level, number), \
        _edge_slot(p2, q2, a2, level, number)


@tack.kernel
def _triangle_counts(active, span, level, marks, refined, hidden, number, values, value, cases,
                     counts):
    # Corner e (leaf e // 8, corner e % 8): its cell's marching-cubes case (a bit
    # a corner at or above ``value``) when its leaf owns the cell and every leaf
    # around is inside, visible and valued (as _owned_cells keeps it), and the
    # case's triangles with three distinct points (a cell's repeated corners, at
    # a hanging corner, collapse some); 0 otherwise.
    for e in range(counts.shape[0]):
        base = (e // 8) * span
        v = e % 8
        n = active[base + span // 2]
        lv = level[n]
        ok = 1
        seen = 0
        case = 0
        for s in range(8):
            q = active[base + _slot_offset(v, s, 3)]
            if q < 0:
                ok = 0
            elif (marks[q] & refined) != 0:
                ok = 0
            elif (marks[q] & hidden) != 0:
                ok = 0
            else:
                if level[q] == lv:
                    if q == n:
                        seen = 1
                    elif seen == 0:
                        ok = 0
                x = values[q]
                if x >= value:
                    case = case | (1 << s)
                if x != x:
                    ok = 0
        if ok == 0:
            case = 0
        count = 0
        for t in range(_ct.HEXAHEDRON_CASES[case]):
            s0, s1, s2 = _triangle_slots(active, base, v, case, t, level, number)
            if s0 != s1 and s1 != s2 and s0 != s2:
                count += 1
        cases[e] = case
        counts[e] = count


@tack.kernel
def _mark_edges(active, span, cases, counts, level, number, used):
    for e in range(counts.shape[0]):
        if counts[e] > 0:
            base = (e // 8) * span
            v = e % 8
            case = cases[e]
            for t in range(_ct.HEXAHEDRON_CASES[case]):
                s0, s1, s2 = _triangle_slots(active, base, v, case, t, level, number)
                if s0 != s1 and s1 != s2 and s0 != s2:
                    used[s0] = 1
                    used[s1] = 1
                    used[s2] = 1


@tack.kernel
def _triangles(active, span, cases, counts, starts, level, number, slots, values, value, lo, size,
               triangles, positions):
    # Each triangle's corners as the points of its dual edges, and each point
    # (every cell meeting it writes the same bits): the crossing interpolated
    # from the end first by position, as ``contour`` interpolates it.
    for e in range(counts.shape[0]):
        if counts[e] > 0:
            base = (e // 8) * span
            v = e % 8
            case = cases[e]
            at = 3 * starts[e]
            for t in range(_ct.HEXAHEDRON_CASES[case]):
                s0, s1, s2 = _triangle_slots(active, base, v, case, t, level, number)
                if s0 != s1 and s1 != s2 and s0 != s2:
                    for i in range(3):
                        p, q, axis = _corner_edge(active, base, v, case, t, i)
                        pid = slots[_edge_slot(p, q, axis, level, number)] - 1
                        triangles[at + i] = pid
                        xp = lo[p] + size[p] * 0.5
                        xq = lo[q] + size[q] * 0.5
                        a = p
                        b = q
                        if _before(xq, xp):
                            a = q
                            b = p
                        xa = lo[a] + size[a] * 0.5
                        xb = lo[b] + size[b] * 0.5
                        w = _crossing(value, values[a], values[b])
                        positions[pid] = xa + w * (xb - xa)
                    at += 3


def _surface(octree, r, value):
    """The contour's triangles from the active leaves' owned cells, in one pass
    (3D)."""
    from tack.algorithms import exclusive_scan, inclusive_scan
    from tack.dataset import ids, shapes
    from tack.dataset.dataset import DataSet, Field
    from tack.dataset.spaces import H1
    from tack.dataset.topology import UnstructuredTopology, uniform_cells

    span = 27
    dtype = octree.roots.dtype
    active, n_active = _active_leaves(octree, r, value)
    with octree.device:
        entries = n_active * 8
        cases = tack.empty((entries,), tack.i32)
        counts = tack.empty((entries,), tack.i32)
        starts = tack.empty((entries,), tack.i64)
        number = tack.empty((octree.num_nodes,), tack.i32)
        used = tack.zeros((n_active * 6,), tack.i32)
        slots = tack.empty((n_active * 6,), tack.i64)
        n_triangles = n_points = 0
        if entries:
            _active_numbers(active, span, n_active, number)
            _triangle_counts(active, span, octree.level, octree.marks, int(Ghosts.REFINEDCELL),
                             int(Ghosts.HIDDENCELL), number, r.values, float(value), cases,
                             counts)
            n_triangles = int(exclusive_scan(counts, starts, entries))
        if n_triangles:
            _mark_edges(active, span, cases, counts, octree.level, number, used)
            n_points = int(inclusive_scan(used, slots, n_active * 6))
        idt = ids.choose(None, max(n_points, 3 * n_triangles))
        triangles = tack.empty((3 * n_triangles,), idt)
        positions = tack.empty((n_points,), tack.vec3(dtype))
        if n_triangles:
            lo, size = octree.geometry()
            _triangles(active, span, cases, counts, starts, octree.level, number, slots, r.values,
                       float(value), lo, size, triangles, positions)
        types, offsets = uniform_cells(shapes.TRIANGLE, n_triangles, idt)
        topology = UnstructuredTopology(types, offsets, triangles, num_points=n_points)
        data = DataSet(topology, positions, dtype=dtype, **octree._frame())
        data.fields[r.name] = Field(H1(data), tack.full((n_points,), float(value), r.values.dtype),
                                    octree.units.get(r.name))
    return data


def contour(octree, name, value, r=None):
    """The contour of cell field ``name`` at ``value``: the dual grid's contour
    (``contour(dual_grid(octree), name, value)``), made from the dual cells the
    surface crosses alone -- in 3D in one pass over them, its points named by
    the faces the dual edges cross. ``r`` (``ranges(octree, name)``) may be
    given to share the field's ranges between contours."""
    from tack.dataset.algorithms.contour import contour as contour_of

    r = ranges(octree, name) if r is None else r
    if octree.dimension == 3:
        return _surface(octree, r, value)
    return contour_of(crossing_cells(octree, r, value), name, value)
