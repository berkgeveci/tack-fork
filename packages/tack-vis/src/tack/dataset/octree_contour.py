# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Contours of a cell-centred octree without its whole dual grid (study,
``study/octree-filters``), as ``vtkHyperTreeGridContour`` makes them: only the
nodes near the surface are visited, and each dual cell is made where it is
needed.

1. **Ranges.** Each node's least and greatest visible leaf value in its
   subtree, a level at a time from the finest (each refined node over its
   children). They depend on the field alone, so every contour value shares
   them.
2. **Descent.** From the roots, a level at a time (on a GPU a thread a
   neighbour's search, on the CPU a thread a node: ``_split_searches``): a node is kept when the
   range over it and its 26 neighbours at its level holds the value (a
   neighbour is the node at that level, or the coarser leaf holding the
   place). A kept refined node's children are the next level's nodes; a kept
   leaf is active. A dual cell's leaves all lie in its finest leaf's
   neighbourhood at that leaf's level, and an ancestor's neighbourhood holds
   its descendants', so every leaf of a cell the surface crosses is kept with
   its ancestors.
3. **Cells.** Each active leaf looks at its corners: the leaves around a corner
   are found by a search of the keys (as ``dual_grid`` finds them), and the
   corner's cell is made by its owner, the finest leaf around it (a corner is
   always a corner of the finest leaves around it), the one in the lowest slot
   on a tie. A cell is kept when every leaf around is inside and visible and
   its values hold the contour value.
4. **The surface.** ``contour`` of those cells alone, on the active leaves'
   centres: the dual grid's contour, polygon for polygon.
"""

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import _compute_type, _unkey
from tack.dataset.octree_topology import (
    _holder,
    _interleave,
    _lower_bound,
    _search_arrays,
    refined_by_level,
    subtree_ends,
)

__all__ = ["OctreeRanges", "contour", "ranges"]


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


@tack.kernel
def _neighbour_ranges(frontier, fcount, entries, tree, carried, level, key, count, nx, ny, nz,
                      finest, dimension, lo, hi, nlo, nhi):
    # Entry e: node e // span's neighbour e % span at its level (x fastest, the
    # node itself in the middle): the range of the node there, or of the coarser
    # leaf holding the place; outside the roots, the node's own. One search a
    # thread, so no thread waits on a chain of 26 (on a GPU the descent's
    # early, small levels were bound by that chain). Launched over a bound; the
    # frontier's count is on the device (``fcount``), and entries past it do
    # nothing.
    for e in range(entries):
        span = 27 if dimension == 3 else 9
        k = e // span
        if k < fcount[0]:
            o = e % span
            n = frontier[k]
            lv = level[n]
            t = tree[n]
            i, j, m = _unkey(key[n], lv, dimension)
            side = tack.i64(1) << tack.i64(lv)
            qi = ((t % nx) << tack.i64(lv)) + tack.i64(i) + tack.i64(o % 3) - 1
            qj = (((t // nx) % ny) << tack.i64(lv)) + tack.i64(j) + tack.i64((o // 3) % 3) - 1
            qk = ((t // (nx * ny)) << tack.i64(lv)) + tack.i64(m)
            if dimension == 3:
                qk = qk + tack.i64(o // 9) - 1
            q = n
            if o != span // 2 and qi >= 0 and qj >= 0 and qk >= 0 and qi < nx * side and \
                    qj < ny * side and qk < nz * side:
                mask = side - 1
                t2 = (qi >> tack.i64(lv)) + nx * ((qj >> tack.i64(lv)) +
                                                  ny * (qk >> tack.i64(lv)))
                code = _interleave(tack.u64(qi & mask), tack.u64(qj & mask),
                                   tack.u64(qk & mask), lv, dimension)
                c = code << tack.u64(dimension * (finest - lv))
                q = _lower_bound(tree, carried, level, count, t2, c, lv + 1) - 1
            nlo[e] = lo[q]
            nhi[e] = hi[q]


@tack.kernel
def _kept_codes(frontier, fcount, bound, nlo, nhi, marks, refined, value, span, packed):
    # A node kept when the range over its neighbourhood holds ``value``: 1 for a
    # leaf, 1 << 32 for a refined node, 0 otherwise (and past the count), so one
    # scan counts both.
    for k in range(bound):
        packed[k] = tack.i64(0)
        if k < fcount[0]:
            a = nlo[k * span]
            b = nhi[k * span]
            for o in range(1, span):
                a = min(a, nlo[k * span + o])
                b = max(b, nhi[k * span + o])
            if a <= value and value <= b:
                packed[k] = tack.i64(1) << tack.i64(32) if (marks[frontier[k]] & refined) != 0 \
                    else tack.i64(1)


@tack.kernel
def _kept_codes_chained(frontier, fcount, bound, tree, carried, level, key, count, nx, ny, nz,
                        finest, dimension, marks, refined, lo, hi, value, packed):
    # The CPU's form of _neighbour_ranges and _kept_codes in one: each thread
    # searches its node's 26 neighbours in turn, the ranges kept in registers.
    for k in range(bound):
        packed[k] = tack.i64(0)
        if k < fcount[0]:
            n = frontier[k]
            lv = level[n]
            t = tree[n]
            i, j, m = _unkey(key[n], lv, dimension)
            gi = ((t % nx) << tack.i64(lv)) + tack.i64(i)
            gj = (((t // nx) % ny) << tack.i64(lv)) + tack.i64(j)
            gk = ((t // (nx * ny)) << tack.i64(lv)) + tack.i64(m)
            side = tack.i64(1) << tack.i64(lv)
            mask = side - 1
            a = lo[n]
            b = hi[n]
            zs = 3 if dimension == 3 else 1
            for dz in range(zs):
                for dy in range(3):
                    for dx in range(3):
                        qi = gi + dx - 1
                        qj = gj + dy - 1
                        qk = gk + dz - 1 if dimension == 3 else gk
                        if qi >= 0 and qj >= 0 and qk >= 0 and qi < nx * side and \
                                qj < ny * side and qk < nz * side:
                            t2 = (qi >> tack.i64(lv)) + nx * ((qj >> tack.i64(lv)) +
                                                              ny * (qk >> tack.i64(lv)))
                            code = _interleave(tack.u64(qi & mask), tack.u64(qj & mask),
                                               tack.u64(qk & mask), lv, dimension)
                            c = code << tack.u64(dimension * (finest - lv))
                            q = _lower_bound(tree, carried, level, count, t2, c, lv + 1) - 1
                            a = min(a, lo[q])
                            b = max(b, hi[q])
            if a <= value and value <= b:
                packed[k] = tack.i64(1) << tack.i64(32) if (marks[n] & refined) != 0 \
                    else tack.i64(1)


def _split_searches(device):
    """Whether the descent and the cells take a thread a search (GPUs: a chain of
    26 or 8 dependent searches a thread left the early, small levels bound by
    memory latency) or a thread a node or corner, the searches in turn (the
    CPU: the split form writes and reads back 27 ranges a node and 64 holders a
    leaf that the chains keep in registers, 15-20% of one thread)."""
    return device.type != "cpu"


#: On a GPU, a level whose bound (its nodes, times 27 neighbours in the split
#: form) is at most this is launched over the bound with its count left on the
#: device; a larger one reads its count first, so as not to launch millions of
#: idle threads. The CPU reads every count (a read costs it nothing).
_READ_ABOVE = 1 << 22

#: On a GPU the descent starts at the first level of at least this many nodes,
#: with every leaf above it: whether a node is kept depends on its own
#: neighbourhood alone (its ancestors' test only saves work), and each level
#: costs a GPU some launches (1-2 ms on an M1 Max) whatever its size. The CPU
#: starts at the roots.
_START_NODES = 1 << 14


@tack.kernel
def _advance(frontier, fcount, bound, packed, numbers, end, dimension, state, active,
             next_frontier):
    # From one scan of the packed counts: a kept leaf goes to ``active`` after the
    # ones before (``state[0]``), a kept refined node's children to the next
    # frontier.
    for k in range(bound):
        if k < fcount[0]:
            p = packed[k]
            if p != 0:
                low = numbers[k] & ((tack.i64(1) << tack.i64(32)) - 1)
                high = numbers[k] >> tack.i64(32)
                n = frontier[k]
                if p == 1:
                    active[state[0] + low - 1] = n
                else:
                    c = n + 1
                    for j in range(1 << dimension):
                        next_frontier[(high - 1) * (1 << dimension) + j] = c
                        c = end[c]


@tack.kernel
def _counts(numbers, fcount, dimension, state, next_fcount):
    # One thread: the level's kept leaves added to ``state[0]``, and the next
    # frontier's count, from the scan's last entry.
    for i in range(1):
        f = fcount[0]
        total = tack.i64(0)
        if f > 0:
            total = numbers[f - 1]
        state[0] = state[0] + (total & ((tack.i64(1) << tack.i64(32)) - 1))
        next_fcount[0] = (total >> tack.i64(32)) << tack.i64(dimension)


def _active_leaves(octree, r, value):
    """The leaves the descent keeps for ``value``: a device array (at least as
    long as their count) and their count. Each level's arrays are sized by a
    bound the host knows (that level's nodes, or the count read back), its count
    kept on the device; on a GPU only levels above ``_READ_ABOVE`` read their
    count, and the descent ends with one read."""
    from tack.algorithms import inclusive_scan

    d = octree.dimension
    span = 27 if d == 3 else 9
    split = _split_searches(octree.device)
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    tree, carried, level, count = _search_arrays(octree)
    finest = octree.finest
    per_level = np.bincount(octree._level, minlength=finest + 2)
    start = 0
    if split:
        deep = np.flatnonzero(per_level[:finest + 1] >= _START_NODES)
        start = int(deep[0]) if len(deep) else finest
    with octree.device:
        first = np.flatnonzero((octree._level == start) |
                               (octree.is_leaf & (octree._level < start))).astype(np.int64)
        frontier = tack.array(first)
        fcount = tack.array(np.array([len(first)], np.int64))
        bound = len(first)
        state = tack.zeros((1,), tack.i64)
        active = tack.empty((max(int(octree.is_leaf.sum()), 1),), tack.i64)
        for lv in range(start, finest + 1):
            if bound == 0:
                break
            packed = tack.empty((bound,), tack.i64)
            if split:
                nlo = tack.empty((bound * span,), r.lo.dtype)
                nhi = tack.empty((bound * span,), r.lo.dtype)
                _neighbour_ranges(frontier, fcount, bound * span, tree, carried, level,
                                  octree.key, count, nx, ny, nz, finest, d, r.lo, r.hi, nlo,
                                  nhi)
                _kept_codes(frontier, fcount, bound, nlo, nhi, octree.marks,
                            int(Ghosts.REFINEDCELL), float(value), span, packed)
            else:
                _kept_codes_chained(frontier, fcount, bound, tree, carried, level, octree.key,
                                    count, nx, ny, nz, finest, d, octree.marks,
                                    int(Ghosts.REFINEDCELL), r.lo, r.hi, float(value), packed)
            numbers = tack.empty((bound,), tack.i64)
            inclusive_scan(packed, numbers, bound, total=False)
            next_bound = int(per_level[lv + 1])
            next_frontier = tack.empty((max(next_bound, 1),), tack.i64)
            _advance(frontier, fcount, bound, packed, numbers, r.ends, d, state, active,
                     next_frontier)
            next_fcount = tack.empty((1,), tack.i64)
            _counts(numbers, fcount, d, state, next_fcount)
            frontier, fcount = next_frontier, next_fcount
            if not split or next_bound * span > _READ_ABOVE:
                next_bound = int(next_fcount.numpy()[0])
            bound = next_bound
        n_active = int(state.numpy()[0])
    return active, n_active


@tack.kernel
def _corner_holders(active, tree, carried, level, key, count, nx, ny, nz, finest, dimension,
                    cells):
    # Entry e: active leaf a's corner v, slot s (a hexahedron's order: x as
    # (s ^ s >> 1) & 1, y as s >> 1 & 1, z as s >> 2 & 1): the leaf holding the
    # lattice cell there, -1 outside. One search a thread.
    for e in range(cells.shape[0]):
        corners = 1 << dimension
        a = e // (corners * corners)
        v = (e // corners) % corners
        s = e % corners
        n = active[a]
        lv = level[n]
        t = tree[n]
        i, j, m = _unkey(key[n], lv, dimension)
        step = tack.i64(1) << tack.i64(finest - lv)
        gi = ((t % nx) << tack.i64(finest)) + (tack.i64(i) + tack.i64(v & 1)) * step + \
            tack.i64((s ^ (s >> 1)) & 1) - 1
        gj = (((t // nx) % ny) << tack.i64(finest)) + \
            (tack.i64(j) + tack.i64((v >> 1) & 1)) * step + tack.i64((s >> 1) & 1) - 1
        gk = ((t // (nx * ny)) << tack.i64(finest)) + tack.i64(m) * step
        if dimension == 3:
            gk = gk + tack.i64((v >> 2) & 1) * step + tack.i64((s >> 2) & 1) - 1
        cells[e] = _holder(gi, gj, gk, nx, ny, nz, finest, dimension, tree, carried, level,
                           count)


@tack.kernel
def _owned_cells(active, cells, level, marks, hidden, values, value, corners, keep):
    # Corner e (leaf e // corners, corner e % corners): kept when its leaf owns
    # it (the finest leaf around, the lowest slot on a tie), every leaf around is
    # inside and visible, and their values hold ``value``.
    for e in range(keep.shape[0]):
        n = active[e // corners]
        ok = 1
        owner = tack.i64(-1)
        deepest = -1
        a_min = values[n]
        a_max = values[n]
        for s in range(corners):
            q = cells[e * corners + s]
            if q < 0:
                ok = 0
            elif (marks[q] & hidden) != 0:
                ok = 0
            else:
                if level[q] > deepest:
                    deepest = level[q]
                    owner = q
                a_min = min(a_min, values[q])
                a_max = max(a_max, values[q])
        keep[e] = 1 if ok == 1 and owner == n and a_min <= value and value <= a_max else 0


@tack.kernel
def _owned_cells_chained(active, tree, carried, level, key, count, nx, ny, nz, finest,
                         dimension, marks, hidden, values, value, cells, keep):
    # The CPU's form of _corner_holders and _owned_cells in one: a thread a
    # corner, its 8 slots searched in turn.
    for e in range(keep.shape[0]):
        corners = 1 << dimension
        n = active[e // corners]
        v = e % corners
        lv = level[n]
        t = tree[n]
        i, j, m = _unkey(key[n], lv, dimension)
        step = tack.i64(1) << tack.i64(finest - lv)
        px = ((t % nx) << tack.i64(finest)) + (tack.i64(i) + tack.i64(v & 1)) * step
        py = (((t // nx) % ny) << tack.i64(finest)) + (tack.i64(j) + tack.i64((v >> 1) & 1)) * step
        pz = ((t // (nx * ny)) << tack.i64(finest)) + tack.i64(m) * step
        if dimension == 3:
            pz = pz + tack.i64((v >> 2) & 1) * step
        ok = 1
        owner = tack.i64(-1)
        deepest = -1
        a_min = values[n]
        a_max = values[n]
        for s in range(corners):
            gi = px + tack.i64((s ^ (s >> 1)) & 1) - 1
            gj = py + tack.i64((s >> 1) & 1) - 1
            gk = pz
            if dimension == 3:
                gk = pz + tack.i64((s >> 2) & 1) - 1
            q = _holder(gi, gj, gk, nx, ny, nz, finest, dimension, tree, carried, level, count)
            cells[e * corners + s] = q
            if q < 0:
                ok = 0
            elif (marks[q] & hidden) != 0:
                ok = 0
            else:
                if level[q] > deepest:
                    deepest = level[q]
                    owner = q
                a_min = min(a_min, values[q])
                a_max = max(a_max, values[q])
        keep[e] = 1 if ok == 1 and owner == n and a_min <= value and value <= a_max else 0


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
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    tree, carried, level, count = _search_arrays(octree)
    dtype = octree.roots.dtype
    active, n_active = _active_leaves(octree, r, value)
    with octree.device:
        entries = n_active * corners
        cells = tack.empty((entries * corners,), tack.i64)
        keep = tack.empty((entries,), tack.i32)
        kept = 0
        if entries and _split_searches(octree.device):
            _corner_holders(active, tree, carried, level, octree.key, count, nx, ny, nz,
                            octree.finest, d, cells)
            _owned_cells(active, cells, level, octree.marks, int(Ghosts.HIDDENCELL), r.values,
                         float(value), corners, keep)
        elif entries:
            _owned_cells_chained(active, tree, carried, level, octree.key, count, nx, ny, nz,
                                 octree.finest, d, octree.marks, int(Ghosts.HIDDENCELL),
                                 r.values, float(value), cells, keep)
        if entries:
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


def contour(octree, name, value, r=None):
    """The contour of cell field ``name`` at ``value``: the dual grid's contour
    (``contour(dual_grid(octree), name, value)``), made from the dual cells the
    surface crosses alone. ``r`` (``ranges(octree, name)``) may be given to share
    the field's ranges between contours."""
    from tack.dataset.algorithms.contour import contour as contour_of

    r = ranges(octree, name) if r is None else r
    return contour_of(crossing_cells(octree, r, value), name, value)
