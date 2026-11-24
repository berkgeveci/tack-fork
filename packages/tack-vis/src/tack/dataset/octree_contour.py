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
2. **Descent.** From the roots, a level at a time: a node is kept when the
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
                refined = ~octree.is_leaf
                for level in range(octree.finest - 1, -1, -1):
                    nodes = np.flatnonzero(refined & (octree._level == level))
                    if len(nodes):
                        _child_ranges(tack.array(nodes.astype(np.int64)), self.ends,
                                      self.lo, self.hi)
            self.values = values


def ranges(octree, name):
    """``OctreeRanges`` of field ``name``: computed once, shared by contours."""
    return OctreeRanges(octree, name)


@tack.kernel
def _kept_nodes(frontier, tree, carried, level, key, count, nx, ny, nz, finest, dimension,
                marks, refined, lo, hi, value, keep):
    # 1 for a kept leaf, 2 for a kept refined node, 0 otherwise: kept when the
    # range over the node and its neighbours at its level holds ``value``.
    for k in range(frontier.shape[0]):
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
        keep[k] = 0
        if a <= value and value <= b:
            keep[k] = 2 if (marks[n] & refined) != 0 else 1


@tack.kernel
def _flag(keep, which, flags):
    for k in range(flags.shape[0]):
        flags[k] = 1 if keep[k] == which else 0


@tack.kernel
def _picked(frontier, flags, numbers, out):
    for k in range(flags.shape[0]):
        if flags[k] == 1:
            out[numbers[k] - 1] = frontier[k]


@tack.kernel
def _children(nodes, end, dimension, out):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        c = n + 1
        for j in range(1 << dimension):
            out[k * (1 << dimension) + j] = c
            c = end[c]


def _compact(octree, ids, flags):
    from tack.algorithms import inclusive_scan

    with octree.device:
        numbers = tack.empty((len(ids),), tack.i64)
        total = int(inclusive_scan(flags, numbers, len(ids))) if len(ids) else 0
        out = tack.empty((total,), tack.i64)
        if total:
            _picked(ids, flags, numbers, out)
    return out


def _active_leaves(octree, r, value):
    """The leaves the descent keeps for ``value``: a device array."""
    d = octree.dimension
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    tree, carried, level, count = _search_arrays(octree)
    with octree.device:
        frontier = tack.array(np.flatnonzero(octree._level == 0).astype(np.int64))
        active = []
        while len(frontier):
            keep = tack.empty((len(frontier),), tack.i32)
            _kept_nodes(frontier, tree, carried, level, octree.key, count, nx, ny, nz,
                        octree.finest, d, octree.marks, int(Ghosts.REFINEDCELL), r.lo, r.hi,
                        float(value), keep)
            flags = tack.empty((len(frontier),), tack.i32)
            _flag(keep, 1, flags)
            leaves = _compact(octree, frontier, flags)
            if len(leaves):
                active.append(leaves)
            _flag(keep, 2, flags)
            refined = _compact(octree, frontier, flags)
            frontier = tack.empty((len(refined) << d,), tack.i64)
            if len(refined):
                _children(refined, r.ends, d, frontier)
        if not active:
            return tack.empty((0,), tack.i64)
        return tack.concat(active) if len(active) > 1 else active[0]


@tack.kernel
def _owned_cells(active, tree, carried, level, key, count, nx, ny, nz, finest, dimension,
                 marks, hidden, values, value, cells, keep):
    # Corner v of active leaf a: its cell (the leaves around it, in a
    # hexahedron's order) when the leaf owns it and the cell holds ``value``.
    for e in range(keep.shape[0]):
        corners = 1 << dimension
        a = e // corners
        v = e % corners
        n = active[a]
        lv = level[n]
        t = tree[n]
        i, j, m = _unkey(key[n], lv, dimension)
        step = tack.i64(1) << tack.i64(finest - lv)
        px = ((t % nx) << tack.i64(finest)) + tack.i64(i) * step + tack.i64(v & 1) * step
        py = (((t // nx) % ny) << tack.i64(finest)) + tack.i64(j) * step + \
            tack.i64((v >> 1) & 1) * step
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
    active = _active_leaves(octree, r, value)
    with octree.device:
        entries = len(active) * corners
        cells = tack.empty((entries * corners,), tack.i64)
        keep = tack.empty((entries,), tack.i32)
        kept = 0
        if entries:
            _owned_cells(active, tree, carried, level, octree.key, count, nx, ny, nz,
                         octree.finest, d, octree.marks, int(Ghosts.HIDDENCELL), r.values,
                         float(value), cells, keep)
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
        node_ids = tack.array(np.flatnonzero(used.numpy()).astype(np.int64)) if points else \
            tack.empty((0,), tack.i64)
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
