# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An octree's topology from its keys (study, ``study/octree-filters``): each
node's parent and the end of its subtree, and the faces between leaves.

Everything is a binary search of the nodes in pre-order, by (tree, key carried
to the finest level, level), on the device; nothing is stored beside the keys.

- **Parent:** the node at (tree, the key's carried prefix, level - 1), which
  exists, and in pre-order comes first among the nodes of that carried key at
  or below its level.
- **Subtree end:** the first node at or after the carried key one subtree
  further on, so ``[n + 1, end)`` are ``n``'s descendants.
- **Face neighbours:** each node's neighbour across each face at its level
  (or the coarser leaf there), carried down from the roots: a child's is its
  sibling, or in its parent's neighbour (``face_neighbours``, no search).
- **Faces:** a leaf's face is recorded when the neighbour across it is as
  large or larger (it then covers the whole face, since boxes are aligned at
  their level); a smaller one finds this leaf from its own side. So every face
  is found once, from its smaller side, and is that side's square; between
  leaves of one size, from the lower side. A face on the domain's boundary has
  no neighbour (-1). (The first form searched the keys for the leaf holding
  the finest cell outside each face: 55 ms of 8 threads at 1M leaves.)
"""

from typing import NamedTuple

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts

__all__ = ["OctreeChildren", "OctreeFaces", "children", "face_neighbours", "faces", "grid_rows",
           "lattice_boxes", "neighbourhoods", "nodes_per_level", "parents", "refined_by_level",
           "root_rows", "subtree_ends"]

#: Above every level, for the last node at a carried key.
_DEEPEST = tack.constant(127, tack.i32)


@tack.func
def _lower_bound(tree, carried, level, count, t, c, lv):
    # The first node whose (tree, carried key, level) is at least (t, c, lv).
    lo = tack.i64(0)
    hi = tack.i64(count)
    while lo < hi:
        mid = (lo + hi) // 2
        if tree[mid] < t or (tree[mid] == t and (carried[mid] < c or (
                carried[mid] == c and level[mid] < lv))):
            lo = mid + 1
        else:
            hi = mid
    return lo


@tack.kernel
def _parents(tree, carried, level, count, dimension, finest, parent):
    for n in range(parent.shape[0]):
        lv = level[n]
        p = tack.i64(-1)
        if lv > 0:
            shift = tack.u64(dimension * (finest - lv + 1))
            c = (carried[n] >> shift) << shift
            p = _lower_bound(tree, carried, level, count, tree[n], c, lv - 1)
        parent[n] = p


@tack.kernel
def _subtree_ends(tree, carried, level, count, dimension, finest, end):
    for n in range(end.shape[0]):
        span = tack.u64(1) << tack.u64(dimension * (finest - level[n]))
        end[n] = _lower_bound(tree, carried, level, count, tree[n], carried[n] + span, 0)


@tack.func
def _interleave(i, j, m, bits, dimension):
    code = tack.u64(0)
    for b in range(bits):
        code = code | (((i >> tack.u64(b)) & tack.u64(1)) << tack.u64(dimension * b))
        code = code | (((j >> tack.u64(b)) & tack.u64(1)) << tack.u64(dimension * b + 1))
        if dimension == 3:
            code = code | (((m >> tack.u64(b)) & tack.u64(1)) << tack.u64(dimension * b + 2))
    return code


@tack.func
def _holder(gi, gj, gk, nx, ny, nz, finest, dimension, tree, carried, level, count):
    # The leaf holding cell (gi, gj, gk) of the finest lattice over every root,
    # or -1 outside the roots.
    found = tack.i64(-1)
    side = tack.i64(1) << tack.i64(finest)
    if gi >= 0 and gj >= 0 and gk >= 0 and gi < nx * side and gj < ny * side and \
            gk < nz * side:
        t = (gi >> tack.i64(finest)) + nx * ((gj >> tack.i64(finest)) +
                                            ny * (gk >> tack.i64(finest)))
        mask = side - 1
        code = _interleave(tack.u64(gi & mask), tack.u64(gj & mask), tack.u64(gk & mask),
                           finest, dimension)
        found = _lower_bound(tree, carried, level, count, t, code, _DEEPEST) - 1
    return found


# Offset o's components in a row of 3^d (x fastest), as 32-bit tables: o % 3 - 1,
# (o // 3) % 3 - 1, o // 9 - 1. A GPU has no integer division (Metal emulates
# 64-bit division, the loops' indices being 64-bit), and these run for every
# entry of every row: on an M1 Max a remesh's rows at 1.9M nodes took about 50
# ms in these kernels with the divisions.
_OX = tack.constant(tuple(o % 3 - 1 for o in range(27)), tack.i32)
_OY = tack.constant(tuple((o // 3) % 3 - 1 for o in range(27)), tack.i32)
_OZ = tack.constant(tuple(o // 9 - 1 for o in range(27)), tack.i32)


@tack.func
def _parent_offset(j, o, dimension):
    # Child j's (its octant's bits, x lowest) neighbour at offset o is in its
    # parent's neighbour at this offset of the parent's row: the child lattice
    # position (u) of the neighbour is -1..2 along each axis, in the parent's
    # neighbour at its half (-1, 0, 0, 1).
    # (u + 2 is 1 to 4, so halving it is a shift.)
    ux = tack.i32(j & 1) + _OX[o]
    uy = tack.i32((j >> 1) & 1) + _OY[o]
    po = ((ux + 2) >> 1) + 3 * ((uy + 2) >> 1)
    if dimension == 3:
        uz = tack.i32((j >> 2) & 1) + _OZ[o]
        po = po + 9 * ((uz + 2) >> 1)
    return po


@tack.func
def _octant(j, o, dimension):
    # ...and, when that neighbour is refined, at this octant of it: the child
    # lattice position's parity.
    ux = tack.i32(j & 1) + _OX[o]
    uy = tack.i32((j >> 1) & 1) + _OY[o]
    c = ((ux + 2) & 1) + 2 * ((uy + 2) & 1)
    if dimension == 3:
        uz = tack.i32((j >> 2) & 1) + _OZ[o]
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
def _root_map(roots, tree, root_of):
    for k in range(roots.shape[0]):
        root_of[tree[roots[k]]] = roots[k]


@tack.kernel
def _grid_rows(roots, tree, root_of, nx, ny, nz, wx, wy, wz, dimension, span, rows):
    # Each root's 3^d neighbours on the roots' grid (x fastest), wrapping along
    # the axes w marks, -1 beyond the grid or where no tree has nodes.
    for k in range(roots.shape[0]):
        t = tack.i32(tree[roots[k]])            # a root's index: 32 bits, a division each
        x = t % nx
        y = (t // nx) % ny
        z = t // (nx * ny)
        for o in range(span):
            a = x + _OX[o]
            b = y + _OY[o]
            c = z
            if dimension == 3:
                c = z + _OZ[o]
            inside = 1
            if wx == 1:
                a = (a + nx) % nx
            elif a < 0 or a >= nx:
                inside = 0
            if wy == 1:
                b = (b + ny) % ny
            elif b < 0 or b >= ny:
                inside = 0
            if wz == 1:
                c = (c + nz) % nz
            elif c < 0 or c >= nz:
                inside = 0
            r = -1
            if inside == 1:
                r = root_of[a + nx * (b + ny * c)]
            rows[k * span + o] = r


@tack.kernel
def _root_neighbourhoods(rows, count, span, centre, out):
    for k in range(count):
        n = rows[k * span + centre]
        for o in range(span):
            out[n * span + o] = rows[k * span + o]


@tack.kernel
def _child_neighbourhoods(nodes, number, table, marks, refined, dimension, span, rows):
    # A refined node's children's rows from its own (as the contour's descent
    # makes them), by node.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        corners = 1 << dimension
        r = (number[n] - 1) * corners
        for j in range(corners):
            c = table[r + j]
            for o in range(span):
                rows[c * span + o] = _descend(rows[n * span + _parent_offset(j, o, dimension)],
                                              _octant(j, o, dimension), dimension, marks,
                                              refined, number, table)


def neighbourhoods(octree):
    """Every node's row: its 3^d neighbours at its level (x fastest, the node in
    the middle), each the node there, or the coarser leaf holding the place,
    or -1 beyond the roots; a device i32 array of ``3^d`` entries a node, from
    the roots' rows down through the children table. Not cached (108 MB at 1M
    leaves): the dual grid builds it and lets it go."""
    d = octree.dimension
    span = 3 ** d
    n = octree.num_nodes
    number, table = children(octree)
    with octree.device:
        out = tack.empty((n * span,), tack.i32)
        if n:
            rows = root_rows(octree)
            _root_neighbourhoods(rows, len(rows) // span, span, span // 2, out)
            for nodes in reversed(refined_by_level(octree)):
                _child_neighbourhoods(nodes, number, table, octree.marks,
                                      int(Ghosts.REFINEDCELL), d, span, out)
    return out


@tack.kernel
def _root_faces(rows, count, span, dimension, out):
    # Each root's 2d face neighbours (axis f // 2, the upper side when f is
    # odd), from its row of 3^d.
    for k in range(count):
        n = rows[k * span + span // 2]
        for f in range(2 * dimension):
            stride = 1
            if f // 2 == 1:
                stride = 3
            if f // 2 == 2:
                stride = 9
            o = span // 2 + stride if f % 2 == 1 else span // 2 - stride
            out[n * (2 * dimension) + f] = rows[k * span + o]


@tack.kernel
def _child_faces(nodes, number, table, marks, refined, dimension, faces):
    # A refined node's children's face neighbours from its own: across the
    # face toward a sibling, the sibling; across the other, the parent's
    # neighbour there -- its child at the mirrored octant when refined, else
    # itself (a coarser leaf, or -1 beyond the roots).
    for k in range(nodes.shape[0]):
        n = nodes[k]
        corners = 1 << dimension
        r = (number[n] - 1) * corners
        for j in range(corners):
            c = table[r + j]
            for f in range(2 * dimension):
                axis = f // 2
                mirrored = j ^ (1 << axis)
                q = table[r + mirrored]
                if ((j >> axis) & 1) == f % 2:
                    q = faces[n * (2 * dimension) + f]
                    if q >= 0:
                        if (marks[q] & refined) != 0:
                            q = table[(number[q] - 1) * corners + mirrored]
                faces[c * (2 * dimension) + f] = q


@tack.kernel
def _leaf_faces(leaf, neighbours, level, marks, refined, dimension, small, large, direction):
    # Slot c * 2d + f: leaf c's face f, or small -1 where the face is found from
    # its other side: a finer neighbour (refined at this level) finds it from
    # its own, an equal one from the lower side.
    for c in range(leaf.shape[0]):
        n = leaf[c]
        lv = level[n]
        for f in range(2 * dimension):
            other = neighbours[n * (2 * dimension) + f]
            slot = c * (2 * dimension) + f
            small[slot] = -1
            if other < 0:
                small[slot] = n
                large[slot] = -1
                direction[slot] = f
            elif (marks[other] & refined) == 0 and (level[other] < lv or
                                                    (level[other] == lv and f % 2 == 1)):
                small[slot] = n
                large[slot] = tack.i64(other)
                direction[slot] = f


@tack.kernel
def _kept(small, flags):
    for s in range(flags.shape[0]):
        flags[s] = 1 if small[s] >= 0 else 0


@tack.kernel
def _compacted(small, large, direction, numbers, out_small, out_large, out_direction):
    for s in range(small.shape[0]):
        if small[s] >= 0:
            k = numbers[s] - 1
            out_small[k] = small[s]
            out_large[k] = large[s]
            out_direction[k] = direction[s]


class OctreeFaces(NamedTuple):
    """The faces between an octree's leaves, each once: ``small`` the node on its
    smaller side (the face is that node's square), ``large`` the node across it
    (-1 on the domain's boundary), ``direction`` the face of ``small`` it is
    (axis ``direction // 2``, the upper side when odd), all device arrays."""

    small: tack.Array
    large: tack.Array
    direction: tack.Array


def _search_arrays(octree):
    return octree.tree, octree._carried(), octree.level, octree.num_nodes


def parents(octree):
    """Each node's parent's index in the nodes' order, -1 at a root: a device i64
    array."""
    with octree.device:
        parent = tack.empty((octree.num_nodes,), tack.i64)
        if octree.num_nodes:
            _parents(*_search_arrays(octree), octree.dimension, octree.finest, parent)
    return parent


def subtree_ends(octree):
    """Where each node's subtree ends: its descendants are the nodes
    ``[n + 1, end[n])``. A device i64 array, made once an octree (it depends on
    the tree alone). Bottom-up, a level at a time from the finest: a leaf's
    subtree ends at the next node, and a refined node's where its last child's
    does, its children found by hopping from the first (the next node) over
    each child's subtree -- 2^d reads, no search. (A binary search for each
    node's end, the first form, took 53 ms of one thread at 1.2M nodes.)"""
    cached = getattr(octree, "_subtree_ends", None)
    if cached is not None:
        return cached
    d = octree.dimension
    with octree.device:
        end = tack.empty((octree.num_nodes,), tack.i64)
        if octree.num_nodes:
            _next_nodes(end)
            for nodes in refined_by_level(octree):
                _last_child_ends(nodes, d, end)
    octree._subtree_ends = end
    return end


def refined_by_level(octree):
    """The refined nodes of each level, finest level first, as device i64 arrays
    (empty levels left out), made once an octree: the order bottom-up passes
    take them in."""
    cached = getattr(octree, "_refined_levels", None)
    if cached is None:
        refined = ~octree.is_leaf
        with octree.device:
            cached = [tack.array(nodes.astype(np.int64)) for nodes in
                      (np.flatnonzero(refined & (octree._level == level))
                       for level in range(octree.finest - 1, -1, -1)) if len(nodes)]
        octree._refined_levels = cached
    return cached


@tack.kernel
def _next_nodes(end):
    for n in range(end.shape[0]):
        end[n] = n + 1


@tack.kernel
def _last_child_ends(nodes, dimension, end):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        c = n + 1
        for j in range(1 << dimension):
            c = end[c]
        end[n] = c


@tack.kernel
def _refined_flags(marks, refined, out):
    for n in range(out.shape[0]):
        out[n] = 1 if (marks[n] & refined) != 0 else 0


@tack.kernel
def _child_table(numbers, marks, refined, end, dimension, children):
    # A refined node's children in key order: the next node, then each child's
    # subtree end.
    for n in range(numbers.shape[0]):
        if (marks[n] & refined) != 0:
            r = numbers[n] - 1
            c = n + 1
            for j in range(1 << dimension):
                children[r * (1 << dimension) + j] = tack.i32(c)
                c = end[c]


class OctreeChildren(NamedTuple):
    """Each refined node's number among the refined nodes, from 1 (``number``,
    i32, one a node, meaningless at leaves) and the children of each refined
    node in key order (``children``, i32, ``2^d`` a refined node): node ``n``'s
    child ``j`` is ``children[(number[n] - 1) * 2^d + j]``."""

    number: tack.Array
    children: tack.Array


def children(octree):
    """The children table of the refined nodes (``OctreeChildren``), made once an
    octree from the subtree ends: a child by its octant in two reads, where
    hopping the ends takes up to ``2^d``. A tenth of the nodes' size in 3D (a
    number a node, eight children a refined node, as i32)."""
    from tack.algorithms import inclusive_scan

    cached = getattr(octree, "_children", None)
    if cached is not None:
        return cached
    n, d = octree.num_nodes, octree.dimension
    refined = int(Ghosts.REFINEDCELL)
    with octree.device:
        number = tack.empty((n,), tack.i32)
        count = 0
        if n:
            flags = tack.empty((n,), tack.i32)
            _refined_flags(octree.marks, refined, flags)
            count = int(inclusive_scan(flags, number, n))
        table = tack.empty((count << d,), tack.i32)
        if count:
            _child_table(number, octree.marks, refined, subtree_ends(octree), d, table)
    octree._children = OctreeChildren(number, table)
    return octree._children


def nodes_per_level(octree):
    """How many nodes each level has, as a host array of ``finest + 2`` counts
    (the last 0), made once an octree: a count over a million levels costs 4
    ms, which the descent paid at every call."""
    cached = getattr(octree, "_nodes_per_level", None)
    if cached is None:
        cached = np.bincount(octree._level, minlength=octree.finest + 2)
        octree._nodes_per_level = cached
    return cached


def root_rows(octree):
    """Each root's neighbourhood: for the nodes of level 0, in their order, the
    root node at each of the 3^d offsets (x fastest, the root itself in the
    middle), -1 beyond the roots' grid or where no tree has nodes; a device i32
    array, made once an octree. The descent starts from these rows."""
    cached = getattr(octree, "_root_rows", None)
    if cached is None:
        cached = octree._root_rows = grid_rows(octree)
    return cached


def grid_rows(octree, periodic=None):
    """``root_rows``, made on the device, wrapping across the roots' grid along
    the axes ``periodic`` (d booleans) marks; not cached."""
    d = octree.dimension
    span = 3 ** d
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    wrap = [int(bool(p)) for p in (tuple(periodic) if periodic is not None else ())]
    wrap = [*wrap, 0, 0, 0][:3]
    roots = np.flatnonzero(octree._level == 0).astype(np.int64)
    with octree.device:
        rows = tack.empty((len(roots) * span,), tack.i32)
        if len(roots):
            roots_d = tack.array(roots, tack.i64)
            root_of = tack.full((nx * ny * nz,), -1, tack.i32)
            _root_map(roots_d, octree.tree, root_of)
            _grid_rows(roots_d, octree.tree, root_of, nx, ny, nz, *wrap, d, span, rows)
    return rows


def _searched_subtree_ends(octree):
    """``subtree_ends`` by a binary search for each node (the first form, kept to
    check the bottom-up one)."""
    with octree.device:
        end = tack.empty((octree.num_nodes,), tack.i64)
        if octree.num_nodes:
            _subtree_ends(*_search_arrays(octree), octree.dimension, octree.finest, end)
    return end


def face_neighbours(octree):
    """Each node's neighbour across each of its 2d faces (axis ``f // 2``, the
    upper side when ``f`` is odd): the node at its level there, or the coarser
    leaf holding the place, or -1 beyond the roots; a device i32 array of
    ``2d`` entries a node, made once an octree, from the roots' rows down a
    level at a time through the children table (a child's neighbour is its
    sibling or in its parent's neighbour), no search."""
    cached = getattr(octree, "_face_neighbours", None)
    if cached is not None:
        return cached
    d = octree.dimension
    n = octree.num_nodes
    number, table = children(octree)
    with octree.device:
        out = tack.empty((n * 2 * d,), tack.i32)
        if n:
            rows = root_rows(octree)
            _root_faces(rows, len(rows) // 3 ** d, 3 ** d, d, out)
            for nodes in reversed(refined_by_level(octree)):
                _child_faces(nodes, number, table, octree.marks, int(Ghosts.REFINEDCELL), d, out)
    octree._face_neighbours = out
    return out


def faces(octree):
    """The faces between the leaves, and the leaves' faces on the domain's
    boundary, each once (``OctreeFaces``), from ``face_neighbours``. Hidden
    leaves are leaves here."""
    from tack.algorithms import inclusive_scan

    d = octree.dimension
    with octree.device:
        leaf = tack.array(np.flatnonzero(octree.is_leaf).astype(np.int64), tack.i64)
        slots = len(leaf) * 2 * d
        small = tack.empty((slots,), tack.i64)
        large = tack.empty((slots,), tack.i64)
        direction = tack.empty((slots,), tack.u8)
        if slots:
            _leaf_faces(leaf, face_neighbours(octree), octree.level, octree.marks,
                        int(Ghosts.REFINEDCELL), d, small, large, direction)
        flags = tack.empty((slots,), tack.i32)
        numbers = tack.empty((slots,), tack.i64)
        total = 0
        if slots:
            _kept(small, flags)
            total = int(inclusive_scan(flags, numbers, slots))
        out = OctreeFaces(tack.empty((total,), tack.i64), tack.empty((total,), tack.i64),
                          tack.empty((total,), tack.u8))
        if total:
            _compacted(small, large, direction, numbers, *out)
    return out


def lattice_boxes(octree):
    """Each node's box on the finest level's lattice over every root, host
    ``(n, 3)`` int64 ``lo`` and ``hi`` (a 2D octree's z from 0 to 1)."""
    d, finest = octree.dimension, octree.finest
    nx, ny, _ = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    tree, level, key = octree._tree, octree._level, octree._key
    index = np.zeros((len(key), 3), np.int64)
    for b in range(int(level.max()) + 1 if len(level) else 0):
        for a in range(d):
            index[:, a] |= ((key >> np.uint64(d * b + a)) & np.uint64(1)).astype(np.int64) << b
    root = np.stack([tree % nx, (tree // nx) % ny, tree // (nx * ny)], axis=1)
    step = (np.int64(1) << (finest - level)).astype(np.int64)
    lo = (root << finest) + index * step[:, None]
    hi = lo + step[:, None]
    if d == 2:
        lo[:, 2], hi[:, 2] = 0, 1
    return lo, hi
