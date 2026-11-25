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
- **Faces:** each leaf looks across each of its faces at the cell of the
  finest lattice just outside the face's lower corner. The leaf holding that
  cell is the neighbour when it is as large or larger (it then covers the whole
  face, since boxes are aligned at their level); a smaller one finds this leaf
  from its own side. So every face is found once, from its smaller side, and is
  that side's square; between leaves of one size, from the lower side. A face
  on the domain's boundary has no neighbour (-1).
"""

from typing import NamedTuple

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import _unkey

__all__ = ["OctreeChildren", "OctreeFaces", "children", "faces", "lattice_boxes",
           "nodes_per_level", "parents", "refined_by_level", "root_rows", "subtree_ends"]

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


@tack.kernel
def _leaf_faces(leaf, tree, carried, level, key, count, nx, ny, nz, dimension, finest,
                small, large, direction):
    # Slot c * 2d + f: leaf c's face f (axis f // 2, the upper side when f is odd),
    # or small -1 where the face is found from its other side.
    for c in range(leaf.shape[0]):
        n = leaf[c]
        t = tree[n]
        lv = level[n]
        i, j, m = _unkey(key[n], lv, dimension)
        step = tack.i64(1) << tack.i64(finest - lv)
        bi = ((t % nx) << tack.i64(finest)) + tack.i64(i) * step
        bj = (((t // nx) % ny) << tack.i64(finest)) + tack.i64(j) * step
        bk = ((t // (nx * ny)) << tack.i64(finest)) + tack.i64(m) * step
        for f in range(2 * dimension):
            axis = f // 2
            upper = f % 2
            gi = bi
            gj = bj
            gk = bk
            if axis == 0:
                gi = bi + step if upper == 1 else bi - 1
            if axis == 1:
                gj = bj + step if upper == 1 else bj - 1
            if axis == 2:
                gk = bk + step if upper == 1 else bk - 1
            other = _holder(gi, gj, gk, nx, ny, nz, finest, dimension, tree, carried, level,
                            count)
            slot = c * (2 * dimension) + f
            small[slot] = -1
            if other < 0:
                small[slot] = n
                large[slot] = -1
                direction[slot] = f
            elif level[other] < lv or (level[other] == lv and upper == 1):
                small[slot] = n
                large[slot] = other
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
    if cached is not None:
        return cached
    d = octree.dimension
    dims = np.array(([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3])
    roots = np.flatnonzero(octree._level == 0)
    root_of = np.full(int(dims.prod()), -1, np.int64)
    root_of[octree._tree[roots]] = roots
    t = octree._tree[roots]
    at = np.stack([t % dims[0], (t // dims[0]) % dims[1], t // (dims[0] * dims[1])], axis=1)
    offsets = np.stack(np.meshgrid(*([np.arange(-1, 2)] * d + [np.zeros(1, np.int64)] * (3 - d)),
                                   indexing="ij"), axis=-1).reshape(-1, 3)
    offsets = offsets[np.lexsort(offsets.T)]                         # x fastest
    q = at[:, None, :] + offsets[None, :, :]
    inside = ((q >= 0) & (q < dims)).all(axis=2)
    q = np.clip(q, 0, dims - 1)
    rows = np.where(inside, root_of[q[..., 0] + dims[0] * (q[..., 1] + dims[1] * q[..., 2])], -1)
    with octree.device:
        octree._root_rows = tack.array(rows.reshape(-1).astype(np.int32), tack.i32)
    return octree._root_rows


def _searched_subtree_ends(octree):
    """``subtree_ends`` by a binary search for each node (the first form, kept to
    check the bottom-up one)."""
    with octree.device:
        end = tack.empty((octree.num_nodes,), tack.i64)
        if octree.num_nodes:
            _subtree_ends(*_search_arrays(octree), octree.dimension, octree.finest, end)
    return end


def faces(octree):
    """The faces between the leaves, and the leaves' faces on the domain's
    boundary, each once (``OctreeFaces``). Hidden leaves are leaves here."""
    from tack.algorithms import inclusive_scan

    d = octree.dimension
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    with octree.device:
        leaf = tack.array(np.flatnonzero(octree.is_leaf).astype(np.int64), tack.i64)
        slots = len(leaf) * 2 * d
        small = tack.empty((slots,), tack.i64)
        large = tack.empty((slots,), tack.i64)
        direction = tack.empty((slots,), tack.u8)
        if slots:
            tree, carried, level, count = _search_arrays(octree)
            _leaf_faces(leaf, tree, carried, level, octree.key, count, nx, ny, nz, d,
                        octree.finest, small, large, direction)
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
