# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Filters that take an octree and give an octree (study,
``study/octree-filters``), nothing interpolated:

- ``average_up``: every node's value as its visible leaves' mean, weighted by
  measure (HyperTreeGrid's ``EvaluateCoarse``): each refined node sums its
  children's sums, a level at a time from the finest. (Prefix sums over the pre-order were tried first: a deep node's
  sum, a difference of two large prefix sums, lost 2e-9 of itself in f64 on a
  10,000-node tree.)
- ``limit_depth``: the nodes down to a level, the new leaves' values averaged
  up (HyperTreeGrid's depth limiter).
- ``threshold``: leaves outside a range hidden, the trees kept (HyperTreeGrid's
  threshold).
- ``axis_slice``: the plane at a coordinate along one axis, as a 2D octree. The
  plane meets one node in each column of every level, so the nodes that hold it
  are a quadtree, each keyed by its key less that axis's bits.
"""

import copy

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet, _root_axes, _unkey
from tack.dataset.octree_topology import _interleave, subtree_ends

__all__ = ["average_up", "axis_slice", "cell_gradients", "limit_depth", "threshold"]


@tack.kernel
def _weighted_leaves(values, measure, marks, refined_or_hidden, vm, m):
    # A visible leaf's value times its measure, and its measure; 0 elsewhere.
    for n in range(vm.shape[0]):
        keep = (marks[n] & refined_or_hidden) == 0
        vm[n] = values[n] * measure[n] if keep else 0.0
        m[n] = measure[n] if keep else 0.0


@tack.kernel
def _child_sums(nodes, end, vm, m):
    # A refined node's sums, its children's: the first child follows it, and each
    # child's subtree ends where the next child starts. In the children's order,
    # so the bits do not depend on the launch.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        s = vm[n] * 0.0
        w = m[n] * 0.0
        c = n + 1
        while c < end[n]:
            s = s + vm[c]
            w = w + m[c]
            c = end[c]
        vm[n] = s
        m[n] = w


@tack.kernel
def _means(vm, m, nan, out):
    for n in range(out.shape[0]):
        out[n] = vm[n] / m[n] if m[n] > 0.0 else nan


def _compute_type(octree):
    from tack.dataset.octree import _compute_type as compute_type
    return compute_type(octree.device, tack.f32)


def _subtree_sums(octree, name, ends=None):
    """Each node's visible leaves' mean of ``name`` weighted by measure (NaN where
    none is visible), and their measure: device arrays. Level by level from the
    finest, each refined node summing its children's sums."""
    values = octree._field(name)
    n = octree.num_nodes
    dtype = _compute_type(octree)
    ends = subtree_ends(octree) if ends is None else ends
    with octree.device:
        measure = tack.array(octree.measures().astype(dtype.numpy_dtype), dtype)
        vm, m, mean = (tack.empty((n,), dtype) for _ in range(3))
        if n:
            _weighted_leaves(values.astype(dtype), measure, octree.marks,
                             int(Ghosts.REFINEDCELL | Ghosts.HIDDENCELL), vm, m)
            refined = ~octree.is_leaf
            for level in range(octree.finest - 1, -1, -1):
                nodes = np.flatnonzero(refined & (octree._level == level))
                if len(nodes):
                    _child_sums(tack.array(nodes.astype(np.int64), tack.i64), ends, vm, m)
            _means(vm, m, float("nan"), mean)
    return mean, m


def _with(octree, fields=None, marks=None):
    """A copy of ``octree`` sharing its nodes, with other fields or marks."""
    out = copy.copy(octree)
    out.fields = dict(octree.fields if fields is None else fields)
    if marks is not None:
        out.marks = marks
    return out


def average_up(octree, names=None):
    """The octree with each named field's (by default every scalar field's) value
    on every node replaced by its visible leaves' mean, weighted by measure; a
    leaf keeps its own, a node with no visible leaf gets NaN. Each node sums its
    children's sums, finest level first, so the bits are the same however the
    launches go."""
    names = [k for k, v in octree.fields.items() if not getattr(v, "_vector_n", None)] \
        if names is None else list(names)
    ends = subtree_ends(octree)
    fields = dict(octree.fields)
    for name in names:
        mean, _ = _subtree_sums(octree, name, ends)
        fields[name] = mean.astype(octree.fields[name].dtype)
    return _with(octree, fields=fields)


def limit_depth(octree, depth):
    """The nodes down to level ``depth``, each field averaged up (``average_up``)
    so a node refined below ``depth`` becomes a leaf of its leaves' mean; one
    whose leaves are all hidden is hidden."""
    keep = octree._level <= int(depth)
    averaged = average_up(octree)
    hidden = (octree.marks.numpy() & Ghosts.HIDDENCELL) != 0
    names = [k for k, v in octree.fields.items() if not getattr(v, "_vector_n", None)]
    if names:
        _, weight = _subtree_sums(octree, names[0])
        hidden = hidden | (np.asarray(weight.numpy()) == 0)
    fields = {k: np.asarray(v.numpy())[keep] for k, v in averaged.fields.items()}
    return OctreeDataSet(octree.roots, octree._tree[keep], octree._level[keep],
                         octree._key[keep], fields=fields, mask=hidden[keep],
                         world=octree.world, units=octree.units)


@tack.kernel
def _hide_outside(values, leaf, marks, lower, upper, hidden, out):
    # A leaf outside [lower, upper] is hidden; refined nodes keep their marks.
    for n in range(out.shape[0]):
        v = values[n]
        out[n] = marks[n] | hidden if leaf[n] != 0 and (v < lower or v > upper) else marks[n]


def threshold(octree, name, lower, upper):
    """The octree with the leaves whose ``name`` is outside ``[lower, upper]``
    hidden (``HIDDENCELL``); the trees are kept, as are the other nodes."""
    values = octree._field(name)
    with octree.device:
        marks = tack.empty((octree.num_nodes,), tack.u8)
        if octree.num_nodes:
            leaf = tack.array(octree.is_leaf.astype(np.uint8), tack.u8)
            _hide_outside(values, leaf, octree.marks, float(lower), float(upper),
                          int(Ghosts.HIDDENCELL), marks)
    return _with(octree, marks=marks)


@tack.kernel
def _dropped(tree, level, key, nx, ny, axis, finest, out_tree, out_level, out_key):
    # Each node dropped onto the plane of the other two axes: its root column,
    # level, and 2D key (the other two axes' bits, in order).
    for n in range(tree.shape[0]):
        lv = level[n]
        i, j, m = _unkey(key[n], lv, 3)
        t = tree[n]
        ti = t % nx
        tj = (t // nx) % ny
        tk = t // (nx * ny)
        if axis == 0:
            out_tree[n] = tj + ny * tk
            out_key[n] = _interleave(j, m, tack.u64(0), lv, 2)
        elif axis == 1:
            out_tree[n] = ti + nx * tk
            out_key[n] = _interleave(i, m, tack.u64(0), lv, 2)
        else:
            out_tree[n] = ti + nx * tj
            out_key[n] = _interleave(i, j, tack.u64(0), lv, 2)
        out_level[n] = tack.i64(lv)


@tack.kernel
def _on_plane(tree, level, key, nx, ny, axis, root, g, finest, flags):
    # The nodes holding the plane: in root ``root`` along the axis, their cell
    # there holding the finest lattice's cell ``g``.
    for n in range(flags.shape[0]):
        lv = level[n]
        i, j, m = _unkey(key[n], lv, 3)
        t = tree[n]
        r = t % nx
        c = i
        if axis == 1:
            r = (t // nx) % ny
            c = j
        if axis == 2:
            r = t // (nx * ny)
            c = m
        flags[n] = 1 if r == root and tack.i64(c) == (g >> (finest - lv)) else 0


@tack.kernel
def _flagged(flags, numbers, out):
    for n in range(flags.shape[0]):
        if flags[n] == 1:
            out[numbers[n] - 1] = n


def axis_slice(octree, axis, position):
    """The plane at ``position`` along ``axis`` (0, 1, 2) of a 3D octree, as a 2D
    octree: the nodes holding the plane, with their values and marks (a leaf on
    the plane's lower side is taken, as ``probe`` takes it). The 2D octree's x
    and y are the two other axes in order, and its plane's z is ``position``, so
    a slice along z is in place. The nodes are found and dropped onto the plane
    on the device; the 2D octree is made from the few that hold it."""
    from tack.algorithms import gather, inclusive_scan
    from tack.dataset.dataset import rectilinear_grid

    if octree.dimension != 3:
        raise ValueError("axis_slice takes a 3D octree")
    axis = int(axis)
    u, v = [a for a in range(3) if a != axis]
    axes = _root_axes(octree.roots)
    origin = np.asarray(octree.origin, np.float64)
    x = float(position) - origin[axis]
    lines = axes[axis]
    if not lines[0] <= x <= lines[-1]:
        raise ValueError(f"the plane at {position} misses the roots along axis {axis}")
    root = int(np.clip(np.searchsorted(lines, x, side="right") - 1, 0, len(lines) - 2))
    finest = octree.finest
    g = int(np.clip(np.floor((x - lines[root]) / (lines[root + 1] - lines[root]) * (1 << finest)),
                    0, (1 << finest) - 1))
    nx, ny, nz = [m - 1 for m in octree.roots.topology.point_dims]
    n = octree.num_nodes
    with octree.device:
        flags = tack.empty((n,), tack.i32)
        numbers = tack.empty((n,), tack.i64)
        _on_plane(octree.tree, octree.level, octree.key, nx, ny, axis, root, g, finest, flags)
        count = int(inclusive_scan(flags, numbers, n)) if n else 0
        held = tack.empty((count,), tack.i64)
        if count:
            _flagged(flags, numbers, held)
        tree2, level2, key2 = (tack.empty((n,), t) for t in (tack.i64, tack.i64, tack.u64))
        _dropped(octree.tree, octree.level, octree.key, nx, ny, axis, finest, tree2, level2, key2)
        tree2, level2, key2 = (gather(a, held).numpy() for a in (tree2, level2, key2))
        fields = {k: gather(f, held).numpy() for k, f in octree.fields.items()}
        hidden = (np.asarray(gather(octree.marks, held).numpy()) & Ghosts.HIDDENCELL) != 0
    roots2 = rectilinear_grid(axes[u] + origin[u], axes[v] + origin[v], (float(position),),
                              dtype=octree.roots.dtype, unit=octree.unit)
    return OctreeDataSet(roots2, tree2, level2, key2, fields=fields, mask=hidden,
                         world=octree.world, units=octree.units)


@tack.kernel
def _face_leaves(leaf, neighbours, number, table, marks, refined, hidden, dimension, write,
                 numbers, counts, out):
    # The visible leaves across each leaf's faces: a neighbour that is a leaf,
    # or the leaves of a refined neighbour's subtree on the facing side (its
    # children with the mirrored octant bit, down to the leaves, by a stack).
    # Run twice: to count (``write`` 0, to ``counts``) and, after a scan, to
    # write them at ``numbers[c] - counts[c]``.
    for c in range(leaf.shape[0]):
        n = leaf[c]
        k = tack.i64(0)
        at = tack.i64(0)
        if write == 1:
            at = numbers[c] - counts[c]
        stack = tack.local_array(tack.i32, 128)
        corners = 1 << dimension
        for f in range(2 * dimension):
            axis = f // 2
            facing = 1 - f % 2
            q = neighbours[n * (2 * dimension) + f]
            if q >= 0:
                stack[0] = q
                top = 1
                while top > 0:
                    top -= 1
                    p = stack[top]
                    if (marks[p] & refined) != 0:
                        r = (number[p] - 1) * corners
                        for j in range(corners):
                            if ((j >> axis) & 1) == facing:
                                stack[top] = table[r + j]
                                top += 1
                    elif (marks[p] & hidden) == 0:
                        if write == 1:
                            out[at + k] = tack.i64(p)
                        k += 1
        if write == 0:
            counts[c] = tack.i32(k)


@tack.kernel
def _lsq_gradients(leaf, numbers, counts, neighbours, lo, size, marks, hidden, values, nan, out):
    # Least squares over the leaves across each leaf's faces, by inverse squared
    # distance: the gradient a linear field has, exactly. NaN on a hidden leaf.
    for c in range(leaf.shape[0]):
        n = leaf[c]
        x = lo[n] + size[n] * 0.5
        v = values[n]
        a00 = 0.0 * v
        a01 = a00
        a02 = a00
        a11 = a00
        a12 = a00
        a22 = a00
        b = tack.vec3(a00)
        for k in range(numbers[c] - counts[c], numbers[c]):
            m = neighbours[k]
            dx = lo[m] + size[m] * 0.5 - x
            w = 1.0 / dx.dot(dx)
            dv = (values[m] - v) * w
            a00 = a00 + w * dx[0] * dx[0]
            a01 = a01 + w * dx[0] * dx[1]
            a02 = a02 + w * dx[0] * dx[2]
            a11 = a11 + w * dx[1] * dx[1]
            a12 = a12 + w * dx[1] * dx[2]
            a22 = a22 + w * dx[2] * dx[2]
            b = b + dx * dv
        a = tack.mat3(a00, a01, a02, a01, a11, a12, a02, a12, a22)
        det = a.determinant()
        if det > 0.0 and (marks[n] & hidden) == 0:
            out[c] = a.inverse() @ b
        else:
            out[c] = tack.vec3(nan)


def cell_gradients(octree, name):
    """The gradient of cell field ``name`` at each visible leaf's centre, by least
    squares over the visible leaves across its faces (from
    ``octree_topology.face_neighbours``), weighted by inverse squared distance:
    exact for a linear field. A device ``vec3`` array, one per leaf in the
    nodes' order (as ``leaves()``); NaN on a hidden leaf, or one whose
    neighbours do not span the space. On the device throughout (the first form
    grouped the face list on the host: 440 ms of the 486 at 1M leaves)."""
    from tack.algorithms import inclusive_scan
    from tack.dataset.octree_topology import children, face_neighbours

    if octree.dimension != 3:
        raise ValueError("cell_gradients takes a 3D octree (a study)")
    neighbours = face_neighbours(octree)
    number, table = children(octree)
    leaf = np.flatnonzero(octree.is_leaf)
    count = len(leaf)
    dtype = _compute_type(octree)
    refined, hidden = int(Ghosts.REFINEDCELL), int(Ghosts.HIDDENCELL)
    with octree.device:
        out = tack.empty((count,), tack.vec3(dtype))
        if count:
            leaf = tack.array(leaf.astype(np.int64))
            counts = tack.empty((count,), tack.i32)
            numbers = tack.empty((count,), tack.i64)
            _face_leaves(leaf, neighbours, number, table, octree.marks, refined, hidden, 3, 0,
                         numbers, counts, numbers)
            total = int(inclusive_scan(counts, numbers, count))
            across = tack.empty((max(total, 1),), tack.i64)
            _face_leaves(leaf, neighbours, number, table, octree.marks, refined, hidden, 3, 1,
                         numbers, counts, across)
            lo, size = octree.geometry()
            _lsq_gradients(leaf, numbers, counts, across, lo, size, octree.marks, hidden,
                           octree._field(name).astype(dtype), float("nan"), out)
    return out
