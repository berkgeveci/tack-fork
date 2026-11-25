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
from tack.dataset.octree_topology import _interleave, refined_by_level, subtree_ends

__all__ = ["average_up", "axis_slice", "cell_gradients", "covering_grid", "curl", "divergence",
           "extract_box", "limit_depth", "threshold", "vector_gradients"]


@tack.kernel
def _weighted_leaves(values, size, dimension, marks, refined_or_hidden, vm, m):
    # A visible leaf's value times its measure (its sides' product), and its
    # measure; 0 elsewhere.
    for n in range(vm.shape[0]):
        keep = (marks[n] & refined_or_hidden) == 0
        measure = size[n][0] * size[n][1]
        if dimension == 3:
            measure = measure * size[n][2]
        vm[n] = values[n] * measure if keep else 0.0
        m[n] = measure if keep else 0.0


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
        _, size = octree.geometry()
        vm, m, mean = (tack.empty((n,), dtype) for _ in range(3))
        if n:
            _weighted_leaves(values.astype(dtype), size, octree.dimension, octree.marks,
                             int(Ghosts.REFINEDCELL | Ghosts.HIDDENCELL), vm, m)
            for nodes in refined_by_level(octree):
                _child_sums(nodes, ends, vm, m)
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
    hidden = (octree.marks.numpy() & Ghosts.HIDDENCELL) != 0
    ends = subtree_ends(octree)
    fields = {}
    weight = None
    for name, values in octree.fields.items():
        if getattr(values, "_vector_n", None):
            fields[name] = np.asarray(values.numpy())[keep]
            continue
        mean, weight = _subtree_sums(octree, name, ends)
        fields[name] = np.asarray(mean.astype(values.dtype).numpy())[keep]
    if weight is not None:
        hidden = hidden | (np.asarray(weight.numpy()) == 0)
    # The kept nodes are a prefix-closed subset of a pre-order, so pre-ordered
    # and whole: no sort, no check.
    return OctreeDataSet(octree.roots, octree._tree[keep], octree._level[keep],
                         octree._key[keep], fields=fields, mask=hidden[keep],
                         world=octree.world, units=octree.units, _ordered=True)


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


def _face_lists(octree):
    """Each leaf's visible leaves across its faces: ``(leaf, numbers, counts,
    across)`` device arrays, leaf ``c``'s at ``across[numbers[c] - counts[c]:
    numbers[c]]``."""
    from tack.algorithms import inclusive_scan
    from tack.dataset.octree_topology import children, face_neighbours

    if octree.dimension != 3:
        raise ValueError("gradients take a 3D octree (a study)")
    neighbours = face_neighbours(octree)
    number, table = children(octree)
    leaf = np.flatnonzero(octree.is_leaf)
    count = len(leaf)
    refined, hidden = int(Ghosts.REFINEDCELL), int(Ghosts.HIDDENCELL)
    with octree.device:
        leaf = tack.array(leaf.astype(np.int64))
        counts = tack.empty((count,), tack.i32)
        numbers = tack.empty((count,), tack.i64)
        across = tack.empty((1,), tack.i64)
        if count:
            _face_leaves(leaf, neighbours, number, table, octree.marks, refined, hidden, 3, 0,
                         numbers, counts, numbers)
            total = int(inclusive_scan(counts, numbers, count))
            across = tack.empty((max(total, 1),), tack.i64)
            _face_leaves(leaf, neighbours, number, table, octree.marks, refined, hidden, 3, 1,
                         numbers, counts, across)
    return leaf, numbers, counts, across


def cell_gradients(octree, name):
    """The gradient of cell field ``name`` at each visible leaf's centre, by least
    squares over the visible leaves across its faces (from
    ``octree_topology.face_neighbours``), weighted by inverse squared distance:
    exact for a linear field. A device ``vec3`` array, one per leaf in the
    nodes' order (as ``leaves()``); NaN on a hidden leaf, or one whose
    neighbours do not span the space. On the device throughout (the first form
    grouped the face list on the host: 440 ms of the 486 at 1M leaves)."""
    leaf, numbers, counts, across = _face_lists(octree)
    dtype = _compute_type(octree)
    with octree.device:
        out = tack.empty((len(leaf),), tack.vec3(dtype))
        if len(leaf):
            lo, size = octree.geometry()
            _lsq_gradients(leaf, numbers, counts, across, lo, size, octree.marks,
                           int(Ghosts.HIDDENCELL), octree._field(name).astype(dtype),
                           float("nan"), out)
    return out


@tack.kernel
def _lsq_vector_gradients(leaf, numbers, counts, neighbours, lo, size, marks, hidden, values,
                          nan, out):
    # _lsq_gradients for a vector field: one normal matrix, three right-hand
    # sides; row i of the result is the gradient of component i.
    for c in range(leaf.shape[0]):
        n = leaf[c]
        x = lo[n] + size[n] * 0.5
        v = values[n]
        a00 = 0.0 * v[0]
        a01 = a00
        a02 = a00
        a11 = a00
        a12 = a00
        a22 = a00
        b0 = tack.vec3(a00)
        b1 = tack.vec3(a00)
        b2 = tack.vec3(a00)
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
            b0 = b0 + dx * dv[0]
            b1 = b1 + dx * dv[1]
            b2 = b2 + dx * dv[2]
        a = tack.mat3(a00, a01, a02, a01, a11, a12, a02, a12, a22)
        det = a.determinant()
        if det > 0.0 and (marks[n] & hidden) == 0:
            inv = a.inverse()
            g0 = inv @ b0
            g1 = inv @ b1
            g2 = inv @ b2
            out[c] = [g0[0], g0[1], g0[2], g1[0], g1[1], g1[2], g2[0], g2[1], g2[2]]
        else:
            out[c] = [nan, nan, nan, nan, nan, nan, nan, nan, nan]


@tack.kernel
def _divergence(gradients, out):
    for c in range(out.shape[0]):
        g = gradients[c]
        out[c] = g[0] + g[4] + g[8]


@tack.kernel
def _curl(gradients, out):
    for c in range(out.shape[0]):
        g = gradients[c]
        out[c] = tack.vec3(g[7] - g[5], g[2] - g[6], g[3] - g[1])


def vector_gradients(octree, name):
    """The gradient of the 3-vector cell field ``name`` at each visible leaf's
    centre, as ``cell_gradients`` takes a scalar's: a device array of 9
    components a leaf, row-major, entry ``3 * i + j`` the derivative of
    component ``i`` along axis ``j``; NaN on a hidden leaf."""
    leaf, numbers, counts, across = _face_lists(octree)
    dtype = _compute_type(octree)
    values = octree._field_values(name)
    if getattr(values, "_vector_n", None) != 3:
        raise TypeError(f"{name!r} is not a 3-vector field")
    with octree.device:
        out = tack.empty((len(leaf),), tack.vec(9, dtype))
        if len(leaf):
            lo, size = octree.geometry()
            _lsq_vector_gradients(leaf, numbers, counts, across, lo, size, octree.marks,
                                  int(Ghosts.HIDDENCELL), values.astype(tack.vec3(dtype)),
                                  float("nan"), out)
    return out


def divergence(octree, name):
    """The divergence of the 3-vector cell field ``name`` at each visible leaf
    (the trace of ``vector_gradients``): a device array, one per leaf."""
    g = vector_gradients(octree, name)
    with octree.device:
        out = tack.empty((len(g),), _compute_type(octree))
        if len(g):
            _divergence(g, out)
    return out


def curl(octree, name):
    """The curl of the 3-vector cell field ``name`` at each visible leaf (from
    ``vector_gradients``): a device ``vec3`` array, one per leaf."""
    g = vector_gradients(octree, name)
    with octree.device:
        out = tack.empty((len(g),), tack.vec3(_compute_type(octree)))
        if len(g):
            _curl(g, out)
    return out


# -- a region ------------------------------------------------------------------------


@tack.kernel
def _meets_box(lo, size, dimension, x0, y0, z0, x1, y1, z1, out):
    # Whether each node's box meets the open region (touching does not count).
    for n in range(out.shape[0]):
        a = lo[n]
        b = lo[n] + size[n]
        inside = a[0] < x1 and b[0] > x0 and a[1] < y1 and b[1] > y0
        if dimension == 3:
            inside = inside and a[2] < z1 and b[2] > z0
        out[n] = 1 if inside else 0


@tack.kernel
def _keep_children(nodes, number, table, dimension, meets, keep):
    # A refined node that meets the region keeps all its children (so the
    # subset's refined nodes are whole); the ones outside become hidden leaves.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        if meets[n] == 1:
            r = (number[n] - 1) * (1 << dimension)
            for j in range(1 << dimension):
                keep[table[r + j]] = 1


def extract_box(octree, lower, upper):
    """The octree's nodes whose box meets the open box ``[lower, upper)`` (world
    coordinates), as an octree: whole subtrees where they meet it, and, to keep
    every refined node's children, the siblings outside as hidden leaves. The
    values are the nodes' own."""
    from tack.algorithms import gather
    from tack.dataset.octree_topology import children, refined_by_level

    d = octree.dimension
    lower = np.asarray(lower, np.float64) - np.asarray(octree.origin, np.float64)
    upper = np.asarray(upper, np.float64) - np.asarray(octree.origin, np.float64)
    if d == 2:
        lower, upper = np.append(lower[:2], -np.inf), np.append(upper[:2], np.inf)
    number, table = children(octree)
    n = octree.num_nodes
    with octree.device:
        lo, size = octree.geometry()
        meets = tack.empty((n,), tack.i32)
        keep = tack.empty((n,), tack.i32)
        if n:
            _meets_box(lo, size, d, *map(float, lower), *map(float, upper), meets)
            tack.copyto(keep, meets)
            for nodes in refined_by_level(octree):
                _keep_children(nodes, number, table, d, meets, keep)
        keep_host = np.asarray(keep.numpy()) == 1
        meets_host = np.asarray(meets.numpy()) == 1
        held = tack.array(np.flatnonzero(keep_host).astype(np.int64))
        fields = {k: np.asarray(gather(v, held).numpy()) for k, v in octree.fields.items()}
    hidden = ((octree.marks.numpy() & Ghosts.HIDDENCELL) != 0) | ~meets_host
    return OctreeDataSet(octree.roots, octree._tree[keep_host], octree._level[keep_host],
                         octree._key[keep_host], fields=fields, mask=hidden[keep_host],
                         world=octree.world, units=octree.units, _ordered=True)


# -- a covering grid -------------------------------------------------------------------


@tack.kernel
def _cover(tree, level, key, marks, refined, hidden, values, nx, ny, at, dimension, nan,
           out):
    # Each node at level ``at``, or a leaf above it, fills its block of the
    # lattice at level ``at`` with its value (NaN when hidden); x fastest.
    for n in range(tree.shape[0]):
        lv = level[n]
        if lv <= at and (lv == at or (marks[n] & refined) == 0):
            i, j, m = _unkey(key[n], lv, dimension)
            t = tree[n]
            span = 1 << (at - lv)
            side_x = nx << at
            side_y = ny << at
            bi = ((t % nx) << at) + (tack.i64(i) << (at - lv))
            bj = (((t // nx) % ny) << at) + (tack.i64(j) << (at - lv))
            bk = ((t // (nx * ny)) << at) + (tack.i64(m) << (at - lv))
            v = values[n]
            if (marks[n] & hidden) != 0:
                v = nan
            zs = span if dimension == 3 else 1
            for dz in range(zs):
                for dy in range(span):
                    for dx in range(span):
                        out[(bi + dx) + side_x * ((bj + dy) + side_y * (bk + dz))] = v


def covering_grid(octree, level, names=None):
    """The octree sampled on the uniform lattice of ``level`` over all its
    roots, as a grid ``DataSet`` with each named field (by default every
    scalar one) as cell data: a lattice cell takes the value of the node at
    ``level`` holding it, or of the coarser leaf there (``average_up`` first to
    have a refined node's value be its leaves' mean); NaN on hidden leaves and
    where no node is. yt's covering grid."""
    from tack.dataset.dataset import Field, rectilinear_grid
    from tack.dataset.spaces import Constant

    d = octree.dimension
    at = int(level)
    if not 0 <= at <= octree.finest:
        raise ValueError(f"level is 0 to {octree.finest}")
    names = [k for k, v in octree.fields.items() if not getattr(v, "_vector_n", None)] \
        if names is None else list(names)
    axes = _root_axes(octree.roots)
    origin = np.asarray(octree.origin, np.float64)
    fine = []
    for a in range(d):
        line = np.asarray(axes[a], np.float64)
        steps = np.linspace(0.0, 1.0, (1 << at) + 1)[:-1]
        fine.append(np.append((line[:-1, None] + steps[None, :] * np.diff(line)[:, None]).reshape(-1),
                              line[-1]) + origin[a])
    if d == 2:
        fine.append((float(axes[2][0] + origin[2]) if len(axes) > 2 else 0.0,))
    grid = rectilinear_grid(*fine, dtype=octree.roots.dtype, unit=octree.unit)
    nx, ny, nz = ([m - 1 for m in octree.roots.topology.point_dims] + [1])[:3]
    cells = (nx << at) * (ny << at) * (nz << at if d == 3 else 1)
    with octree.device:
        for name in names:
            values = octree._field(name)
            out = tack.full((cells,), float("nan"), values.dtype)
            if octree.num_nodes:
                _cover(octree.tree, octree.level, octree.key, octree.marks,
                       int(Ghosts.REFINEDCELL), int(Ghosts.HIDDENCELL), values, nx, ny, at, d,
                       float("nan"), out)
            grid.fields[name] = Field(Constant(grid.topology), out, octree.units.get(name))
    return grid
