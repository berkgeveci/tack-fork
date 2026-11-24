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
from tack.dataset.octree import OctreeDataSet, _root_axes
from tack.dataset.octree_topology import subtree_ends

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


def _coordinates(octree):
    """Each node's integer coordinates in its tree at its level, ``(n, 3)``."""
    d = octree.dimension
    key, level = octree._key, octree._level
    index = np.zeros((len(key), 3), np.int64)
    for b in range(int(level.max()) + 1 if len(level) else 0):
        for a in range(d):
            index[:, a] |= ((key >> np.uint64(d * b + a)) & np.uint64(1)).astype(np.int64) << b
    return index


def axis_slice(octree, axis, position):
    """The plane at ``position`` along ``axis`` (0, 1, 2) of a 3D octree, as a 2D
    octree: the nodes holding the plane, with their values and marks (a leaf on
    the plane's lower side is taken, as ``probe`` takes it). The 2D octree's x
    and y are the two other axes in order, and its plane's z is ``position``, so
    a slice along z is in place. Host arrays (a study)."""
    from tack.dataset.dataset import rectilinear_grid
    from tack.dataset.octree import _interleave

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
    dims = [n - 1 for n in octree.roots.topology.point_dims]
    tree = octree._tree
    roots_ijk = np.stack([tree % dims[0], (tree // dims[0]) % dims[1],
                          tree // (dims[0] * dims[1])], axis=1)
    level = octree._level
    index = _coordinates(octree)
    keep = (roots_ijk[:, axis] == root) & (index[:, axis] == (g >> (finest - level)))
    tree2 = roots_ijk[keep, u] + dims[u] * roots_ijk[keep, v]
    key2 = _interleave(np.stack([index[keep, u], index[keep, v], np.zeros(keep.sum(), np.int64)],
                                axis=1), 2)
    roots2 = rectilinear_grid(axes[u] + origin[u], axes[v] + origin[v], (float(position),),
                              dtype=octree.roots.dtype, unit=octree.unit)
    fields = {k: np.asarray(f.numpy())[keep] for k, f in octree.fields.items()}
    hidden = (octree.marks.numpy()[keep] & Ghosts.HIDDENCELL) != 0
    return OctreeDataSet(roots2, tree2, level[keep], key2, fields=fields, mask=hidden,
                         world=octree.world, units=octree.units)


@tack.kernel
def _lsq_gradients(leaf, offsets, neighbours, centers, values, nan, out):
    # Least squares over the leaves across each leaf's faces, by inverse squared
    # distance: the gradient a linear field has, exactly.
    for c in range(leaf.shape[0]):
        n = leaf[c]
        x = centers[n]
        v = values[n]
        a00 = 0.0 * v
        a01 = a00
        a02 = a00
        a11 = a00
        a12 = a00
        a22 = a00
        b = tack.vec3(a00)
        for k in range(offsets[c], offsets[c + 1]):
            m = neighbours[k]
            dx = centers[m] - x
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
        if det > 0.0:
            out[c] = a.inverse() @ b
        else:
            out[c] = tack.vec3(nan)


def cell_gradients(octree, name):
    """The gradient of cell field ``name`` at each visible leaf's centre, by least
    squares over the visible leaves across its faces (``octree_topology.faces``),
    weighted by inverse squared distance: exact for a linear field. A device
    ``vec3`` array, one per leaf in the nodes' order (as ``leaves()``); NaN on a
    hidden leaf, or one whose neighbours do not span the space."""
    from tack.dataset.octree_topology import faces

    if octree.dimension != 3:
        raise ValueError("cell_gradients takes a 3D octree (a study)")
    found = faces(octree)
    small, large = (np.asarray(a.numpy()) for a in (found.small, found.large))
    visible = (octree.marks.numpy() & Ghosts.HIDDENCELL) == 0
    pair = (large >= 0) & visible[small] & visible[np.maximum(large, 0)]
    a, b = small[pair], large[pair]
    number = np.cumsum(octree.is_leaf) - 1
    owner = np.concatenate([number[a], number[b]])
    other = np.concatenate([b, a])
    order = np.argsort(owner, kind="stable")
    count = int(octree.is_leaf.sum())
    offsets = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=count))])
    leaf = np.flatnonzero(octree.is_leaf)
    dtype = _compute_type(octree)
    with octree.device:
        out = tack.empty((count,), tack.vec3(dtype))
        if count:
            centers = tack.array(octree.centers().astype(dtype.numpy_dtype), tack.vec3(dtype))
            _lsq_gradients(tack.array(leaf.astype(np.int64)), tack.array(offsets.astype(np.int64)),
                           tack.array(other[order].astype(np.int64)), centers,
                           octree._field(name).astype(dtype), float("nan"), out)
    hidden = ~visible[leaf]
    if hidden.any():
        values = np.asarray(out.numpy())
        values[hidden] = np.nan
        with octree.device:
            out = tack.array(values, tack.vec3(dtype))
    return out
