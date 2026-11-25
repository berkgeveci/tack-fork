# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A 3D octree's cell field integrated along an axis, as a 2D octree (study,
``study/octree-filters``): a column density, as yt projects RAMSES data, exact
and as adaptive as the columns.

Each node of the 3D octree, dropped onto the plane of the other two axes, is a
node of a quadtree: its key less that axis's bits, its level, the root column
it is in. Dropping every node gives a whole quadtree (a node's ancestors are
nodes, and a refined node's 2^3 children cover its 2^2 children on the plane),
whose leaves are where the columns are finest. Each visible 3D leaf adds its
value times its length along the axis to the quadtree node it drops to; a
quadtree leaf's column is the sum over itself and its ancestors. Refined
quadtree nodes get their leaves' mean (``average_up``).

With ``weight``, the column's mean weighted by that field along the line of
sight, ``sum(value * weight * length) / sum(weight * length)`` (yt's weighted
projection). Grouped on the device by a sort (the host's ``np.unique`` took
1.2 s at 1M leaves).
"""

import numpy as np

import tack
from tack.dataset import units as _units
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet, _compute_type, _root_axes, _run_starts
from tack.dataset.octree_filters import _dropped, average_up
from tack.dataset.octree_topology import children, refined_by_level

__all__ = ["project_along"]


@tack.kernel
def _contributions(order, marks, refined_or_hidden, values, weights, size, axis, sums, wsums):
    # Sorted entry s: its node's value times weight times length along the
    # axis, and weight times length, for a visible leaf; 0 otherwise.
    for s in range(order.shape[0]):
        n = order[s]
        sums[s] = 0.0
        wsums[s] = 0.0
        if (marks[n] & refined_or_hidden) == 0:
            length = size[n][axis]
            sums[s] = values[n] * weights[n] * length
            wsums[s] = weights[n] * length


@tack.kernel
def _runs(flags, numbers, st, sl, sk, count, out_tree, out_level, out_key, offsets):
    # Each run's first entry: its 2D node, and where it starts.
    for s in range(flags.shape[0]):
        if flags[s] == 1:
            k = numbers[s] - 1
            out_tree[k] = st[s]
            out_level[k] = sl[s]
            out_key[k] = sk[s]
            offsets[k] = s
        if s == 0:
            offsets[count] = flags.shape[0]


@tack.kernel
def _down_to_children(nodes, number, table, dimension, total, wtotal):
    # A refined node's column so far added to its children's.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        r = (number[n] - 1) * (1 << dimension)
        for j in range(1 << dimension):
            c = table[r + j]
            total[c] = total[c] + total[n]
            wtotal[c] = wtotal[c] + wtotal[n]


@tack.kernel
def _column_values(total, wtotal, marks, refined, weighted, nan, out):
    for n in range(out.shape[0]):
        out[n] = nan
        if (marks[n] & refined) == 0:
            if weighted == 0:
                out[n] = total[n]
            elif wtotal[n] > 0:
                out[n] = total[n] / wtotal[n]


def project_along(octree, name, axis, weight=None):
    """The column integral of cell field ``name`` along ``axis`` (0, 1, 2) of a 3D
    octree, as a 2D octree over the other two axes in order (its plane at the
    roots' lower side along ``axis``) with the field ``name``: on each leaf the
    integral over its column, on refined nodes their leaves' mean. With
    ``weight`` (a field's name), the column's mean of ``name`` weighted by it.
    Hidden leaves add nothing. On the device: the nodes dropped onto the plane
    and grouped by a sort of their (column, level, key), each group's
    contributions summed by segments; the 2D octree's columns then accumulated
    down through its children table."""
    from tack.algorithms import argsort, gather, inclusive_scan, segment_reduce
    from tack.dataset.dataset import rectilinear_grid

    if octree.dimension != 3:
        raise ValueError("project_along takes a 3D octree")
    axis = int(axis)
    u, v = [a for a in range(3) if a != axis]
    n = octree.num_nodes
    nx, ny, nz = [m - 1 for m in octree.roots.topology.point_dims]
    dtype = _compute_type(octree.device, tack.f32)
    with octree.device:
        tree2 = tack.empty((n,), tack.i64)
        level2 = tack.empty((n,), tack.i64)
        key2 = tack.empty((n,), tack.u64)
        _dropped(octree.tree, octree.level, octree.key, nx, ny, axis, octree.finest, tree2,
                 level2, key2)
        order = argsort(key2, index_dtype=tack.i64)
        order = gather(order, argsort(gather(level2, order), index_dtype=tack.i64))
        order = gather(order, argsort(gather(tree2, order), index_dtype=tack.i64))
        st, sl, sk = (gather(c, order) for c in (tree2, level2, key2))
        flags = tack.empty((n,), tack.i32)
        _run_starts(st, sl, sk, flags)
        numbers = tack.empty((n,), tack.i64)
        count = int(inclusive_scan(flags, numbers, n))
        utree, ulevel, ukey = (tack.empty((count,), t) for t in (tack.i64, tack.i64, tack.u64))
        offsets = tack.empty((count + 1,), tack.i64)
        _runs(flags, numbers, st, sl, sk, count, utree, ulevel, ukey, offsets)
        values = octree._field(name).astype(dtype)
        weights = tack.ones((n,), dtype) if weight is None else octree._field(weight).astype(dtype)
        _, size = octree.geometry()
        sums, wsums = tack.empty((n,), dtype), tack.empty((n,), dtype)
        _contributions(order, octree.marks, int(Ghosts.REFINEDCELL | Ghosts.HIDDENCELL), values,
                       weights, size, axis, sums, wsums)
        total = segment_reduce(sums, offsets)
        wtotal = segment_reduce(wsums, offsets)

    axes = _root_axes(octree.roots)
    origin = np.asarray(octree.origin, np.float64)
    roots = rectilinear_grid(axes[u] + origin[u], axes[v] + origin[v],
                             (float(axes[axis][0] + origin[axis]),),
                             dtype=octree.roots.dtype, unit=octree.unit)
    out = OctreeDataSet(roots, utree.numpy(), ulevel.numpy(), ukey.numpy(),
                        fields={"sum": total.numpy(), "weight": wtotal.numpy()},
                        world=octree.world)
    # Each node's column, down to it: its own plus its ancestors', a level at a
    # time from the roots, through the children table.
    number, table = children(out)
    with out.device:
        total, wtotal = out.fields["sum"], out.fields["weight"]
        for nodes in reversed(refined_by_level(out)):
            _down_to_children(nodes, number, table, 2, total, wtotal)
        column = tack.empty((out.num_nodes,), dtype)
        _column_values(total, wtotal, out.marks, int(Ghosts.REFINEDCELL),
                       0 if weight is None else 1, float("nan"), column)
    unit = octree.units.get(name)
    if weight is None:
        unit = _units.derived((unit, 1), (octree.unit, 1))
    out.fields = {name: column}
    out.units = {name: unit} if unit is not None else {}
    return average_up(out, [name])
