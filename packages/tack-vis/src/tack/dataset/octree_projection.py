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
projection). Grouping on the host (a study).
"""

import numpy as np

import tack
from tack.dataset import units as _units
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet, _interleave, _root_axes
from tack.dataset.octree_filters import _coordinates, average_up
from tack.dataset.octree_topology import parents

__all__ = ["project_along"]


def _drop(octree, axis):
    """Each node dropped onto the plane: (2D tree, level, 2D key), host arrays."""
    u, v = [a for a in range(3) if a != axis]
    dims = [n - 1 for n in octree.roots.topology.point_dims]
    tree = octree._tree
    root = np.stack([tree % dims[0], (tree // dims[0]) % dims[1], tree // (dims[0] * dims[1])],
                    axis=1)
    index = _coordinates(octree)
    tree2 = root[:, u] + dims[u] * root[:, v]
    key2 = _interleave(np.stack([index[:, u], index[:, v], np.zeros(len(tree), np.int64)],
                                axis=1), 2)
    return tree2, octree._level.copy(), key2


def project_along(octree, name, axis, weight=None):
    """The column integral of cell field ``name`` along ``axis`` (0, 1, 2) of a 3D
    octree, as a 2D octree over the other two axes in order (its plane at the
    roots' lower side along ``axis``) with the field ``name``: on each leaf the
    integral over its column, on refined nodes their leaves' mean. With
    ``weight`` (a field's name), the column's mean of ``name`` weighted by it.
    Hidden leaves add nothing."""
    from tack.dataset.dataset import rectilinear_grid

    if octree.dimension != 3:
        raise ValueError("project_along takes a 3D octree")
    axis = int(axis)
    u, v = [a for a in range(3) if a != axis]
    tree2, level2, key2 = _drop(octree, axis)
    rows = np.stack([tree2, level2, key2.astype(np.int64)], axis=1)
    nodes, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    marks = octree.marks.numpy()
    adds = octree.is_leaf & ((marks & Ghosts.HIDDENCELL) == 0)
    _, size = octree.geometry()
    length = np.asarray(size.numpy(), np.float64)[:, axis]
    values = np.asarray(octree._field(name).numpy(), np.float64)
    w = np.ones(len(values)) if weight is None else \
        np.asarray(octree._field(weight).numpy(), np.float64)
    on = np.flatnonzero(adds)
    sums = np.bincount(inverse[on], values[on] * w[on] * length[on], minlength=len(nodes))
    weights = np.bincount(inverse[on], w[on] * length[on], minlength=len(nodes))

    axes = _root_axes(octree.roots)
    origin = np.asarray(octree.origin, np.float64)
    roots = rectilinear_grid(axes[u] + origin[u], axes[v] + origin[v],
                             (float(axes[axis][0] + origin[axis]),),
                             dtype=octree.roots.dtype, unit=octree.unit)
    out = OctreeDataSet(roots, nodes[:, 0], nodes[:, 1], nodes[:, 2].astype(np.uint64),
                        fields={"sum": sums, "weight": weights}, world=octree.world)
    # Each node's column, down to it: its own plus its ancestors', a level at a
    # time from the roots.
    parent = np.asarray(parents(out).numpy())
    total = {k: np.asarray(out.fields[k].numpy(), np.float64).copy() for k in ("sum", "weight")}
    for lv in range(1, out.finest + 1):
        here = np.flatnonzero(out._level == lv)
        for k in total:
            total[k][here] += total[k][parent[here]]
    column = total["sum"] if weight is None else \
        np.divide(total["sum"], total["weight"], out=np.full(out.num_nodes, np.nan),
                  where=total["weight"] > 0)
    column[~out.is_leaf] = np.nan
    unit = octree.units.get(name)
    if weight is None:
        unit = _units.derived((unit, 1), (octree.unit, 1))
    out.fields = {name: tack.array(column, tack.f64, device=out.device)}
    out.units = {name: unit} if unit is not None else {}
    return average_up(out, [name])
