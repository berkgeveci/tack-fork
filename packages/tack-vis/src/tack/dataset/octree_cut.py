# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A plane through a cell-centred octree, as polygons (study,
``study/octree-filters``): each visible leaf the plane crosses gives the
convex polygon of its box's section, 3 to 6 vertices, with the leaf's values
as cell data -- the crisp AMR cut, as ``vtkHyperTreeGridPlaneCutter`` makes
it without its dual option. (The dual grid's cut, smooth but half a leaf short
of the boundary, is ``slice`` of ``dual_grid``.)

A descent from the roots through the children table keeps the subtrees the
plane cuts (a node's section holds its descendants'); each crossed leaf
alone then: its corners' signed distances to the plane, a crossing on each
edge whose ends differ in sign (a corner on the plane counts as above,
and the crossings there coincide and are kept once), the crossings ordered by
angle about their mean in the plane. Counted, scanned, written; nothing is
merged between leaves.
"""

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import _compute_type

__all__ = ["plane_cut"]

#: A box's 12 edges as pairs of corners (x the lowest bit, then y, then z).
_EDGES = tack.constant((0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7),
                       tack.i32)


@tack.func
def _distance(lo, size, j, nx, ny, nz, w):
    # Corner j's signed distance to the plane n . x = w.
    x = lo[0] + size[0] * ((j & 1) * 1.0)
    y = lo[1] + size[1] * (((j >> 1) & 1) * 1.0)
    z = lo[2] + size[2] * (((j >> 2) & 1) * 1.0)
    return nx * x + ny * y + nz * z - w


@tack.kernel
def _section(leaf, lo, size, marks, hidden, nx, ny, nz, w, ux, uy, uz, vx, vy, vz, write, proto,
             numbers, counts, points):
    # Leaf c's polygon: its distinct edge crossings, in order about their mean
    # (by angle in the plane's basis (u, v)); 0 vertices when the plane misses
    # it, or touches a corner or an edge alone, or the leaf is hidden. Run
    # twice: counting (``write`` 0) and, after a scan, writing the vertices at
    # ``numbers[c] - counts[c]``. ``proto`` gives the locals their type.
    for c in range(counts.shape[0]):
        n = leaf[c]
        count = 0
        if (marks[n] & hidden) == 0:
            box = lo[n]
            side = size[n]
            px = tack.local_array_like(proto, 12)
            py = tack.local_array_like(proto, 12)
            pz = tack.local_array_like(proto, 12)
            for e in range(12):
                a = _EDGES[2 * e]
                b = _EDGES[2 * e + 1]
                fa = _distance(box, side, a, nx, ny, nz, w)
                fb = _distance(box, side, b, nx, ny, nz, w)
                if (fa < 0.0) != (fb < 0.0):
                    t = fa / (fa - fb)
                    x = box[0] + side[0] * (((a & 1) * 1.0) + t * (((b & 1) - (a & 1)) * 1.0))
                    y = box[1] + side[1] * ((((a >> 1) & 1) * 1.0) +
                                            t * ((((b >> 1) & 1) - ((a >> 1) & 1)) * 1.0))
                    z = box[2] + side[2] * ((((a >> 2) & 1) * 1.0) +
                                            t * ((((b >> 2) & 1) - ((a >> 2) & 1)) * 1.0))
                    new = 1
                    for k in range(count):
                        if px[k] == x and py[k] == y and pz[k] == z:
                            new = 0
                    if new == 1:
                        px[count] = x
                        py[count] = y
                        pz[count] = z
                        count += 1
            if count < 3:
                count = 0
            if write == 1 and count > 0:
                cx = 0.0 * w
                cy = cx
                cz = cx
                for k in range(count):
                    cx += px[k]
                    cy += py[k]
                    cz += pz[k]
                cx = cx / count
                cy = cy / count
                cz = cz / count
                angle = tack.local_array_like(proto, 12)
                for k in range(count):
                    dx = px[k] - cx
                    dy = py[k] - cy
                    dz = pz[k] - cz
                    angle[k] = atan2(dx * vx + dy * vy + dz * vz, dx * ux + dy * uy + dz * uz)
                at = numbers[c] - counts[c]
                for k in range(count):
                    # The k-th smallest angle: insertion into the output.
                    best = -1
                    for i in range(count):
                        if angle[i] < 1e30:
                            if best < 0:
                                best = i
                            elif angle[i] < angle[best]:
                                best = i
                    points[at + k] = tack.vec3(px[best], py[best], pz[best])
                    angle[best] = 1e30
        if write == 0:
            counts[c] = count


@tack.kernel
def _straddling(frontier, lo, size, marks, refined, hidden, nx, ny, nz, w, big, packed):
    # A node kept when the plane cuts its box with some area (a corner below,
    # and one above or three on it): 1 for a visible leaf, 1 << 32 for a refined
    # node, 0 otherwise, so one scan counts both. A node's section holds its
    # descendants', so a pruned subtree has no polygon.
    for k in range(frontier.shape[0]):
        n = frontier[k]
        box = lo[n]
        side = size[n]
        fmin = big
        fmax = -big
        on = 0
        for j in range(8):
            f = _distance(box, side, j, nx, ny, nz, w)
            fmin = min(fmin, f)
            fmax = max(fmax, f)
            if f == 0.0:
                on += 1
        packed[k] = tack.i64(0)
        if fmin < 0.0 and (fmax > 0.0 or (fmax == 0.0 and on >= 3)):
            if (marks[n] & refined) != 0:
                packed[k] = tack.i64(1) << tack.i64(32)
            elif (marks[n] & hidden) == 0:
                packed[k] = tack.i64(1)


@tack.kernel
def _advance_nodes(frontier, packed, numbers, number, table, dimension, start, active,
                   next_frontier):
    # From the scan: a kept leaf to ``active`` after ``start``, a kept refined
    # node's children to the next frontier.
    for k in range(frontier.shape[0]):
        p = packed[k]
        if p == 1:
            active[start + (numbers[k] & ((tack.i64(1) << tack.i64(32)) - 1)) - 1] = frontier[k]
        elif p != 0:
            n = frontier[k]
            high = numbers[k] >> tack.i64(32)
            corners = 1 << dimension
            for j in range(corners):
                next_frontier[(high - 1) * corners + j] = table[(number[n] - 1) * corners + j]


def _crossed_leaves(octree, n, w):
    """The visible leaves whose box the plane ``n . x = w`` cuts with some area:
    a device i64 array, by a descent from the roots through the children
    table, a level at a time, pruning subtrees the plane misses."""
    from tack.algorithms import inclusive_scan
    from tack.dataset.octree_topology import children

    d = octree.dimension
    number, table = children(octree)
    leaves = octree.num_nodes - (len(table) >> d)
    big = float(np.finfo(np.float64).max)
    with octree.device:
        lo, size = octree.geometry()
        frontier = tack.array(np.flatnonzero(octree._level == 0).astype(np.int64))
        active = tack.empty((max(leaves, 1),), tack.i64)
        found = 0
        while len(frontier):
            count = len(frontier)
            packed = tack.empty((count,), tack.i64)
            _straddling(frontier, lo, size, octree.marks, int(Ghosts.REFINEDCELL),
                        int(Ghosts.HIDDENCELL), *map(float, n), w, big, packed)
            numbers = tack.empty((count,), tack.i64)
            total = int(inclusive_scan(packed, numbers, count))
            kept_leaves, kept_refined = total & ((1 << 32) - 1), total >> 32
            next_frontier = tack.empty((kept_refined << d,), tack.i64)
            _advance_nodes(frontier, packed, numbers, number, table, d, found, active,
                           next_frontier)
            found += kept_leaves
            frontier = next_frontier
    return active, found


@tack.kernel
def _cut_leaves(leaf, counts, numbers, out):
    for c in range(counts.shape[0]):
        if counts[c] > 0:
            out[numbers[c] - 1] = leaf[c]


@tack.kernel
def _loop_offsets(counts, numbers, polygons, out):
    # The kept leaves' polygons' offsets: leaf c's polygon (the polygons[c]-th)
    # starts where its vertices do.
    for c in range(counts.shape[0]):
        if counts[c] > 0:
            out[polygons[c] - 1] = numbers[c] - counts[c]
        if c == 0:
            out[polygons[counts.shape[0] - 1]] = numbers[counts.shape[0] - 1]


@tack.kernel
def _flags(counts, out):
    for c in range(out.shape[0]):
        out[c] = 1 if counts[c] > 0 else 0


def plane_cut(octree, point, normal):
    """The plane through ``point`` with ``normal`` (world coordinates) cut
    through the visible leaves of a 3D octree: a ``DataSet`` of polygons, one
    per leaf the plane crosses (its box's section, 3 to 6 vertices, wound
    counterclockwise seen from the normal), every field as cell data, the
    points unmerged."""
    from tack.algorithms import gather, inclusive_scan
    from tack.dataset.dataset import DataSet, Field
    from tack.dataset.polyhedra import PolygonalTopology
    from tack.dataset.spaces import Constant

    if octree.dimension != 3:
        raise ValueError("plane_cut takes a 3D octree (a study)")
    n = np.asarray(normal, np.float64)
    if not np.isfinite(n).all() or not np.linalg.norm(n) > 0:
        raise ValueError("the normal is a nonzero vector")
    n = n / np.linalg.norm(n)
    w = float(n @ (np.asarray(point, np.float64) - np.asarray(octree.origin, np.float64)))
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    dtype = _compute_type(octree.device, tack.f32)
    out_dtype = octree.roots.dtype
    hidden = int(Ghosts.HIDDENCELL)
    leaf, count = _crossed_leaves(octree, n, w)
    with octree.device:
        lo, size = octree.geometry()
        counts = tack.empty((count,), tack.i32)
        numbers = tack.empty((count,), tack.i64)
        proto = tack.empty((1,), dtype)
        vertices = polygons = 0
        if count:
            _section(leaf, lo, size, octree.marks, hidden, *map(float, n), w, *map(float, u),
                     *map(float, v), 0, proto, numbers, counts, lo)
            vertices = int(inclusive_scan(counts, numbers, count))
        points = tack.empty((max(vertices, 1),), tack.vec3(dtype))
        flags = tack.empty((count,), tack.i32)
        polygon_numbers = tack.empty((count,), tack.i64)
        if vertices:
            _section(leaf, lo, size, octree.marks, hidden, *map(float, n), w, *map(float, u),
                     *map(float, v), 1, proto, numbers, counts, points)
            _flags(counts, flags)
            polygons = int(inclusive_scan(flags, polygon_numbers, count))
        cut = tack.empty((polygons,), tack.i64)
        offsets = tack.empty((polygons + 1,), tack.i64)
        if polygons:
            _cut_leaves(leaf, counts, polygon_numbers, cut)
            _loop_offsets(counts, numbers, polygon_numbers, offsets)
        else:
            offsets = tack.zeros((1,), tack.i64)
        loop_points = tack.arange(vertices, dtype=tack.i64)
        topology = PolygonalTopology(offsets, loop_points, num_points=vertices)
        positions = points.astype(tack.vec3(out_dtype)) if dtype != out_dtype else points
        if vertices == 0:
            positions = tack.empty((0,), tack.vec3(out_dtype))
        data = DataSet(topology, positions, dtype=out_dtype, **octree._frame())
        for name, values in octree.fields.items():
            data.fields[name] = Field(Constant(topology), gather(values, cut),
                                      octree.units.get(name))
    return data
