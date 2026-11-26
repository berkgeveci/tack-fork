# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A cell-centred octree volume rendered by Tack's march (study,
``study/octree-filters``): the sampler the AMR volume work's march takes
(``tack_march.march`` on ``proto/webgpu-surfaces``, ``packages/tack-vis/
examples/amr_volume``: rays marched by a Tack kernel on Tack's device and
blended into VTK's frame), written for an octree.

- **The lookup is a descent.** From the root cell holding the point, down the
  children table by the point's octant bits, one i32 read a level: no keys, no
  brick tables, nothing 64-bit (``OctreeTents.leaf_at``, ``node_at``).
- **The step is the leaf.** ``sample(p, first, stop)`` returns the value and the
  leaf's size, which sets the march's step.
- **Values with sentinels.** One f32 a node: the value on a visible leaf,
  ``REFINED`` (descend) on a refined node, ``HIDDEN`` (nothing here) on a hidden
  leaf -- one load a node, as the AMR sampler marks them.
- **Reconstruction.** ``blend=0``: the leaf's value, piecewise constant, exactly
  what ``probe`` gives. ``blend=2``: trilinear in each leaf's eight corner
  values, the tents evaluated there once (one descent and two cache lines a
  sample; a crack where a fine leaf's corner hangs on a coarse face).
  ``blend=1``: the AMR sampler's tents: every visible
  leaf is a tent its own size wide about its centre, and the value is the
  weighted mean of the tents reaching the point. The cells of level ``l``
  whose tents reach the point are the eight around it on level ``l``'s
  lattice, so the sum runs over the levels the macrocell's word gives
  (``first`` up to ``stop``: the levels of the leaves reaching it) and, per
  level, the eight lattice cells, each found by a descent that stops where a
  leaf of another level holds it. Within a uniform region this is trilinear
  interpolation of the centres; across a level change it blends the coarse
  and fine tents, continuously.
- **Empty space.** Per macrocell of a lattice ``depth`` levels below the roots,
  the least and greatest value of the leaves whose tents reach it, and their
  least and greatest level, from the per-node ranges the contour keeps
  (``ranges``) and the same ranges over the leaves' levels: a node at that
  level or a coarser leaf writes its subtree's ranges over its block, widened
  by half its size, with atomics; the AMR march's ``visible`` and
  ``macro_words`` then make the distance words as for AMR. The march's finer
  skipping (``sub_grids``, divided macrocells) is not used yet.

The AMR modules are imported from a directory the caller names
(``OctreeRenderer(..., amr_volume=<path>)``): a snapshot of their committed
state, since their worktree changes under a run.
"""

import sys
from math import floor

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import _root_axes, _root_cells, _unkey
from tack.dataset.octree_contour import _child_ranges, _leaf_ranges, ranges, subtree_ends
from tack.dataset.octree_topology import children, refined_by_level

__all__ = ["OctreeRenderer", "OctreeTents", "corner_values", "macro_ranges", "octree_tents"]

#: A refined node's value (descend), a hidden leaf's (nothing), and what a visible
#: leaf's value is above.
REFINED = tack.constant(-3.0e38)
HIDDEN = tack.constant(-2.0e38)
ACTIVE = tack.constant(-1.0e38)


@tack.kernel
def _node_values(values, marks, refined, hidden, out):
    for n in range(out.shape[0]):
        v = tack.f32(values[n])
        if (marks[n] & refined) != 0:
            v = REFINED
        elif (marks[n] & hidden) != 0:
            v = HIDDEN
        out[n] = v


@tack.device_class
class OctreeTents:
    """An octree as the march's sampler: ``sample(p, first, stop)``, the value at
    ``p`` and the size of the leaf there."""

    def __init__(self, octree, name, blend=1, balanced=0, chunk=None, corners=None):
        if octree.dimension != 3:
            raise ValueError("the volume sampler takes a 3D octree")
        number, table = children(octree)
        self.number = number
        self.table = table
        nx, ny, nz = [m - 1 for m in octree.roots.topology.point_dims]
        roots = np.full(nx * ny * nz, -1, np.int32)
        level0 = np.flatnonzero(octree._level == 0)
        roots[octree._tree[level0]] = level0
        axes = _root_axes(octree.roots)
        origin = np.asarray(octree.origin, np.float64)
        _, sizes = _root_cells(octree.roots)
        if not all(np.allclose(np.diff(a), np.diff(a)[0]) for a in axes):
            raise ValueError("the volume sampler takes roots of one size (uniform axes)")
        with octree.device:
            self.roots = tack.array(roots, tack.i32)
            values = tack.empty((octree.num_nodes,), tack.f32)
            if octree.num_nodes:
                _node_values(octree._field(name), octree.marks, int(Ghosts.REFINEDCELL),
                             int(Ghosts.HIDDENCELL), values)
            self.values = values
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.ox, self.oy, self.oz = (float(axes[a][0] + origin[a]) for a in range(3))
        self.sx, self.sy, self.sz = (float(s) for s in sizes[0])
        self.coarse = float(min(sizes[0]))
        self.finest = float(min(sizes[0]) / (1 << octree.finest))
        self.blend = int(blend)
        #: ``blend=2``: each leaf's eight corner values (x fastest), the exact
        #: tents evaluated there once, and a sample is trilinear in them: one
        #: descent and two cache lines a sample, at the price of a crack where a
        #: fine leaf's corner hangs on a coarse face.
        self.corners = tack.empty((1,), tack.f32)
        if int(blend) == 2 and corners is not None:
            # Made elsewhere (``corner_values``, two fields' in one pass).
            self.corners = corners
        elif int(blend) == 2 and int(balanced) == 1:
            self.corners = corner_values(octree, name, chunk=chunk)
        elif int(blend) == 2:
            exact = OctreeTents(octree, name, blend=1, balanced=balanced)
            with octree.device:
                self.corners = tack.empty((max(octree.num_nodes, 1) * 8,), tack.f32)
                if octree.num_nodes:
                    lo, size = octree.geometry()
                    _corner_values(exact, lo, size, octree.marks, int(Ghosts.REFINEDCELL),
                                   octree.finest + 1, self.corners)
        #: Whether neighbouring leaves differ by at most one level (xRAGE, RAMSES):
        #: then no tent finer than one level below the leaf reaches a point.
        self.balanced = int(balanced)
        #: What ``OctreeRenderer`` needs beside the sampler.
        self.bounds = tuple(float(v) for v in np.stack([
            [axes[a][0] + origin[a] for a in range(3)],
            [axes[a][-1] + origin[a] for a in range(3)]], axis=1).reshape(-1))

    @tack.func
    def leaf_at(self, p):
        """The leaf holding ``p`` (-1 beyond the roots) and its smallest side, by a
        descent from the root cell."""
        rx = (p[0] - self.ox) / self.sx
        ry = (p[1] - self.oy) / self.sy
        rz = (p[2] - self.oz) / self.sz
        ti = min(max(int(floor(rx)), 0), self.nx - 1)  # noqa: RUF046 (floor is f32 here)
        tj = min(max(int(floor(ry)), 0), self.ny - 1)  # noqa: RUF046
        tk = min(max(int(floor(rz)), 0), self.nz - 1)  # noqa: RUF046
        fx = min(max(rx - tack.f32(ti), 0.0), 0.999999)
        fy = min(max(ry - tack.f32(tj), 0.0), 0.999999)
        fz = min(max(rz - tack.f32(tk), 0.0), 0.999999)
        n = self.roots[ti + self.nx * (tj + self.ny * tk)]
        size = self.coarse
        depth = 0
        while n >= 0 and self.values[n] == REFINED:
            jx = 1 if fx >= 0.5 else 0
            jy = 1 if fy >= 0.5 else 0
            jz = 1 if fz >= 0.5 else 0
            fx = 2.0 * fx - tack.f32(jx)
            fy = 2.0 * fy - tack.f32(jy)
            fz = 2.0 * fz - tack.f32(jz)
            size = 0.5 * size
            depth += 1
            n = self.table[(self.number[n] - 1) * 8 + jx + 2 * jy + 4 * jz]
        # The point's position in the leaf, 0 to 1 along each axis, and how far it
        # is from the nearest face, in units of the leaf's side.
        inside = min(fx, 1.0 - fx, fy, 1.0 - fy, fz, 1.0 - fz)
        return n, size, depth, inside, fx, fy, fz

    @tack.func
    def node_at(self, level, i, j, k):
        """The node of level ``level`` at lattice cell ``(i, j, k)`` of that level, or
        -1 where a leaf of a coarser level holds the cell, or beyond the roots."""
        n = tack.i32(-1)
        ti = i >> level
        tj = j >> level
        tk = k >> level
        if i >= 0 and j >= 0 and k >= 0 and ti < self.nx and tj < self.ny and tk < self.nz:
            n = self.roots[ti + self.nx * (tj + self.ny * tk)]
            d = level
            while n >= 0 and d > 0:
                if self.values[n] != REFINED:
                    n = tack.i32(-1)
                else:
                    d -= 1
                    n = self.table[(self.number[n] - 1) * 8 + ((i >> d) & 1) +
                                   2 * ((j >> d) & 1) + 4 * ((k >> d) & 1)]
        return n

    @tack.func
    def holder_at(self, level, i, j, k):
        """The node holding lattice cell ``(i, j, k)`` of ``level``: the node of that
        level there, or the coarser leaf holding it, and its depth; -1 beyond
        the roots."""
        n = tack.i32(-1)
        d = 0
        ti = i >> level
        tj = j >> level
        tk = k >> level
        if i >= 0 and j >= 0 and k >= 0 and ti < self.nx and tj < self.ny and tk < self.nz:
            n = self.roots[ti + self.nx * (tj + self.ny * tk)]
            left = level
            while n >= 0 and left > 0 and self.values[n] == REFINED:
                left -= 1
                d += 1
                n = self.table[(self.number[n] - 1) * 8 + ((i >> left) & 1) +
                               2 * ((j >> left) & 1) + 4 * ((k >> left) & 1)]
        return n, d

    @tack.func
    def sample(self, p, first, stop):
        n, size, depth, inside, fx, fy, fz = self.leaf_at(p)
        value = 0.0
        if n >= 0:
            v = self.values[n]
            if v > ACTIVE:
                value = v
        last = stop
        if n >= 0:
            # A finer leaf's tent reaches half its size beyond its face, so none
            # reaches a point deeper inside the leaf than half the finest cell;
            # in a balanced tree no neighbour is more than one level finer.
            if inside > 0.25:
                last = min(last, depth + 1)
            if self.balanced == 1:
                last = min(last, depth + 2)
        if self.blend == 2 and n >= 0:
            # Trilinear in the leaf's corner values (the tents, once, at its corners).
            c = n * 8
            v00 = self.corners[c] * (1.0 - fx) + self.corners[c + 1] * fx
            v10 = self.corners[c + 2] * (1.0 - fx) + self.corners[c + 3] * fx
            v01 = self.corners[c + 4] * (1.0 - fx) + self.corners[c + 5] * fx
            v11 = self.corners[c + 6] * (1.0 - fx) + self.corners[c + 7] * fx
            value = (v00 * (1.0 - fy) + v10 * fy) * (1.0 - fz) + (v01 * (1.0 - fy) + v11 * fy) * fz
        if self.blend == 1 and last > first:
            num = 0.0
            den = 0.0
            for wide in range(first, last):
                level = tack.i32(wide)
                # Level's lattice: cell sides, and the point in cell-centre units.
                hx = self.sx / tack.f32(1 << level)
                hy = self.sy / tack.f32(1 << level)
                hz = self.sz / tack.f32(1 << level)
                ux = (p[0] - self.ox) / hx - 0.5
                uy = (p[1] - self.oy) / hy - 0.5
                uz = (p[2] - self.oz) / hz - 0.5
                i0 = int(floor(ux))  # noqa: RUF046
                j0 = int(floor(uy))  # noqa: RUF046
                k0 = int(floor(uz))  # noqa: RUF046
                for s in range(8):
                    i = i0 + (s & 1)
                    j = j0 + ((s >> 1) & 1)
                    k = k0 + ((s >> 2) & 1)
                    m = self.node_at(level, i, j, k)
                    if m >= 0:
                        u = self.values[m]
                        if u > ACTIVE:
                            w = (1.0 - abs(ux - tack.f32(i))) * (1.0 - abs(uy - tack.f32(j))) * \
                                (1.0 - abs(uz - tack.f32(k)))
                            num += w * u
                            den += w
            if den > 0.0:
                value = num / den
        return value, size


@tack.kernel
def _corner_leaves(tents, values_b, tree, level, key, marks, refined, nx, ny, first, count, two,
                   out_a, out_b):
    # Each leaf's eight corners, the tents there in a balanced tree: at a corner,
    # the eight cells of the leaf's level around it (among the leaf's 27
    # neighbours, each found once by a descent), each a leaf's tent with its own
    # box, a refined one's leaves down the chain of cells at the corner, each
    # weighing an eighth; a coarser leaf holding several cells counted once. For
    # the field of ``tents`` and, with ``two``, ``values_b`` (``_node_values`` of
    # another field): the same cells and weights, sums of their own. The nodes
    # first .. first + count.
    for k in range(count):
        n = first + k
        if (marks[n] & refined) == 0:
            lv = level[n]
            i, j, m = _unkey(key[n], lv, 3)
            t = tree[n]
            ci = ((t % nx) << lv) + tack.i32(i)
            cj = (((t // nx) % ny) << lv) + tack.i32(j)
            ck = ((t // (nx * ny)) << lv) + tack.i32(m)
            near = tack.local_array(tack.i32, 27)
            for o in range(27):
                # The node of level lv there, or the coarser leaf holding it.
                h, _ = tents.holder_at(lv, ci + o % 3 - 1, cj + (o // 3) % 3 - 1,
                                       ck + o // 9 - 1)
                near[o] = h
            f = tack.f32(1 << lv)
            hx = tents.sx / f
            hy = tents.sy / f
            hz = tents.sz / f
            for c in range(8):
                cx = c & 1
                cy = (c >> 1) & 1
                cz = (c >> 2) & 1
                vx = tents.ox + tack.f32(ci + cx) * hx
                vy = tents.oy + tack.f32(cj + cy) * hy
                vz = tents.oz + tack.f32(ck + cz) * hz
                num = 0.0
                den = 0.0
                num_b = 0.0
                den_b = 0.0
                for s in range(8):
                    sx = s & 1
                    sy = (s >> 1) & 1
                    sz = (s >> 2) & 1
                    q = near[cx + sx + 3 * (cy + sy) + 9 * (cz + sz)]
                    if q >= 0:
                        # Counted once: a coarser leaf holds several of the cells.
                        new = 1
                        for e in range(s):
                            if near[cx + (e & 1) + 3 * (cy + ((e >> 1) & 1)) +
                                    9 * (cz + ((e >> 2) & 1))] == q:
                                new = 0
                        if new == 1:
                            u = tents.values[q]
                            if u == REFINED:
                                # Down the chain of cells at the corner: the child in
                                # the octant facing the corner, 7 - s.
                                d = q
                                while d >= 0 and tents.values[d] == REFINED:
                                    d = tents.table[(tents.number[d] - 1) * 8 + (7 - s)]
                                if d >= 0:
                                    u = tents.values[d]
                                    if u > ACTIVE:
                                        num += 0.125 * u
                                        den += 0.125
                                    if two == 1:
                                        ub = values_b[d]
                                        if ub > ACTIVE:
                                            num_b += 0.125 * ub
                                            den_b += 0.125
                            else:
                                ub = u
                                if two == 1:
                                    ub = values_b[q]
                                if u > ACTIVE or ub > ACTIVE:
                                    g = tack.f32(1 << level[q])
                                    ax = tents.sx / g
                                    ay = tents.sy / g
                                    az = tents.sz / g
                                    # q's box from the cell's centre, which lies in it.
                                    px = tents.ox + (tack.f32(ci + cx - 1 + sx) + 0.5) * hx
                                    py = tents.oy + (tack.f32(cj + cy - 1 + sy) + 0.5) * hy
                                    pz = tents.oz + (tack.f32(ck + cz - 1 + sz) + 0.5) * hz
                                    lx = tents.ox + floor((px - tents.ox) / ax) * ax
                                    ly = tents.oy + floor((py - tents.oy) / ay) * ay
                                    lz = tents.oz + floor((pz - tents.oz) / az) * az
                                    w = max(1.0 - abs(vx - lx - 0.5 * ax) / ax, 0.0) * \
                                        max(1.0 - abs(vy - ly - 0.5 * ay) / ay, 0.0) * \
                                        max(1.0 - abs(vz - lz - 0.5 * az) / az, 0.0)
                                    if u > ACTIVE:
                                        num += w * u
                                        den += w
                                    if two == 1 and ub > ACTIVE:
                                        num_b += w * ub
                                        den_b += w
                value = 0.0
                if den > 0.0:
                    value = num / den
                out_a[n * 8 + c] = value
                if two == 1:
                    value_b = 0.0
                    if den_b > 0.0:
                        value_b = num_b / den_b
                    out_b[n * 8 + c] = value_b


def corner_values(octree, name, second=None, chunk=None):
    """Each leaf's eight corner values of field ``name`` (the sampler's for
    ``blend=2``, a balanced octree) and, with ``second``, those of another field
    in the same pass: each leaf's 27 neighbours found once, both fields' sums
    from them (on an M1 Max, 78M nodes: 2.3 s for two fields, against 4.6 for
    two passes of a corner at a time, eight descents a corner; a table of every
    node's neighbours, ``octree_topology.neighbourhoods``, cost more to make,
    2.9 s and 8.4 GB, than the descents it spared). ``chunk`` nodes a launch,
    each waited for, so a viewer's frames are not held behind one launch of
    seconds. Device f32 arrays of 8 a node, ``(a, b)`` with ``second``."""
    exact = OctreeTents(octree, name, blend=1, balanced=1)
    n = octree.num_nodes
    with octree.device:
        out_a = tack.empty((max(n, 1) * 8,), tack.f32)
        out_b = tack.empty((max(n, 1) * 8,) if second is not None else (1,), tack.f32)
        values_b = exact.values
        if second is not None:
            values_b = tack.empty((n,), tack.f32)
            if n:
                _node_values(octree._field(second), octree.marks, int(Ghosts.REFINEDCELL),
                             int(Ghosts.HIDDENCELL), values_b)
        step = max(n, 1) if chunk is None else max(int(chunk), 1)
        for start in range(0, n, step):
            _corner_leaves(exact, values_b, octree.tree, octree.level, octree.key, octree.marks,
                           int(Ghosts.REFINEDCELL), exact.nx, exact.ny, start,
                           min(step, n - start), int(second is not None), out_a, out_b)
            if chunk is not None:
                tack.sync()
    return (out_a, out_b) if second is not None else out_a


@tack.kernel
def _corner_values(exact, lo, size, marks, refined, stop, out):
    # A leaf's corners, nudged inward by a thousandth of the leaf so the point
    # lies in it: the exact tents there.
    for n in range(marks.shape[0]):
        if (marks[n] & refined) == 0:
            for c in range(8):
                q = lo[n] + size[n] * tack.vec3(0.001 + 0.998 * tack.f32(c & 1),
                                                0.001 + 0.998 * tack.f32((c >> 1) & 1),
                                                0.001 + 0.998 * tack.f32((c >> 2) & 1))
                v, s = exact.sample(q, 0, stop)
                out[n * 8 + c] = v


def octree_tents(octree, name, blend=1):
    """``OctreeTents`` of field ``name``: the march's sampler."""
    return OctreeTents(octree, name, blend)


# -- the macrocells ---------------------------------------------------------------------


@tack.kernel
def _macro_ranges(tree, level, key, marks, refined, lo, hi, llo, lhi, nx, ny, nz, depth, lows,
                  highs, firsts, lasts):
    # Each node at ``depth``, or a leaf above it, writes its subtree's ranges (of
    # value and of leaf level) over its block of the lattice at ``depth``,
    # widened by half its size (its tents reach that far), by atomics.
    for n in range(tree.shape[0]):
        lv = level[n]
        if lv <= depth and (lv == depth or (marks[n] & refined) == 0) and lo[n] <= hi[n]:
            i, j, m = _unkey(key[n], lv, 3)
            t = tree[n]
            span = 1 << (depth - lv)
            # A leaf's neighbours shape its corner values, so its values reach a
            # whole neighbour into it, and a finer leaf's less than one cell.
            reach = max(span, 1)
            gx = nx << depth
            gy = ny << depth
            gz = nz << depth
            bi = ((t % nx) << depth) + (tack.i64(i) << (depth - lv))
            bj = (((t // nx) % ny) << depth) + (tack.i64(j) << (depth - lv))
            bk = ((t // (nx * ny)) << depth) + (tack.i64(m) << (depth - lv))
            a = lo[n]
            b = hi[n]
            la = tack.i32(llo[n])
            lb = tack.i32(lhi[n])
            for dz in range(max(bk - reach, 0), min(bk + span + reach, gz)):
                for dy in range(max(bj - reach, 0), min(bj + span + reach, gy)):
                    for dx in range(max(bi - reach, 0), min(bi + span + reach, gx)):
                        cell = dx + gx * (dy + gy * dz)
                        tack.atomic_min(lows, cell, a)
                        tack.atomic_max(highs, cell, b)
                        tack.atomic_min(firsts, cell, la)
                        tack.atomic_max(lasts, cell, lb)


@tack.kernel
def _leaf_levels(level, out):
    for n in range(out.shape[0]):
        out[n] = tack.f32(level[n])


def _level_ranges(octree):
    """Each node's least and greatest visible leaf level over its subtree, as the
    value ranges are made: f32 device arrays."""
    n = octree.num_nodes
    with octree.device:
        levels = tack.empty((n,), tack.f32)
        lo, hi = tack.empty((n,), tack.f32), tack.empty((n,), tack.f32)
        if n:
            _leaf_levels(octree.level, levels)
            _leaf_ranges(levels, octree.marks, int(Ghosts.REFINEDCELL | Ghosts.HIDDENCELL),
                         3.0e38, lo, hi)
            ends = subtree_ends(octree)
            for nodes in refined_by_level(octree):
                _child_ranges(nodes, ends, lo, hi)
    return lo, hi


def macro_ranges(octree, r, most=64):
    """The macrocells of the octree: the lattice ``depth`` levels below the roots,
    the deepest with at most ``most`` cells along the longest axis (at least the
    roots), as the march's grid (``origin``, ``size``, ``dims``); per macrocell
    the least and greatest value of the leaves whose tents reach it (``r``:
    ``ranges(octree, name)``), f32 device arrays (greatest < least where none
    does); and their least and greatest level, i32 (the march's words hold
    levels up to 15)."""
    nx, ny, nz = [m - 1 for m in octree.roots.topology.point_dims]
    depth = 0
    while (max(nx, ny, nz) << (depth + 1)) <= most and depth < octree.finest:
        depth += 1
    dims = (nx << depth, ny << depth, nz << depth)
    axes = _root_axes(octree.roots)
    origin = np.asarray(octree.origin, np.float64)
    _, sizes = _root_cells(octree.roots)
    size = tuple(float(s) / (1 << depth) for s in sizes[0])
    count = int(np.prod(dims))
    llo, lhi = _level_ranges(octree)
    with octree.device:
        lows = tack.full((count,), 3.0e38, tack.f32)
        highs = tack.full((count,), -3.0e38, tack.f32)
        firsts = tack.full((count,), 1 << 20, tack.i32)
        lasts = tack.full((count,), -1, tack.i32)
        lo = r.lo if r.lo.dtype == tack.f32 else r.lo.astype(tack.f32)
        hi = r.hi if r.hi.dtype == tack.f32 else r.hi.astype(tack.f32)
        if octree.num_nodes:
            _macro_ranges(octree.tree, octree.level, octree.key, octree.marks,
                          int(Ghosts.REFINEDCELL), lo, hi, llo, lhi, nx, ny, nz, depth, lows,
                          highs, firsts, lasts)
    grid = {"origin": tuple(float(axes[a][0] + origin[a]) for a in range(3)), "size": size,
            "dims": tuple(int(d) for d in dims)}
    return grid, lows, highs, firsts, lasts


# -- the divided macrocells -------------------------------------------------------------


@tack.kernel
def _sub_holders(divided, gx, gy, depth, shift, roots, nx, ny, number, table, values, lo, hi, lows,
                 highs):
    # Each cell of each divided macrocell: the range of the values a sample in
    # it can take -- the node holding it (the node at the cells' level, or the
    # coarser leaf) and the holders of the 26 lattice cells around that node at
    # its own level, since a leaf's neighbours shape its corner values. Cells
    # are x fastest within a macrocell.
    for e in range(lows.shape[0]):
        across = 1 << shift
        k = e >> (3 * shift)
        local = e & ((1 << (3 * shift)) - 1)
        m = divided[k]
        mi = m % gx
        mj = (m // gx) % gy
        mk = m // (gx * gy)
        level = depth + shift
        ci = (mi << shift) + (local & (across - 1))
        cj = (mj << shift) + ((local >> shift) & (across - 1))
        ck = (mk << shift) + (local >> (2 * shift))
        a = 3.0e38
        b = -3.0e38
        n = tack.i32(-1)
        d = 0
        if ci >= 0:
            # The holder: the descent from the root cell, stopping at a leaf.
            n = roots[(ci >> level) + nx * ((cj >> level) + ny * (ck >> level))]
            left = level
            while n >= 0 and left > 0 and values[n] == REFINED:
                left -= 1
                d += 1
                n = table[(number[n] - 1) * 8 + ((ci >> left) & 1) + 2 * ((cj >> left) & 1) +
                          4 * ((ck >> left) & 1)]
        if n >= 0:
            # The holder's cell at its level, and the 27 around it.
            hi_ = ci >> (level - d)
            hj = cj >> (level - d)
            hk = ck >> (level - d)
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        qi = hi_ + dx
                        qj = hj + dy
                        qk = hk + dz
                        if qi >= 0 and qj >= 0 and qk >= 0 and (qi >> d) < nx and \
                                (qj >> d) < ny and (qk >> d) < (roots.shape[0] // (nx * ny)):
                            q = roots[(qi >> d) + nx * ((qj >> d) + ny * (qk >> d))]
                            left = d
                            while q >= 0 and left > 0 and values[q] == REFINED:
                                left -= 1
                                q = table[(number[q] - 1) * 8 + ((qi >> left) & 1) +
                                          2 * ((qj >> left) & 1) + 4 * ((qk >> left) & 1)]
                            if q >= 0 and lo[q] <= hi[q]:
                                a = min(a, lo[q])
                                b = max(b, hi[q])
        lows[e] = a
        highs[e] = b


def sub_ranges(octree, tents, r, grid, lasts, lows, highs, shift=2):
    """The macrocells a level reaches whose leaves are at most half a divided cell
    wide, divided ``1 << shift`` cells an axis, with each cell's range of the
    values a sample in it can take: the march's ``sub`` (``index``, ``lows``,
    ``highs``, ``shift``, ``count``), its ``sub_away`` then giving the skips."""
    nx, ny, nz = [m - 1 for m in octree.roots.topology.point_dims]
    gx, gy, gz = grid["dims"]
    depth = 0
    while (nx << depth) < gx:
        depth += 1
    finest = np.asarray(lasts.numpy())
    divided = np.flatnonzero(finest >= depth + shift + 1)
    # The march's word holds a divided macrocell's index + 1 above bit 14 of a
    # signed i32: 2^17 - 1 at most (its session caps sub_grids the same).
    most = (1 << 17) - 1
    if len(divided) > most:
        divided = divided[:most]
    index = np.full(len(finest), -1, np.int32)
    index[divided] = np.arange(len(divided), dtype=np.int32)
    count = max(len(divided), 1) << (3 * shift)
    with octree.device:
        divided_dev = tack.array(divided.astype(np.int32) if len(divided) else np.zeros(1, np.int32),
                                 tack.i32)
        a, b = tack.empty((count,), tack.f32), tack.empty((count,), tack.f32)
        lo = r.lo if r.lo.dtype == tack.f32 else r.lo.astype(tack.f32)
        hi = r.hi if r.hi.dtype == tack.f32 else r.hi.astype(tack.f32)
        if len(divided):
            _sub_holders(divided_dev, gx, gy, depth, shift, tents.roots, nx, ny, tents.number,
                         tents.table, tents.values, lo, hi, a, b)
        sub_lows, sub_highs = a, b
    return {"index": index, "lows": sub_lows, "highs": sub_highs, "shift": int(shift),
            "count": len(divided)}


# -- the renderer ------------------------------------------------------------------------


@tack.kernel
def _blend(volume, frame, out):
    # The volume (premultiplied RGBA) over the frame's 8-bit colour, 8-bit RGBA
    # out, as tack_composite.composite blends on the host.
    for k in range(volume.shape[0]):
        v = volume[k]
        for c in range(3):
            x = v[c] + (1.0 - v[3]) * tack.f32(frame[4 * k + c]) * (1.0 / 255.0)
            out[4 * k + c] = tack.u8(min(max(x, 0.0), 1.0) * 255.0 + 0.5)
        out[4 * k + 3] = tack.u8(255)


class OctreeRenderer:
    """An octree's cell field ``name`` as a volume Tack renders with the AMR work's
    march: ``render`` gives the picture as premultiplied RGBA, to blend into a
    frame (``tack_composite.composite``). ``amr_volume`` is the directory of the
    AMR volume modules (``tack_march``, ``tack_composite``, ``volume_tack``)."""

    def __init__(self, octree, name, amr_volume, blend=1, macrocells=64, divide=-1, balanced=0,
                 chunk=None, corners=None):
        if str(amr_volume) not in sys.path:
            sys.path.insert(0, str(amr_volume))
        import tack_composite
        import tack_march
        import volume_tack

        self._modules = (tack_march, tack_composite, volume_tack)
        self.octree = octree
        # ``balanced``: the sampler for octrees whose neighbouring leaves differ by
        # at most a level (``OctreeTents``); made once here, as replacing it after
        # cost a second sampler (5 of a 94M-node octree's 9 s on Metal).
        self.tents = OctreeTents(octree, name, blend, balanced=balanced, chunk=chunk,
                                 corners=corners)
        self.ranges = ranges(octree, name)
        self.grid, self.lows, self.highs, self.firsts, self.lasts = macro_ranges(
            octree, self.ranges, macrocells)
        self.bounds = self.tents.bounds
        self.cell = self.tents.finest
        #: The field's range over the visible leaves.
        with octree.device:
            lows = self.ranges.lo.numpy()
            highs = self.ranges.hi.numpy()
        held = lows <= highs
        self.range = (float(lows[held].min()), float(highs[held].max())) if held.any() else (0.0, 1.0)
        #: The divided macrocells (``divide`` the shift, -1 for none).
        self.divide = int(divide)
        self.sub = sub_ranges(octree, self.tents, self.ranges, self.grid, self.lasts, self.lows,
                              self.highs, divide) if divide >= 0 else None
        self._seen = (None, None, None)

    def table(self, color, opacity, lo, hi, size=256):
        """VTK's transfer functions sampled over ``[lo, hi]``: ``(size, 4)`` f32."""
        x = np.linspace(lo, hi, size)
        rgb = np.array([color.GetColor(v) for v in x])
        a = np.array([opacity.GetValue(v) for v in x])
        return np.concatenate([rgb, a[:, None]], axis=1).astype(np.float32)

    def render(self, camera, size, table, lo, hi, depth=None, sample_distance=0.5,
               unit_distance=1.0, min_step=0.0, jitter=0.0, aspect=None, **_):
        """The volume seen by ``camera`` (a ``vtkCamera``) at ``size`` (w, h):
        ``(h, w, 4)`` f32, premultiplied, row 0 at the top; as
        ``tack_composite.TackRenderer.render`` (``aspect``: the viewport's width
        over its height where ``size`` is a smaller picture of it). ``timings``
        holds the parts."""
        import time

        out = self._march(camera, size, table, lo, hi, depth, sample_distance, unit_distance,
                          min_step, jitter, aspect)
        t = time.perf_counter()
        with self.octree.device:
            picture = out.numpy().reshape(size[1], size[0], 4)
        self.timings["read_ms"] = 1000 * (time.perf_counter() - t)
        return picture

    def render_over(self, frame, camera, size, table, lo, hi, depth=None, sample_distance=0.5,
                    unit_distance=1.0, min_step=0.0, jitter=0.0, aspect=None):
        """The volume blended over ``frame`` (``(h, w, 4)`` u8, row 0 at the top, the
        colour VTK drew) on the device: ``(h, w, 4)`` u8. The volume's RGBA never
        comes to the host; ``timings["blend_ms"]`` is the upload, the kernel
        and the read."""
        import time

        out = self._march(camera, size, table, lo, hi, depth, sample_distance, unit_distance,
                          min_step, jitter, aspect)
        t = time.perf_counter()
        with self.octree.device:
            colour = tack.array(np.ascontiguousarray(frame, np.uint8).reshape(-1), tack.u8)
            blended = tack.empty((len(colour),), tack.u8)
            _blend(out, colour, blended)
            picture = blended.numpy().reshape(size[1], size[0], 4)
        self.timings["blend_ms"] = 1000 * (time.perf_counter() - t)
        return picture

    def _march(self, camera, size, table, lo, hi, depth, sample_distance, unit_distance,
               min_step, jitter, aspect=None):
        """The march: the volume's premultiplied RGBA, a device array of ``w * h``."""
        import time

        tack_march, tack_composite, volume_tack = self._modules
        t0 = time.perf_counter()
        w, h = size
        count = w * h
        m = np.array([[camera.GetCompositeProjectionTransformMatrix(aspect or w / h, 0, 1)
                       .GetElement(i, j) for j in range(4)] for i in range(4)])
        f32 = np.float32
        with self.octree.device:
            inverse = tack.array(np.linalg.inv(m).astype(f32).reshape(-1))
            if depth is None:
                depths, ends = tack.empty((1,), tack.f32), 0
            else:
                depths, ends = tack.array(np.ascontiguousarray(depth, f32).reshape(-1)), 1
            starts = tack.empty((count,), tack.vec3(tack.f32))
            dirs = tack.empty((count,), tack.vec3(tack.f32))
            tmins = tack.empty((count,), tack.f32)
            tmaxs = tack.empty((count,), tack.f32)
            tack_composite._rays(inverse, depths, ends, w, h, *self.bounds, starts, dirs, tmins,
                                 tmaxs)
            key = (table.tobytes(), lo, hi)
            if self._seen[0] != key:
                seen = volume_tack.visible(self.lows, self.highs, table, lo, hi)
                if self.sub is None:
                    divided = tack.full((len(seen),), -1, tack.i32)
                    near = tack.zeros((1,), tack.i32)
                else:
                    divided = tack.array(self.sub["index"], tack.i32)
                    near = tack_march.sub_away(self.sub, table, lo, hi)
                words = tack_march.macro_words(seen, self.firsts, self.lasts, self.grid["dims"],
                                               divided)
                self._seen = (key, seen, (words, near))
            out = tack.empty((count,), tack.vec4(tack.f32))
            words, near = self._seen[2]
            asked = self.divide
            if self.sub is not None and min_step >= min(self.grid["size"]) / (1 << self.divide):
                asked = -1
            tack.sync()
            t1 = time.perf_counter()
            tack_march.march(self.tents, starts, dirs, tmins, tmaxs, tack.array(table, tack.vec4()),
                             words, float(lo), float(hi), float(sample_distance),
                             float(min_step), float(unit_distance * self.cell), *self.grid["dims"],
                             *self.grid["origin"], *self.grid["size"], float(0.01 * self.cell),
                             float(jitter), int(w), near, asked, out)
            tack.sync()
            t2 = time.perf_counter()
        self.timings = {"rays_ms": 1000 * (t1 - t0), "march_ms": 1000 * (t2 - t1)}
        return out
