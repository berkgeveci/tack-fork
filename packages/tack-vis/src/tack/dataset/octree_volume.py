# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A cell-centred octree volume rendered by Tack's march (study,
``study/octree-filters``): the sampler the AMR volume work's march takes
(``tack_march.march`` on ``proto/webgpu-surfaces``, ``packages/tack-vis/
examples/amr_volume``: rays marched by a Tack kernel on Tack's device and
blended into VTK's frame), written for an octree.

- **The lookup is a descent.** From the root cell holding the point, down the
  children table by the point's octant bits, one i32 read a level: no keys, no
  brick tables, nothing 64-bit (``OctreeTents.leaf_at``).
- **The step is the leaf.** ``sample(p, first, stop)`` returns the value and the
  leaf's size, which sets the march's step; the levels ``first``, ``stop`` the
  macrocell word carries are not needed.
- **Values with sentinels.** One f32 a node: the value on a visible leaf,
  ``REFINED`` (descend) on a refined node, ``HIDDEN`` (nothing here) on a hidden
  leaf -- one load a node, as the AMR sampler marks them.
- **Reconstruction.** ``blend=0``: the leaf's value, piecewise constant, exactly
  what ``probe`` gives. ``blend=1``: tent basis functions, as the AMR sampler
  blends levels: every visible leaf is a tent its own size wide about its
  centre, and the value is the weighted mean of the tents reaching the point:
  the leaf's and those of the leaves holding the points one cell over on the
  point's side along each axis (seven more descents), each counted once.
  Within a uniform region this is trilinear interpolation of the centres;
  across a level change it blends the coarse and fine tents.
- **Empty space.** Per macrocell of a lattice ``depth`` levels below the roots,
  the least and greatest value of the leaves whose tents reach it, from the
  per-node ranges the contour keeps (``ranges``): a node at that level or a
  coarser leaf writes its subtree's range over its block, widened by half its
  size, with float atomics; the AMR march's ``visible`` and ``macro_words``
  then make the distance words as for AMR.

The AMR modules are imported from their worktree
(``OctreeRenderer(..., amr_volume=<path>)``), not copied, so their march flows
through unchanged.
"""

import sys
from math import floor

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import _root_axes, _root_cells, _unkey
from tack.dataset.octree_contour import ranges
from tack.dataset.octree_topology import children

__all__ = ["OctreeRenderer", "OctreeTents", "macro_ranges", "octree_tents"]

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

    def __init__(self, octree, name, blend=1):
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
            #: Per node, whether its subtree can show (1) under the transfer
            #: function: every node until ``set_visible``.
            self.seen = tack.ones((max(octree.num_nodes, 1),), tack.i32)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.ox, self.oy, self.oz = (float(axes[a][0] + origin[a]) for a in range(3))
        self.sx, self.sy, self.sz = (float(s) for s in sizes[0])
        self.coarse = float(min(sizes[0]))
        self.finest = float(min(sizes[0]) / (1 << octree.finest))
        self.blend = int(blend)
        #: What ``OctreeRenderer`` needs beside the sampler.
        self.bounds = tuple(float(v) for v in np.stack([
            [axes[a][0] + origin[a] for a in range(3)],
            [axes[a][-1] + origin[a] for a in range(3)]], axis=1).reshape(-1))

    @tack.func
    def leaf_at(self, p):
        """The leaf holding ``p`` (-1 beyond the roots) and its box: lower corner and
        sides, by a descent from the root cell."""
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
        lx = self.ox + tack.f32(ti) * self.sx
        ly = self.oy + tack.f32(tj) * self.sy
        lz = self.oz + tack.f32(tk) * self.sz
        hx = self.sx
        hy = self.sy
        hz = self.sz
        while n >= 0 and self.values[n] == REFINED:
            jx = 1 if fx >= 0.5 else 0
            jy = 1 if fy >= 0.5 else 0
            jz = 1 if fz >= 0.5 else 0
            fx = 2.0 * fx - tack.f32(jx)
            fy = 2.0 * fy - tack.f32(jy)
            fz = 2.0 * fz - tack.f32(jz)
            hx = 0.5 * hx
            hy = 0.5 * hy
            hz = 0.5 * hz
            lx = lx + tack.f32(jx) * hx
            ly = ly + tack.f32(jy) * hy
            lz = lz + tack.f32(jz) * hz
            n = self.table[(self.number[n] - 1) * 8 + jx + 2 * jy + 4 * jz]
        return n, lx, ly, lz, hx, hy, hz

    @tack.func
    def hidden(self, p, first, stop):
        """Whether the point is in a subtree that cannot show (1 or 0), and that
        subtree's box: the first node on the descent to ``p`` whose range the
        transfer function gives no opacity (``seen`` 0), the whole tree's
        skipping at every scale; 0 and the leaf's box otherwise."""
        rx = (p[0] - self.ox) / self.sx
        ry = (p[1] - self.oy) / self.sy
        rz = (p[2] - self.oz) / self.sz
        ti = min(max(int(floor(rx)), 0), self.nx - 1)  # noqa: RUF046
        tj = min(max(int(floor(ry)), 0), self.ny - 1)  # noqa: RUF046
        tk = min(max(int(floor(rz)), 0), self.nz - 1)  # noqa: RUF046
        fx = min(max(rx - tack.f32(ti), 0.0), 0.999999)
        fy = min(max(ry - tack.f32(tj), 0.0), 0.999999)
        fz = min(max(rz - tack.f32(tk), 0.0), 0.999999)
        n = self.roots[ti + self.nx * (tj + self.ny * tk)]
        lx = self.ox + tack.f32(ti) * self.sx
        ly = self.oy + tack.f32(tj) * self.sy
        lz = self.oz + tack.f32(tk) * self.sz
        hx = self.sx
        hy = self.sy
        hz = self.sz
        hides = 0
        if n < 0:
            hides = 1
        else:
            while self.seen[n] == 1 and self.values[n] == REFINED:
                jx = 1 if fx >= 0.5 else 0
                jy = 1 if fy >= 0.5 else 0
                jz = 1 if fz >= 0.5 else 0
                fx = 2.0 * fx - tack.f32(jx)
                fy = 2.0 * fy - tack.f32(jy)
                fz = 2.0 * fz - tack.f32(jz)
                hx = 0.5 * hx
                hy = 0.5 * hy
                hz = 0.5 * hz
                lx = lx + tack.f32(jx) * hx
                ly = ly + tack.f32(jy) * hy
                lz = lz + tack.f32(jz) * hz
                n = self.table[(self.number[n] - 1) * 8 + jx + 2 * jy + 4 * jz]
            if self.seen[n] == 0:
                hides = 1
        return hides, lx, ly, lz, hx, hy, hz

    @tack.func
    def tent(self, p, lx, ly, lz, hx, hy, hz):
        """A leaf's tent at ``p``: 1 at its centre, 0 one cell away."""
        wx = max(1.0 - abs(p[0] - lx - 0.5 * hx) / hx, 0.0)
        wy = max(1.0 - abs(p[1] - ly - 0.5 * hy) / hy, 0.0)
        wz = max(1.0 - abs(p[2] - lz - 0.5 * hz) / hz, 0.0)
        return wx * wy * wz

    @tack.func
    def sample(self, p, first, stop):
        n, lx, ly, lz, hx, hy, hz = self.leaf_at(p)
        size = self.coarse
        value = 0.0
        if n >= 0:
            size = min(hx, hy, hz)
            v = self.values[n]
            if v > ACTIVE:
                value = v
            if self.blend == 1:
                num = 0.0
                den = 0.0
                if v > ACTIVE:
                    w = self.tent(p, lx, ly, lz, hx, hy, hz)
                    num = w * v
                    den = w
                seen = tack.local_array(tack.i32, 8)
                seen[0] = n
                count = 1
                cx = lx + 0.5 * hx
                cy = ly + 0.5 * hy
                cz = lz + 0.5 * hz
                dx = hx if p[0] >= cx else -hx
                dy = hy if p[1] >= cy else -hy
                dz = hz if p[2] >= cz else -hz
                for s in range(1, 8):
                    q = tack.vec3(cx + dx * tack.f32(s & 1), cy + dy * tack.f32((s >> 1) & 1),
                                  cz + dz * tack.f32((s >> 2) & 1))
                    m, mx, my, mz, ax, ay, az = self.leaf_at(q)
                    if m >= 0:
                        new = 1
                        for i in range(count):
                            if seen[i] == m:
                                new = 0
                        if new == 1:
                            seen[count] = m
                            count += 1
                            u = self.values[m]
                            if u > ACTIVE:
                                w = self.tent(p, mx, my, mz, ax, ay, az)
                                num += w * u
                                den += w
                if den > 0.0:
                    value = num / den
        return value, size


def octree_tents(octree, name, blend=1):
    """``OctreeTents`` of field ``name``: the march's sampler."""
    return OctreeTents(octree, name, blend)


# -- the macrocells ---------------------------------------------------------------------


@tack.kernel
def _macro_ranges(tree, level, key, marks, refined, lo, hi, nx, ny, nz, depth, lows, highs):
    # Each node at ``depth``, or a leaf above it, writes its subtree's range over
    # its block of the lattice at ``depth``, widened by half its size (its tents
    # reach that far), by atomics.
    for n in range(tree.shape[0]):
        lv = level[n]
        if lv <= depth and (lv == depth or (marks[n] & refined) == 0) and lo[n] <= hi[n]:
            i, j, m = _unkey(key[n], lv, 3)
            t = tree[n]
            span = 1 << (depth - lv)
            reach = max(span >> 1, 1)      # a finer leaf's tent reaches less than one cell
            gx = nx << depth
            gy = ny << depth
            gz = nz << depth
            bi = ((t % nx) << depth) + (tack.i64(i) << (depth - lv))
            bj = (((t // nx) % ny) << depth) + (tack.i64(j) << (depth - lv))
            bk = ((t // (nx * ny)) << depth) + (tack.i64(m) << (depth - lv))
            a = lo[n]
            b = hi[n]
            for dz in range(max(bk - reach, 0), min(bk + span + reach, gz)):
                for dy in range(max(bj - reach, 0), min(bj + span + reach, gy)):
                    for dx in range(max(bi - reach, 0), min(bi + span + reach, gx)):
                        cell = dx + gx * (dy + gy * dz)
                        tack.atomic_min(lows, cell, a)
                        tack.atomic_max(highs, cell, b)


def macro_ranges(octree, r, most=64):
    """The macrocells of the octree: the lattice ``depth`` levels below the roots,
    the deepest with at most ``most`` cells along the longest axis (at least the
    roots), as the march's grid (``origin``, ``size``, ``dims``), and per
    macrocell the least and greatest value of the leaves whose tents reach it
    (``r``: ``ranges(octree, name)``), f32 device arrays (greatest < least
    where none does)."""
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
    with octree.device:
        lows = tack.full((count,), 3.0e38, tack.f32)
        highs = tack.full((count,), -3.0e38, tack.f32)
        lo = r.lo if r.lo.dtype == tack.f32 else r.lo.astype(tack.f32)
        hi = r.hi if r.hi.dtype == tack.f32 else r.hi.astype(tack.f32)
        if octree.num_nodes:
            _macro_ranges(octree.tree, octree.level, octree.key, octree.marks,
                          int(Ghosts.REFINEDCELL), lo, hi, nx, ny, nz, depth, lows, highs)
    grid = {"origin": tuple(float(axes[a][0] + origin[a]) for a in range(3)), "size": size,
            "dims": tuple(int(d) for d in dims)}
    return grid, lows, highs


# -- the renderer ------------------------------------------------------------------------


class OctreeRenderer:
    """An octree's cell field ``name`` as a volume Tack renders with the AMR work's
    march: ``render`` gives the picture as premultiplied RGBA, to blend into a
    frame (``tack_composite.composite``). ``amr_volume`` is the directory of the
    AMR volume modules (``tack_march``, ``tack_composite``, ``volume_tack``)."""

    def __init__(self, octree, name, amr_volume, blend=1, macrocells=64):
        if amr_volume not in sys.path:
            sys.path.insert(0, str(amr_volume))
        import tack_composite
        import tack_march
        import volume_tack

        self._modules = (tack_march, tack_composite, volume_tack)
        self.octree = octree
        self.tents = octree_tents(octree, name, blend)
        self.ranges = ranges(octree, name)
        self.grid, self.lows, self.highs = macro_ranges(octree, self.ranges, macrocells)
        self.bounds = self.tents.bounds
        self.cell = self.tents.finest
        self._seen = (None, None, None)

    def table(self, color, opacity, lo, hi, size=256):
        """VTK's transfer functions sampled over ``[lo, hi]``: ``(size, 4)`` f32."""
        x = np.linspace(lo, hi, size)
        rgb = np.array([color.GetColor(v) for v in x])
        a = np.array([opacity.GetValue(v) for v in x])
        return np.concatenate([rgb, a[:, None]], axis=1).astype(np.float32)

    def render(self, camera, size, table, lo, hi, depth=None, sample_distance=0.5,
               unit_distance=1.0, min_step=0.0, jitter=0.0):
        """The volume seen by ``camera`` (a ``vtkCamera``) at ``size`` (w, h):
        ``(h, w, 4)`` f32, premultiplied, row 0 at the top; as
        ``tack_composite.TackRenderer.render``. ``timings`` holds the parts."""
        import time

        tack_march, tack_composite, volume_tack = self._modules
        t0 = time.perf_counter()
        w, h = size
        count = w * h
        m = np.array([[camera.GetCompositeProjectionTransformMatrix(w / h, 0, 1)
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
                n = len(seen)
                words = tack_march.macro_words(seen, tack.zeros((n,), tack.i32),
                                               tack.ones((n,), tack.i32), self.grid["dims"])
                # Per node too, for the sampler's ``hidden``.
                r = self.ranges
                self.tents.seen = volume_tack.visible(
                    r.lo if r.lo.dtype == tack.f32 else r.lo.astype(tack.f32),
                    r.hi if r.hi.dtype == tack.f32 else r.hi.astype(tack.f32), table, lo, hi)
                self._seen = (key, seen, words)
            out = tack.empty((count,), tack.vec4(tack.f32))
            tack.sync()
            t1 = time.perf_counter()
            tack_march.march(self.tents, starts, dirs, tmins, tmaxs, tack.array(table, tack.vec4()),
                             self._seen[2], float(lo), float(hi), float(sample_distance),
                             float(min_step), float(unit_distance * self.cell), *self.grid["dims"],
                             *self.grid["origin"], *self.grid["size"], float(0.01 * self.cell),
                             float(jitter), int(w), out)
            tack.sync()
            t2 = time.perf_counter()
            picture = out.numpy().reshape(h, w, 4)
        self.timings = {"rays_ms": 1000 * (t1 - t0), "march_ms": 1000 * (t2 - t1),
                        "read_ms": 1000 * (time.perf_counter() - t2)}
        return picture
