# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The AMR volume's march, made for Tack's kernels rather than shared with a shader.

The reconstruction is ``volume_tack.AMRVolume``'s (tent basis functions of every active
cell); what differs is how a sample finds its cells and how a ray steps:

- **One record per box, one per level** (``Tents``): a box's corners and where its
  values start are eight ints side by side, a level's brick table and cell sizes
  likewise, where ``AMRVolume`` reads a dozen small arrays.
- **Refined cells are a value**, ``REFINED``, so a cell is one load, not a value and a
  bit.
- **Bricks are a power of two wide**, so a cell's brick is a shift.
- **32-bit indices**: a ``range`` loop's variable is 64-bit on GPUs, and so was every
  index made from a level or a corner.
- **One lookup a step**: the sample also says how fine the data is there, which sets
  the step; ``AMRVolume`` looked that up apart.
- **Empty space by distance**: each macrocell holds how many macrocells away the
  nearest one that can show is, and a ray leaves that whole block in one step.
- **Empty space within a macrocell**: a macrocell the fine levels reach is divided
  again (``sub_grids``: ``1 << shift`` cells along each axis), each cell holding how
  many of them away, within its macrocell, the nearest that can show is. The macrocell's
  word says where its cells are, so asking costs one load.

``march`` samples at the start of each step, where ``volume_tack._march`` samples at
its middle, so pictures agree closely, not exactly; ``Tents.sample`` gives
``AMRVolume.sample``'s values.
"""

from math import ceil, floor, pow

import numpy as np
import volume_tack
from volume_tack import MAX_STEPS, OPAQUE, _jitter, _safe_inverse, _transfer

import tack

#: A refined cell's value, and what an active cell's value is above.
REFINED = tack.constant(-3.0e38)
ACTIVE = tack.constant(-1.0e38)
#: The farthest a macrocell's distance to one that can show is counted (6 bits).
FAR = tack.constant(60)


@tack.kernel
def _mark_refined(volume, out):
    """Each cell's value, ``REFINED`` where a finer level covers it."""
    for n in range(out.shape[0]):
        value, active, x, y, z, sx, sy, sz, level = volume.cell_info(n)
        out[n] = value if active == 1 else REFINED


@tack.device_class
class Tents:
    """``volume_tack.AMRVolume``'s tents, laid out for the march (see the module).

    - ``values``: every box's cells one after another, ``REFINED`` where refined;
    - ``boxes``: 8 ints a box, its first cell, one past its last, and where its values
      start;
    - ``levels``: 8 ints a level, its bricks' first cell, the shift of a brick's width,
      the bricks along each axis, and where its bricks start in ``bricks``;
    - ``sizes``: 4 floats a level, one over the cell size per axis, and the smallest;
    - ``bricks``: each brick's box, or -1."""

    def __init__(self, volume):
        host = {name: getattr(volume, name).numpy() for name in
                ("box_lo", "box_dims", "box_level", "box_values", "spacing")}
        lo = host["box_lo"].reshape(-1, 3).astype(np.int64)
        hi = lo + host["box_dims"].reshape(-1, 3)
        count = len(lo)
        boxes = np.zeros((count, 8), np.int32)
        boxes[:, 0:3], boxes[:, 3:6], boxes[:, 6] = lo, hi, host["box_values"]
        levels = np.zeros((volume.nlevels, 8), np.int32)
        sizes = np.zeros((volume.nlevels, 4), np.float32)
        spacing = host["spacing"].reshape(-1, 3).astype(np.float64)
        bricks = []
        at = 0
        for level in range(volume.nlevels):
            ids = np.flatnonzero(host["box_level"] == level)
            llo, lhi = lo[ids], hi[ids]
            # The widest power of two dividing every corner: a brick lies in one box or
            # none.
            shared = int(np.gcd.reduce(np.concatenate([llo.ravel(), lhi.ravel()])))
            shift = 0
            while shared > 0 and shared % (2 << shift) == 0:
                shift += 1
            start = llo.min(axis=0)
            dims = (lhi.max(axis=0) - start) >> shift
            table = np.full(tuple(int(d) for d in dims[::-1]), -1, np.int32)
            for box, blo, bhi in zip(ids, llo, lhi):
                a, z = (blo - start) >> shift, (bhi - start) >> shift
                table[a[2]:z[2], a[1]:z[1], a[0]:z[0]] = box
            levels[level] = (*start, shift, *dims, at)
            sizes[level] = (*(1.0 / spacing[level]), spacing[level].min())
            bricks.append(table.reshape(-1))
            at += table.size
        values = tack.empty((len(volume.values),), tack.f32)
        _mark_refined(volume, values)
        self.values = values
        self.boxes = tack.array(boxes.reshape(-1))
        self.levels = tack.array(levels.reshape(-1))
        self.sizes = tack.array(sizes.reshape(-1))
        self.bricks = tack.array(np.concatenate(bricks))
        self.ox, self.oy, self.oz = volume.ox, volume.oy, volume.oz
        self.coarse = float(spacing[0].min())
    @tack.func
    def box_of(self, level, i, j, k):
        """The box of ``level`` holding cell ``(i, j, k)``, or -1."""
        r = 8 * level
        shift = self.levels[r + 3]
        nx = self.levels[r + 4]
        ny = self.levels[r + 5]
        bi = (i - self.levels[r]) >> shift
        bj = (j - self.levels[r + 1]) >> shift
        bk = (k - self.levels[r + 2]) >> shift
        box = -1
        if bi >= 0 and bi < nx and bj >= 0 and bj < ny and bk >= 0 and bk < self.levels[r + 6]:
            box = self.bricks[self.levels[r + 7] + bi + nx * (bj + ny * bk)]
        return box

    @tack.func
    def cell(self, level, i, j, k):
        """Cell ``(i, j, k)`` of ``level``: its value, ``REFINED`` where it is refined
        or in no box."""
        out = REFINED
        box = self.box_of(level, i, j, k)
        if box >= 0:
            q = 8 * box
            dx = self.boxes[q + 3] - self.boxes[q]
            dy = self.boxes[q + 4] - self.boxes[q + 1]
            out = self.values[self.boxes[q + 6] + (i - self.boxes[q])
                              + dx * ((j - self.boxes[q + 1]) + dy * (k - self.boxes[q + 2]))]
        return out

    @tack.func
    def level_sum(self, level, p):
        """The tents of ``level`` at ``p``: the sum of weight times value and of weight
        over its active cells around ``p``, and whether the cell holding ``p`` is one
        of them (1 or 0)."""
        s = 4 * level
        ux = (p[0] - self.ox) * self.sizes[s] - 0.5
        uy = (p[1] - self.oy) * self.sizes[s + 1] - 0.5
        uz = (p[2] - self.oz) * self.sizes[s + 2] - 0.5
        lx = floor(ux)
        ly = floor(uy)
        lz = floor(uz)
        x1 = ux - lx
        y1 = uy - ly
        z1 = uz - lz
        x0 = 1.0 - x1
        y0 = 1.0 - y1
        z0 = 1.0 - z1
        ix = int(lx)
        iy = int(ly)
        iz = int(lz)
        v000 = REFINED
        v100 = REFINED
        v010 = REFINED
        v110 = REFINED
        v001 = REFINED
        v101 = REFINED
        v011 = REFINED
        v111 = REFINED
        box = self.box_of(level, ix, iy, iz)
        q = 8 * max(box, 0)
        if (box >= 0 and ix + 1 < self.boxes[q + 3] and iy + 1 < self.boxes[q + 4]
                and iz + 1 < self.boxes[q + 5]):
            # All eight in the first's box: read them directly.
            dx = self.boxes[q + 3] - self.boxes[q]
            dxy = dx * (self.boxes[q + 4] - self.boxes[q + 1])
            n = (self.boxes[q + 6] + (ix - self.boxes[q])
                 + dx * (iy - self.boxes[q + 1]) + dxy * (iz - self.boxes[q + 2]))
            v000 = self.values[n]
            v100 = self.values[n + 1]
            v010 = self.values[n + dx]
            v110 = self.values[n + dx + 1]
            v001 = self.values[n + dxy]
            v101 = self.values[n + dxy + 1]
            v011 = self.values[n + dxy + dx]
            v111 = self.values[n + dxy + dx + 1]
        else:
            # Unless the eight share a brick that holds no box, look each up.
            r = 8 * level
            last = (1 << self.levels[r + 3]) - 1
            if (box >= 0 or ((ix - self.levels[r]) & last) == last
                    or ((iy - self.levels[r + 1]) & last) == last
                    or ((iz - self.levels[r + 2]) & last) == last
                    or self.box_of(level, ix + 1, iy + 1, iz + 1) >= 0):
                v000 = self.cell(level, ix, iy, iz)
                v100 = self.cell(level, ix + 1, iy, iz)
                v010 = self.cell(level, ix, iy + 1, iz)
                v110 = self.cell(level, ix + 1, iy + 1, iz)
                v001 = self.cell(level, ix, iy, iz + 1)
                v101 = self.cell(level, ix + 1, iy, iz + 1)
                v011 = self.cell(level, ix, iy + 1, iz + 1)
                v111 = self.cell(level, ix + 1, iy + 1, iz + 1)
        num = 0.0
        den = 0.0
        if v000 > ACTIVE:
            num += x0 * y0 * z0 * v000
            den += x0 * y0 * z0
        if v100 > ACTIVE:
            num += x1 * y0 * z0 * v100
            den += x1 * y0 * z0
        if v010 > ACTIVE:
            num += x0 * y1 * z0 * v010
            den += x0 * y1 * z0
        if v110 > ACTIVE:
            num += x1 * y1 * z0 * v110
            den += x1 * y1 * z0
        if v001 > ACTIVE:
            num += x0 * y0 * z1 * v001
            den += x0 * y0 * z1
        if v101 > ACTIVE:
            num += x1 * y0 * z1 * v101
            den += x1 * y0 * z1
        if v011 > ACTIVE:
            num += x0 * y1 * z1 * v011
            den += x0 * y1 * z1
        if v111 > ACTIVE:
            num += x1 * y1 * z1 * v111
            den += x1 * y1 * z1
        # The cell holding p is the nearest of the eight.
        a0 = v000 if x1 < 0.5 else v100
        a1 = v010 if x1 < 0.5 else v110
        a2 = v001 if x1 < 0.5 else v101
        a3 = v011 if x1 < 0.5 else v111
        b0 = a0 if y1 < 0.5 else a1
        b1 = a2 if y1 < 0.5 else a3
        holding = b0 if z1 < 0.5 else b1
        return num, den, 1 if holding > ACTIVE else 0

    @tack.func
    def sample(self, p, first, stop):
        """The value at ``p`` from levels ``first`` up to ``stop`` (the levels whose
        tents reach ``p``'s macrocell), and the cell size of the finest level with an
        active cell holding ``p`` (level 0's where none does)."""
        num = 0.0
        den = 0.0
        size = self.coarse
        for wide in range(first, stop):
            level = tack.i32(wide)
            n, d, holds = self.level_sum(level, p)
            num += n
            den += d
            if holds == 1:
                size = self.sizes[4 * level + 3]
        out = 0.0
        if den > 0.0:
            out = num / den
        return out, size


@tack.kernel
def _sub_ranges(tents, sub_macro, firsts, lasts, gx, gy, ox, oy, oz, mx, my, mz, shift, lows,
                highs):
    """Per cell of each divided macrocell (``sub_macro`` its macrocell, ``1 << shift``
    cells along each axis, x fastest) the least and greatest value of the tents that
    reach it: the active cells, of the levels reaching the macrocell, within a cell
    of it (and a hundredth, for rounding)."""
    for n in range(lows.shape[0]):
        local = tack.i32(n)
        across = 1 << shift
        m = sub_macro[local >> (3 * shift)]
        wx = mx / tack.f32(across)
        wy = my / tack.f32(across)
        wz = mz / tack.f32(across)
        x0 = ox + tack.f32(m % gx) * mx + tack.f32(local & (across - 1)) * wx - tents.ox
        y0 = (oy + tack.f32((m // gx) % gy) * my
              + tack.f32((local >> shift) & (across - 1)) * wy - tents.oy)
        z0 = (oz + tack.f32(m // (gx * gy)) * mz
              + tack.f32((local >> (2 * shift)) & (across - 1)) * wz - tents.oz)
        low = 3.0e38
        high = -3.0e38
        for other in range(firsts[m], lasts[m] + 1):
            c = tack.i32(other)
            i0 = int(floor(x0 * tents.sizes[4 * c] - 1.51)) + 1
            i1 = int(ceil((x0 + wx) * tents.sizes[4 * c] + 0.51))
            j0 = int(floor(y0 * tents.sizes[4 * c + 1] - 1.51)) + 1
            j1 = int(ceil((y0 + wy) * tents.sizes[4 * c + 1] + 0.51))
            k0 = int(floor(z0 * tents.sizes[4 * c + 2] - 1.51)) + 1
            k1 = int(ceil((z0 + wz) * tents.sizes[4 * c + 2] + 0.51))
            for k in range(k0, k1):
                for j in range(j0, j1):
                    for i in range(i0, i1):
                        v = tents.cell(c, tack.i32(i), tack.i32(j), tack.i32(k))
                        if v > ACTIVE:
                            low = min(low, v)
                            high = max(high, v)
        lows[n] = low
        highs[n] = high


@tack.kernel
def _nearest_sub(src, along, shift, out):
    """Along one axis (``along`` 0, 1 or 2) of each divided macrocell: the least of each
    cell and its two neighbours, a neighbour in another macrocell counting as 0."""
    for n in range(src.shape[0]):
        stride = 1 << (along * shift)
        c = (tack.i32(n) >> (along * shift)) & ((1 << shift) - 1)
        least = src[n]
        if c > 0:
            least = min(least, src[n - stride])
        else:
            least = 0
        if c < (1 << shift) - 1:
            least = min(least, src[n + stride])
        else:
            least = 0
        out[n] = least


def sub_grids(tents, grid, firsts, lasts, shift=3):
    """The macrocells of ``grid`` divided again, ``1 << shift`` cells along each axis:
    those a level reaches (``lasts``) whose cells are at most half such a cell. Gives
    per macrocell its place among the divided ones or -1 (``index``, on the host), and
    per cell of those the range of the tents reaching it (``lows``, ``highs``)."""
    sizes = tents.sizes.numpy().reshape(-1, 4)[:, 3]
    finest = lasts.numpy()
    width = min(grid["size"]) / (1 << shift)
    divided = np.flatnonzero((finest >= 0) & (sizes[np.maximum(finest, 0)] * 2
                                              <= width * (1 + 1e-6)))
    index = np.full(len(finest), -1, np.int32)
    index[divided] = np.arange(len(divided), dtype=np.int32)
    count = max(len(divided), 1) << (3 * shift)
    lows, highs = tack.empty((count,), tack.f32), tack.empty((count,), tack.f32)
    macros = tack.array(divided.astype(np.int32) if len(divided) else np.zeros(1, np.int32))
    _sub_ranges(tents, macros, firsts, lasts, grid["dims"][0], grid["dims"][1],
                *(float(v) for v in grid["origin"]), *(float(v) for v in grid["size"]),
                int(shift), lows, highs)
    return {"index": index, "lows": lows, "highs": highs, "shift": int(shift),
            "count": len(divided)}


def sub_away(sub, table, lo, hi):
    """Per cell of the divided macrocells: 0 where the transfer function ``table`` over
    ``[lo, hi]`` can show it, else how many cells away, within its macrocell, the
    nearest it can show is, the macrocell's faces counting as one."""
    seen = volume_tack.visible(sub["lows"], sub["highs"], table, lo, hi)
    count = len(seen)
    far = 1 << max(sub["shift"] - 1, 0)
    away = tack.empty((count,), tack.i32)
    _start(seen, far, away)
    a = tack.empty((count,), tack.i32)
    b = tack.empty((count,), tack.i32)
    for _ in range(far):
        _nearest_sub(away, 0, sub["shift"], a)
        _nearest_sub(a, 1, sub["shift"], b)
        _nearest_sub(b, 2, sub["shift"], a)
        _further(a, away)
    return away


@tack.kernel
def _sample_at(tents, ps, first, stop, values, sizes):
    for n in range(ps.shape[0]):
        value, size = tents.sample(ps[n], first, stop)
        values[n] = value
        sizes[n] = size


# The macrocells: how far the nearest that can show is, and the levels reaching each.

@tack.kernel
def _nearest(src, stride, count, out):
    """Along one axis (``count`` cells, ``stride`` apart): the least of each cell and
    its two neighbours."""
    for m in range(src.shape[0]):
        c = (tack.i32(m) // stride) % count
        least = src[m]
        if c > 0:
            least = min(least, src[m - stride])
        if c < count - 1:
            least = min(least, src[m + stride])
        out[m] = least


@tack.kernel
def _further(spread, distance):
    for m in range(distance.shape[0]):
        distance[m] = min(distance[m], spread[m] + 1)


@tack.kernel
def _start(seen, far, distance):
    for m in range(seen.shape[0]):
        distance[m] = 0 if seen[m] == 1 else far


@tack.kernel
def _pack(distance, firsts, lasts, divided, out):
    """A macrocell's word: the distance (6 bits), the first level reaching it and one
    past the last (4 bits each; none: 0 and 0), and one more than its place among the
    divided macrocells (0: not divided)."""
    for m in range(out.shape[0]):
        first = min(firsts[m], 15)
        stop = max(lasts[m] + 1, 0)
        if stop == 0:
            first = 0
        out[m] = distance[m] | (first << 6) | (stop << 10) | ((divided[m] + 1) << 14)


def macro_words(seen, firsts, lasts, dims, divided):
    """Per macrocell of a grid of ``dims``, one word: in its low byte how many
    macrocells away (the largest of the axes' differences, at most ``FAR``) the nearest
    with ``seen`` 1 is, then the levels whose tents reach it and where its cells are
    (``divided``: ``sub_grids``' index)."""
    gx, gy, gz = dims
    count = len(seen)
    distance = tack.empty((count,), tack.i32)
    _start(seen, int(FAR), distance)
    a = tack.empty((count,), tack.i32)
    b = tack.empty((count,), tack.i32)
    for _ in range(int(FAR)):
        _nearest(distance, 1, gx, a)
        _nearest(a, gx, gy, b)
        _nearest(b, gx * gy, gz, a)
        _further(a, distance)
    out = tack.empty((count,), tack.i32)
    _pack(distance, firsts, lasts, tack.array(divided), out)
    return out


@tack.func
def _leave(q, d, inv, lo, hi, m):
    """Where the ray ``q + t d`` leaves macrocells ``lo`` up to ``hi`` (``m`` wide) on
    one axis."""
    face = tack.f32(lo) * m
    if d >= 0.0:
        face = tack.f32(hi) * m
    return (face - q) * inv


@tack.kernel
def march(tents, starts, dirs, tmins, tmaxs, tf, macro, lo, hi, factor, min_step, unit, gx, gy,
          gz, ox, oy, oz, mx, my, mz, skip, jitter, width, near, shift, out):
    """Each ray's premultiplied colour and opacity: steps of ``factor`` local cells (at
    least ``min_step``), the sample at each step's start, opacity per ``unit`` of
    length, blocks of macrocells that cannot show left in one step, and within a
    divided macrocell blocks of its cells (``near``: ``sub_away``; ``shift`` -1: not
    asked)."""
    for k in range(starts.shape[0]):
        p0 = starts[k]
        d = dirs[k]
        ix = _safe_inverse(d[0])
        iy = _safe_inverse(d[1])
        iz = _safe_inverse(d[2])
        qx = p0[0] - ox
        qy = p0[1] - oy
        qz = p0[2] - oz
        t = tmins[k]
        t1 = tmaxs[k]
        r = 0.0
        g = 0.0
        b = 0.0
        a = 0.0
        steps = 0
        pixel = tack.i32(k)
        offset = jitter * _jitter(pixel % width, pixel // width)
        fresh = 1
        scale = 1.0 / (hi - lo)
        first = 0
        stop = 0
        while t < t1 and steps < MAX_STEPS:
            # On to the next point that can show. A loop of its own: the threads of a
            # GPU's group run in step, and this way they sample together, where one
            # skipping while another samples would make each wait for the other.
            seeking = 1
            while seeking == 1 and t < t1 and steps < MAX_STEPS:
                steps += 1
                mi = min(max(int(floor((qx + d[0] * t) / mx)), 0), gx - 1)
                mj = min(max(int(floor((qy + d[1] * t) / my)), 0), gy - 1)
                mk = min(max(int(floor((qz + d[2] * t) / mz)), 0), gz - 1)
                word = macro[mi + gx * (mj + gy * mk)]
                away = word & 63
                if away > 0:
                    reach = away - 1
                    ex = _leave(qx, d[0], ix, mi - reach, mi + reach + 1, mx)
                    ey = _leave(qy, d[1], iy, mj - reach, mj + reach + 1, my)
                    ez = _leave(qz, d[2], iz, mk - reach, mk + reach + 1, mz)
                    t = max(min(ex, ey, ez), t) + skip
                    fresh = 1
                else:
                    first = (word >> 6) & 15
                    stop = (word >> 10) & 15
                    seeking = 0
                    sub = (word >> 14) - 1
                    if sub >= 0 and shift >= 0:
                        across = 1 << shift
                        wx = mx / tack.f32(across)
                        wy = my / tack.f32(across)
                        wz = mz / tack.f32(across)
                        # From the macrocell's corner.
                        cx = qx - tack.f32(mi) * mx
                        cy = qy - tack.f32(mj) * my
                        cz = qz - tack.f32(mk) * mz
                        si = min(max(int(floor((cx + d[0] * t) / wx)), 0), across - 1)
                        sj = min(max(int(floor((cy + d[1] * t) / wy)), 0), across - 1)
                        sk = min(max(int(floor((cz + d[2] * t) / wz)), 0), across - 1)
                        hides = near[(sub << (3 * shift)) + si + (sj << shift)
                                     + (sk << (2 * shift))]
                        if hides > 0:
                            reach = hides - 1
                            ex = _leave(cx, d[0], ix, si - reach, si + reach + 1, wx)
                            ey = _leave(cy, d[1], iy, sj - reach, sj + reach + 1, wy)
                            ez = _leave(cz, d[2], iz, sk - reach, sk + reach + 1, wz)
                            t = max(min(ex, ey, ez), t) + skip
                            fresh = 1
                            seeking = 1
            if seeking == 0:
                value, size = tents.sample(p0 + d * t, first, stop)
                h = max(factor * size, min_step)
                if fresh == 1:
                    fresh = 0
                    if offset > 0.0:
                        t += offset * h
                        value, size = tents.sample(p0 + d * t, first, stop)
                        h = max(factor * size, min_step)
                c = _transfer(tf, (value - lo) * scale)
                if c[3] > 0.0:
                    alpha = 1.0 - pow(1.0 - c[3], h / unit)
                    w = (1.0 - a) * alpha
                    r += w * c[0]
                    g += w * c[1]
                    b += w * c[2]
                    a += w
                    if a >= OPAQUE:
                        break
                t += h
        out[k] = tack.vec4(r, g, b, a)


def check_samples(volume, tents, bounds, count=200_000, seed=0):
    """The largest difference between ``tents``' and ``volume``'s values, and between
    their cell sizes, at ``count`` random points in ``bounds``."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    points = tack.array((lo + rng.random((count, 3)) * (hi - lo)).astype(np.float32),
                        tack.vec3())
    values, sizes = tack.empty((count,), tack.f32), tack.empty((count,), tack.f32)
    _sample_at(tents, points, 0, volume.nlevels, values, sizes)
    expected = tack.empty((count,), tack.f32)
    volume_tack._sample_at(volume, points, expected)
    expected_sizes = tack.empty((count,), tack.f32)
    volume_tack._cell_size_at(volume, points, expected_sizes)
    return (float(np.abs(values.numpy() - expected.numpy()).max()),
            float(np.abs(sizes.numpy() - expected_sizes.numpy()).max()))
