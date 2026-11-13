# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a volume's field *is*, in Tack: ``sample(p)`` at any point, exported as Slang
for the renderer's marcher, and a reference marcher that runs in Tack itself.

Step 1 (one box): ``CellBox`` holds cell-centred values on a uniform box and samples
them trilinearly over the cell centres, clamped at the boundary (within half a cell of
a face the value is the face cells' own, interpolated along the face).

- ``sample_functions(box)``: ``tack_sample(pool, p)`` as Slang (``slang_gen``) and the
  pool's words.
- ``reference(...)``: the same rays marched by a Tack kernel with the same ``sample``,
  transfer function, step and compositing as the Slang hook (``volume_mapper.HOOK``).
"""

import os
import sys

import numpy as np

import tack

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slang_surfaces"))

from slang_gen import export_shared, joined

#: The largest number of steps a ray takes (the hook's too).
MAX_STEPS = tack.constant(8192)
#: A ray stops once its opacity reaches this (the hook's too).
OPAQUE = tack.constant(0.99)


@tack.func
def _corner(u, n):
    """The lower of the two cell centres around index-space coordinate ``u`` (clamped
    to the centres, ``n`` of them) and the fraction towards the upper."""
    x = min(max(u, 0.0), tack.f32(n - 1))
    i = max(min(int(x), n - 2), 0)
    return i, x - tack.f32(i)


@tack.device_class
class CellBox:
    """Cell-centred values (``values``, x fastest) on a uniform box of ``dims`` cells
    whose first point is ``origin``, cells ``spacing`` wide. At least two cells along
    each axis."""

    def __init__(self, values, dims, origin, spacing):
        self.values = values
        self.nx, self.ny, self.nz = (int(d) for d in dims)
        self.ox, self.oy, self.oz = (float(o) for o in origin)
        self.sx, self.sy, self.sz = (float(s) for s in spacing)

    @tack.func
    def value(self, i, j, k):
        return self.values[i + self.nx * (j + self.ny * k)]

    @tack.func
    def sample(self, p):
        i, fx = _corner((p[0] - self.ox) / self.sx - 0.5, self.nx)
        j, fy = _corner((p[1] - self.oy) / self.sy - 0.5, self.ny)
        k, fz = _corner((p[2] - self.oz) / self.sz - 0.5, self.nz)
        c00 = self.value(i, j, k) * (1.0 - fx) + self.value(i + 1, j, k) * fx
        c10 = self.value(i, j + 1, k) * (1.0 - fx) + self.value(i + 1, j + 1, k) * fx
        c01 = self.value(i, j, k + 1) * (1.0 - fx) + self.value(i + 1, j, k + 1) * fx
        c11 = self.value(i, j + 1, k + 1) * (1.0 - fx) + self.value(i + 1, j + 1, k + 1) * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        return c0 * (1.0 - fz) + c1 * fz

    @tack.func
    def cell_size(self, p):
        return min(self.sx, self.sy, self.sz)

    @tack.func
    def classify(self, tf, lo, hi, p):
        return _transfer(tf, (self.sample(p) - lo) / (hi - lo))

    @tack.func
    def cell_info(self, n):
        """Cell ``n``: its value, whether it is active (1), its centre, size and level."""
        i = n % self.nx
        j = (n // self.nx) % self.ny
        k = n // (self.nx * self.ny)
        return (self.values[n], 1, self.ox + (tack.f32(i) + 0.5) * self.sx,
                self.oy + (tack.f32(j) + 0.5) * self.sy, self.oz + (tack.f32(k) + 0.5) * self.sz,
                self.sx, self.sy, self.sz, 0)


@tack.device_class
class AMRVolume:
    """Cell-centred values on the boxes of a block-structured AMR hierarchy, sampled by
    tent basis functions (``sample``). Made by ``amr_volume`` from an ``AMRDataSet``.

    Every *active* cell (one no finer level covers: not ``REFINEDCELL``) is a tent,
    ``prod_a max(0, 1 - |p_a - x_a| / s_a)`` about its centre ``x`` with its level's
    cell size ``s``; the value at ``p`` is the tents' weighted mean of their cells'
    values. Within one level's active cells away from its boxes' edges that is
    trilinear interpolation (the weights sum to one: step 1's ``CellBox``). Across a
    coarse/fine boundary both levels' tents overlap and the value is continuous: a fine
    cell's tent reaches half a fine cell into the coarse region, an active coarse
    cell's half a coarse cell into the fine. At the domain's faces the mean of the
    cells there (``CellBox``'s clamp).

    At ``p`` each level has at most eight tents: the cells around ``p`` in the level's
    index space. A cell's box is one load from the level's dense table of bricks
    (``brick_box``): bricks of ``bin_size`` cells over the level's boxes, the greatest
    common divisor of their corners, so a brick lies in one box or none (AMReX's
    blocking factor makes it 8 or more), each holding that box or -1. Refined cells
    are one bit each (``refined``, 32 to a word).

    ``blend`` 0 instead takes the finest level with an active cell containing ``p``
    and only that level's cells: per-level trilinear interpolation, clamped at its
    boxes' edges (the naive scheme, which shows seams)."""

    def __init__(self, arrays, origin, levels, boxes, blend=1):
        for name, array in arrays.items():
            setattr(self, name, array)
        self.ox, self.oy, self.oz = (float(o) for o in origin)
        self.nlevels = int(levels)
        self.nboxes = int(boxes)
        self.blend = int(blend)
        # The levels whose tents reach each macrocell (``attach_levels``); until then one
        # macrocell, every level.
        self.attach_levels({"origin": origin, "size": (3.0e38,) * 3, "dims": (1, 1, 1)},
                           tack.array(np.zeros(1, np.int32)),
                           tack.array(np.full(1, levels - 1, np.int32)))

    def attach_levels(self, grid, firsts, lasts):
        """Per macrocell of ``grid`` the coarsest (``firsts``) and finest (``lasts``) level
        whose active cells' tents reach into it: a sample there sums those levels alone,
        which is exact (no other level's tent reaches it)."""
        self.mgx, self.mgy, self.mgz = (int(d) for d in grid["dims"])
        self.mox, self.moy, self.moz = (float(v) for v in grid["origin"])
        self.msx, self.msy, self.msz = (float(v) for v in grid["size"])
        self.macro_first = firsts
        self.macro_last = lasts

    @tack.func
    def levels_at(self, p):
        """The coarsest and finest level whose tents can reach ``p``."""
        i = min(max(int(floor((p[0] - self.mox) / self.msx)), 0), self.mgx - 1)
        j = min(max(int(floor((p[1] - self.moy) / self.msy)), 0), self.mgy - 1)
        k = min(max(int(floor((p[2] - self.moz) / self.msz)), 0), self.mgz - 1)
        m = i + self.mgx * (j + self.mgy * k)
        return self.macro_first[m], self.macro_last[m]

    @tack.func
    def value_at(self, n):
        """Value ``n`` of the boxes' values (all boxes' cells one after another)."""
        return self.values[n]

    @tack.func
    def classify(self, tf, lo, hi, p):
        """The colour and opacity at ``p``: the value, through the table ``tf``."""
        return _transfer(tf, (self.sample(p) - lo) / (hi - lo))

    @tack.func
    def bin_slot(self, level, i, j, k):
        """The bin of ``level`` holding cell ``(i, j, k)``, or -1 outside the bins."""
        b = self.bin_size[level]
        bi = (i - self.bin_origin[3 * level]) // b
        bj = (j - self.bin_origin[3 * level + 1]) // b
        bk = (k - self.bin_origin[3 * level + 2]) // b
        nx = self.bin_dims[3 * level]
        ny = self.bin_dims[3 * level + 1]
        nz = self.bin_dims[3 * level + 2]
        slot = -1
        if bi >= 0 and bi < nx and bj >= 0 and bj < ny and bk >= 0 and bk < nz:
            slot = self.bin_base[level] + bi + nx * (bj + ny * bk)
        return slot

    @tack.func
    def find(self, level, i, j, k):
        """The box of ``level`` holding cell ``(i, j, k)``, or -1."""
        out = -1
        slot = self.bin_slot(level, i, j, k)
        if slot >= 0:
            out = self.brick_box[slot]
        return out

    @tack.func
    def read(self, box, i, j, k):
        """Cell ``(i, j, k)`` of ``box`` (which holds it): its value, and whether a finer
        level covers it (0 or 1)."""
        dx = self.box_dims[3 * box]
        dy = self.box_dims[3 * box + 1]
        n = (i - self.box_lo[3 * box]) + dx * ((j - self.box_lo[3 * box + 1])
                                               + dy * (k - self.box_lo[3 * box + 2]))
        word = self.refined[self.box_refined[box] + n // 32]
        return self.value_at(self.box_values[box] + n), (word >> (n % 32)) & 1

    @tack.func
    def cell(self, level, i, j, k):
        """The value of cell ``(i, j, k)`` of ``level``, whether it is in a box (0 or
        1), and whether a finer level covers it (0 or 1)."""
        value = 0.0
        found = 0
        refined = 0
        box = self.find(level, i, j, k)
        if box >= 0:
            value, refined = self.read(box, i, j, k)
            found = 1
        return value, found, refined

    @tack.func
    def level_sum(self, level, p, active_only):
        """The tents of ``level`` at ``p``: the sum of weight times value and of weight,
        over the level's cells around ``p`` that are in a box (and active, if
        ``active_only``).

        The common cases cost one lookup: the eight cells in the box holding the first
        (read directly), or all in one bin that holds no box (none). Otherwise each
        cell is looked up."""
        ux = (p[0] - self.ox) / self.spacing[3 * level] - 0.5
        uy = (p[1] - self.oy) / self.spacing[3 * level + 1] - 0.5
        uz = (p[2] - self.oz) / self.spacing[3 * level + 2] - 0.5
        fx = ux - floor(ux)
        fy = uy - floor(uy)
        fz = uz - floor(uz)
        ix = int(floor(ux))
        iy = int(floor(uy))
        iz = int(floor(uz))
        num = 0.0
        den = 0.0
        # 1: the box of the first cell holds all eight; -1: none can be in a box; 0: look.
        mode = 0
        box = self.find(level, ix, iy, iz)
        if box >= 0:
            if (ix + 1 < self.box_lo[3 * box] + self.box_dims[3 * box]
                    and iy + 1 < self.box_lo[3 * box + 1] + self.box_dims[3 * box + 1]
                    and iz + 1 < self.box_lo[3 * box + 2] + self.box_dims[3 * box + 2]):
                mode = 1
        else:
            first = self.bin_slot(level, ix, iy, iz)
            last = self.bin_slot(level, ix + 1, iy + 1, iz + 1)
            b = self.bin_size[level]
            # The eight cells in one bin (no bin boundary between them on any axis).
            one = ((ix - self.bin_origin[3 * level]) % b != b - 1
                   and (iy - self.bin_origin[3 * level + 1]) % b != b - 1
                   and (iz - self.bin_origin[3 * level + 2]) % b != b - 1)
            if first < 0 and last < 0:
                # Outside the bins on some axis at both corners: no box near.
                if (ix + 1 < self.bin_origin[3 * level] or iy + 1 < self.bin_origin[3 * level + 1]
                        or iz + 1 < self.bin_origin[3 * level + 2] or ix >= self.bin_origin[3 * level]
                        + b * self.bin_dims[3 * level] or iy >= self.bin_origin[3 * level + 1]
                        + b * self.bin_dims[3 * level + 1] or iz >= self.bin_origin[3 * level + 2]
                        + b * self.bin_dims[3 * level + 2]):
                    mode = -1
            elif first >= 0 and one:
                if self.brick_box[first] < 0:
                    mode = -1
        if mode >= 0:
            for corner in range(8):
                cx = corner & 1
                cy = (corner >> 1) & 1
                cz = corner >> 2
                w = ((tack.f32(cx) * fx + tack.f32(1 - cx) * (1.0 - fx))
                     * (tack.f32(cy) * fy + tack.f32(1 - cy) * (1.0 - fy))
                     * (tack.f32(cz) * fz + tack.f32(1 - cz) * (1.0 - fz)))
                if w > 0.0:
                    v = 0.0
                    found = 0
                    refined = 0
                    if mode == 1:
                        v, refined = self.read(box, ix + cx, iy + cy, iz + cz)
                        found = 1
                    else:
                        v, found, refined = self.cell(level, ix + cx, iy + cy, iz + cz)
                    if found == 1 and (active_only == 0 or refined == 0):
                        num += w * v
                        den += w
        return num, den

    @tack.func
    def containing(self, level, p):
        """Whether ``p`` lies in an active cell of ``level`` (0 or 1)."""
        i = int(floor((p[0] - self.ox) / self.spacing[3 * level]))
        j = int(floor((p[1] - self.oy) / self.spacing[3 * level + 1]))
        k = int(floor((p[2] - self.oz) / self.spacing[3 * level + 2]))
        v, found, refined = self.cell(level, i, j, k)
        out = 0
        if found == 1 and refined == 0:
            out = 1
        return out

    @tack.func
    def sample(self, p):
        num = 0.0
        den = 0.0
        first, last = self.levels_at(p)
        if self.blend == 1:
            for level in range(first, last + 1):
                n, d = self.level_sum(level, p, 1)
                num += n
                den += d
        else:
            done = 0
            for back in range(last - first + 1):
                level = last - back
                if done == 0 and self.containing(level, p) == 1:
                    num, den = self.level_sum(level, p, 0)
                    done = 1
        out = 0.0
        if den > 0.0:
            out = num / den
        return out

    @tack.func
    def cell_size(self, p):
        """The cell size (smallest axis) of the finest level with an active cell holding
        ``p``; level 0's where none does."""
        out = min(self.spacing[0], self.spacing[1], self.spacing[2])
        done = 0
        first, last = self.levels_at(p)
        for back in range(last - first + 1):
            level = last - back
            if done == 0 and self.containing(level, p) == 1:
                out = min(self.spacing[3 * level], self.spacing[3 * level + 1],
                          self.spacing[3 * level + 2])
                done = 1
        return out


    @tack.func
    def cell_info(self, n):
        """Cell ``n`` of the values (the boxes' cells one after another): its value,
        whether it is active (1), its centre, its level's cell size and its level."""
        first = 0
        last = self.nboxes
        while last - first > 1:
            middle = (first + last) // 2
            if self.box_values[middle] <= n:
                first = middle
            else:
                last = middle
        box = first
        local = n - self.box_values[box]
        dx = self.box_dims[3 * box]
        dy = self.box_dims[3 * box + 1]
        i = local % dx
        j = (local // dx) % dy
        k = local // (dx * dy)
        level = self.box_level[box]
        sx = self.spacing[3 * level]
        sy = self.spacing[3 * level + 1]
        sz = self.spacing[3 * level + 2]
        word = self.refined[self.box_refined[box] + local // 32]
        return (self.value_at(n), 1 - ((word >> (local % 32)) & 1),
                self.ox + (tack.f32(self.box_lo[3 * box] + i) + 0.5) * sx,
                self.oy + (tack.f32(self.box_lo[3 * box + 1] + j) + 0.5) * sy,
                self.oz + (tack.f32(self.box_lo[3 * box + 2] + k) + 0.5) * sz, sx, sy, sz, level)




@tack.device_class
class QuantizedAMRVolume(AMRVolume):
    """An ``AMRVolume`` whose values are ``bits``-bit codes (16 or 8), packed
    ``32 // bits`` to a word (``codes``): code ``c`` stands for ``qlo + c * qstep``, or
    its exponential where ``qlog`` is 1 (a logarithmic scale). Made by ``amr_volume``
    with ``quantize``."""

    @tack.func
    def value_at(self, n):
        per = 32 // self.bits
        word = self.codes[n // per]
        code = (word >> ((n % per) * self.bits)) & ((1 << self.bits) - 1)
        x = self.qlo + tack.f32(code) * self.qstep
        if self.qlog == 1:
            x = exp(x)
        return x


#: Bins of a stand-in cell's histogram, one byte each, four words a cell (two spare).
BINS = tack.constant(14)


@tack.device_class
class HistogramAMRVolume(AMRVolume):
    """An ``AMRVolume`` some of whose boxes are missing their finer data, each coarse
    cell standing in for it (a *stand-in*) carrying a histogram of the finer cells'
    values beneath it, weighted by volume: ``BINS`` bins over a logarithmic scale
    (``hlo``, ``hstep``), each a byte holding the square root of its fraction (so a
    single small fine cell under a coarse one still registers). ``box_hist`` is a
    box's first word in ``hist`` (four a cell), or -1.

    ``classify`` treats a stand-in as the distribution its histogram gives: every
    bin's value through the table, weighted by the bin's fraction, which keeps peaks
    a coarse value averages away (a sparse-PDF-like summary). The other cells are
    interpolated and classified as usual, and the two parts mixed by their tents'
    weights."""

    @tack.func
    def bins_of(self, box, n):
        """The first word of cell ``n``'s histogram in ``box``, or -1 if it has none."""
        out = -1
        base = self.box_hist[box]
        if base >= 0:
            off = base + 4 * n
            if (self.hist[off] | self.hist[off + 1] | self.hist[off + 2]
                    | self.hist[off + 3]) != 0:
                out = off
        return out

    @tack.func
    def mixture(self, p):
        """At ``p``: the interpolated value of the ordinary cells, their tents' weight,
        and the stand-ins' tents' weight per bin (``BINS``): 16 numbers."""
        acc = tack.local_array(tack.f32, 16)
        for i in range(16):
            acc[i] = 0.0
        first, last = self.levels_at(p)
        for level in range(first, last + 1):
            ux = (p[0] - self.ox) / self.spacing[3 * level] - 0.5
            uy = (p[1] - self.oy) / self.spacing[3 * level + 1] - 0.5
            uz = (p[2] - self.oz) / self.spacing[3 * level + 2] - 0.5
            fx = ux - floor(ux)
            fy = uy - floor(uy)
            fz = uz - floor(uz)
            ix = int(floor(ux))
            iy = int(floor(uy))
            iz = int(floor(uz))
            for corner in range(8):
                cx = corner & 1
                cy = (corner >> 1) & 1
                cz = corner >> 2
                w = ((tack.f32(cx) * fx + tack.f32(1 - cx) * (1.0 - fx))
                     * (tack.f32(cy) * fy + tack.f32(1 - cy) * (1.0 - fy))
                     * (tack.f32(cz) * fz + tack.f32(1 - cz) * (1.0 - fz)))
                if w > 0.0:
                    i = ix + cx
                    j = iy + cy
                    k = iz + cz
                    box = self.find(level, i, j, k)
                    if box >= 0:
                        dx = self.box_dims[3 * box]
                        dy = self.box_dims[3 * box + 1]
                        n = (i - self.box_lo[3 * box]) + dx * (
                            (j - self.box_lo[3 * box + 1]) + dy * (k - self.box_lo[3 * box + 2]))
                        word = self.refined[self.box_refined[box] + n // 32]
                        if ((word >> (n % 32)) & 1) == 0:
                            off = self.bins_of(box, n)
                            if off >= 0:
                                total = 0.0
                                for b in range(BINS):
                                    q = tack.f32((self.hist[off + b // 4] >> ((b % 4) * 8)) & 255)
                                    total += q * q
                                for b in range(BINS):
                                    q = tack.f32((self.hist[off + b // 4] >> ((b % 4) * 8)) & 255)
                                    acc[2 + b] += w * q * q / total
                            else:
                                acc[0] += w * self.value_at(self.box_values[box] + n)
                                acc[1] += w
        value = 0.0
        if acc[1] > 0.0:
            value = acc[0] / acc[1]
        return [value, acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7], acc[8], acc[9],
                acc[10], acc[11], acc[12], acc[13], acc[14], acc[15]]

    @tack.func
    def classify(self, tf, lo, hi, p):
        """The colour and opacity at ``p``: the ordinary cells' interpolated value and
        each stand-in bin's value through the table, mixed by weight (opacity-weighted
        colour)."""
        m = self.mixture(p)
        r = 0.0
        g = 0.0
        b = 0.0
        a = 0.0
        total = m[1]
        if m[1] > 0.0:
            c = _transfer(tf, (m[0] - lo) / (hi - lo))
            r += m[1] * c[0] * c[3]
            g += m[1] * c[1] * c[3]
            b += m[1] * c[2] * c[3]
            a += m[1] * c[3]
        for k in range(BINS):
            wk = m[2 + k]
            if wk > 0.0:
                c = _transfer(tf, (exp(self.hlo + (tack.f32(k) + 0.5) * self.hstep) - lo)
                              / (hi - lo))
                r += wk * c[0] * c[3]
                g += wk * c[1] * c[3]
                b += wk * c[2] * c[3]
                a += wk * c[3]
                total += wk
        out = tack.vec4(0.0, 0.0, 0.0, 0.0)
        if total > 0.0 and a > 0.0:
            out = tack.vec4(r / a, g / a, b / a, a / total)
        return out


@tack.kernel
def _histograms(full, held, cells, hlo, hstep, acc):
    for n in range(cells):
        v, active, cx, cy, cz, sx, sy, sz, level = full.cell_info(n)
        if active == 1:
            # The finest held level with a box holding the cell's centre: below the
            # cell's own level, the box's cell there stands in for it.
            found = -1
            box = -1
            i = 0
            j = 0
            k = 0
            for back in range(held.nlevels):
                lh = held.nlevels - 1 - back
                if found < 0:
                    ti = int(floor((cx - held.ox) / held.spacing[3 * lh]))
                    tj = int(floor((cy - held.oy) / held.spacing[3 * lh + 1]))
                    tk = int(floor((cz - held.oz) / held.spacing[3 * lh + 2]))
                    tb = held.find(lh, ti, tj, tk)
                    if tb >= 0:
                        found = lh
                        box = tb
                        i = ti
                        j = tj
                        k = tk
            if found >= 0 and found < level and held.box_hist[box] >= 0:
                dx = held.box_dims[3 * box]
                dy = held.box_dims[3 * box + 1]
                local = (i - held.box_lo[3 * box]) + dx * (
                    (j - held.box_lo[3 * box + 1]) + dy * (k - held.box_lo[3 * box + 2]))
                weight = (sx * sy * sz) / (held.spacing[3 * found] * held.spacing[3 * found + 1]
                                           * held.spacing[3 * found + 2])
                which = min(max(int(floor((log(max(v, 1e-30)) - hlo) / hstep)), 0), BINS - 1)
                tack.atomic_add(acc, (held.box_hist[box] // 4 + local) * BINS + which, weight)


@tack.kernel
def _pack_histograms(acc, words):
    for c in range(words.shape[0] // 4):
        total = 0.0
        for b in range(BINS):
            total += acc[c * BINS + b]
        for w in range(4):
            word = 0
            for k in range(4):
                b = 4 * w + k
                if b < BINS and total > 0.0:
                    q = min(int(floor(255.0 * sqrt(acc[c * BINS + b] / total) + 0.5)), 255)
                    word = word | (q << (8 * k))
            words[4 * c + w] = word


def histogram_volume(amr, spacings, origin, field, keep, blend=1):
    """A ``HistogramAMRVolume`` of the boxes ``keep`` of ``amr``: each held box under
    which finer boxes are left out carries a histogram per cell of the finer data
    beneath it (``_histograms``, from the whole hierarchy), on a log scale from the
    data's 1st percentile to its largest value."""
    held = amr_volume(amr, spacings, origin, field, blend, keep)
    full = amr_volume(amr, spacings, origin, field)
    levels = held.host["levels"]
    # Which held boxes need histograms: those a left-out finer box meets.
    box_hist, at = [], 0
    boxes_held = [amr.boxes(level)[np.asarray(sorted(keep[level]), int)] if keep is not None
                  else amr.boxes(level) for level in range(levels)]
    for level in range(levels):
        missing = np.zeros((0, 2, 3), np.int64)
        for finer in range(level + 1, amr.num_levels):
            chosen = np.zeros(len(amr.boxes(finer)), bool)
            if finer < levels and keep is not None:
                chosen[np.asarray(keep[finer], int)] = True
            elif finer < levels:
                chosen[:] = True
            scale = 2 ** (finer - level)
            gone = amr.boxes(finer)[~chosen]
            missing = np.concatenate([missing, np.stack([gone[:, 0] // scale,
                                                         -(-gone[:, 1] // scale)], axis=1)])
        for lo, hi in boxes_held[level]:
            meets = len(missing) and np.any(np.all((missing[:, 0] < hi) & (missing[:, 1] > lo),
                                                   axis=1))
            if meets:
                box_hist.append(4 * at)
                at += int(np.prod(hi - lo))
            else:
                box_hist.append(-1)
    values = np.concatenate([piece.fields[field].values.numpy().reshape(-1)[::97]
                             for _, piece in amr])
    positive = values[values > 0]
    hlo = float(np.log(np.percentile(positive, 1)))
    hhi = float(np.log(positive.max()))
    hstep = (hhi - hlo) / int(BINS)
    acc = tack.array(np.zeros(max(at, 1) * int(BINS), np.float32))
    out = HistogramAMRVolume.__new__(HistogramAMRVolume)
    out.__dict__.update(held.__dict__)
    out.box_hist = tack.array(np.asarray(box_hist, np.int32))
    out.hist = tack.array(np.zeros(max(at, 1) * 4, np.int32))
    out.hlo, out.hstep = hlo, hstep
    _histograms(full, out, full.host["cells"], hlo, hstep, acc)
    _pack_histograms(acc, out.hist)
    out.host = dict(held.host, stand_in_cells=at)
    return out


@tack.kernel
def _encode(values, lo, step, log_scale, top, codes):
    for n in range(values.shape[0]):
        x = values[n]
        if log_scale == 1:
            x = log(max(x, 1e-30))
        codes[n] = min(max(int(floor((x - lo) / step + 0.5)), 0), top)


@tack.kernel
def _pack(codes, count, bits, words):
    for w in range(words.shape[0]):
        per = 32 // bits
        word = 0
        for k in range(per):
            n = w * per + k
            if n < count:
                word = word | (codes[n] << (k * bits))
        words[w] = word


def quantize(values, bits, scale="linear"):
    """``values`` (a Tack f32 array) as ``bits``-bit codes packed in i32 words, and
    the decoding ``(qlo, qstep, qlog)``: linear over the values' range, or over their
    logarithms (``scale="log"``, positive values)."""
    host = values.numpy()
    log_scale = scale == "log"
    x = np.log(np.maximum(host, 1e-30)) if log_scale else host
    lo, hi = float(x.min()), float(x.max())
    top = (1 << bits) - 1
    step = (hi - lo) / top if hi > lo else 1.0
    codes = tack.empty((len(host),), tack.i32)
    _encode(values, lo, step, int(log_scale), top, codes)
    per = 32 // bits
    words = tack.empty((-(-len(host) // per),), tack.i32)
    _pack(codes, len(host), bits, words)
    return words, (lo, step, int(log_scale))


def _refined_by(lo, hi, children, ratio):
    """Which cells of the box ``[lo, hi)`` the boxes ``children`` (``(n, 2, 3)``, the
    next level's) cover: a bool array shaped z, y, x."""
    out = np.zeros(tuple(int(v) for v in (hi - lo)[::-1]), bool)
    if len(children):
        clo = children[:, 0] // ratio
        chi = -(-children[:, 1] // ratio)
        meet = np.all((clo < hi) & (chi > lo), axis=1)
        for a, b in zip(np.maximum(clo[meet], lo), np.minimum(chi[meet], hi)):
            a, b = a - lo, b - lo
            out[a[2]:b[2], a[1]:b[1], a[0]:b[0]] = True
    return out


def amr_volume(amr, spacings, origin, field="values", blend=1, keep=None, quantize_bits=None,
               scale="linear"):
    """An ``AMRVolume`` of ``amr`` (a ``tack.dataset.AMRDataSet``) and its cell field
    ``field``: ``spacings`` each level's cell size (three per level), ``origin`` the
    domain's first point. The bins and boxes are the hierarchy's few numbers, made on
    the host.

    ``keep`` says which boxes the volume holds: per level the indices of its boxes
    (``amr.boxes(level)``'s order), all where None; a level with none ends the
    hierarchy. A cell is refined where a finer box *held* covers it, so a coarse cell
    stands in for a finer box left out. The finer boxes held must be properly nested
    in the coarser ones held (``select_boxes`` makes them so).

    ``quantize_bits`` (16 or 8) stores the values as codes (``QuantizedAMRVolume``),
    linear over their range or logarithmic (``scale="log"``)."""
    levels = amr.num_levels
    pieces = [[] for _ in range(levels)]
    for level, piece in amr:
        pieces[level].append(piece)
    held = []
    for level in range(levels):
        count = len(amr.boxes(level))
        chosen = np.arange(count) if keep is None else np.asarray(sorted(keep[level]), int)
        if len(chosen) == 0:
            break
        held.append(chosen)
    levels = len(held)
    box_lo, box_dims, box_values, box_refined, box_level = [], [], [], [], []
    values, refined = [], []
    bin_origin, bin_size, bin_dims, bin_base, brick_box = [], [], [], [], []
    at_value = at_word = first = 0
    for level in range(levels):
        boxes = amr.boxes(level)[held[level]]
        lo, hi = boxes[:, 0], boxes[:, 1]
        finer = amr.boxes(level + 1)[held[level + 1]] if level + 1 < levels else np.zeros(
            (0, 2, 3), np.int64)
        ratio = np.asarray(amr.ratios[level] if level < len(amr.ratios) else 1, np.int64)
        for index, blo, bhi in zip(held[level], lo, hi, strict=True):
            cells = int(np.prod(bhi - blo))
            box_lo.extend(int(v) for v in blo)
            box_dims.extend(int(v) for v in bhi - blo)
            box_values.append(at_value)
            box_refined.append(at_word)
            box_level.append(level)
            piece = pieces[level][index]
            values.append(piece.fields[field].values.numpy().reshape(-1).astype(np.float32))
            bits = _refined_by(blo, bhi, finer, ratio).reshape(-1)
            words = -(-cells // 32)
            packed = np.zeros(words * 32, bool)
            packed[:cells] = bits
            refined.append(np.packbits(packed, bitorder="little").view(np.int32))
            at_value += cells
            at_word += words
        # The level's bricks: the corners' greatest common divisor, so that a brick lies
        # in one box or none; a dense table of each brick's box over the level's boxes.
        size = int(np.gcd.reduce(np.concatenate([lo.ravel(), hi.ravel()]).astype(np.int64)))
        size = max(size, 1)
        start = lo.min(axis=0)
        dims = (hi.max(axis=0) - start) // size
        table = np.full(tuple(int(d) for d in dims[::-1]), -1, np.int32)
        for b, (blo, bhi) in enumerate(zip(lo, hi)):
            a, z = (blo - start) // size, (bhi - start) // size
            table[a[2]:z[2], a[1]:z[1], a[0]:z[0]] = first + b
        bin_origin.extend(int(v) for v in start)
        bin_size.append(size)
        bin_dims.extend(int(v) for v in dims)
        bin_base.append(sum(len(t) for t in brick_box))
        brick_box.append(table.reshape(-1))
        first += len(boxes)

    def ints(v):
        return tack.array(np.asarray(v, np.int32))

    arrays = {
        "spacing": tack.array(np.asarray(spacings[:levels], np.float32).reshape(-1)),
        "bin_origin": ints(bin_origin), "bin_size": ints(bin_size), "bin_dims": ints(bin_dims),
        "bin_base": ints(bin_base), "brick_box": ints(np.concatenate(brick_box)),
        "box_lo": ints(box_lo), "box_dims": ints(box_dims), "box_values": ints(box_values),
        "box_refined": ints(box_refined), "box_level": ints(box_level),
        "refined": tack.array(np.concatenate(refined)),
    }
    flat = tack.array(np.concatenate(values))
    if quantize_bits is None:
        arrays["values"] = flat
        out = AMRVolume(arrays, origin, levels, len(box_level), blend)
    else:
        arrays["codes"], (qlo, qstep, qlog) = quantize(flat, quantize_bits, scale)
        out = QuantizedAMRVolume(arrays, origin, levels, len(box_level), blend)
        out.bits, out.qlo, out.qstep, out.qlog = int(quantize_bits), qlo, qstep, qlog
    # What the hierarchy holds, for the host (not the kernels): its cells and range.
    out.host = {"cells": len(flat), "range": (flat.min(), flat.max()), "levels": levels,
                "boxes": [len(h) for h in held]}
    return out


def select_boxes(amr, spacings, origin, camera, size, base, pixels=1.0):
    """The boxes a view resolves, per level (``amr_volume``'s ``keep``): every box of
    levels 0 to ``base``; of a finer level, each box in the view frustum whose cells
    project to at least ``pixels`` at its nearest point to the camera, and whose
    coarser boxes (those its region coarsened meets) are all held."""
    w, h = size
    m = np.array([[camera.GetCompositeProjectionTransformMatrix(w / h, -1, 1).GetElement(i, j)
                   for j in range(4)] for i in range(4)])
    eye = np.array(camera.GetPosition())
    tan = np.tan(np.radians(camera.GetViewAngle()) / 2)
    origin = np.asarray(origin, np.float64)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    keep = []
    for level in range(amr.num_levels):
        boxes = amr.boxes(level)
        if level <= base:
            keep.append(np.arange(len(boxes)))
            continue
        cell = np.asarray(spacings[level], np.float64)
        a = origin + boxes[:, 0] * cell
        b = origin + boxes[:, 1] * cell
        # In the frustum: not all eight corners outside one clip plane.
        points = a[:, None, :] + corners[None] * (b - a)[:, None, :]
        clip = np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=2) @ m.T
        x, y, z, q = (clip[..., i] for i in range(4))
        outside = (np.all(x > q, 1) | np.all(x < -q, 1) | np.all(y > q, 1) | np.all(y < -q, 1)
                   | np.all(z > q, 1) | np.all(z < -q, 1))
        # The cells' size on the screen at the box's nearest point.
        distance = np.linalg.norm(np.clip(eye, a, b) - eye, axis=1)
        projected = cell.min() / (2 * np.maximum(distance, 1e-9) * tan) * h
        wanted = ~outside & (projected >= pixels)
        # Nested: every coarser box the region meets is held.
        ratio = np.asarray(amr.ratios[level - 1], np.int64)
        coarse = amr.boxes(level - 1)
        held = np.zeros(len(coarse), bool)
        held[keep[level - 1]] = True
        clo, chi = boxes[:, 0] // ratio, -(-boxes[:, 1] // ratio)
        for i in np.nonzero(wanted)[0]:
            meet = np.all((coarse[:, 0] < chi[i]) & (coarse[:, 1] > clo[i]), axis=1)
            wanted[i] = bool(held[meet].all())
        keep.append(np.nonzero(wanted)[0])
    return keep


@tack.kernel
def _sample_at(box, ps, out):
    for k in range(ps.shape[0]):
        out[k] = box.sample(ps[k])


@tack.kernel
def _cell_size_at(box, ps, out):
    for k in range(ps.shape[0]):
        out[k] = box.cell_size(ps[k])


def sample_functions(box):
    """``tack_sample(pool, p)`` and ``tack_cell_size(pool, p)`` as Slang over one pool
    (and ``tack_mixture(pool, p)`` for a ``HistogramAMRVolume``: its ``mixture``, a
    ``float4x4`` row by row), and the pool's words (``np.uint32``)."""
    ps = tack.array(np.zeros((1, 3), np.float32), tack.vec3())
    slots = {"ps": ("float", 3), "out": ("float", 1)}
    specs = [
        (_sample_at, (box, ps, tack.empty((1,), tack.f32)), "tack_sample", slots,
         ("float", "{out}[0]")),
        (_cell_size_at, (box, ps, tack.empty((1,), tack.f32)), "tack_cell_size", slots,
         ("float", "{out}[0]")),
    ]
    if isinstance(box, HistogramAMRVolume):
        specs.append((_mixture_at, (box, ps, tack.empty((16,), tack.f32)), "tack_mixture",
                      {"ps": ("float", 3), "out": ("float", 16)},
                      ("float4x4", "float4x4(" + ", ".join(f"{{out}}[{i}]" for i in range(16))
                       + ")")))
    outs, words = export_shared(specs)
    return joined(*outs), words


@tack.kernel
def _mixture_at(box, ps, out):
    for k in range(ps.shape[0]):
        m = box.mixture(ps[k])
        for i in range(16):
            out[16 * k + i] = m[i]


# Empty-space skipping: a coarse grid of the values each macrocell can take.

@tack.kernel
def _macro_ranges(volume, cells, gx, gy, gz, ox, oy, oz, mx, my, mz, lows, highs, firsts,
                  lasts):
    for n in range(cells):
        v, active, cx, cy, cz, sx, sy, sz, level = volume.cell_info(n)
        if active == 1:
            # The cell's tent reaches one cell from its centre: every macrocell it meets
            # can take its value.
            i0 = max(int(floor((cx - sx - ox) / mx)), 0)
            i1 = min(int(floor((cx + sx - ox) / mx)), gx - 1)
            j0 = max(int(floor((cy - sy - oy) / my)), 0)
            j1 = min(int(floor((cy + sy - oy) / my)), gy - 1)
            k0 = max(int(floor((cz - sz - oz) / mz)), 0)
            k1 = min(int(floor((cz + sz - oz) / mz)), gz - 1)
            for k in range(k0, k1 + 1):
                for j in range(j0, j1 + 1):
                    for i in range(i0, i1 + 1):
                        m = i + gx * (j + gy * k)
                        tack.atomic_min(lows, m, v)
                        tack.atomic_max(highs, m, v)
                        tack.atomic_min(firsts, m, level)
                        tack.atomic_max(lasts, m, level)


def macro_grid(volume, cells, bounds, coarse, most=64):
    """The macrocells over ``bounds``: each ``coarse`` (level 0's cell size, per axis)
    times a whole number, at most ``most`` along the longest axis; and per macrocell
    the smallest and largest value of the active cells whose tents meet it (largest <
    smallest where none does). ``cells`` is how many cells ``volume`` holds. An
    ``AMRVolume`` is also given each macrocell's range of levels (``attach_levels``),
    so export it after this."""
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    across = (hi - lo) / np.asarray(coarse)
    factor = max(1, int(np.ceil(across.max() / most)))
    size = np.asarray(coarse, np.float64) * factor
    dims = np.maximum(np.ceil((hi - lo) / size - 1e-9).astype(int), 1)
    count = int(np.prod(dims))
    lows = tack.array(np.full(count, 3.0e38, np.float32))
    highs = tack.array(np.full(count, -3.0e38, np.float32))
    firsts = tack.array(np.full(count, 1 << 20, np.int32))
    lasts = tack.array(np.full(count, -1, np.int32))
    _macro_ranges(volume, int(cells), *(int(d) for d in dims), *(float(v) for v in lo),
                  *(float(v) for v in size), lows, highs, firsts, lasts)
    grid = {"origin": tuple(float(v) for v in lo), "size": tuple(float(v) for v in size),
            "dims": tuple(int(d) for d in dims)}
    if hasattr(volume, "attach_levels"):
        volume.attach_levels(grid, firsts, lasts)
    return grid, lows, highs


@tack.kernel
def _visible(lows, highs, table, lo, hi, out):
    for m in range(lows.shape[0]):
        a = lows[m]
        b = highs[m]
        seen = 0
        if a <= b:
            n = table.shape[0]
            x0 = min(max((a - lo) / (hi - lo), 0.0), 1.0) * tack.f32(n - 1)
            x1 = min(max((b - lo) / (hi - lo), 0.0), 1.0) * tack.f32(n - 1)
            for i in range(int(floor(x0)), min(int(ceil(x1)), n - 1) + 1):
                if table[i][3] > 0.0:
                    seen = 1
        out[m] = seen


def visible(lows, highs, table, lo, hi):
    """Per macrocell 1 if the transfer function (``table``, ``(n, 4)`` RGBA over
    ``[lo, hi]``) gives any opacity to a value it can take, else 0 (``i32``)."""
    out = tack.empty((len(lows),), tack.i32)
    _visible(lows, highs, tack.array(np.asarray(table, np.float32), tack.vec4()), float(lo),
             float(hi), out)
    return out


# The reference: the hook's march, in Tack.

@tack.func
def _transfer(tf, t):
    n = tf.shape[0]
    x = min(max(t, 0.0), 1.0) * tack.f32(n - 1)
    i = min(int(x), n - 2)
    f = x - tack.f32(i)
    return tf[i] * (1.0 - f) + tf[i + 1] * f


@tack.func
def _safe_inverse(x):
    y = x
    if abs(x) < 1e-20:
        y = 1e-20
        if x < 0.0:
            y = -1e-20
    return 1.0 / y


@tack.func
def _jitter(px, py):
    """A pixel's offset in [0, 1): a PCG hash of its position (the hook's
    ``tack_jitter``)."""
    state = (tack.u32(px) + tack.u32(4096) * tack.u32(py)) * tack.u32(747796405) \
        + tack.u32(2891336453)
    word = ((state >> ((state >> tack.u32(28)) + tack.u32(4))) ^ state) * tack.u32(277803737)
    word = (word >> tack.u32(22)) ^ word
    return tack.f32(word >> tack.u32(8)) * (1.0 / 16777216.0)


@tack.func
def _exit(q, d, inv, mi, m):
    """Where the ray ``q + t d`` (from the macrocells' origin) leaves macrocell ``mi``
    (``m`` wide) on one axis: its face ahead."""
    face = tack.f32(mi) * m
    if d >= 0.0:
        face = tack.f32(mi + 1) * m
    return (face - q) * inv


@tack.kernel
def _march(volume, starts, dirs, tmins, tmaxs, tf, seen, lo, hi, factor, min_step, unit, gx,
           gy, gz, ox, oy, oz, mx, my, mz, skip, jitter, width, out):
    for k in range(starts.shape[0]):
        p0 = starts[k]
        d = dirs[k]
        ix = _safe_inverse(d[0])
        iy = _safe_inverse(d[1])
        iz = _safe_inverse(d[2])
        # The ray from the macrocells' origin: o + t d.
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
        offset = jitter * _jitter(k % width, k // width)
        fresh = 1
        while t < t1 and steps < MAX_STEPS:
            steps += 1
            mi = min(max(int(floor((qx + d[0] * t) / mx)), 0), gx - 1)
            mj = min(max(int(floor((qy + d[1] * t) / my)), 0), gy - 1)
            mk = min(max(int(floor((qz + d[2] * t) / mz)), 0), gz - 1)
            if seen[mi + gx * (mj + gy * mk)] == 0:
                ex = _exit(qx, d[0], ix, mi, mx)
                ey = _exit(qy, d[1], iy, mj, my)
                ez = _exit(qz, d[2], iz, mk, mz)
                t = max(min(ex, ey, ez), t) + skip
                fresh = 1
            else:
                h = max(factor * volume.cell_size(p0 + d * t), min_step)
                if fresh == 1:
                    t += offset * h
                    fresh = 0
                c = volume.classify(tf, lo, hi, p0 + d * (t + 0.5 * h))
                alpha = 1.0 - pow(1.0 - c[3], h / unit)
                w = (1.0 - a) * alpha
                r += w * c[0]
                g += w * c[1]
                b += w * c[2]
                a += w
                t += h
                if a >= OPAQUE:
                    break
        out[k] = tack.vec4(r, g, b, a)


def rays(camera, size, bounds):
    """Each pixel's ray (row 0 at the top): its start on the near plane, unit
    direction, and the distances along it where it enters and leaves ``bounds``."""
    w, h = size
    m = np.array([[camera.GetCompositeProjectionTransformMatrix(w / h, -1, 1).GetElement(i, j)
                   for j in range(4)] for i in range(4)])
    inverse = np.linalg.inv(m)
    j, i = np.mgrid[0:h, 0:w]
    ndc = np.stack([(i + 0.5) / w * 2 - 1, 1 - (j + 0.5) / h * 2], axis=-1).reshape(-1, 2)

    def unproject(z):
        clip = np.concatenate([ndc, np.full((len(ndc), 1), z), np.ones((len(ndc), 1))], axis=1)
        p = clip @ inverse.T
        return p[:, :3] / p[:, 3:]

    near, far = unproject(-1.0), unproject(1.0)
    d = far - near
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-20, 1e-20, d)
    ta, tb = (lo - near) * inv, (hi - near) * inv
    tmin = np.maximum(np.minimum(ta, tb).max(axis=1), 0.0)
    tmax = np.maximum(ta, tb).min(axis=1)
    return near, d, tmin, tmax


def reference(volume, camera, size, bounds, tf, lo, hi, factor, unit, grid, seen, skip,
              background, min_step=0.0, jitter=0.0):
    """The picture the hook should draw (``(h, w, 4)`` u8, row 0 at the top): the
    rays marched in Tack as the hook marches them (steps ``factor`` times the local
    cell size and at least ``min_step``, macrocells ``grid`` skipped where ``seen`` is
    0), composited over ``background`` as the A-buffer does."""
    near, d, tmin, tmax = rays(camera, size, bounds)
    f32 = np.float32
    out = tack.empty((len(near),), tack.vec4(tack.f32))
    _march(volume, tack.array(near.astype(f32), tack.vec3()),
           tack.array(d.astype(f32), tack.vec3()), tack.array(tmin.astype(f32)),
           tack.array(tmax.astype(f32)), tack.array(np.asarray(tf, f32), tack.vec4()), seen,
           float(lo), float(hi), float(factor), float(min_step), float(unit), *grid["dims"],
           *grid["origin"],
           *grid["size"], float(skip), float(jitter), int(size[0]), out)
    rgba = out.numpy().astype(np.float64)
    a = rgba[:, 3:]
    # The hook writes (C / A, A) through an 8-bit target; the resolve multiplies back.
    colour = np.where(a > 0, rgba[:, :3] / np.maximum(a, 1e-30), 0.0)
    colour = np.round(np.clip(colour, 0, 1) * 255) / 255
    alpha = np.round(np.clip(a, 0, 1) * 255) / 255
    picture = colour * alpha + np.asarray(background, np.float64) * (1 - alpha)
    w, hgt = size
    rgb = np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8).reshape(hgt, w, 3)
    return np.concatenate([rgb, np.full((hgt, w, 1), 255, np.uint8)], axis=2)
