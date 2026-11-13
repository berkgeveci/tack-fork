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
        return self.values[self.box_values[box] + n], (word >> (n % 32)) & 1

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
        return (self.values[n], 1 - ((word >> (local % 32)) & 1),
                self.ox + (tack.f32(self.box_lo[3 * box] + i) + 0.5) * sx,
                self.oy + (tack.f32(self.box_lo[3 * box + 1] + j) + 0.5) * sy,
                self.oz + (tack.f32(self.box_lo[3 * box + 2] + k) + 0.5) * sz, sx, sy, sz, level)




def amr_volume(amr, spacings, origin, field="values", blend=1):
    """An ``AMRVolume`` of ``amr`` (a ``tack.dataset.AMRDataSet``) and its cell field
    ``field``: ``spacings`` each level's cell size (three per level), ``origin`` the
    domain's first point. The bins and boxes are the hierarchy's few numbers, made on
    the host; the values and the refined marks are the boxes' own."""
    from tack.dataset import Ghosts

    levels = amr.num_levels
    box_lo, box_dims, box_values, box_refined, box_level = [], [], [], [], []
    values, refined = [], []
    bin_origin, bin_size, bin_dims, bin_base, brick_box = [], [], [], [], []
    at_value = at_word = first = 0
    pieces = [[] for _ in range(levels)]
    for level, piece in amr:
        pieces[level].append(piece)
    for level in range(levels):
        boxes = amr.boxes(level)
        lo, hi = boxes[:, 0], boxes[:, 1]
        for (blo, bhi), piece in zip(zip(lo, hi), pieces[level], strict=True):
            cells = int(np.prod(bhi - blo))
            box_lo.extend(int(v) for v in blo)
            box_dims.extend(int(v) for v in bhi - blo)
            box_values.append(at_value)
            box_refined.append(at_word)
            box_level.append(level)
            values.append(piece.fields[field].values.numpy().reshape(-1).astype(np.float32))
            marks = piece.ghosts.cells.numpy().reshape(-1)
            bits = (marks & Ghosts.REFINEDCELL) != 0
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
        "spacing": tack.array(np.asarray(spacings, np.float32).reshape(-1)),
        "bin_origin": ints(bin_origin), "bin_size": ints(bin_size), "bin_dims": ints(bin_dims),
        "bin_base": ints(bin_base), "brick_box": ints(np.concatenate(brick_box)),
        "box_lo": ints(box_lo), "box_dims": ints(box_dims), "box_values": ints(box_values),
        "box_refined": ints(box_refined), "box_level": ints(box_level),
        "values": tack.array(np.concatenate(values)),
        "refined": tack.array(np.concatenate(refined)),
    }
    return AMRVolume(arrays, origin, levels, len(box_level), blend)


@tack.kernel
def _sample_at(box, ps, out):
    for k in range(ps.shape[0]):
        out[k] = box.sample(ps[k])


@tack.kernel
def _cell_size_at(box, ps, out):
    for k in range(ps.shape[0]):
        out[k] = box.cell_size(ps[k])


def sample_functions(box):
    """``tack_sample(pool, p)`` and ``tack_cell_size(pool, p)`` as Slang over one pool,
    and the pool's words (``np.uint32``)."""
    ps = tack.array(np.zeros((1, 3), np.float32), tack.vec3())
    slots = {"ps": ("float", 3), "out": ("float", 1)}
    outs, words = export_shared([
        (_sample_at, (box, ps, tack.empty((1,), tack.f32)), "tack_sample", slots,
         ("float", "{out}[0]")),
        (_cell_size_at, (box, ps, tack.empty((1,), tack.f32)), "tack_cell_size", slots,
         ("float", "{out}[0]")),
    ])
    return joined(*outs), words


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
                v = volume.sample(p0 + d * (t + 0.5 * h))
                c = _transfer(tf, (v - lo) / (hi - lo))
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
