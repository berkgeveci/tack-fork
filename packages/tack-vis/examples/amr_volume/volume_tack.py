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
    index space. A cell is found through the level's bins: a uniform grid over the
    level's boxes (``bin_size`` cells, at most a box's smallest extent, so a bin meets
    at most eight boxes), each listing the boxes it meets (CSR). Refined cells are one
    bit each (``refined``, 32 to a word).

    ``blend`` 0 instead takes the finest level with an active cell containing ``p``
    and only that level's cells: per-level trilinear interpolation, clamped at its
    boxes' edges (the naive scheme, which shows seams)."""

    def __init__(self, arrays, origin, levels, blend=1):
        for name, array in arrays.items():
            setattr(self, name, array)
        self.ox, self.oy, self.oz = (float(o) for o in origin)
        self.nlevels = int(levels)
        self.blend = int(blend)

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
            for q in range(self.bin_offsets[slot], self.bin_offsets[slot + 1]):
                box = self.bin_boxes[q]
                a = i - self.box_lo[3 * box]
                c = j - self.box_lo[3 * box + 1]
                e = k - self.box_lo[3 * box + 2]
                if (a >= 0 and a < self.box_dims[3 * box] and c >= 0
                        and c < self.box_dims[3 * box + 1] and e >= 0
                        and e < self.box_dims[3 * box + 2]):
                    out = box
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
                if self.bin_offsets[first] == self.bin_offsets[first + 1]:
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
        if self.blend == 1:
            for level in range(self.nlevels):
                n, d = self.level_sum(level, p, 1)
                num += n
                den += d
        else:
            done = 0
            for back in range(self.nlevels):
                level = self.nlevels - 1 - back
                if done == 0 and self.containing(level, p) == 1:
                    num, den = self.level_sum(level, p, 0)
                    done = 1
        out = 0.0
        if den > 0.0:
            out = num / den
        return out


def amr_volume(amr, spacings, origin, field="values", blend=1):
    """An ``AMRVolume`` of ``amr`` (a ``tack.dataset.AMRDataSet``) and its cell field
    ``field``: ``spacings`` each level's cell size (three per level), ``origin`` the
    domain's first point. The bins and boxes are the hierarchy's few numbers, made on
    the host; the values and the refined marks are the boxes' own."""
    from tack.dataset import Ghosts

    levels = amr.num_levels
    box_lo, box_dims, box_values, box_refined = [], [], [], []
    values, refined = [], []
    bin_origin, bin_size, bin_dims, bin_base, bin_offsets, bin_boxes = [], [], [], [], [], []
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
            values.append(piece.fields[field].values.numpy().reshape(-1).astype(np.float32))
            marks = piece.ghosts.cells.numpy().reshape(-1)
            bits = (marks & Ghosts.REFINEDCELL) != 0
            words = -(-cells // 32)
            packed = np.zeros(words * 32, bool)
            packed[:cells] = bits
            refined.append(np.packbits(packed, bitorder="little").view(np.int32))
            at_value += cells
            at_word += words
        # The level's bins: a box's smallest extent (or less), over the level's boxes.
        size = int(max(1, (hi - lo).min()))
        start = lo.min(axis=0)
        dims = -(-(hi.max(axis=0) - start) // size)
        members = [[] for _ in range(int(np.prod(dims)))]
        for b, (blo, bhi) in enumerate(zip(lo, hi)):
            a = (blo - start) // size
            z = -(-(bhi - start) // size)
            for bk in range(a[2], z[2]):
                for bj in range(a[1], z[1]):
                    for bi in range(a[0], z[0]):
                        members[bi + dims[0] * (bj + dims[1] * bk)].append(first + b)
        bin_origin.extend(int(v) for v in start)
        bin_size.append(size)
        bin_dims.extend(int(v) for v in dims)
        bin_base.append(len(bin_offsets))
        for m in members:
            bin_offsets.append(len(bin_boxes))
            bin_boxes.extend(m)
        bin_offsets.append(len(bin_boxes))
        first += len(boxes)

    def ints(v):
        return tack.array(np.asarray(v, np.int32))

    arrays = {
        "spacing": tack.array(np.asarray(spacings, np.float32).reshape(-1)),
        "bin_origin": ints(bin_origin), "bin_size": ints(bin_size), "bin_dims": ints(bin_dims),
        "bin_base": ints(bin_base), "bin_offsets": ints(bin_offsets),
        "bin_boxes": ints(bin_boxes or [0]),
        "box_lo": ints(box_lo), "box_dims": ints(box_dims), "box_values": ints(box_values),
        "box_refined": ints(box_refined),
        "values": tack.array(np.concatenate(values)),
        "refined": tack.array(np.concatenate(refined)),
    }
    return AMRVolume(arrays, origin, levels, blend)


@tack.kernel
def _sample_at(box, ps, out):
    for k in range(ps.shape[0]):
        out[k] = box.sample(ps[k])


def sample_functions(box):
    """``tack_sample(pool, p)`` as Slang, and the pool's words (``np.uint32``)."""
    ps = tack.array(np.zeros((1, 3), np.float32), tack.vec3())
    outs, words = export_shared([
        (_sample_at, (box, ps, tack.empty((1,), tack.f32)), "tack_sample",
         {"ps": ("float", 3), "out": ("float", 1)}, ("float", "{out}[0]")),
    ])
    return joined(*outs), words


# The reference: the hook's march, in Tack.

@tack.func
def _transfer(tf, t):
    n = tf.shape[0]
    x = min(max(t, 0.0), 1.0) * tack.f32(n - 1)
    i = min(int(x), n - 2)
    f = x - tack.f32(i)
    return tf[i] * (1.0 - f) + tf[i + 1] * f


@tack.kernel
def _march(box, starts, dirs, tmins, tmaxs, tf, lo, hi, h, unit, out):
    for k in range(starts.shape[0]):
        p0 = starts[k]
        d = dirs[k]
        t0 = tmins[k]
        n = 0
        if tmaxs[k] > t0:
            n = min(int((tmaxs[k] - t0) / h), MAX_STEPS)
        r = 0.0
        g = 0.0
        b = 0.0
        a = 0.0
        for j in range(n):
            v = box.sample(p0 + d * (t0 + (tack.f32(j) + 0.5) * h))
            c = _transfer(tf, (v - lo) / (hi - lo))
            alpha = 1.0 - pow(1.0 - c[3], h / unit)
            w = (1.0 - a) * alpha
            r += w * c[0]
            g += w * c[1]
            b += w * c[2]
            a += w
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


def reference(box, camera, size, bounds, tf, lo, hi, h, unit, background):
    """The picture the hook should draw (``(h, w, 4)`` u8, row 0 at the top): the
    rays marched in Tack, composited over ``background`` as the A-buffer does."""
    near, d, tmin, tmax = rays(camera, size, bounds)
    f32 = np.float32
    out = tack.empty((len(near),), tack.vec4(tack.f32))
    _march(box, tack.array(near.astype(f32), tack.vec3()), tack.array(d.astype(f32), tack.vec3()),
           tack.array(tmin.astype(f32)), tack.array(tmax.astype(f32)),
           tack.array(np.asarray(tf, f32), tack.vec4()), float(lo), float(hi), float(h),
           float(unit), out)
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
