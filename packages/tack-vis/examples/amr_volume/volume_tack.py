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
