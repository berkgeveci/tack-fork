# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A volume rendered by Tack alone and blended into a frame VTK has drawn.

Nothing here is a shader: the rays are marched by ``volume_tack._march``, a Tack kernel
on Tack's device (the CPU or a GPU, whatever rasterizes the frame), and what crosses to
VTK and back is two host arrays:

- in, the frame's depth (``vtkRenderWindow.GetZbufferData``), which ends each ray at
  the opaque geometry in front of it;
- out, the volume's premultiplied RGBA, to blend over the frame's colour.

``TackRenderer`` holds the Tack side (the sampler, the macrocells) and ``render`` makes
one image. ``composite`` blends it over a frame.
"""

from math import sqrt

import numpy as np
import volume_tack
from tack_volume import TackVolume

import tack


def frame_depth(window):
    """The window's depth buffer as ``(h, w)`` f32 in [0, 1], row 0 at the top (VTK's
    rows start at the bottom)."""
    w, h = window.GetSize()
    from vtkmodules.vtkCommonCore import vtkFloatArray

    z = vtkFloatArray()
    window.GetZbufferData(0, 0, w - 1, h - 1, z)
    return np.asarray(z).reshape(h, w)[::-1]


def depth_distances(camera, size, depth):
    """Per pixel (row 0 at the top), how far along its ray from the near plane the
    frame's ``depth`` lies: the depth buffer's [0, 1] is the camera's near to far."""
    w, h = size
    m = np.array([[camera.GetCompositeProjectionTransformMatrix(w / h, 0, 1).GetElement(i, j)
                   for j in range(4)] for i in range(4)])
    inverse = np.linalg.inv(m)
    j, i = np.mgrid[0:h, 0:w]
    x = ((i + 0.5) / w * 2 - 1).reshape(-1)
    y = (1 - (j + 0.5) / h * 2).reshape(-1)

    def unproject(z):
        clip = np.stack([x, y, z, np.ones_like(x)], axis=1)
        p = clip @ inverse.T
        return p[:, :3] / p[:, 3:]

    z = np.asarray(depth, np.float64).reshape(-1)
    return np.linalg.norm(unproject(z) - unproject(np.zeros_like(z)), axis=1)


@tack.func
def _unproject(inverse, x, y, z):
    """The world point of normalized device coordinates ``(x, y, z)``, by the inverse
    (16 values, row major) of the camera's world-to-device matrix."""
    px = inverse[0] * x + inverse[1] * y + inverse[2] * z + inverse[3]
    py = inverse[4] * x + inverse[5] * y + inverse[6] * z + inverse[7]
    pz = inverse[8] * x + inverse[9] * y + inverse[10] * z + inverse[11]
    pw = inverse[12] * x + inverse[13] * y + inverse[14] * z + inverse[15]
    return tack.vec3(px / pw, py / pw, pz / pw)


@tack.func
def _slab(p, d, lo, hi):
    """Where the ray ``p + t d`` crosses the planes ``lo`` and ``hi`` of one axis: the
    nearer and the farther ``t``."""
    inv = volume_tack._safe_inverse(d)
    a = (lo - p) * inv
    b = (hi - p) * inv
    return min(a, b), max(a, b)


@tack.kernel
def _rays(inverse, depth, ends, w, h, x0, x1, y0, y1, z0, z1, starts, dirs, tmins, tmaxs):
    """Each pixel's ray (row 0 at the top): its start on the near plane, its unit
    direction, and where it enters and leaves the box, leaving no later than the
    frame's ``depth`` there (``ends`` 1; the depth buffer's [0, 1] is near to far)."""
    for k in range(starts.shape[0]):
        x = (tack.f32(k % w) + 0.5) / tack.f32(w) * 2.0 - 1.0
        y = 1.0 - (tack.f32(k // w) + 0.5) / tack.f32(h) * 2.0
        near = _unproject(inverse, x, y, 0.0)
        far = _unproject(inverse, x, y, 1.0)
        d = far - near
        d = d / sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        ax, bx = _slab(near[0], d[0], x0, x1)
        ay, by = _slab(near[1], d[1], y0, y1)
        az, bz = _slab(near[2], d[2], z0, z1)
        t1 = min(bx, by, bz)
        if ends == 1:
            e = _unproject(inverse, x, y, depth[k]) - near
            t1 = min(t1, sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]))
        starts[k] = near
        dirs[k] = d
        tmins[k] = max(ax, ay, az, 0.0)
        tmaxs[k] = t1


class TackRenderer:
    """The cell array ``array`` of ``data`` (a ``vtkImageData`` or a
    ``vtkOverlappingAMR``) as a volume Tack renders: the sampler ``TackVolume`` makes
    (tents across levels with ``blend=1``) and its macrocells, made once."""

    def __init__(self, data, array="values", blend=1, macrocells=64):
        prepare = TackVolume(array, blend, macrocells)
        self.box, self.bounds, cells, self.range, self.cell, coarse = prepare._sampler(data)
        self.grid, self.lows, self.highs = volume_tack.macro_grid(self.box, cells, self.bounds,
                                                                  coarse, macrocells)
        self.amr = getattr(prepare, "amr", None)
        self._seen = (None, None)

    def table(self, color, opacity, lo, hi, size=256):
        """VTK's transfer functions sampled over ``[lo, hi]``: ``(size, 4)`` f32."""
        x = np.linspace(lo, hi, size)
        rgb = np.array([color.GetColor(v) for v in x])
        a = np.array([opacity.GetValue(v) for v in x])
        return np.concatenate([rgb, a[:, None]], axis=1).astype(np.float32)

    def render(self, camera, size, table, lo, hi, depth=None, sample_distance=0.5,
               unit_distance=1.0, min_step=0.0, jitter=0.0):
        """The volume seen by ``camera`` at ``size`` (w, h): ``(h, w, 4)`` f32,
        premultiplied, row 0 at the top. With ``depth`` (the frame's, ``frame_depth``)
        each ray ends at the frame's surface. Steps are ``sample_distance`` local cells
        and at least ``min_step``; ``table`` (over ``[lo, hi]``) gives opacity per
        ``unit_distance`` finest cells."""
        import time

        t0 = time.perf_counter()
        w, h = size
        count = w * h
        m = np.array([[camera.GetCompositeProjectionTransformMatrix(w / h, 0, 1)
                       .GetElement(i, j) for j in range(4)] for i in range(4)])
        f32 = np.float32
        inverse = tack.array(np.linalg.inv(m).astype(f32).reshape(-1))
        if depth is None:
            depths, ends = tack.empty((1,), tack.f32), 0
        else:
            depths, ends = tack.array(np.ascontiguousarray(depth, f32).reshape(-1)), 1
        starts = tack.empty((count,), tack.vec3(tack.f32))
        dirs = tack.empty((count,), tack.vec3(tack.f32))
        tmins = tack.empty((count,), tack.f32)
        tmaxs = tack.empty((count,), tack.f32)
        _rays(inverse, depths, ends, w, h, *(float(v) for v in self.bounds), starts, dirs,
              tmins, tmaxs)
        key = (table.tobytes(), lo, hi)
        if self._seen[0] != key:
            self._seen = (key, volume_tack.visible(self.lows, self.highs, table, lo, hi))
        out = tack.empty((count,), tack.vec4(tack.f32))
        grid = self.grid
        arrays = (starts, dirs, tmins, tmaxs, tack.array(table, tack.vec4()),
                  tack.array(np.zeros((int(volume_tack.MAX_BINS), 4), f32), tack.vec4()))
        t1 = t0
        tack.sync()
        t2 = time.perf_counter()
        volume_tack._march(
            self.box, *arrays,
            self._seen[1], float(lo), float(hi), float(sample_distance), float(min_step),
            float(unit_distance * self.cell), *grid["dims"], *grid["origin"], *grid["size"],
            float(0.01 * self.cell), float(jitter), int(size[0]), out)
        tack.sync()
        t3 = time.perf_counter()
        picture = out.numpy().reshape(size[1], size[0], 4)
        #: The last render's parts, in ms: the rays (a kernel, with the uploads of the
        #: depth and the tables), the march (the kernel alone), the read.
        self.timings = {"rays_ms": 1000 * (t2 - t1),
                        "march_ms": 1000 * (t3 - t2),
                        "read_ms": 1000 * (time.perf_counter() - t3)}
        return picture


def composite(frame, volume):
    """``volume`` (premultiplied f32 RGBA) over ``frame`` (u8 RGBA, the same rows):
    u8 RGBA."""
    rgb = volume[..., :3] + (1.0 - volume[..., 3:]) * (frame[..., :3].astype(np.float32) / 255)
    out = frame.copy()
    out[..., :3] = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return out
