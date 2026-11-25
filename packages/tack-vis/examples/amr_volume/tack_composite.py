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
        near, d, tmin, tmax = volume_tack.rays(camera, size, self.bounds)
        if depth is not None:
            tmax = np.minimum(tmax, depth_distances(camera, size, depth))
        key = (table.tobytes(), lo, hi)
        if self._seen[0] != key:
            self._seen = (key, volume_tack.visible(self.lows, self.highs, table, lo, hi))
        f32 = np.float32
        out = tack.empty((len(near),), tack.vec4(tack.f32))
        grid = self.grid
        volume_tack._march(
            self.box, tack.array(near.astype(f32), tack.vec3()),
            tack.array(d.astype(f32), tack.vec3()), tack.array(tmin.astype(f32)),
            tack.array(tmax.astype(f32)), tack.array(table, tack.vec4()),
            tack.array(np.zeros((int(volume_tack.MAX_BINS), 4), f32), tack.vec4()),
            self._seen[1], float(lo), float(hi), float(sample_distance), float(min_step),
            float(unit_distance * self.cell), *grid["dims"], *grid["origin"], *grid["size"],
            float(0.01 * self.cell), float(jitter), int(size[0]), out)
        return out.numpy().reshape(size[1], size[0], 4)


def composite(frame, volume):
    """``volume`` (premultiplied f32 RGBA) over ``frame`` (u8 RGBA, the same rows):
    u8 RGBA."""
    rgb = volume[..., :3] + (1.0 - volume[..., 3:]) * (frame[..., :3].astype(np.float32) / 255)
    out = frame.copy()
    out[..., :3] = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return out
