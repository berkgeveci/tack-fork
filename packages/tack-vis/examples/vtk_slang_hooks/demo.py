# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tack's surface through VTK's WebGPU mapper and Slang hooks (``tack_hooks``): a
picture, a check, frame times.

The check: four renders write, per pixel, the field's value, u, v and the face (probes
in the ``output`` hook, 24-bit numbers in the colour); Tack then evaluates the field at
each covered pixel's (face, u, v) on the CPU, and the shader's value must agree to f32
(and the 24-bit encoding of the range). So the pixels are Tack's own evaluation.

    python demo.py [straight|curved|tets] [n] [place: 0 cached geometry, 1 the vertex hook,
                                               2 instanced templates]
"""

import os
import sys
import time

import numpy as np
from tack_hooks import _value_at, rr, surface_actors

import tack
import tack.dataset as td
from scenes import camera, surface, wave  # noqa: E402 (on the path through tack_hooks)
from vtkmodules.util.numpy_support import vtk_to_numpy  # noqa: E402

W, H = 1920, 1080
OUT = os.environ.get("OUT", ".")


def tets_surface(n):
    """The cube's image tetrahedralized (5 a hexahedron), an order-3 L2 field: the
    surface is triangles."""
    image = td.uniform_grid((n + 1,) * 3, spacing=(1.0 / n,) * 3, dtype=tack.f64)
    data = td.tetrahedralize(image)
    data.fields["f"] = td.interpolate(data, wave, td.L2(data, order=3))
    return td.extract_surface(data, fields=["f"])


def window_for(actors):
    renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
    for a in actors:
        renderer.AddActor(a.actor)
    window = rr.vtkRenderWindow(size=(W, H), off_screen_rendering=True)
    window.AddRenderer(renderer)
    return window, renderer


def frame(window, renderer, k, frames):
    eye, centre = camera(k, frames)
    cam = renderer.active_camera
    cam.position, cam.focal_point, cam.view_up = tuple(eye), tuple(centre), (0.0, 0.0, 1.0)
    cam.view_angle = 30.0
    cam.SetClippingRange(0.5, 6.0)
    window.Render()
    window.WaitForCompletion()


def grab(window):
    g = rr.vtkWindowToImageFilter(input=window, input_buffer_type=3)
    g.Update()
    img = g.GetOutput()
    w, h, _ = img.GetDimensions()
    return vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[..., :3].astype(np.int64)


def numbers(img):
    return (img[..., 0] << 16) | (img[..., 1] << 8) | img[..., 2]


def check(surf, actors, lo, hi, window, renderer):
    planes = {}
    renderer.background = (0.0, 0.0, 0.0)      # no face: number 0
    for probe in (1, 2, 3, 4):
        for a in actors:
            a.set_probe(probe)
        frame(window, renderer, 0, 30)
        img = grab(window)
        if os.environ.get("PROBE_PNG"):
            np.save(os.path.join(OUT, f"probe{probe}.npy"), img)
        planes[probe] = numbers(img)
    for a in actors:
        a.set_probe(0)
    renderer.background = (1.0, 1.0, 1.0)
    faces = planes[4] - 1
    scale = float(2 ** 24 - 1)
    # At a face's edge the interpolated (u, v) can leave [0, 1] by a hair; the shader
    # evaluates there, but the probe clamps it: such pixels are not compared.
    outside = ((planes[2] == 0) | (planes[2] == scale) | (planes[3] == 0)
               | (planes[3] == scale)) & (faces >= 0)
    covered = (faces >= 0) & ~outside
    check.outside = int(outside.sum())
    shader = lo + planes[1][covered] / scale * (hi - lo)
    u, v = planes[2][covered] / scale, planes[3][covered] / scale
    face = faces[covered]
    tack_value = np.empty(len(face), np.float64)
    field = surf.fields["f"]
    for a in actors:
        mine = (face >= a.base) & (face < a.base + a.count)
        if not mine.any():
            continue
        ids = tack.array((face[mine] - a.base).astype(np.int32))
        pcs = np.zeros((int(mine.sum()), 3), np.float32)
        pcs[:, 0], pcs[:, 1] = u[mine], v[mine]
        out = tack.empty((len(pcs),), tack.f32)
        _value_at(field.view(a.group), ids, tack.array(pcs, tack.vec3()), out)
        tack_value[mine] = out.numpy()
    big = float(np.abs(field.values.numpy()).max())
    err = np.abs(shader - tack_value) / big
    if os.environ.get("PROBE_PNG"):
        np.savez(os.path.join(OUT, "check.npz"), face=face, u=u, v=v, shader=shader,
                 tack=tack_value, where=np.argwhere(covered))
    return int(covered.sum()), len(np.unique(face)), float(err.max()), float(np.median(err))


def main(kind="straight", n=16, place=0, frames=30):
    tack.set_default_device("cpu")
    surf = tets_surface(n) if kind == "tets" else surface(kind, n)
    start = time.perf_counter()
    actors, (lo, hi) = surface_actors(surf, "f", place=bool(place),
                                      instanced=True if place == 2 else None)
    setup = time.perf_counter() - start
    window, renderer = window_for(actors)
    start = time.perf_counter()
    frame(window, renderer, 0, frames)
    first = time.perf_counter() - start
    times = []
    for k in range(1, frames + 1):
        start = time.perf_counter()
        frame(window, renderer, k, frames)
        times.append(time.perf_counter() - start)
    tris = sum(a.triangles for a in actors)
    print(f"{kind} n={n} place={place}: {surf.num_cells} faces, {len(actors)} groups, "
          f"{tris} triangles; setup {1e3 * setup:.0f} ms, first frame {1e3 * first:.0f} ms, "
          f"steady frame {1e3 * float(np.median(times)):.2f} ms", flush=True)

    frame(window, renderer, 0, frames)
    from vtkmodules.vtkIOImage import vtkPNGWriter
    png = os.path.join(OUT, f"hooks-{kind}-{n}-place{place}.png")
    shot = rr.vtkWindowToImageFilter(input=window)   # held: a port does not own its producer
    writer = vtkPNGWriter(file_name=png)
    writer.SetInputConnection(shot.output_port)
    writer.Write()
    pixels, faces, worst, median = check(surf, actors, lo, hi, window, renderer)
    print(f"  check: {pixels} pixels on {faces} faces ({check.outside} at a clamped u or v "
          f"left out); shader vs Tack, relative to the field's largest value: max "
          f"{worst:.2e}, median {median:.2e}  ({png})", flush=True)

    # A timestep: a new field of the same space (mirrored in the range), uploaded into
    # the field buffers.
    old = surf.fields["f"]
    surf.fields["f"] = td.Field(old.space, tack.array((lo + hi) - old.values.numpy()))
    steps = []
    for _ in range(5):
        start = time.perf_counter()
        for a in actors:
            a.update_field(surf, "f")
        frame(window, renderer, 0, frames)
        steps.append(time.perf_counter() - start)
    pixels, faces, worst, median = check(surf, actors, lo, hi, window, renderer)
    print(f"  after a new field (update + frame {1e3 * float(np.median(steps)):.1f} ms): "
          f"max {worst:.2e}, median {median:.2e}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "straight", int(a[1]) if len(a) > 1 else 16,
         int(a[2]) if len(a) > 2 else 0)
