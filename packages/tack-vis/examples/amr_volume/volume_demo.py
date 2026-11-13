# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Step 1 of the AMR volume rendering plan: one box, volume rendered by VTK's WebGPU
renderer through a Python mapper whose Slang hook marches the rays, and checked
against the same march in Tack.

    image (cell data) ─► TackVolume ─► VolumeMapper ─► actor (ForceTranslucent)

The field is made by a Tack kernel on the cells' centres: three blobs, a ring and a
faint ripple. The picture and the reference go to OUTDIR.

    python volume_demo.py [OUTDIR] [--cells N] [--size WxH] [--dolly D]
"""

import argparse
import os
import sys
import time

import numpy as np
import volume_tack
import vtkmodules.vtkCommonDataModel
import vtkmodules.vtkRenderingCore as rr
import vtkmodules.vtkRenderingWebGPU  # noqa: F401
from tack_volume import TackVolume
from volume_mapper import VolumeMapper
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkImageData

import tack

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "webgpu_surfaces"))

from renderer import write_png


@tack.kernel
def _field(out, nx, ny, nz, origin, spacing):
    for k, j, i in tack.ndindex(nz, ny, nx):
        x = origin + (tack.f32(i) + 0.5) * spacing
        y = origin + (tack.f32(j) + 0.5) * spacing
        z = origin + (tack.f32(k) + 0.5) * spacing
        blobs = (exp(-((x - 0.35) ** 2 + (y - 0.25) ** 2 + (z + 0.1) ** 2) / 0.04)
                 + 0.8 * exp(-((x + 0.4) ** 2 + (y + 0.35) ** 2 + (z - 0.3) ** 2) / 0.07)
                 + 0.6 * exp(-((x + 0.1) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / 0.02))
        ring = 0.7 * exp(-((sqrt(x * x + y * y) - 0.6) ** 2 + z * z) / 0.008)
        ripple = 0.12 * sin(7.0 * x) * sin(7.0 * y) * sin(7.0 * z)
        out[i + nx * (j + ny * k)] = blobs + ring + ripple


def image(cells):
    """``cells``^3 cells over [-1, 1]^3 with the field as cell data ``values``."""
    spacing = 2.0 / cells
    values = tack.empty((cells ** 3,), tack.f32)
    _field(values, cells, cells, cells, -1.0, spacing)
    out = vtkImageData(dimensions=(cells + 1,) * 3, spacing=(spacing,) * 3, origin=(-1.0,) * 3)
    array = vtkFloatArray(values.numpy())
    array.SetName("values")
    out.GetCellData().AddArray(array)
    return out


def grab(window):
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

    g = vtkWindowToImageFilter(input=window, input_buffer_type=3)
    g.Update()
    img = g.GetOutput()
    w, h, _ = img.GetDimensions()
    rgb = np.asarray(img.GetPointData().GetScalars()).reshape(h, w, -1)[..., :3]
    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
    return np.ascontiguousarray(rgba[::-1])           # row 0 at the top, RGBA


def reference(volume, mapper, camera, size, background, interactive=False):
    """The picture the mapper's hook should draw, marched in Tack (``interactive``: as
    while the camera moves)."""
    march = mapper.march(volume.GetOutputDataObject(0))
    factor, min_step, jitter = march["factor"], 0.0, 0.0
    if interactive:
        changed = mapper.interactive(march)
        factor, min_step, jitter = changed["factor"], changed["min_step"], changed["jitter"]
    return volume_tack.reference(volume.box, camera, size, march["bounds"], march["table"],
                                 march["lo"], march["hi"], factor, march["unit"],
                                 march["grid"], march["seen"], march["skip"], background,
                                 min_step, jitter, march["bins"])


def transfer_functions(mapper, lo, hi):
    r = hi - lo
    c = mapper.color
    c.SetColorSpaceToDiverging()
    c.AddRGBPoint(lo, 0.23, 0.30, 0.75)
    c.AddRGBPoint(lo + 0.5 * r, 0.86, 0.86, 0.86)
    c.AddRGBPoint(hi, 0.71, 0.02, 0.15)
    o = mapper.opacity
    o.AddPoint(lo, 0.0)
    o.AddPoint(lo + 0.3 * r, 0.0)
    o.AddPoint(lo + 0.55 * r, 0.08)
    o.AddPoint(hi, 0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=".")
    parser.add_argument("--cells", type=int, default=96)
    parser.add_argument("--size", default="1280x800")
    parser.add_argument("--dolly", type=float, default=1.3,
                        help="the camera's dolly from the reset view (6: inside the box)")
    args = parser.parse_args()
    tack.set_default_device("cpu")
    size = tuple(int(v) for v in args.size.split("x"))
    background = (0.32, 0.34, 0.43)

    source = image(args.cells)
    volume = TackVolume("values")
    volume.SetInputDataObject(source)
    mapper = VolumeMapper()
    mapper.SetInputConnection(volume.GetOutputPort())
    lo, hi = source.GetCellData().GetArray("values").GetRange()
    transfer_functions(mapper, lo, hi)
    actor = rr.vtkActor(mapper=mapper)
    actor.ForceTranslucentOn()

    renderer = rr.vtkRenderer(background=background)
    renderer.AddActor(actor)
    window = rr.vtkRenderWindow(size=size, off_screen_rendering=True)
    window.AddRenderer(renderer)
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Azimuth(35.0)
    camera.Elevation(25.0)
    camera.Dolly(args.dolly)
    renderer.ResetCameraClippingRange()

    t = time.perf_counter()
    window.Render()
    print(f"first frame {1000 * (time.perf_counter() - t):.0f} ms (Tack "
          f"{volume.timings['tack_ms']:.0f} ms; RequestSlang built {mapper.preparations}x)",
          flush=True)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        window.Render()
        window.WaitForCompletion()
        times.append(1000 * (time.perf_counter() - t))
    print(f"frames: median {np.median(times):.1f} ms at {size[0]}x{size[1]}, "
          f"{args.cells}^3 cells", flush=True)
    picture = grab(window)
    tag = "" if args.dolly == 1.3 else f"_dolly{args.dolly:g}"
    write_png(os.path.join(args.out, f"volume{tag}.png"), picture)

    t = time.perf_counter()
    expected = reference(volume, mapper, camera, size, background)
    print(f"reference (Tack, CPU) {1000 * (time.perf_counter() - t):.0f} ms", flush=True)
    write_png(os.path.join(args.out, f"volume{tag}_reference.png"), expected)
    d = np.abs(picture[..., :3].astype(int) - expected[..., :3].astype(int)).max(axis=2)
    print(f"VTK vs Tack: {100 * float((d > 2).mean()):.3f}% of pixels differ by more than "
          f"2/255, {100 * float((d > 8).mean()):.3f}% by more than 8 (max {int(d.max())})",
          flush=True)
    print(f"  pictures in {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
