# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Step 2 of the AMR volume rendering plan: a three-level hierarchy (``amr_case``)
volume rendered through the Python mapper, sampled by tent basis functions across
levels, against the naive per-level scheme, each checked against the same march in
Tack.

    vtkOverlappingAMR ─► TackVolume (AMRDataSet, AMRVolume) ─► VolumeMapper ─► actor

Also measured, in Tack: the reconstruction against the analytic field at random
points (both schemes, and level 0 alone), and the largest jump between neighbouring
samples along a line crossing a coarse/fine boundary through the ring.

    python amr_demo.py [OUTDIR] [--size WxH] [--outlines] [--close]
"""

import argparse
import os
import time

import amr_case
import numpy as np
import volume_tack
import vtkmodules.vtkRenderingCore as rr
from tack_volume import TackVolume
from volume_demo import grab, transfer_functions, write_png
from volume_mapper import VolumeMapper
from vtkmodules.vtkFiltersSources import vtkOutlineSource

import tack


def sample(sampler, points):
    out = tack.empty((len(points),), tack.f32)
    volume_tack._sample_at(sampler, tack.array(np.asarray(points, np.float32), tack.vec3()),
                           out)
    return out.numpy()


def measure(amr):
    """The reconstruction's error and continuity, in Tack."""
    spacings = [[amr_case.spacing(level)] * 3 for level in range(amr.num_levels)]
    origin = (amr_case.ORIGIN,) * 3
    points = np.random.default_rng(0).uniform(-1, 1, (200_000, 3)).astype(np.float32)
    truth = amr_case.at(points)
    x = np.linspace(-0.2, 0.6, 8001)
    line = np.stack([x, np.full_like(x, -0.5), np.zeros_like(x)], axis=1)
    rows = []
    level0 = volume_tack.amr_volume(amr_case.hierarchy(1), spacings[:1], origin)
    for name, sampler in (("tents", volume_tack.amr_volume(amr, spacings, origin, blend=1)),
                          ("naive", volume_tack.amr_volume(amr, spacings, origin, blend=0)),
                          ("level 0 alone", level0)):
        e = np.abs(sample(sampler, points) - truth)
        jumps = np.abs(np.diff(sample(sampler, line)))
        rows.append((name, e.max(), e.mean(), jumps.max(), x[jumps.argmax()]))
    print("  reconstruction vs the analytic field (200k random points), and the largest "
          "jump between samples 1e-4 apart on y = -0.5, z = 0 (crossing x = 0.25):", flush=True)
    for name, emax, emean, jump, where in rows:
        print(f"    {name:14s} error max {emax:.4f} mean {emean:.5f}; jump {jump:.4f} at "
              f"x = {where:.4f}", flush=True)


def outlines(amr):
    """The boxes' outlines, coloured by level, as actors."""
    colours = [(0.9, 0.9, 0.9), (1.0, 0.85, 0.3), (0.3, 1.0, 0.5)]
    actors = []
    for level in range(amr.num_levels):
        h = amr_case.spacing(level)
        for lo, hi in amr.boxes(level):
            a = amr_case.ORIGIN + lo * h
            b = amr_case.ORIGIN + hi * h
            source = vtkOutlineSource(bounds=(a[0], b[0], a[1], b[1], a[2], b[2]))
            mapper = rr.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = rr.vtkActor(mapper=mapper)
            actor.GetProperty().SetColor(*colours[level % len(colours)])
            actors.append((source, actor))
    return actors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=".")
    parser.add_argument("--size", default="1280x800")
    parser.add_argument("--outlines", action="store_true")
    parser.add_argument("--close", action="store_true",
                        help="look closely at where the boundary at x = 0.25 cuts the ring")
    args = parser.parse_args()
    tack.set_default_device("cpu")
    size = tuple(int(v) for v in args.size.split("x"))
    background = (0.32, 0.34, 0.43)

    amr = amr_case.hierarchy()
    source = amr_case.as_vtk(amr)
    print(f"{amr}: {sum(len(amr.boxes(level)) for level in range(amr.num_levels))} boxes",
          flush=True)
    measure(amr)

    volume = TackVolume("values")
    volume.SetInputDataObject(source)
    mapper = VolumeMapper()
    mapper.SetInputConnection(volume.GetOutputPort())
    volume.Update()
    lo, hi = volume.GetOutputDataObject(0).GetFieldData().GetArray("TackRange").GetTuple2(0)
    transfer_functions(mapper, lo, hi)
    actor = rr.vtkActor(mapper=mapper)
    actor.ForceTranslucentOn()
    renderer = rr.vtkRenderer(background=background)
    renderer.AddActor(actor)
    kept = outlines(amr) if args.outlines else []
    for _, outline in kept:
        renderer.AddActor(outline)
    window = rr.vtkRenderWindow(size=size, off_screen_rendering=True)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    if args.close:
        camera.SetFocalPoint(0.25, -0.56, 0.0)
        camera.SetPosition(0.55, -1.5, 0.45)
    else:
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(1.6, -3.6, 2.4)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetViewAngle(30.0)
    renderer.ResetCameraClippingRange()

    suffix = ("_close" if args.close else "") + ("_outlines" if args.outlines else "")
    for blend, name in ((1, "tents"), (0, "naive")):
        volume.SetBlend(blend)
        t = time.perf_counter()
        window.Render()
        first = 1000 * (time.perf_counter() - t)
        times = []
        for _ in range(5):
            t = time.perf_counter()
            window.Render()
            window.WaitForCompletion()
            times.append(1000 * (time.perf_counter() - t))
        timings = volume.timings
        print(f"{name}: Tack {timings['convert_ms']:.0f} ms converting, "
              f"{timings['export_ms']:.0f} ms exporting, pool "
              f"{timings['pool_bytes'] / 1024:.0f} KiB; first frame {first:.0f} ms, then "
              f"median {np.median(times):.1f} ms at {size[0]}x{size[1]}", flush=True)
        picture = grab(window)
        write_png(os.path.join(args.out, f"amr_{name}{suffix}.png"), picture)
        if args.outlines:
            continue
        cell = volume.GetOutputDataObject(0).GetFieldData().GetArray(
            "TackCellSize").GetValue(0)
        h, unit = mapper.steps(cell)
        t = time.perf_counter()
        bounds = volume.GetOutputDataObject(0).GetFieldData().GetArray("TackBounds").GetTuple(0)
        expected = volume_tack.reference(volume.box, camera, size, bounds,
                                         mapper.table(lo, hi), lo, hi, h, unit, background)
        reference_ms = 1000 * (time.perf_counter() - t)
        write_png(os.path.join(args.out, f"amr_{name}{suffix}_reference.png"), expected)
        d = np.abs(picture[..., :3].astype(int) - expected[..., :3].astype(int)).max(axis=2)
        print(f"  VTK vs Tack ({reference_ms / 1000:.1f} s): "
              f"{100 * float((d > 2).mean()):.3f}% of pixels differ by more than 2/255, "
              f"{100 * float((d > 8).mean()):.3f}% by more than 8 (max {int(d.max())})",
              flush=True)
    print(f"  pictures in {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
