# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Step 3 of the AMR volume rendering plan: AMR-Wind's plotfile (``plt255900``: five
levels, 15,598 boxes, 183M cells) volume rendered through the Python mapper.

    AMReX plotfile ─► read_plotfile (~/Work/amrex-reader) ─► vtkOverlappingAMR
                   ─► TackVolume (AMRDataSet, AMRVolume, macrocells) ─► VolumeMapper

Timed: the read, Tack's conversion, export and macrocells, the first frame and the
next ones, with and without empty-space skipping; and checked against the same march
in Tack at a smaller size.

    python amrwind_demo.py [OUTDIR] [--plotfile P] [--field F] [--max-level L]
                           [--size WxH] [--check WxH]
"""

import argparse
import os
import sys
import time

import numpy as np
import vtkmodules.vtkRenderingCore as rr
from tack_volume import TackVolume
from volume_demo import grab, reference, write_png
from volume_mapper import VolumeMapper

import tack

sys.path.insert(0, os.path.expanduser("~/Work/amrex-reader"))

from amrex_reader import read_plotfile

PLOTFILE = "/Volumes/T7/data/plt255900"


def transfer_functions(mapper, lo, hi):
    """Vorticity: transparent below ``lo``, warmer and more opaque up to ``hi``."""
    c = mapper.color
    c.SetColorSpaceToDiverging()
    c.AddRGBPoint(lo, 0.23, 0.30, 0.75)
    c.AddRGBPoint(0.5 * (lo + hi), 0.86, 0.86, 0.86)
    c.AddRGBPoint(hi, 0.71, 0.02, 0.15)
    o = mapper.opacity
    o.AddPoint(0.0, 0.0)
    o.AddPoint(lo, 0.0)
    o.AddPoint(0.5 * (lo + hi), 0.05)
    o.AddPoint(hi, 0.4)


def frames(window, count=5):
    times = []
    for _ in range(count):
        t = time.perf_counter()
        window.Render()
        window.WaitForCompletion()
        times.append(1000 * (time.perf_counter() - t))
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=".")
    parser.add_argument("--plotfile", default=PLOTFILE)
    parser.add_argument("--field", default="mag_vorticity")
    parser.add_argument("--max-level", type=int, default=None)
    parser.add_argument("--size", default="1280x800")
    parser.add_argument("--check", default="640x400",
                        help="the size of the comparison with Tack's march ('' for none)")
    parser.add_argument("--macrocells", type=int, default=64,
                        help="macrocells along the longest axis, at most")
    parser.add_argument("--sample-distance", type=float, default=0.5,
                        help="a step, in local cells")
    parser.add_argument("--blend", type=int, default=1, help="1: tents; 0: naive")
    parser.add_argument("--close", action="store_true",
                        help="look at the finest level's boxes, from nearer")
    parser.add_argument("--opaque-from", type=float, default=0.4)
    parser.add_argument("--opaque-to", type=float, default=2.0)
    args = parser.parse_args()
    tack.set_default_device("cpu")
    size = tuple(int(v) for v in args.size.split("x"))
    background = (0.32, 0.34, 0.43)

    t = time.perf_counter()
    source = read_plotfile(args.plotfile, fields=[args.field], max_level=args.max_level)
    read_s = time.perf_counter() - t
    boxes = [source.GetNumberOfBlocks(level) for level in range(source.GetNumberOfLevels())]
    print(f"read {read_s:.1f} s: {len(boxes)} levels, boxes {boxes}", flush=True)

    volume = TackVolume(args.field, blend=args.blend, macrocells=args.macrocells)
    volume.SetInputDataObject(source)
    t = time.perf_counter()
    volume.Update()
    timings = volume.timings
    poly = volume.GetOutputDataObject(0)
    cells = len(volume.box.values)
    print(f"Tack {time.perf_counter() - t:.1f} s: convert {timings['convert_ms'] / 1000:.1f} s, "
          f"export {timings['export_ms'] / 1000:.1f} s, macrocells "
          f"{timings['macro_ms'] / 1000:.1f} s {timings['macrocells']}; {cells / 1e6:.1f}M "
          f"cells, pool {timings['pool_bytes'] / 2**20:.0f} MiB", flush=True)

    mapper = VolumeMapper(sample_distance=args.sample_distance)
    mapper.SetInputConnection(volume.GetOutputPort())
    transfer_functions(mapper, args.opaque_from, args.opaque_to)
    actor = rr.vtkActor(mapper=mapper)
    actor.ForceTranslucentOn()
    renderer = rr.vtkRenderer(background=background)
    renderer.AddActor(actor)
    window = rr.vtkRenderWindow(size=size, off_screen_rendering=True)
    window.AddRenderer(renderer)
    b = poly.GetFieldData().GetArray("TackBounds").GetTuple(0)
    centre = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    if args.close:
        # The finest level's boxes: their bounding box in world coordinates.
        amr, finest = volume.amr, volume.amr.num_levels - 1
        boxes = amr.boxes(finest)
        h = np.array(volume.box.spacing.numpy()[3 * finest:3 * finest + 3], np.float64)
        lo = np.array(b[::2]) + boxes[:, 0].min(axis=0) * h
        hi = np.array(b[::2]) + boxes[:, 1].max(axis=0) * h
        centre, span = 0.5 * (lo + hi), 0.9 * float(np.max(hi - lo))
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*centre)
    camera.SetPosition(*(centre + np.array([-0.9, -1.3, 0.9]) * span))
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetViewAngle(30.0)
    renderer.ResetCameraClippingRange()

    t = time.perf_counter()
    window.Render()
    first = 1000 * (time.perf_counter() - t)
    skipping = frames(window)
    picture = grab(window)
    tag = f"amrwind_L{source.GetNumberOfLevels() - 1}" + ("_close" if args.close else "")
    write_png(os.path.join(args.out, f"{tag}.png"), picture)
    seen = mapper.march(poly)["seen"].numpy()
    mapper.skip_empty = False
    window.Render()
    marching = frames(window)
    mapper.skip_empty = True
    window.Render()
    print(f"frames at {size[0]}x{size[1]}: first {first:.0f} ms; then {skipping:.1f} ms with "
          f"empty-space skipping ({100 * seen.mean():.0f}% of macrocells visible), "
          f"{marching:.1f} ms without", flush=True)
    # As an interactor style does while the camera moves.
    window.SetDesiredUpdateRate(15.0)
    window.Render()
    moving = frames(window)
    write_png(os.path.join(args.out, f"{tag}_moving.png"), grab(window))
    window.SetDesiredUpdateRate(0.0001)
    window.Render()
    print(f"while the camera moves: {moving:.1f} ms (steps of "
          f"{mapper.interactive_sample_distance:g} local cells, at most about "
          f"{mapper.interactive_steps} a ray)", flush=True)

    if args.check:
        # A window of its own: one resized after rendering came out empty.
        small = tuple(int(v) for v in args.check.split("x"))
        renderer.RemoveActor(actor)
        check_renderer = rr.vtkRenderer(background=background)
        check_renderer.AddActor(actor)
        check_renderer.SetActiveCamera(camera)
        check_window = rr.vtkRenderWindow(size=small, off_screen_rendering=True)
        check_window.AddRenderer(check_renderer)
        check_window.Render()
        check_window.Render()
        picture = grab(check_window)
        t = time.perf_counter()
        expected = reference(volume, mapper, camera, small, background)
        d = np.abs(picture[..., :3].astype(int) - expected[..., :3].astype(int)).max(axis=2)
        print(f"VTK vs Tack at {small[0]}x{small[1]} ({time.perf_counter() - t:.1f} s): "
              f"{100 * float((d > 2).mean()):.3f}% of pixels differ by more than 2/255, "
              f"{100 * float((d > 8).mean()):.3f}% by more than 8 (max {int(d.max())})",
              flush=True)
        check_window.SetDesiredUpdateRate(15.0)
        check_window.Render()
        check_window.Render()
        moving = grab(check_window)
        expected_moving = reference(volume, mapper, camera, small, background, True)
        d = np.abs(moving[..., :3].astype(int) - expected_moving[..., :3].astype(int)).max(axis=2)
        print(f"  while moving: {100 * float((d > 2).mean()):.3f}% of pixels differ by more "
              f"than 2/255 (max {int(d.max())})", flush=True)
        write_png(os.path.join(args.out, f"{tag}_check.png"), picture)
        write_png(os.path.join(args.out, f"{tag}_reference.png"), expected)
    print(f"  pictures in {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
