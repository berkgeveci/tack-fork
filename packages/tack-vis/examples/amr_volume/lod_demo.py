# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a client needs of AMR-Wind (the client data plan, items 1 and 2): pictures made
from less than the whole hierarchy, against the picture from all of it.

For the full view and the close view, VTK draws the whole hierarchy in f32, then the
coarse levels alone (0-1, then 0-2), the finer levels left out:

- as values: a coarse cell stands in for the finer data beneath it with its own value;
- with histograms: each stand-in carries a histogram of the finer values beneath it
  (``volume_tack.HistogramAMRVolume``), classified bin by bin: 13 or 8 bins of 8 or 4
  bits, and with a floor (``--floor``, the transfer function's transparent threshold)
  below which no histogram is kept.

(``select_boxes`` and quantization were measured with an earlier version of this
script: the client data plan's "First results".)

Each: the cells held, the pool's size, the frame time and how the picture differs
from the whole hierarchy's (pixels over 2/255 and 8/255, the largest difference, the
PSNR).

    python lod_demo.py [OUTDIR] [--plotfile P] [--size WxH]
"""

import argparse
import os
import sys
import time

import numpy as np
import vtkmodules.vtkRenderingCore as rr
from amrwind_demo import PLOTFILE, frames, transfer_functions
from tack_volume import TackVolume
from volume_demo import grab, write_png
from volume_mapper import VolumeMapper

import tack

sys.path.insert(0, os.path.expanduser("~/Work/amrex-reader"))

from amrex_reader import read_plotfile


def views(volume):
    """The full view and the close view (the finest level's boxes) as camera settings."""
    poly = volume.GetOutputDataObject(0)
    b = poly.GetFieldData().GetArray("TackBounds").GetTuple(0)
    lo, hi = np.array(b[::2]), np.array(b[1::2])
    full = (0.5 * (lo + hi), float(np.max(hi - lo)))
    finest = volume.amr.num_levels - 1
    boxes = volume.amr.boxes(finest)
    h = np.array(volume.spacings[finest], np.float64)
    a = lo + boxes[:, 0].min(axis=0) * h
    z = lo + boxes[:, 1].max(axis=0) * h
    close = (0.5 * (a + z), 0.9 * float(np.max(z - a)))
    return {"full": full, "close": close}


def place(camera, renderer, centre, span):
    camera.SetFocalPoint(*centre)
    camera.SetPosition(*(centre + np.array([-0.9, -1.3, 0.9]) * span))
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetViewAngle(30.0)
    renderer.ResetCameraClippingRange()


def difference(picture, reference):
    d = np.abs(picture[..., :3].astype(int) - reference[..., :3].astype(int))
    worst = d.max(axis=2)
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    return (f"{100 * float((worst > 2).mean()):.2f}% > 2/255, "
            f"{100 * float((worst > 8).mean()):.2f}% > 8, max {int(worst.max())}, "
            f"PSNR {psnr:.1f} dB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=".")
    parser.add_argument("--plotfile", default=PLOTFILE)
    parser.add_argument("--field", default="mag_vorticity")
    parser.add_argument("--size", default="1280x800")
    parser.add_argument("--floor", type=float, default=0.4,
                        help="no histograms where the finer data is all below this "
                             "(the transfer function is transparent there)")
    args = parser.parse_args()
    tack.set_default_device("cpu")
    size = tuple(int(v) for v in args.size.split("x"))
    background = (0.32, 0.34, 0.43)

    source = read_plotfile(args.plotfile, fields=[args.field])
    volume = TackVolume(args.field, macrocells=128)
    volume.SetInputDataObject(source)
    mapper = VolumeMapper()
    mapper.SetInputConnection(volume.GetOutputPort())
    transfer_functions(mapper, 0.4, 2.0)
    actor = rr.vtkActor(mapper=mapper)
    actor.ForceTranslucentOn()
    renderer = rr.vtkRenderer(background=background)
    renderer.AddActor(actor)
    window = rr.vtkRenderWindow(size=size, off_screen_rendering=True)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    volume.Update()
    places = views(volume)
    levels = len(volume.amr.boxes(0)) and volume.amr.num_levels
    def up_to(base):
        return [np.arange(len(volume.amr.boxes(level))) if level <= base
                else np.array([], int) for level in range(levels)]

    def draw(name, label, view, keep=None, average=False, bits=None, histograms=None):
        volume.SetKeep(keep)
        volume.SetAverageDown(average)
        volume.SetQuantize(bits, "log")
        volume.SetHistograms(histograms is not None, *(histograms or ()))
        t = time.perf_counter()
        volume.Update()
        tack_s = time.perf_counter() - t
        window.Render()
        ms = frames(window)
        picture = grab(window)
        write_png(os.path.join(args.out, f"lod_{view}_{name}.png"), picture)
        timings = volume.timings
        held = volume.box.host["boxes"]
        stand_ins = volume.box.host.get("stand_in_cells")
        label = label if stand_ins is None else f"{label} ({stand_ins / 1e6:.1f}M)"
        print(f"  {label:52s} {timings['cells'] / 1e6:6.1f}M cells "
              f"{timings['pool_bytes'] / 2**20:6.0f} MiB  boxes {held}  Tack {tack_s:4.1f} s  "
              f"frame {ms:5.0f} ms", flush=True)
        return picture

    for view, (centre, span) in places.items():
        place(camera, renderer, centre, span)
        print(f"{view} view:", flush=True)
        reference = draw("all", "all five levels, f32", view)
        rows = [("l01", "levels 0-1, values", up_to(1), False, None, None),
                ("l02", "levels 0-2, values", up_to(2), False, None, None)]
        for bins, hbits in ((13, 8), (8, 8), (8, 4)):
            for floor_value in (None, args.floor):
                tag = f"l01h{bins}x{hbits}" + ("f" if floor_value is not None else "")
                label = (f"levels 0-1 + histograms {bins}x{hbits} bits"
                         + (f", floor {floor_value:g}" if floor_value is not None else ""))
                rows.append((tag, label, up_to(1), False, None, (bins, hbits, floor_value)))
        for name, label, keep, average, bits, histograms in rows:
            picture = draw(name, label, view, keep, average, bits, histograms)
            print(f"  {'':44s} vs all: {difference(picture, reference)}", flush=True)
    print(f"  pictures in {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
