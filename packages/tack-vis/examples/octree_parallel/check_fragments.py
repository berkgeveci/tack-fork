# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The fragments' composite against one picture of the whole forest (study, one
process): ranks holding consecutive xRAGE pieces, a fragment a box of each, front
to back in the kd-tree's order.

    check_fragments.py PIECES.npz OCTREE.bp [--field tev] [--ranks 2 8 64]
                       [--device metal] [--size 640x400] [--step 0.5] [--out DIR]

``PIECES.npz`` holds each piece's root cells (``r<k>``, column-major root ids), as
the pieces scan writes it; ``OCTREE.bp`` is the step's octree (a playback copy
will do). AMR is the AMR session's volume modules beside the Deep Water tools."""

import argparse
import os
import sys
import time

import numpy as np
from fragments import bsp_order, composite, run_boxes
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction, vtkRenderer

import tack
from tack.dataset import octree_io as oio
from tack.dataset import octree_volume as ov

AMR = "/Volumes/T7/deepwaterimpact/tools/amr_volume"
ROOTS = (230, 140, 120)                      # yA31's root grid
ORIGIN = np.array([-2300000.0, -500000.0, -1200000.0])
SIDE = 20000.0                               # a root cell, cm
BLOCKS = tuple(n // 2 for n in ROOTS)        # xRAGE's 2x2x2 blocks of root cells


def piece_runs(path):
    """Each piece's first and last block (column-major block ids)."""
    z = np.load(path)
    nx, ny, _ = ROOTS
    bx, by, _ = BLOCKS
    firsts, lasts = [], []
    k = 0
    while f"r{k}" in z:
        r = z[f"r{k}"]
        b = ((r % nx) // 2) + bx * (((r // nx) % ny) // 2 + by * ((r // (nx * ny)) // 2))
        firsts.append(int(b.min()))
        lasts.append(int(b.max()))
        k += 1
    return np.array(firsts), np.array(lasts)


def rank_boxes(firsts, lasts, n):
    """Each of n ranks' boxes in world coordinates, the ranks holding consecutive
    pieces; a block two ranks share goes whole to the earlier (the octant
    fragments come with the local forests)."""
    cuts = np.linspace(0, len(firsts), n + 1).round().astype(int)
    out = []
    end = -1
    for r in range(n):
        b0 = max(int(firsts[cuts[r]]), end + 1)
        b1 = int(lasts[cuts[r + 1] - 1])
        end = b1
        for i0, i1, j0, j1, k0, k1 in run_boxes(b0, b1, BLOCKS):
            w = 2 * SIDE
            out.append((r, (ORIGIN[0] + i0 * w, ORIGIN[0] + i1 * w, ORIGIN[1] + j0 * w,
                            ORIGIN[1] + j1 * w, ORIGIN[2] + k0 * w, ORIGIN[2] + k1 * w)))
    return out


def tev_functions(lo, hi):
    color, opacity = vtkColorTransferFunction(), vtkPiecewiseFunction()
    for x, c in ((0.0, (0.2, 0.0, 0.3)), (0.1, (0.8, 0.1, 0.1)), (0.4, (1.0, 0.6, 0.1)),
                 (1.0, (1.0, 1.0, 0.8))):
        color.AddRGBPoint(lo + x * (hi - lo), *c)
    for x, a in ((0.0, 0.0), (0.05, 0.05), (0.3, 0.4), (1.0, 0.9)):
        opacity.AddPoint(lo + x * (hi - lo), a)
    return color, opacity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pieces")
    parser.add_argument("octree")
    parser.add_argument("--field", default="tev")
    parser.add_argument("--range", type=float, nargs=2, default=(0.03, 1.0))
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 8, 64])
    parser.add_argument("--device", default="metal")
    parser.add_argument("--size", default="640x400")
    parser.add_argument("--step", type=float, default=0.5, help="sample distance, in cells")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    tack.set_default_device(args.device)
    size = tuple(int(v) for v in args.size.split("x"))
    t = time.perf_counter()
    octree = oio.read_octree(args.octree, fields=[args.field], device=tack.device(args.device))
    renderer = ov.OctreeRenderer(octree, args.field, AMR, blend=2, macrocells=256, divide=-1,
                                 balanced=1)
    print(f"{octree.num_nodes} nodes, renderer in {time.perf_counter() - t:.1f} s")
    lo, hi = args.range
    table = renderer.table(*tev_functions(lo, hi), lo, hi)
    view = vtkRenderer()
    camera = view.GetActiveCamera()
    camera.SetFocalPoint(0.0, 3.0e5, 0.0)
    camera.SetPosition(3.0e6, 1.4e6, 3.4e6)
    camera.SetViewUp(0, 1, 0)
    camera.SetViewAngle(30)
    view.ResetCamera(*renderer.bounds)
    camera.Zoom(1.5)
    eye = camera.GetPosition()
    options = {"sample_distance": args.step, "unit_distance": 1.0, "aspect": size[0] / size[1]}
    whole = renderer.render(camera, size, table, lo, hi, **options)
    firsts, lasts = piece_runs(args.pieces)
    for n in args.ranks:
        boxes = rank_boxes(firsts, lasts, n)
        t = time.perf_counter()
        fragments = [renderer.render(camera, size, table, lo, hi, region=box, **options)
                     for _, box in boxes]
        march = time.perf_counter() - t
        order = bsp_order([box for _, box in boxes], eye)
        picture = composite(fragments, order)
        d = np.abs(picture - whole)
        per_rank = np.bincount([r for r, _ in boxes], minlength=n)
        print(f"{n:3d} ranks: {len(boxes)} fragments ({per_rank.max()} a rank at most), marched "
              f"in {march:.1f} s; against the whole: max {d.max():.4f}, pixels over 1/255 "
              f"{int((d.max(axis=2) > 1 / 255).sum())}, over 8/255 "
              f"{int((d.max(axis=2) > 8 / 255).sum())} of {size[0] * size[1]}")
        if args.out:
            from tack_view import write_png
            os.makedirs(args.out, exist_ok=True)
            for name, image in (("whole", whole), (f"ranks{n}", picture)):
                rgb = np.clip(image[..., :3] + (1 - image[..., 3:4]) * 0.1, 0, 1)
                write_png(os.path.join(args.out, f"{name}.png"),
                          (np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis=2)
                           * 255).astype(np.uint8))


if __name__ == "__main__":
    sys.path.insert(0, AMR)
    main()
