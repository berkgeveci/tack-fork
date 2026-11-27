# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An xRAGE octree forest volume rendered across MPI ranks (study, step 4 of the
octree forests' plan), on ``tack.communication``: a block a rank in a
``World(comm)``.

1. **Load:** every rank makes the same metadata from xRAGE's pieces (a rank holds
   consecutive pieces: its root cells' owner, its boxes), reads only the file's
   blocks holding its trees (``read_octree(blocks=)``) and keeps its own trees.
2. **Ring:** the one-leaf ghost ring by ``Routing`` (``ring.py``), the ranks'
   forests completed on the device, a renderer each.
3. **Fragments:** a rank renders each of its boxes, rays clipped to it.
4. **Composite by direct send:** the picture's rows are cut into a band a rank;
   each rank sends each band of its fragments that shows anything to the band's
   rank (``Blocks.exchange``), which composites them front to back in the
   kd-tree's order (a Tack kernel a fragment); the bands go to rank 0 the same
   way.

    mpiexec -n 4 python mpi_render.py PIECES.npz OCTREE.bp [--field tev]
        [--device metal] [--size 640x400] [--step 0.5] [--out picture.png] [--check]

``--check`` has rank 0 also render the whole forest alone and compare."""

import argparse
import sys
import time

import numpy as np
from check_fragments import AMR, ORIGIN, ROOTS, SIDE, piece_runs, rank_boxes, tev_functions
from check_ranks import rank_of_roots
from fragments import bsp_order
from mpi4py import MPI
from ring import own_forest, ring, ringed_forest
from vtkmodules.vtkRenderingCore import vtkRenderer

import tack
from tack.communication import Blocks, World
from tack.dataset import octree_io as oio
from tack.dataset import octree_volume as ov


@tack.kernel
def _over(fragments, first, out):
    # The fragment at ``first`` of a message's stack behind what is composited
    # so far (premultiplied RGBA).
    for k in range(out.shape[0]):
        a = out[k][3]
        out[k] = out[k] + (1.0 - a) * fragments[first + k]


def camera_for(bounds):
    view = vtkRenderer()
    camera = view.GetActiveCamera()
    camera.SetFocalPoint(0.0, 3.0e5, 0.0)
    camera.SetPosition(3.0e6, 1.4e6, 3.4e6)
    camera.SetViewUp(0, 1, 0)
    camera.SetViewAngle(30)
    view.ResetCamera(*bounds)
    camera.Zoom(1.5)
    return view, camera


def my_blocks(path, trees):
    """The file's blocks holding any of ``trees`` (sorted root ids)."""
    nodes, held, _ = oio.block_info(path)
    out = []
    for b, (first, last) in enumerate(held):
        if nodes[b] and first >= 0:
            i = np.searchsorted(trees, first)
            if i < len(trees) and trees[i] <= last:
                out.append(b)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pieces")
    parser.add_argument("octree")
    parser.add_argument("--field", default="tev")
    parser.add_argument("--range", type=float, nargs=2, default=(0.03, 1.0))
    parser.add_argument("--device", default="metal")
    parser.add_argument("--size", default="640x400")
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    n = comm.size
    tack.set_default_device(args.device)
    device = tack.device(args.device)
    blocks = Blocks([device], World(comm))
    g = blocks.local[0]
    w, h = (int(v) for v in args.size.split("x"))
    times = {}

    def lap(name, start):
        tack.sync()
        times[name] = time.perf_counter() - start
        return time.perf_counter()

    # 1. Metadata, the same on every rank, and this rank's trees.
    t = time.perf_counter()
    firsts, lasts = piece_runs(args.pieces)
    owner = rank_of_roots(firsts, lasts, n)
    boxes = rank_boxes(firsts, lasts, n)
    trees = np.flatnonzero(owner == g)
    chosen = my_blocks(args.octree, trees)
    part = oio.read_octree(args.octree, blocks=chosen, fields=[args.field], device=device)
    with device:
        owner_d = tack.array(owner.astype(np.int32), tack.i32)
    own = own_forest(part, owner_d, g, args.field)
    del part
    t = lap("load", t)

    # 2. The ring, the forest, the renderer.
    leaves, probes = ring(blocks, {g: own}, owner, owner_d, ROOTS, ORIGIN, SIDE, args.field)
    t = lap("ring", t)
    forest, ring_nodes = ringed_forest(own, leaves[g], args.field)
    renderer = ov.OctreeRenderer(forest, args.field, AMR, blend=2, macrocells=256, divide=-1,
                                 balanced=1)
    t = lap("forest and renderer", t)

    # 3. A fragment a box of this rank's.
    lo, hi = args.range
    table = renderer.table(*tev_functions(lo, hi), lo, hi)
    _, camera = camera_for(renderer.bounds)
    options = {"sample_distance": args.step, "unit_distance": 1.0, "aspect": w / h}
    mine = [i for i, (rank, _) in enumerate(boxes) if rank == g]
    fragments = {i: renderer.render(camera, (w, h), table, lo, hi, region=boxes[i][1], **options)
                 for i in mine}
    t = lap("march", t)

    # 4. Direct send: a band of rows a rank; each band of a fragment that shows
    # anything to its rank, composited there in the kd-tree's order.
    bands = np.linspace(0, h, n + 1).round().astype(int)
    outgoing = {g: {}}
    for r in range(n):
        rows = slice(bands[r], bands[r + 1])
        parts = [(i, f[rows]) for i, f in fragments.items() if f[rows][..., 3].max() > 0]
        if parts:
            with device:
                outgoing[g][r] = {
                    "ids": tack.array(np.array([i for i, _ in parts], np.int32), tack.i32),
                    "rgba": tack.array(np.stack([p for _, p in parts]).reshape(-1, 4)
                                       .astype(np.float32), tack.vec4(tack.f32))}
    incoming = blocks.exchange(outgoing)
    order = bsp_order([box for _, box in boxes], camera.GetPosition())
    place = {i: k for k, i in enumerate(order)}
    rows = bands[g + 1] - bands[g]
    got = []
    for message in incoming[g].values():
        ids = message["ids"].numpy()
        rgba = message["rgba"]
        for k, i in enumerate(ids):
            got.append((place[int(i)], rgba, k))
    got.sort(key=lambda item: item[0])
    with device:
        band = tack.zeros((rows * w,), tack.vec4(tack.f32))
        for _, rgba, k in got:
            _over(rgba, k * rows * w, band)
    t = lap("composite", t)
    gathered = blocks.exchange({g: {0: {"band": band, "rank": tack.array(np.array([g], np.int32),
                                                                         tack.i32)}}})
    t = lap("gather", t)

    every = blocks.all_gather({g: (times, sum(probes.values()), ring_nodes, own.num_nodes,
                                   forest.num_nodes, len(mine), len(got))})
    if g == 0:
        picture = np.zeros((h, w, 4), np.float32)
        for message in gathered[0].values():
            r = int(message["rank"].numpy()[0])
            picture[bands[r]:bands[r + 1]] = message["band"].numpy().reshape(-1, w, 4)
        print(f"{n} ranks, {args.size}, step {args.step}, Tack on {args.device} (one GPU for "
              f"every rank)")
        names = list(times)
        print("  rank  " + "  ".join(f"{k:>18s}" for k in names) +
              "   probes  ring nodes  own nodes  held  boxes  fragments in")
        for r in range(n):
            ti, pr, rn, on, fn, bx, fi = every[r]
            print(f"  {r:4d}  " + "  ".join(f"{1000 * ti[k]:16.0f}ms" for k in names) +
                  f"  {pr:7d}  {rn:10d}  {on:9d}  {fn / on:4.2f}  {bx:5d}  {fi:12d}")
        if args.out:
            sys.path.insert(0, AMR)
            from tack_view import write_png
            rgb = np.clip(picture[..., :3] + (1 - picture[..., 3:4]) * 0.1, 0, 1)
            write_png(args.out, (np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis=2)
                                 * 255).astype(np.uint8))
        if args.check:
            whole_tree = oio.read_octree(args.octree, fields=[args.field], device=device)
            whole = ov.OctreeRenderer(whole_tree, args.field, AMR, blend=2, macrocells=256,
                                      divide=-1, balanced=1)
            reference = whole.render(camera, (w, h), whole.table(*tev_functions(lo, hi), lo, hi),
                                     lo, hi, **options)
            d = np.abs(picture - reference)
            print(f"  against one rank's picture of the whole: max {d.max():.4f}, pixels over "
                  f"8/255 {int((d.max(axis=2) > 8 / 255).sum())} of {w * h}")


if __name__ == "__main__":
    sys.path.insert(0, AMR)
    main()
