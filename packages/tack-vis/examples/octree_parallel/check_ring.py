# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ring by ``Routing`` (study, one process, step 3 of the octree forests'
plan): each rank holds only its own trees, fetches the one-leaf ring its corner
values need from the ranks owning it (blocks of one ``World``, a block a rank),
makes its renderer from its trees and the ring, renders its boxes; the fragments
composite in kd order. Checks as ``check_ranks.py``: own leaves' corner values
bit for bit the whole forest's, and the composite against one picture.

    check_ring.py PIECES.npz OCTREE.bp [--ranks 2 8] [--field tev] [--step 0.1]"""

import argparse
import sys
import time

import numpy as np
from check_fragments import AMR, ORIGIN, ROOTS, SIDE, piece_runs, rank_boxes, tev_functions
from check_ranks import rank_of_roots
from fragments import bsp_order, composite
from ring import own_forest, ring, ringed_forest
from vtkmodules.vtkRenderingCore import vtkRenderer

import tack
from tack.communication import Blocks
from tack.dataset import octree_io as oio
from tack.dataset import octree_volume as ov
from tack.dataset.ghosts import Ghosts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pieces")
    parser.add_argument("octree")
    parser.add_argument("--field", default="tev")
    parser.add_argument("--range", type=float, nargs=2, default=(0.03, 1.0))
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--device", default="metal")
    parser.add_argument("--size", default="640x400")
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    tack.set_default_device(args.device)
    device = tack.device(args.device)
    size = tuple(int(v) for v in args.size.split("x"))
    octree = oio.read_octree(args.octree, fields=[args.field], device=device)
    whole = ov.OctreeRenderer(octree, args.field, AMR, blend=2, macrocells=256, divide=-1,
                              balanced=1)
    lo, hi = args.range
    table = whole.table(*tev_functions(lo, hi), lo, hi)
    view = vtkRenderer()
    camera = view.GetActiveCamera()
    camera.SetFocalPoint(0.0, 3.0e5, 0.0)
    camera.SetPosition(3.0e6, 1.4e6, 3.4e6)
    camera.SetViewUp(0, 1, 0)
    camera.SetViewAngle(30)
    view.ResetCamera(*whole.bounds)
    camera.Zoom(1.5)
    options = {"sample_distance": args.step, "unit_distance": 1.0, "aspect": size[0] / size[1]}
    reference = whole.render(camera, size, table, lo, hi, **options)
    corners = whole.tents.corners.numpy().reshape(-1, 8)
    del whole
    firsts, lasts = piece_runs(args.pieces)
    for n in args.ranks:
        owner = rank_of_roots(firsts, lasts, n)
        boxes = rank_boxes(firsts, lasts, n)
        blocks = Blocks([device] * n)
        with device:
            owner_d = tack.array(owner.astype(np.int32), tack.i32)
        own = {g: own_forest(octree, owner_d, g, args.field) for g in blocks.local}
        tack.sync()
        t = time.perf_counter()
        leaves, probes = ring(blocks, own, owner, owner_d, ROOTS, ORIGIN, SIDE, args.field)
        tack.sync()
        t_ring = time.perf_counter() - t
        fragments, held, equal, ring_count, t_forest = [None] * len(boxes), 0, True, 0, 0.0
        for g in blocks.local:
            t = time.perf_counter()
            forest, ring_leaves = ringed_forest(own[g], leaves[g], args.field)
            tack.sync()
            t_forest += time.perf_counter() - t
            renderer = ov.OctreeRenderer(forest, args.field, AMR, blend=2, macrocells=256,
                                         divide=-1, balanced=1)
            held += forest.num_nodes
            ring_count += ring_leaves
            mine = owner[forest._tree] == g
            leaf = (np.asarray(forest.marks.numpy()) & Ghosts.REFINEDCELL) == 0
            positions = np.flatnonzero(owner[octree._tree] == g)
            whole_leaf = (np.asarray(octree.marks.numpy())[positions] & Ghosts.REFINEDCELL) == 0
            equal &= np.array_equal(renderer.tents.corners.numpy().reshape(-1, 8)[mine & leaf],
                                    corners[positions[whole_leaf]])
            for i, (rank, box) in enumerate(boxes):
                if rank == g:
                    fragments[i] = renderer.render(camera, size, table, lo, hi, region=box,
                                                   **options)
            del renderer, forest
        picture = composite(fragments, bsp_order([box for _, box in boxes],
                                                 camera.GetPosition()))
        d = np.abs(picture - reference)
        print(f"{n:3d} ranks: ring {ring_count} nodes from {sum(probes.values())} probes "
              f"(ring {t_ring:.1f} s, forests {t_forest:.1f} s); "
              f"{held / octree.num_nodes:.3f}x the nodes held; own corners "
              f"bit for bit: {equal}; against the whole: max {d.max():.4f}, over 8/255 "
              f"{int((d.max(axis=2) > 8 / 255).sum())}")


if __name__ == "__main__":
    sys.path.insert(0, AMR)
    main()
