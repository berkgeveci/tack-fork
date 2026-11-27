# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Local forests a rank (study, one process, step 2 of the octree forests'
plan): each rank holds only the trees of its blocks and a halo of copies (the
trees within one root cell of its region), makes its own renderer from them,
and renders its boxes; the ranks' fragments composite in kd order. Checks:
every rank's own leaves' corner values against one forest's, bit for bit (the
halo reaches every neighbour a corner reads), and the composite against one
picture of the whole forest.

    check_ranks.py PIECES.npz OCTREE.bp [--ranks 2 8] [--field tev] [--step 0.1]

The halo here is whole root cells, more than the one-leaf ring the plan's
exchange will send."""

import argparse
import sys
import time

import numpy as np
from check_fragments import AMR, BLOCKS, ROOTS, piece_runs, rank_boxes, tev_functions
from fragments import bsp_order, composite
from vtkmodules.vtkRenderingCore import vtkRenderer

import tack
from tack.dataset import octree_io as oio
from tack.dataset import octree_volume as ov
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet


def rank_of_roots(firsts, lasts, n):
    """Each root cell's rank: the rank whose run of blocks holds its block (a block
    two ranks share goes to the earlier, as ``rank_boxes``)."""
    cuts = np.linspace(0, len(firsts), n + 1).round().astype(int)
    starts, end = [], -1
    for r in range(n):
        starts.append(max(int(firsts[cuts[r]]), end + 1))
        end = int(lasts[cuts[r + 1] - 1])
    nx, ny, nz = ROOTS
    bx, by, _ = BLOCKS
    t = np.arange(nx * ny * nz)
    block = (t % nx) // 2 + bx * (((t // nx) % ny) // 2 + by * ((t // (nx * ny)) // 2))
    return np.searchsorted(np.array(starts), block, side="right") - 1


def halo_of(mine):
    """The root cells within one of ``mine`` (a boolean grid, z y x) and not in it."""
    # Grown by one along each axis in turn: the 26-neighbourhood.
    near = mine.copy()
    for axis in range(3):
        grown = near.copy()
        for shift in (-1, 1):
            moved = np.roll(near, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            moved[tuple(edge)] = False                  # no wrapping across the domain
            grown |= moved
        near = grown
    return near & ~mine


def local_forest(octree, trees, field):
    """The octree's trees ``trees`` (a sorted array of root ids), whole."""
    keep = np.isin(octree._tree, trees)
    hidden = (np.asarray(octree.marks.numpy()) & Ghosts.HIDDENCELL) != 0
    return OctreeDataSet(octree.roots, octree._tree[keep], octree._level[keep],
                         octree._key[keep],
                         fields={field: np.asarray(octree.fields[field].numpy())[keep]},
                         mask=hidden[keep], _ordered=True)


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
    size = tuple(int(v) for v in args.size.split("x"))
    octree = oio.read_octree(args.octree, fields=[args.field], device=tack.device(args.device))
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
    leaf = (np.asarray(octree.marks.numpy()) & Ghosts.REFINEDCELL) == 0
    firsts, lasts = piece_runs(args.pieces)
    nx, ny, nz = ROOTS
    for n in args.ranks:
        owner = rank_of_roots(firsts, lasts, n)
        boxes = rank_boxes(firsts, lasts, n)
        fragments, held, own_cells, equal = [None] * len(boxes), 0, 0, True
        t = time.perf_counter()
        for r in range(n):
            mine = (owner == r).reshape(nz, ny, nx)
            halo = halo_of(mine)
            trees = np.flatnonzero((mine | halo).ravel())
            forest = local_forest(octree, trees, args.field)
            renderer = ov.OctreeRenderer(forest, args.field, AMR, blend=2, macrocells=256,
                                         divide=-1, balanced=1)
            held += forest.num_nodes
            # Own leaves' corners against the whole forest's: their nodes are the
            # whole's nodes of the same trees, in the same order.
            own = np.isin(forest._tree, np.flatnonzero(mine.ravel()))
            local_leaf = (np.asarray(forest.marks.numpy()) & Ghosts.REFINEDCELL) == 0
            pick = own & local_leaf
            positions = np.flatnonzero(np.isin(octree._tree, trees))   # the whole's nodes held
            same = np.array_equal(renderer.tents.corners.numpy().reshape(-1, 8)[pick],
                                  corners[positions[pick]])
            equal &= same
            own_cells += int(pick.sum())
            for i, (rank, box) in enumerate(boxes):
                if rank == r:
                    fragments[i] = renderer.render(camera, size, table, lo, hi, region=box,
                                                   **options)
            del renderer, forest
        order = bsp_order([box for _, box in boxes], camera.GetPosition())
        picture = composite(fragments, order)
        d = np.abs(picture - reference)
        print(f"{n:3d} ranks: {held / octree.num_nodes:.2f}x the nodes held (halo included), "
              f"own leaves {own_cells} of {int(leaf.sum())}; own corners bit for bit: {equal}; "
              f"{time.perf_counter() - t:.1f} s; against the whole: max {d.max():.4f}, "
              f"over 8/255 {int((d.max(axis=2) > 8 / 255).sum())}")


if __name__ == "__main__":
    sys.path.insert(0, AMR)
    main()
