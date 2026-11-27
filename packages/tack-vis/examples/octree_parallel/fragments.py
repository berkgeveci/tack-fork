# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An octree forest volume rendered across ranks by fragments (study): xRAGE's
partition is a run of 2x2x2 blocks of root cells in column-major order a piece, so
a rank holding consecutive pieces holds at most five boxes (a partial row, rows,
block planes, rows, a partial row), and every rank's boxes sit in one kd-tree.
Each rank renders a fragment a box (rays clipped to it); the fragments composite
front to back in the kd-tree's order for the eye. Nothing moves.

Here, the pieces of the plan's first step in one process: the boxes from a run of
blocks (``run_boxes``), their order (``bsp_order``), the composite (``composite``).
See the vault's "Parallel volume rendering - plan", "Octree forests across ranks"."""

import numpy as np


def run_boxes(b0, b1, blocks):
    """The run of blocks ``b0 .. b1`` (inclusive; column-major, x fastest, over a
    grid of ``blocks = (bx, by, bz)``) as boxes in block coordinates, each
    ``(i0, i1, j0, j1, k0, k1)`` half open: a partial row, whole rows, whole planes,
    whole rows, a partial row (at most five)."""
    bx, by, _ = blocks
    out = []
    b = b0
    while b <= b1:
        x, y, z = b % bx, (b // bx) % by, b // (bx * by)
        if x == 0 and y == 0 and b + bx * by - 1 <= b1:           # whole planes
            n = (b1 - b + 1) // (bx * by)
            out.append((0, bx, 0, by, z, z + n))
            b += n * bx * by
        elif x == 0 and b + bx - 1 <= b1:                         # whole rows of a plane
            n = min((b1 - b + 1) // bx, by - y)
            out.append((0, bx, y, y + n, z, z + 1))
            b += n * bx
        else:                                                     # part of a row
            n = min(b1 - b + 1, bx - x)
            out.append((x, x + n, y, y + 1, z, z + 1))
            b += n
    return out


def bsp_order(boxes, eye):
    """The indices of ``boxes`` (disjoint, each ``(x0, x1, y0, y1, z0, z1)``) front
    to back for ``eye``: split by an axis plane no box crosses, the eye's side
    first, recursively. Boxes cut from one raster order always split so."""
    def order(ids):
        if len(ids) <= 1:
            return list(ids)
        for axis in (2, 1, 0):
            lo = np.array([boxes[i][2 * axis] for i in ids])
            hi = np.array([boxes[i][2 * axis + 1] for i in ids])
            for c in np.unique(lo)[1:]:
                below = hi <= c
                above = lo >= c
                if (below | above).all() and below.any() and above.any():
                    near, far = ([i for i, b in zip(ids, below) if b],
                                 [i for i, a in zip(ids, above) if a])
                    if eye[axis] >= c:
                        near, far = far, near
                    return order(near) + order(far)
        raise ValueError("the boxes have no plane between them")
    return order(list(range(len(boxes))))


def composite(fragments, front_to_back):
    """Premultiplied RGBA fragments ``(h, w, 4)`` over one another, front first."""
    out = np.zeros_like(fragments[0])
    for i in front_to_back:
        out += (1.0 - out[..., 3:4]) * fragments[i]
    return out
