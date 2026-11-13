# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A test hierarchy for step 2: three levels over [-1, 1]^3 whose boxes cut through the
features of an analytic field, so that coarse/fine boundaries cross them.

- level 0: 32^3 cells in two boxes (x halves);
- level 1 (ratio 2): the ring's left part, a slab across it (its boundary at x = 0.25
  cuts the ring), the blob at the top, and the region of the level-2 box;
- level 2 (ratio 2): a box around the sharpest blob, cutting it at z = 0.

The field (``analytic``) is three blobs, a ring and a faint ripple; each box holds it
at its cells' centres, so a finer box holds the same field better resolved.
"""

import numpy as np

import tack
import tack.dataset as td

#: Level 0's cells along each axis, the domain's first point and its size.
ROOT, ORIGIN, SIZE = 32, -1.0, 2.0

#: Each level's boxes as (lo, hi) cell indices of the level.
BOXES = [
    [((0, 0, 0), (16, 32, 32)), ((16, 0, 0), (32, 32, 32))],
    [((8, 8, 24), (40, 56, 40)),          # the ring's left part (x < 0.25), |z| < 0.25
     ((40, 24, 20), (56, 56, 36)),        # around the level-2 box
     ((20, 40, 40), (36, 56, 56))],       # the top blob
    [((72, 64, 48), (104, 96, 64))],      # the sharp blob, cut at z = 0
]


@tack.func
def analytic(x, y, z):
    blobs = (exp(-((x - 0.35) ** 2 + (y - 0.25) ** 2 + (z + 0.1) ** 2) / 0.02)
             + 0.8 * exp(-((x + 0.4) ** 2 + (y + 0.35) ** 2 + (z - 0.3) ** 2) / 0.07)
             + 0.6 * exp(-((x + 0.1) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / 0.02))
    ring = 0.7 * exp(-((sqrt(x * x + y * y) - 0.6) ** 2 + z * z) / 0.008)
    ripple = 0.12 * sin(7.0 * x) * sin(7.0 * y) * sin(7.0 * z)
    return blobs + ring + ripple


@tack.kernel
def _field(out, nx, ny, nz, ox, oy, oz, spacing):
    for k, j, i in tack.ndindex(nz, ny, nx):
        out[i + nx * (j + ny * k)] = analytic(ox + (tack.f32(i) + 0.5) * spacing,
                                              oy + (tack.f32(j) + 0.5) * spacing,
                                              oz + (tack.f32(k) + 0.5) * spacing)


@tack.kernel
def _analytic_at(points, out):
    for k in range(points.shape[0]):
        p = points[k]
        out[k] = analytic(p[0], p[1], p[2])


def at(points):
    """``analytic`` at ``points`` (``(n, 3)``), in Tack."""
    out = tack.empty((len(points),), tack.f32)
    _analytic_at(tack.array(np.asarray(points, np.float32), tack.vec3()), out)
    return out.numpy()


def spacing(level):
    return SIZE / (ROOT * 2 ** level)


def box(lo, hi, h, field="values"):
    """An image box of cells ``[lo, hi)`` of the level whose cells are ``h`` wide, with
    ``analytic`` at its cells' centres as cell field ``field``."""
    cells = tuple(int(b - a) for a, b in zip(lo, hi))
    corner = tuple(ORIGIN + a * h for a in lo)
    grid = td.uniform_grid(tuple(n + 1 for n in cells), origin=corner, spacing=(h,) * 3)
    out = td.DataSet(td.StructuredTopology(tuple(n + 1 for n in cells), offset=tuple(lo)),
                     grid.geometry.values)
    values = tack.empty((int(np.prod(cells)),), tack.f32)
    _field(values, *cells, *corner, h)
    out.fields[field] = td.Field(td.Constant(out), values)
    return out


def hierarchy(levels=None):
    """The test hierarchy (its first ``levels`` levels) as an ``AMRDataSet``."""
    boxes = BOXES[:levels]
    made = [[box(lo, hi, spacing(level)) for lo, hi in level_boxes]
            for level, level_boxes in enumerate(boxes)]
    return td.AMRDataSet(made, ratios=[2] * (len(made) - 1))


def as_vtk(amr):
    """The hierarchy as a ``vtkOverlappingAMR``, as a reader would give it."""
    from tack.interop.vtk import dataset_to_vtk

    return dataset_to_vtk(amr)

