# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The LIC prototype's data: a linear hexahedral mesh of the unit cube (geometry p=1)
with a linear vector field (H1 p=1) interpolated from an analytic flow, cut by an
oblique plane. The cut's triangles carry the field at their points, where p=1 is exact
and linear between them, so a renderer interpolating per vertex draws the field
exactly."""

import numpy as np

import tack
import tack.dataset as td

TAU = tack.constant(6.283185307179586)
PI = tack.constant(3.141592653589793)


@tack.func
def flow(x):
    """Cells of counter-rotating vortices in x and y (Taylor-Green), with saddles where
    four cells meet, and a weak z component."""
    return tack.vec3(-tack.sin(TAU * x[0]) * tack.cos(TAU * x[1]),
                     tack.cos(TAU * x[0]) * tack.sin(TAU * x[1]),
                     0.3 * tack.sin(PI * x[2]))


def volume(n=24):
    image = td.uniform_grid((n + 1,) * 3, spacing=(1.0 / n,) * 3, dtype=tack.f32)
    data = td.extract_cells(image, np.arange(image.num_cells))
    data.fields["v"] = td.interpolate(data, flow, td.H1(data, order=1), dtype=tack.vec3())
    return data


def slice_triangles(data, origin=(0.5, 0.5, 0.5), normal=(0.3, 0.4, 1.0), vector="v",
                    scalar=None):
    """The cut as arrays: points ``(m, 3)``, triangles ``(k, 3)``, the vector field at
    the points ``(m, 3)`` (all f32 / u32), the plane's unit normal, and the scalar field
    at the points (``(m,)``, or None)."""
    fields = [vector] + ([scalar] if scalar else [])
    cut = td.slice_plane(data, origin, normal, fields=fields)
    from tack.interop.vtk import dataset_to_vtk

    grid = dataset_to_vtk(cut)
    points = np.asarray(grid.GetPoints().GetData()).astype(np.float32)
    cells = grid.GetCells()
    conn = np.asarray(cells.GetConnectivityArray()).astype(np.uint32)
    offsets = np.asarray(cells.GetOffsetsArray())
    if not np.all(np.diff(offsets) == 3):
        raise ValueError("the cut is not all triangles")
    vectors = np.asarray(grid.GetPointData().GetArray(vector)).astype(np.float32)
    values = (np.asarray(grid.GetPointData().GetArray(scalar)).astype(np.float32).reshape(-1)
              if scalar else None)
    unit = np.asarray(normal, float) / np.linalg.norm(normal)
    return points, conn.reshape(-1, 3), vectors, unit.astype(np.float32), values
