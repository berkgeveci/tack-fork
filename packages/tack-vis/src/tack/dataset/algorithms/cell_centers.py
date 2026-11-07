# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``cell_centers``: each cell's center, where the geometry maps its parametric center, or on a cell with no reference element the mean of its points: a field on cells."""

import tack
from tack.arrays import materialize
from tack.dataset.algorithms._common import _vectors
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['cell_centers']


@tack.kernel
def _centers(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.position(c, cells.parametric_center())


@on_inputs_device
def cell_centers(data):
    """Each cell's center, where the geometry maps its parametric center: a field on cells.
    That is ``values_at_centers(data, data.geometry)``, and not the centroid on a
    cell that is not affine; ``cell_geometry(data).centroid`` is the centroid, on
    every topology, polyhedra included.

    A polyhedron, polygon or polyline has no parametric center: its center is its
    *cell point*, the mean of its distinct points, where its interpolation fans
    its faces from. That is not the centroid either, unless the points are
    spread evenly."""
    out = _vectors(data.num_cells, data.dtype)
    if getattr(data.topology, "reference_cells", True):
        for_each(_centers, data, "cells", out)
        return Field(Values(data, "cells"), out, data.unit)
    from tack.dataset.algorithms.project import _distinct_point_means

    positions = materialize(data.geometry.values)
    for group in data.launch_groups("cells"):
        if group.count:
            view = data.domain_view("cells", group)
            if group.shape.REFERENCE:          # a shape beside polyhedra
                _centers(view, out)
            else:
                _distinct_point_means(view, positions, out, float("nan"))
    return Field(Values(data, "cells"), out, data.unit)
