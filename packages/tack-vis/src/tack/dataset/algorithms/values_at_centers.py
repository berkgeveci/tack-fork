# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``values_at_centers``: ``field`` evaluated at each cell's parametric center, through its basis, or on a cell with no reference element at the mean of its points."""

import tack
from tack.arrays import materialize
from tack.dataset.algorithms._common import _field, _with_basis
from tack.dataset.carry import _like
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import H1, Constant, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['values_at_centers']


@tack.kernel
def _at_centers(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = u.value(c, cells.parametric_center())


@on_inputs_device
def values_at_centers(data, field):
    """``field`` evaluated at each cell's parametric center, through its basis.

    On a polyhedron, polygon or polyline, which have no reference element, the
    center is the cell point (see ``cell_centers``) and point data (``H1``)
    takes its value there, the mean of the cell's distinct points' values: the
    value of the cell's interpolant, linear on tetrahedra fanned from that
    point. Cell data (``Constant``) is its own value."""
    field = _with_basis(data, _field(data, field))
    out = _like(field.values, data.num_cells)
    if getattr(data.topology, "reference_cells", True):
        for_each(_at_centers, data, "cells", field, out)
        return Field(Values(data, "cells"), out, field.unit)
    if not ((isinstance(field.space, H1) and field.space.order == 1)
            or isinstance(field.space, Constant)):
        raise NotImplementedError(f"values_at_centers of {field.space!r} on polyhedra, "
                                  "polygons or polylines: only point data (H1, order 1) and "
                                  "cell data (Constant) have values there")
    from tack.dataset.algorithms.project import _distinct_point_means

    values = materialize(field.values)
    if isinstance(field.space, Constant) and not data.topology.mixed:
        tack.copyto(out, values)
    for group in data.launch_groups("cells", [field]):
        if not group.count:
            continue
        view = data.domain_view("cells", group)
        if group.shape.REFERENCE:
            # Polyhedra beside the shapes: each shape group through its basis.
            _at_centers(view, field.view(group), out)
        elif isinstance(field.space, H1):
            _distinct_point_means(view, values, out, float("nan"))
        elif data.topology.mixed:
            _own_values(view, values, out)
    return Field(Values(data, "cells"), out, field.unit)


@tack.kernel
def _own_values(cells, values, out):
    for c in cells:
        e = cells.entity_id(c)
        out[e] = values[e]
