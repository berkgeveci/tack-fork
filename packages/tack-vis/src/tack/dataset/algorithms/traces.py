# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``traces``: ``field`` -- any field with a basis on the cells: ``H1``, ``L2`` (an order per cell too), ``Constant`` -- evaluated by each 3D cell at each of its faces' points: a ``SideTraces`` field. Each cell writes its own side, at the face's points in the face's own order (``face_position``), so the two sides of a face line up point by point. Boundary faces leave side 1 zero."""

import tack
from tack import arrays
from tack.dataset.algorithms._common import _field
from tack.dataset.dataset import Field
from tack.dataset.spaces import SideTraces, reference_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['traces']


@tack.kernel
def _traces(cells, u, out):
    for c in cells:
        for f in range(cells.NUM_FACES):
            first = (cells.face_id(c, f) * 2 + cells.face_side(c, f)) * 4
            for k in range(cells.face_num_points(f)):
                pc = cells.parametric_point(cells.face_corner(f, k))
                out[first + cells.face_position(c, f, k)] = u.value(c, pc)


@on_inputs_device
def traces(data, field):
    """``field`` -- any field with a basis on the cells: ``H1``, ``L2`` (an order per
    cell too), ``Constant`` -- evaluated by each 3D cell at each of its faces'
    points: a ``SideTraces`` field. Each cell writes its own side, at the face's
    points in the face's own order (``face_position``), so the two sides of a
    face line up point by point. Boundary faces leave side 1 zero."""
    field = _field(data, field)
    space = field.space
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"traces are taken from a field with a basis on the cells, "
                        f"not {space!r}")
    topology = data.topology
    if topology.mixed:
        if any(g.count and not g.shape.REFERENCE for g in topology.faces().groups()):
            raise NotImplementedError("traces of a mesh whose polyhedra have faces of more "
                                      "than four points, which SideTraces does not hold")
    else:
        reference_cells(topology, "traces")
    topology.faces()
    out_space = SideTraces(data)
    width = arrays.width_of(field.values)
    out = (tack.empty((out_space.size,), tack.vec(width, arrays.dtype_of(field.values)))
           if width else tack.empty((out_space.size,), arrays.dtype_of(field.values)))
    if out_space.size:
        out.fill(0)
    starts = topology.faces().group_starts
    for group in data.launch_groups("cells", [field]):
        if not group.count:
            continue
        if not group.shape.REFERENCE:
            # Polyhedra beside the shapes: a point field's values at the face's
            # points, in the face's row (its stored walk); a cell field's own.
            on_points = space.on == "points"
            _polyhedral_traces(group.view(), arrays.materialize(field.values),
                               1 if on_points else 0, out)
        elif group.root in starts:
            _traces(data.domain_view("cells", group, incidence="faces"), field.view(group),
                    out)
    return Field(out_space, out, field.unit)


@tack.kernel
def _polyhedral_traces(cells, values, on_points, out):
    for c in cells:
        e = cells.entity_id(c)
        for k in range(cells.num_faces(c)):
            f = cells.face_id(c, k)
            first = (f * 2 + cells.face_side(c, k)) * 4
            begin = cells.face_offsets[f]
            for j in range(cells.face_offsets[f + 1] - begin):
                if on_points == 1:
                    out[first + j] = values[cells.face_points[begin + j]]
                else:
                    out[first + j] = values[e]
