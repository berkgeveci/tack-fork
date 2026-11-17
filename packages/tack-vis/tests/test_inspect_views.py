# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``tack.inspect_template`` on dataset views: each view says how ``view_class``
composed it, and every method of it names the one class that defines it
(``docs/design/plans/inspect-template.md``, D5)."""

import numpy as np
import pytest

import tack
import tack.arrays as ta
import tack.dataset as td
from tack.dataset import shapes
from tack.dataset.carry import _node_entries
from tack.dataset.topology import uniform_cells
from tack.lang.template_rewrite import template_func_owners

pytestmark = pytest.mark.quick


def _hexes():
    return td.extract_cells(td.uniform_grid((3, 3, 3)), np.arange(8))


def _wedges():
    hexes = _hexes()
    conn = []
    for h in hexes.topology.connectivity.numpy().reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    return td.DataSet(td.UnstructuredTopology(
        types, offsets, tack.array(np.array(conn, np.int32).reshape(-1)),
        num_points=hexes.num_points), hexes.geometry.values)


def _mixed():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
                       [1, 1, 1], [0, 1, 1], [2, 0, 0], [2, 0, 1]], np.float32)
    topology = td.UnstructuredTopology(
        types=np.array([shapes.HEXAHEDRON, shapes.WEDGE], np.uint8),
        offsets=np.array([0, 8, 14]),
        connectivity=np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 8, 2, 5, 9, 6]), num_points=10)
    return td.DataSet(topology, points)


def _cube_faces():
    return [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def _polyhedra():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
    return td.DataSet(td.PolyhedralTopology.from_cell_faces([_cube_faces()], 8), points)


def _polygons():
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    points = np.stack([np.cos(angles), np.sin(angles), 0 * angles], 1).astype(np.float32)
    return td.DataSet(td.PolygonalTopology([0, 6], np.arange(6)), points)


def _polylines():
    return td.DataSet(td.PolylineTopology([0, 3], [0, 1, 2]),
                      np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], np.float32))


def _table_of(data, space):
    """The element-to-DOF table numbering ``space``'s values as it numbers them itself."""
    index, _, _, total = _node_entries(space, tack.zeros((space.size,), tack.f64))
    return index.numpy()[:total].reshape(data.num_cells, -1)


# (name, dataset, space or values for the field, cell set or None)
_CASES = {
    "image, H1": (lambda: td.uniform_grid((3, 3, 3)), td.H1, None),
    "unstructured hexahedra, H1(3)": (_hexes, lambda d: td.H1(d, order=3), None),
    "unstructured hexahedra, H1(3) through a table": (
        _hexes, lambda d: td.H1(d, order=3, dofs=_table_of(d, td.H1(d, order=3))), None),
    "unstructured hexahedra, serendipity": (
        _hexes, lambda d: td.H1(d, order=2, basis="serendipity"), None),
    "tetrahedra, H1(2)": (lambda: td.tetrahedralize(td.uniform_grid((3, 3, 3))),
                          lambda d: td.H1(d, order=2), None),
    "triangles, L2(2)": (lambda: td.triangulate(td.uniform_grid((4, 4))),
                         lambda d: td.L2(d, order=2), None),
    "wedges, H1(2, bernstein)": (_wedges, lambda d: td.H1(d, order=2, basis="bernstein"),
                                 None),
    "quads, L2(3, cells)": (lambda: td.uniform_grid((4, 4)),
                            lambda d: td.L2(d, order=3, layout="cells"), None),
    "mixed shapes, H1": (_mixed, td.H1, None),
    "selected cells": (_hexes, td.H1, [0, 3]),
    "varying order": (lambda: td.uniform_grid((4, 3)),
                      lambda d: td.L2(d, order=np.array([1, 0, 1, 0, 1, 1])), None),
    "polyhedra": (_polyhedra, td.H1, None),
    "selected polyhedra": (_polyhedra, td.H1, [0]),
    "polygons": (_polygons, td.H1, None),
    "polylines": (_polylines, td.H1, None),
}


def _views(case):
    make, space, members = _CASES[case]
    data = make()
    values = space(data)
    field = td.Field(values, tack.zeros((values.size,)))
    data.fields["u"] = field
    domain = "cells"
    if members is not None:
        data.sets["some"] = td.Set(data, "cells", members)
        domain = "some"
    group = next(g for g in data.launch_groups(domain, [field]) if g.count)
    return data.domain_view(domain, group), field.view(group)


def _check(view):
    cls = type(view)
    kind, shape, mixins = vars(cls)["_tack_composition"]
    text = tack.inspect_template(view)
    assert (f"  view: {kind.__qualname__} of {shape.__qualname__}, parts "
            + ", ".join(m.__qualname__ for m in mixins)) in text
    for name, (_, owner, static) in template_func_owners(cls).items():
        first = tack.inspect_template(view, name).splitlines()[0]
        assert first == f"{name}: defined by {owner.__qualname__}" + (" (static)" if static
                                                                         else "")


@pytest.mark.parametrize("case", list(_CASES))
def test_each_view_says_its_composition_and_where_each_method_is(case):
    for view in _views(case):
        _check(view)


def test_faces_edges_and_field_storage_views():
    data = td.uniform_grid((3, 3))
    implicit = td.Field(td.H1(data), ta.CountingArray(data.num_points, dtype=tack.f32))
    soa = td.Field(td.H1(data), tack.zeros((data.num_points,), tack.vec3(tack.f32),
                                             layout="soa"))
    for field in (implicit, soa):
        group = next(g for g in data.launch_groups("cells", [field]) if g.count)
        _check(field.view(group))
    for domain in ("faces", "edges"):
        group = next(g for g in data.launch_groups(domain) if g.count)
        _check(data.domain_view(domain, group))
    text = tack.inspect_template(soa.view(group := next(
        g for g in data.launch_groups("cells", [soa]) if g.count)))
    assert ", soa, cpu" in text


def test_a_field_views_corner_value_names_the_dof_it_reads():
    data = _hexes()
    data.fields["T"] = td.Field(td.H1(data), tack.zeros((data.num_points,)))
    temperature = data.fields["T"]
    group = next(g for g in data.launch_groups("cells", [temperature]) if g.count)
    text = tack.inspect_template(temperature.view(group), "corner_value")
    assert text.splitlines()[0] == "corner_value: defined by _Interpolated"
    assert "tack/dataset/views.py:" in text.splitlines()[1]
    assert "self.dof(i, j)  -> _H1Field.dof  tack/dataset/views.py:" in text


def test_the_composition_is_on_the_class_not_on_views():
    data = _hexes()
    view = next(data.domain_view("cells", g) for g in data.launch_groups("cells") if g.count)
    assert "_tack_composition" not in vars(view)
    assert "_tack_composition" not in tack.inspect_template(view).split("compiled in:")[1]
