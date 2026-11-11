# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``tack.interop.fides``: Fides' finite-element IR (cells in blocks of one shape,
fields in spaces) as Tack datasets, sharing the IR's buffers. The IR is built here
as a Conduit node in its layout; no VTK."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes

conduit = pytest.importorskip("conduit")

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def _cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


@tack.func
def _cubic(x):
    return 1.0 + x[0] * x[0] * x[1] - 0.5 * x[2] * x[2] * x[2] + x[0] * x[1] * x[2]


def _blocks():
    """A 2 x 2 x 2 cube of hexahedra: four kept, two cut into tetrahedra, two into
    wedges; the points and each block's rows, and the same mesh mixed."""
    grid = td.uniform_grid((3, 3, 3), spacing=(0.5, 0.5, 0.5), dtype=tack.f64)
    hexes = td.extract_cells(grid, np.arange(grid.num_cells))
    rows = hexes.topology.connectivity.numpy().reshape(-1, 8)
    points = hexes.positions()
    # Hexahedra 4 and 5 as tetrahedra on the same points (extract_cells would renumber them).
    every = td.tetrahedralize(hexes).topology.connectivity.numpy().reshape(-1, 4)
    centres = points[every].mean(axis=1)
    low, high = points[rows[4]].min(axis=0), points[rows[5]].max(axis=0)
    tets = every[np.all((centres > low) & (centres < high), axis=1)]
    wedges = np.concatenate([[h[[0, 1, 3, 4, 5, 7]], h[[1, 2, 3, 5, 6, 7]]] for h in rows[6:]])
    blocks = [("hexahedron", rows[:4]), ("tetrahedron", tets), ("wedge", wedges)]
    types = np.concatenate([np.full(len(r), {"hexahedron": 12, "tetrahedron": 10,
                                             "wedge": 13}[s], np.uint8) for s, r in blocks])
    sizes = np.concatenate([np.full(len(r), r.shape[1]) for _, r in blocks])
    mixed = td.DataSet(td.UnstructuredTopology(
        types, np.r_[0, np.cumsum(sizes)], np.concatenate([r.ravel() for _, r in blocks]),
        num_points=len(points)), points, dtype=tack.f64)
    return points, blocks, mixed


class _IR:
    """A Conduit node in the IR's layout and the arrays its leaves point into, which
    it keeps alive (Fides' IR keeps its buffers' owners)."""

    def __init__(self):
        self.node = conduit.Node()
        self.keep = []

    def external(self, leaf, values):
        values = np.ascontiguousarray(values).reshape(-1)
        self.keep.append(values)
        leaf.set_external(values)


def _ir(points, blocks, fields):
    """The IR of one item and one partition. ``fields`` is a list of ``(name, space,
    units, point values or None, per-block dict of arrays)``."""
    ir = _IR()
    node = ir.node
    p = node["items"].append()["partitions"].append()
    p["coordinate_system/type"] = "array"
    ir.external(p["coordinate_system/values"], points)
    p["coordinate_system/values_num_components"] = np.int32(3)
    p["cell_set/type"] = "blocks"
    for shape, rows in blocks:
        b = p["cell_set/blocks"].append()
        b["shape"] = shape
        ir.external(b["connectivity"], rows)
    for name, space, units, values, per_block in fields:
        f = p["fields"].append()
        f["name"] = name
        for key, value in space.items():
            f[f"space/{key}"] = value if isinstance(value, str) else np.int32(value)
        if units:
            f["units"] = units
        if values is not None:
            ir.external(f["values"], values)
            if values.ndim == 2:
                f["values_num_components"] = np.int32(values.shape[1])
        for b in range(len(blocks) if per_block else 0):
            block = f["blocks"].append()
            for key, arrays in per_block.items():
                ir.external(block[key], arrays[b])
    return ir


def _split(values, space, blocks):
    """A field's values on the mixed twin, cell values block by block."""
    offsets = space.offsets.numpy() if hasattr(space, "offsets") else np.arange(len(values) + 1)
    out, first = [], 0
    for _, rows in blocks:
        out.append(np.ascontiguousarray(values[offsets[first]:offsets[first + len(rows)]]))
        first += len(rows)
    return out


def test_the_cubes_blocks_fill_it():
    points, blocks, mixed = _blocks()
    np.testing.assert_allclose(td.integrate(mixed).measure, 1.0, rtol=1e-14)
    assert len(blocks[1][1]) == 12                    # two hexahedra's tetrahedra


def test_blocks_and_fields_come_as_they_are():
    """The points, every block's rows and every block's values are the IR's memory;
    the cells are the blocks' in order, each block a shape group; an L2 field of an
    order per cell, a cell field and a point field read as the twin built mixed."""
    from tack.interop.fides import to_datasets

    points, blocks, mixed = _blocks()
    rng = np.random.default_rng(0)
    orders = np.concatenate([rng.integers(2, 5, len(blocks[0][1])),
                             np.full(len(blocks[1][1]), 3), np.full(len(blocks[2][1]), 2)])
    twin = td.L2(mixed, order=orders, basis="equispaced")
    w = td.interpolate(mixed, _cubic, twin, dtype=tack.f64).values.numpy()
    c = np.arange(mixed.num_cells, dtype=np.float64)
    v = np.ascontiguousarray(points * 2.0)
    first = np.cumsum([0] + [len(r) for _, r in blocks])
    order_blocks = [orders[first[b]:first[b + 1]].astype(np.uint8) for b in range(len(blocks))]
    node = _ir(points, blocks, [
        ("w", {"family": "L2", "order": "per_cell", "basis": "equispaced"}, "K", None,
         {"values": _split(w, twin, blocks), "orders": order_blocks}),
        ("c", {"family": "constant"}, None, None,
         {"values": [c[first[b]:first[b + 1]] for b in range(len(blocks))]}),
        ("v", {"family": "points"}, "m/s", v, None)])
    (pieces,) = to_datasets(node)
    data = pieces[0]
    groups = data.topology.groups()
    assert [(g.shape, g.start, g.count) for g in groups] == [
        (shapes.Hexahedron, 0, 4), (shapes.Tetra, 4, len(blocks[1][1])),
        (shapes.Wedge, first[2], 4)]
    held = node.node["items"][0]["partitions"][0]
    for b, g in enumerate(groups):
        rows = held["cell_set/blocks"][b]["connectivity"]
        assert g.args[0]._buffer.address == rows.ctypes.data          # the rows, as given
    assert data.geometry.values._buffer.address == held["coordinate_system/values"].ctypes.data
    parts = data.fields["w"].values._block_parts[0]
    assert parts[1]._buffer.address == held["fields"][0]["blocks"][1]["values"].ctypes.data
    assert data.fields["w"].unit == "K" and data.fields["v"].unit == "m/s"
    mixed.fields["w"] = td.Field(twin, tack.array(w))
    np.testing.assert_allclose(td.integrate(data, fields=["w"]).values["w"],
                               td.integrate(mixed, fields=["w"]).values["w"], rtol=1e-14)
    np.testing.assert_array_equal(td.values_at_centers(data, "c").values.numpy(), c)
    pts = np.random.default_rng(5).random((200, 3)) * 0.98 + 0.01
    np.testing.assert_allclose(td.probe(data, pts).fields["w"].values.numpy(),
                               td.probe(mixed, pts).fields["w"].values.numpy(), rtol=0,
                               atol=1e-13)
    assert td.contour(data, "w", [1.1]).num_cells == td.contour(mixed, "w", [1.1]).num_cells
    assert td.extract_surface(data).num_cells == td.extract_surface(mixed).num_cells


def test_an_h1_field_by_its_dof_tables():
    """An H1 field above order 1 comes as its values (the points' first) and each
    block's dof table: read as H1(dofs=table), it is the field (numbered by entity
    on the twin)."""
    from tack.dataset.carry import _node_entries
    from tack.interop.fides import to_datasets

    grid = td.uniform_grid((3, 3, 3), spacing=(0.5, 0.5, 0.5), dtype=tack.f64)
    hexes = td.extract_cells(grid, np.arange(grid.num_cells))
    rows = hexes.topology.connectivity.numpy().reshape(-1, 8)
    blocks = [("hexahedron", rows[:3]), ("hexahedron", rows[3:])]
    entity = td.H1(hexes, order=2, basis="equispaced")
    u = td.interpolate(hexes, _cubic, entity, dtype=tack.f64).values.numpy()
    index, entry, _, total = _node_entries(entity, tack.array(u))
    table = index.numpy()[:total].reshape(-1, 27)
    node = _ir(hexes.positions(), blocks, [
        ("u", {"family": "H1", "order": 2, "basis": "equispaced"}, None, u,
         {"dof_table": [np.ascontiguousarray(table[:3]), np.ascontiguousarray(table[3:])]})])
    data = to_datasets(node)[0][0]
    field = data.fields["u"]
    assert field.space.dofs is not None and field.space.order == 2
    pts = np.random.default_rng(1).random((100, 3)) * 0.98 + 0.01
    hexes.fields["u"] = td.Field(entity, tack.array(u))
    np.testing.assert_allclose(td.probe(data, pts).fields["u"].values.numpy(),
                               td.probe(hexes, pts).fields["u"].values.numpy(), rtol=0,
                               atol=1e-13)


def test_what_the_prototype_refuses():
    from tack.interop.fides import to_dataset

    points, blocks, _ = _blocks()
    node = _ir(points, blocks, [("n", {"family": "L2", "order": 2, "basis": "nodal"}, None,
                                 None, {"values": [np.zeros(len(r) * 27) for _, r in blocks]})])
    with pytest.raises(NotImplementedError, match="nodal"):
        to_dataset(node.node["items"][0]["partitions"][0])


def test_from_a_producer_through_fides():
    """End to end: a producer's Conduit node in Fides' finite_element layout, read by
    fides.Reader into the IR and by to_datasets into Tack: the blocks' rows and the
    fields' blocks are the producer's memory, and the fields are the twin's."""
    import json

    fides = pytest.importorskip("fides")
    from tack.interop.fides import to_datasets

    points, blocks, mixed = _blocks()
    rng = np.random.default_rng(3)
    first = np.cumsum([0] + [len(r) for _, r in blocks])
    orders = np.concatenate([rng.integers(1, 5, len(blocks[0][1])),
                             np.full(len(blocks[1][1]), 3), np.full(len(blocks[2][1]), 2)])
    twin = td.L2(mixed, order=orders, basis="equispaced")
    w = td.interpolate(mixed, _cubic, twin, dtype=tack.f64).values.numpy()
    node, keep = conduit.Node(), []

    def variable(name, values):
        values = np.ascontiguousarray(values)
        keep.append(values)
        node.fetch(f"variables/{name}/shape").set(np.array(values.shape, np.int64))
        node.fetch(f"variables/{name}/data").set_external(values.reshape(-1))

    variable("points", points)
    node["finite_element/coordinates"] = "points"
    names = node["finite_element/blocks/names"]
    f = "finite_element/fields/w/"
    node[f + "space/family"] = "L2"
    node[f + "space/order"] = "per_cell"
    node[f + "space/basis"] = "equispaced"
    node[f + "units"] = "K"
    for b, ((shape, rows), part) in enumerate(zip(blocks, _split(w, twin, blocks),
                                                  strict=True)):
        name = f"b{b}"
        names.append().set(name)
        node[f"finite_element/blocks/{name}/shape"] = shape
        variable(f"{name}_rows", rows.astype(np.int64))
        node[f"finite_element/blocks/{name}/connectivity"] = f"{name}_rows"
        variable(f"w_{name}", part)
        variable(f"w_{name}_orders", orders[first[b]:first[b + 1]].astype(np.uint8))
        node[f + f"blocks/{name}/values"] = f"w_{name}"
        node[f + f"blocks/{name}/orders"] = f"w_{name}_orders"
    schema = {"finite_element": {"data_sources": [{"name": "source", "type": "conduit"}],
                                 "fields": [{"name": "w"}]}}
    reader = fides.Reader(json.dumps(schema), fides.InputType.JSONString)
    reader.set_data_source_node("source", node)
    data = to_datasets(reader.read())[0][0]
    for b, g in enumerate(data.topology.groups()):
        assert g.args[0]._buffer.address == keep[1 + 3 * b].ctypes.data   # the producer's rows
    assert data.fields["w"].values._block_parts[0][0]._buffer.address == keep[2].ctypes.data
    mixed.fields["w"] = td.Field(twin, tack.array(w))
    np.testing.assert_allclose(td.integrate(data, fields=["w"]).values["w"],
                               td.integrate(mixed, fields=["w"]).values["w"], rtol=1e-14)
    assert td.contour(data, "w", [1.1]).num_cells == td.contour(mixed, "w", [1.1]).num_cells
