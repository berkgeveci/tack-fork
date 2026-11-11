# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fides' finite-element IR as Tack datasets, sharing its buffers: Python and Tack,
no VTK (the vault's "Fides finite-element IR" proposal; a prototype).

The IR is a Conduit tree, ``items/[i]/partitions/[j]/...``: a partition's points
(``coordinate_system/values``), its cells in blocks of one shape
(``cell_set/type: "blocks"``, ``cell_set/blocks/[b]/{shape, connectivity, ghosts}``),
optional ``ghosts/points``, and fields, each in a space
(``fields/[k]/space/{family, order, basis}``) with its point-held ``values`` or its
per-block ``blocks/[b]/{values, orders, dof_table}``.

``to_datasets(ir)`` gives one ``PartitionedDataSet`` per item. Every array is the
IR's own memory, read-only, kept alive by the Tack array that wraps it: the blocks'
connectivity is the topology's rows (``UnstructuredTopology.from_blocks``), a
field's per-block values its values (``tack.dataset.blocks``). What is copied:
an order per cell (one byte a cell), cells' ghost marks, and an ``H1`` field's dof
tables (Tack keeps one table of one node count)."""

import numpy as np

import tack
from tack import arrays

__all__ = ["to_dataset", "to_datasets"]

_SCALARS = {np.dtype(np.float32): tack.f32, np.dtype(np.float64): tack.f64,
            np.dtype(np.int32): tack.i32, np.dtype(np.int64): tack.i64,
            np.dtype(np.uint8): tack.u8}

_SHAPES = {"line": "Line", "triangle": "Triangle", "quad": "Quad", "tetrahedron": "Tetra",
           "hexahedron": "Hexahedron", "wedge": "Wedge", "pyramid": "Pyramid"}


class _Tree:
    """A partition's node, read as Tack needs it: values, strings, children, and
    arrays as NumPy views of the IR's memory, each keeping ``owner`` (the IR) alive.
    ``fides.Data`` serves arrays through its ``array`` (each owning its buffer)."""

    def __init__(self, node, owner, path=""):
        self.node, self.owner, self.path = node, owner, path

    def has(self, key):
        return self.node.has_path(key)

    def __getitem__(self, key):
        return _Tree(self.node.fetch(key), self.owner, f"{self.path}/{key}".lstrip("/"))

    def value(self, key, default=None):
        if not self.has(key):
            return default
        v = self.node[key]
        if isinstance(v, np.ndarray):
            return v.reshape(-1)[0].item() if v.size == 1 else v
        return v

    def children(self, key):
        node = self.node.fetch(key)
        return [_Tree(node.child(i), self.owner, f"{self.path}/{key}/{i}".lstrip("/"))
                for i in range(node.number_of_children())]

    def array(self, key):
        """The leaf at ``key`` as a NumPy array of its tuples (``key_num_components``)."""
        full = f"{self.path}/{key}".lstrip("/")
        if hasattr(self.owner, "array"):
            values = np.asarray(self.owner.array(full))       # owns its buffer (fides.Data)
        else:
            values = np.asarray(self.node[key])               # a view: the node keeps it
        width = int(self.value(key + "_num_components", 1) or 1)
        return values.reshape(-1, width) if width > 1 else values.reshape(-1)


def _wrapped(values, owner, element=None):
    """``values`` (a contiguous NumPy array) as a read-only Tack array on the CPU,
    sharing its memory: ``element`` a vector type for rows of components."""
    values = np.ascontiguousarray(values)          # a no-op for the IR's own buffers
    if element is None:
        element = _SCALARS[values.dtype]
    count = values.shape[0] if values.ndim else 1
    if count == 0:
        return tack.empty((0,), element, device="cpu")
    out = tack.array_from_ptr(values.ctypes.data, (count,), element, device="cpu")
    out._fides_owner = (values, owner)             # the memory lives as long as the array
    return out


def _vector(values, owner):
    """Rows of components (``(n, k)``) as a Tack array of k-vectors, or scalars."""
    dtype = {np.dtype(np.float32): tack.f32, np.dtype(np.float64): tack.f64}.get(values.dtype)
    if values.ndim == 2 and values.shape[1] > 1:
        if dtype is None:
            raise TypeError(f"a vector field of {values.dtype} values")
        return _wrapped(values, owner, tack.vec(values.shape[1], dtype))
    return _wrapped(values.reshape(-1), owner)


def to_datasets(ir):
    """Every item of the IR (``fides.Data``, or a ``conduit.Node`` in its layout) as a
    ``PartitionedDataSet`` of its partitions."""
    from tack.dataset.partitioned import PartitionedDataSet

    node = getattr(ir, "node", ir)
    root = _Tree(node, ir)
    out = []
    for item in root.children("items"):
        out.append(PartitionedDataSet([to_dataset(p) for p in item.children("partitions")]))
    return out


def to_dataset(partition):
    """One partition (a ``_Tree``, or a ``conduit.Node`` of one) as a Tack ``DataSet``."""
    import tack.dataset as td
    from tack.dataset import shapes
    from tack.dataset.blocks import block_values
    from tack.dataset.ghosts import Ghosts

    if not isinstance(partition, _Tree):
        partition = _Tree(partition, partition)
    owner = partition.owner
    if partition.value("cell_set/type") != "blocks":
        raise NotImplementedError("tack.interop.fides reads partitions of cell blocks "
                                  f"(cell_set/type 'blocks'), not {partition.value('cell_set/type')!r}")
    points = partition.array("coordinate_system/values")
    positions = _vector(points.reshape(-1, 3), owner)
    blocks, cells = [], []
    first = 0
    for block in partition.children("cell_set/blocks"):
        name = block.value("shape")
        if name not in _SHAPES:
            raise NotImplementedError(f"a block of {name!r} cells")
        shape = getattr(shapes, _SHAPES[name])
        rows = block.array("connectivity").reshape(-1)
        blocks.append((shape, _wrapped(rows, owner)))
        count = rows.size // shape.NUM_POINTS
        cells.append((first, count))
        first += count
    topology = td.UnstructuredTopology.from_blocks(blocks, num_points=len(points))
    data = td.DataSet(topology, positions, dtype=arrays.dtype_of(positions))
    marks = {}
    if partition.has("ghosts/points"):
        marks["points"] = _wrapped(partition.array("ghosts/points"), owner)
    by_block = [b.array("ghosts") if b.has("ghosts") else np.zeros(n, np.uint8)
                for b, (_, n) in zip(partition.children("cell_set/blocks"), cells)]
    if any(b.has("ghosts") for b in partition.children("cell_set/blocks")):
        marks["cells"] = tack.array(np.concatenate(by_block).astype(np.uint8))   # copied
    if marks:
        data.ghosts = Ghosts(**marks)
    if partition.has("geometry"):
        data.geometry = _field(data, partition["geometry"], cells, owner, block_values)
    if partition.has("fields"):
        for field in partition.children("fields"):
            data.fields[field.value("name")] = _field(data, field, cells, owner, block_values)
    return data


def _field(data, field, cells, owner, block_values):
    """One IR field (or the geometry) as a Tack ``Field`` of ``data``."""
    import tack.dataset as td

    family = field.value("space/family")
    order = field.value("space/order", 1)
    basis = field.value("space/basis", "gauss_lobatto")
    unit = field.value("units")
    if basis == "nodal":
        raise NotImplementedError("a nodal basis given by its nodes is resampled into one of "
                                  "Tack's: not in this prototype yet")
    blocks = field.children("blocks") if field.has("blocks") else []

    def per_block(key):
        parts = [_vector(b.array(key), owner) for b in blocks]
        return block_values(parts, cells)

    if family == "points":
        return td.Field(td.Values(data, "points"), _vector(field.array("values"), owner), unit)
    if family in ("constant", "cells"):
        space = td.Constant(data) if family == "constant" else td.Values(data, "cells")
        return td.Field(space, per_block("values"), unit)
    if family == "H1":
        values = _vector(field.array("values"), owner)
        if int(order) == 1:
            return td.Field(td.H1(data), values, unit)
        tables = [b.array("dof_table").reshape(-1) for b in blocks]
        table = np.concatenate(tables) if tables else np.zeros(0, np.int64)   # copied
        space = td.H1(data, order=int(order), basis=basis, dofs=table)
        return td.Field(space, values, unit)
    if family == "L2":
        if order == "per_cell":
            orders = np.concatenate([b.array("orders").reshape(-1) for b in blocks])  # copied
            space = td.L2(data, order=orders.astype(np.int32), basis=basis)
        else:
            space = td.L2(data, order=int(order), basis=basis, layout="blocks")
        values = per_block("values")
        if arrays.size_of(values) != space.size:
            raise ValueError(f"{field.value('name', 'the geometry')!r}: {space!r} holds "
                             f"{space.size} values, the blocks {arrays.size_of(values)}")
        return td.Field(space, values, unit)
    raise NotImplementedError(f"a field of space family {family!r}")

