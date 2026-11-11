# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A DG solver's data as a Conduit node in Fides' ``finite_element`` layout: what a
solver's Catalyst adaptor would hand Fides (the vault's Fides finite-element IR
proposal). NumPy and Conduit only: no Tack, no VTK.

The mesh is an image's cells, ``n`` a side over the unit cube: those with ``x <
1/2`` hexahedra, the others each cut into six tetrahedra on the same points, two
blocks. Straight geometry (the points). Fields:

- ``pressure``: DG, order 2 (equispaced), a scalar evaluated at each cell's nodes;
- ``velocity``: DG, order 2, a 3-vector (a swirl about the cube's axis);
- ``block``: one value a cell (the cell's block);
- ``temperature``: one value a point.

Each block's per-cell values are the block's own array, cell after cell, each
cell's nodes in Tack's local order: a hexahedron's tensor lattice, x fastest; a
tetrahedron's lattice ``i + j + k <= p``, ``i`` fastest."""

import conduit
import numpy as np

#: A hexahedron cut into six tetrahedra about its diagonal 0-6 (VTK corner order).
_SIX = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6],
                 [0, 5, 1, 6]])


def pressure(x):
    return np.sin(np.pi * x[..., 0]) * np.cos(np.pi * x[..., 1]) + x[..., 2] ** 2


def velocity(x):
    c = x - 0.5
    return np.stack([-c[..., 1], c[..., 0], 0.25 * np.sin(2 * np.pi * x[..., 2])], axis=-1)


def temperature(x):
    return 300.0 + 50.0 * x[..., 0] * x[..., 1]


def image(n):
    """The image's points, ``(n + 1)^3`` of them x fastest, and its cells' corners in
    VTK's order, ``(n^3, 8)``."""
    g = np.linspace(0.0, 1.0, n + 1)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def pid(i, j, k):
        return i + (n + 1) * (j + (n + 1) * k)

    cells = [[pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k),
              pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1),
              pid(i, j + 1, k + 1)]
             for k in range(n) for j in range(n) for i in range(n)]
    return points, np.array(cells, np.int64)


def hex_nodes(corners, p):
    """A straight hexahedron's order-``p`` equispaced nodes, ``((p + 1)^3, 3)``: the
    tensor lattice, x fastest, by the trilinear map of its corners."""
    t = np.linspace(0.0, 1.0, p + 1)
    u, s, r = (a.reshape(-1) for a in np.meshgrid(t, t, t, indexing="ij"))
    w = [(1 - r) * (1 - s) * (1 - u), r * (1 - s) * (1 - u), r * s * (1 - u),
         (1 - r) * s * (1 - u), (1 - r) * (1 - s) * u, r * (1 - s) * u, r * s * u,
         (1 - r) * s * u]
    return sum(wi[:, None] * corners[i] for i, wi in enumerate(w))


def tet_nodes(corners, p):
    """A tetrahedron's order-``p`` equispaced nodes: the lattice ``(i, j, k) / p``,
    ``i + j + k <= p``, ``i`` fastest, mapped from its corners."""
    out = []
    for k in range(p + 1):
        for j in range(p + 1 - k):
            for i in range(p + 1 - j - k):
                r, s, t = i / p, j / p, k / p
                out.append(corners[0] + r * (corners[1] - corners[0])
                           + s * (corners[2] - corners[0]) + t * (corners[3] - corners[0]))
    return np.array(out)


def produce(n=6, order=2):
    """The Conduit node and the arrays its leaves point into: a solver's own memory,
    set external, which must outlive the node."""
    points, cells = image(n)
    centres = points[cells].mean(axis=1)
    hexes = cells[centres[:, 0] < 0.5]
    tets = np.concatenate([h[_SIX] for h in cells[centres[:, 0] >= 0.5]])
    blocks = [("hexes", "hexahedron", hexes, hex_nodes),
              ("tets", "tetrahedron", tets, tet_nodes)]

    node, keep = conduit.Node(), []

    def variable(name, values):
        values = np.ascontiguousarray(values)
        keep.append(values)
        node.fetch(f"variables/{name}/shape").set(np.array(values.shape, np.int64))
        node.fetch(f"variables/{name}/data").set_external(values.reshape(-1))
        return name

    fe = "finite_element/"
    node[fe + "coordinates"] = variable("points", points)
    node[fe + "fields/temperature/space/family"] = "points"
    node[fe + "fields/temperature/units"] = "K"
    node[fe + "fields/temperature/values"] = variable("temperature", temperature(points))
    for name, family, unit in (("pressure", "L2", "Pa"), ("velocity", "L2", "m/s"),
                               ("block", "constant", None)):
        node[fe + f"fields/{name}/space/family"] = family
        if family == "L2":
            node[fe + f"fields/{name}/space/order"] = np.int32(order)
            node[fe + f"fields/{name}/space/basis"] = "equispaced"
        if unit:
            node[fe + f"fields/{name}/units"] = unit
    names = node[fe + "blocks/names"]
    for b, (name, shape, rows, nodes_of) in enumerate(blocks):
        names.append().set(name)
        node[fe + f"blocks/{name}/shape"] = shape
        node[fe + f"blocks/{name}/connectivity"] = variable(f"{name}_connectivity", rows)
        # What the solver holds: each cell's values at its nodes, cell after cell.
        at = np.concatenate([nodes_of(points[r], order) for r in rows])
        node[fe + f"fields/pressure/blocks/{name}/values"] = variable(
            f"pressure_{name}", pressure(at))
        node[fe + f"fields/velocity/blocks/{name}/values"] = variable(
            f"velocity_{name}", velocity(at))
        node[fe + f"fields/block/blocks/{name}/values"] = variable(
            f"block_{name}", np.full(len(rows), float(b)))
    return node, keep
