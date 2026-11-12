# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One curved cell with an order-5 DG field, drawn three ways, a hexahedron in the top
row and a tetrahedron in the bottom one:

1. as a linear renderer sees it: the corners alone (flat faces, the colour
   interpolated from the corner values);
2. high order: the order-2 geometry on 16 x 16 templates, the field evaluated per
   pixel;
3. the same on 4 x 4 templates with the template's triangles drawn (white): the
   field varies inside every triangle, since it is evaluated per pixel, not per
   vertex.

Writes ``single_cell.png`` to OUTDIR (wgpu-py on ``PYTHONPATH``)::

    .venv/bin/python packages/tack-vis/examples/webgpu_surfaces/single_cell.py OUTDIR
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faces import surface_records
from renderer import Renderer, write_png

import tack
import tack.dataset as td


@tack.func
def ripples(x):
    return (tack.sin(2.5 * 3.14159265 * x[0]) * tack.cos(2.0 * 3.14159265 * x[1])
            + 0.6 * tack.sin(3.0 * 3.14159265 * x[2]))


def warp(p):
    """A twist about the cell's vertical axis and a bulge at its middle height."""
    c = p - 0.5
    angle = 0.9 * c[:, 2]
    x = c[:, 0] * np.cos(angle) - c[:, 1] * np.sin(angle)
    y = c[:, 0] * np.sin(angle) + c[:, 1] * np.cos(angle)
    bulge = 1.0 + 1.2 * (0.25 - c[:, 2] ** 2)
    return np.c_[x * bulge, y * bulge, c[:, 2]] + 0.5


def cell(kind):
    """One cell of ``kind`` (``"hexahedron"``, ``"tetrahedron"``) on the unit cube's
    corners, its order-2 geometry warped and an order-5 field on it."""
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float64)
    if kind == "hexahedron":
        topology = td.UnstructuredTopology(np.array([12], np.uint8), np.array([0, 8]),
                                           np.arange(8))
        points = corners
    else:
        topology = td.UnstructuredTopology(np.array([10], np.uint8), np.array([0, 4]),
                                           np.arange(4))
        points = corners[[0, 1, 3, 4]]
    data = td.DataSet(topology, tack.array(points, tack.vec3(tack.f64)))
    space = td.H1(data, order=2)
    nodes = td.project(data, data.geometry, space).values.numpy().reshape(-1, 3)
    data.geometry = td.Field(space, tack.array(warp(nodes), tack.vec3(tack.f64)))
    data.fields["f"] = td.interpolate(data, ripples, td.L2(data, order=5))
    return data


def linear(data):
    """What a linear renderer has of ``data``: its corners, and the field's values
    there."""
    flat = td.DataSet(data.topology, td.project(data, data.geometry, td.H1(data)).values)
    flat.fields["f"] = td.project(data, data.fields["f"], td.L2(data, order=1))
    return flat


def main(out):
    tack.set_default_device("cpu")
    os.makedirs(out, exist_ok=True)
    renderer = Renderer()
    size = (640, 576)
    rows = []
    for kind in ("hexahedron", "tetrahedron"):
        data = cell(kind)
        high = surface_records(data, "f")
        flat = surface_records(linear(data), "f")
        centre = high.origin
        camera = {"eye": centre + np.array([2.2, -2.6, 1.9]), "target": centre, "size": size,
                  "edges": 1.5, "field_range": high.range}
        panels = [renderer.render(flat, **camera)[0],
                  renderer.render(high, curved_level=16, **camera)[0],
                  renderer.render(high, curved_level=4, tessellation=0.8, **camera)[0]]
        rows.append(np.concatenate(panels, axis=1))
        print(f"{kind}: {len(high.faces)} faces, field order 5 "
              f"({len(high.field) // len(high.faces)} coefficients a quad face"
              if kind == "hexahedron" else
              f"{kind}: {len(high.faces)} faces, field order 5 "
              f"({len(high.field) // len(high.faces)} coefficients a triangle face", end="")
        print(f"), geometry order 2; linear: {len(flat.field) // len(flat.faces)} values a face")
    path = os.path.join(out, "single_cell.png")
    write_png(path, np.concatenate(rows, axis=0))
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
