# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""End to end, Fides' finite-element IR: a DG solver's Conduit node (``producer``)
read by ``fides.Reader`` into the IR, the IR written out as YAML (structure in full,
arrays summarized), then Tack datasets sharing the IR's memory, a few filters, and
a picture rendered by VTK.

From the checkout, with the Fides prototype's and Conduit's Python modules and a
VTK build on ``PYTHONPATH``::

    PYTHONPATH=packages/tack-core/src:packages/tack-vis/src:\\
    /usr/local/scratch/builds/fides/fe-ir-prototype/python:\\
    /usr/local/scratch/installs/conduit/develop/python-modules:\\
    /usr/local/scratch/builds/VTK/tack-dev/lib/python3.13/site-packages \\
    .venv/bin/python packages/tack-vis/examples/fides_prototype/driver.py OUTDIR
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fides  # noqa: E402
from producer import pressure, produce  # noqa: E402

import tack  # noqa: E402
import tack.dataset as td  # noqa: E402
from tack.interop.fides import to_datasets  # noqa: E402


def describe(node, indent=0, lines=None):
    """A Conduit tree as YAML-ish lines: every key and string, numbers in full when
    few, an array as its type, shape and first values."""
    lines = [] if lines is None else lines
    pad = "  " * indent
    dtype = node.dtype()
    if dtype.is_list():
        for i in range(node.number_of_children()):
            child = node.child(i)
            if child.number_of_children():
                lines.append(f"{pad}- [{i}]")
                describe(child, indent + 1, lines)
            else:
                lines.append(f"{pad}- {_leaf(child)}")
        return lines
    for name in node.child_names():
        child = node.fetch(name)
        if child.number_of_children():
            lines.append(f"{pad}{name}:")
            describe(child, indent + 1, lines)
        else:
            lines.append(f"{pad}{name}: {_leaf(child)}")
    return lines


def _leaf(node):
    if node.dtype().is_string():
        return repr(node.value())
    values = np.asarray(node.value())
    if values.size <= 4:
        return values.tolist() if values.size > 1 else values.reshape(-1)[0].item()
    first = ", ".join(f"{v:.4g}" if values.dtype.kind == "f" else str(v)
                      for v in values.reshape(-1)[:6])
    return f"<{values.dtype}[{values.size}]> [{first}, ...]"


def render(path, surface, plane, outline):
    """The contour (grey) and the slice coloured by pressure, in the cube's outline:
    a PNG by VTK, offscreen."""
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkDataSetMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )

    from tack.interop.vtk import dataset_to_vtk

    renderer = vtkRenderer(background=(1.0, 1.0, 1.0))
    sliced = dataset_to_vtk(plane)
    sliced.point_data.SetActiveScalars("pressure")
    lo, hi = sliced.point_data["pressure"].GetRange()
    renderer.AddActor(vtkActor(mapper=vtkDataSetMapper(input_data=sliced, scalar_range=(lo, hi))))
    contour = vtkActor(mapper=vtkDataSetMapper(input_data=dataset_to_vtk(surface),
                                               scalar_visibility=False))
    contour.property.color = (0.75, 0.75, 0.78)
    contour.property.opacity = 0.6
    renderer.AddActor(contour)
    edges = vtkActor(mapper=vtkDataSetMapper(input_data=dataset_to_vtk(outline),
                                             scalar_visibility=False))
    edges.property.color = (0.2, 0.2, 0.2)
    renderer.AddActor(edges)
    camera = renderer.active_camera
    camera.position = (2.6, -1.8, 2.2)
    camera.focal_point = (0.5, 0.5, 0.5)
    camera.view_up = (0.0, 0.0, 1.0)
    renderer.ResetCameraClippingRange()
    window = vtkRenderWindow(size=(900, 700), off_screen_rendering=True)
    window.AddRenderer(renderer)
    window.Render()
    grab = vtkWindowToImageFilter(input=window)
    writer = vtkPNGWriter(file_name=path)
    writer.SetInputConnection(grab.output_port)
    writer.Write()


def main(out):
    tack.set_default_device("cpu")
    os.makedirs(out, exist_ok=True)

    # 1. The solver's node, and Fides' read of it (the schema names only the data model).
    node, keep = produce(n=6, order=2)
    schema = {"finite_element": {"data_sources": [{"name": "source", "type": "conduit"}]}}
    reader = fides.Reader(json.dumps(schema), fides.InputType.JSONString)
    reader.set_data_source_node("source", node)
    ir = reader.read()

    # 2. The IR, as Fides hands it to a consumer.
    text = "\n".join(describe(ir.node))
    with open(os.path.join(out, "ir.yaml"), "w") as f:
        f.write(text + "\n")
    print(text)

    # 3. Tack, sharing the IR's memory (which is the solver's).
    data = to_datasets(ir)[0][0]
    rows = [g.args[0] for g in data.topology.groups()]
    shared = [r._buffer.address == k.ctypes.data
              for r, k in zip(rows, [keep[2], keep[6]], strict=True)]
    print(f"\nTack: {data.num_points} points, {data.num_cells} cells in "
          f"{[(g.shape.__name__, g.count) for g in data.topology.groups()]}; "
          f"rows shared with the solver: {shared}")
    for name, field in data.fields.items():
        print(f"  {name}: {field.space!r}, {tack.arrays.size_of(field.values)} values"
              + (f", in {field.unit}" if field.unit else ""))

    # 4. Filters.
    total = td.integrate(data, fields=["pressure"]).values["pressure"]
    print(f"\n∫ pressure = {total:.6f} (the exact integral is 1/3 = {1 / 3:.6f})")
    points = np.random.default_rng(0).random((5, 3))
    probed = td.probe(data, points).fields["pressure"].values.numpy()
    print(f"probe: within {np.abs(probed - pressure(points)).max():.1e} of the solver's "
          "function (an order-2 field of a sine)")
    surface = td.contour(data, "pressure", [0.5])
    # A straight geometry's slice needs no subdivision, but a DG field goes onto the
    # slice's points only through it: 4 steps a cell, the field evaluated at each.
    plane = td.slice_plane(data, (0.5, 0.5, 0.5), (0.0, 0.0, 1.0), fields=["pressure"],
                           subdivision=4)
    parts = td.clip(data, "pressure", 0.5)
    print(f"contour at 0.5: {surface.num_cells} triangles; slice z = 0.5: "
          f"{plane.num_cells} triangles; clip above 0.5: "
          f"{[p.num_cells for p in parts]} cells (whole, cut)")
    outline = td.extract_edges(td.extract_surface(td.uniform_grid((2, 2, 2))))
    path = os.path.join(out, "pressure.png")
    render(path, surface, plane, outline)
    print(f"\nwrote {os.path.join(out, 'ir.yaml')} and {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
