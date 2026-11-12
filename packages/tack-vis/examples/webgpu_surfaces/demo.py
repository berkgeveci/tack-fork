# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""High-order surfaces drawn by WebGPU from Tack's face records (the vault's WebGPU
high-order surfaces proposal). Three scenes, each a PNG and a check:

1. ``fides``: the Fides prototype's DG solver data (hexahedra and tetrahedra,
   straight, DG p=2 pressure), read through Fides' finite-element IR;
2. ``curved``: a shell of hexahedra with order-2 geometry, a DG field of order 4;
3. ``adaptive``: an order per cell, 1 to 5, on hexahedra and tetrahedra, coloured by
   the field and by each face's order.

Each check reads, per pixel, the field, the face and (u, v) the shader used, and
compares the field with a NumPy evaluation of the same records (the WGSL) and with
Tack's own evaluation of the solid at that point (``probe``: the whole path).

From the checkout, with wgpu-py, and for the first scene the Fides prototype and
Conduit, on ``PYTHONPATH``::

    .venv/bin/python packages/tack-vis/examples/webgpu_surfaces/demo.py OUTDIR
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "fides_prototype"))

from faces import evaluate, surface_records
from renderer import Renderer, write_png

import tack
import tack.dataset as td


@tack.func
def wave(x):
    return tack.sin(3.0 * x[0]) * tack.cos(2.0 * x[1]) + 0.5 * x[2]


def fides_scene():
    import fides
    from producer import produce

    from tack.interop.fides import to_datasets

    node, keep = produce(n=6, order=2)
    schema = {"finite_element": {"data_sources": [{"name": "source", "type": "conduit"}]}}
    reader = fides.Reader(json.dumps(schema), fides.InputType.JSONString)
    reader.set_data_source_node("source", node)
    data = to_datasets(reader.read())[0][0]
    data._fides_keep = keep                       # the solver's memory, as long as the data
    return data, "pressure", {"eye": (2.6, -1.8, 2.2), "target": (0.5, 0.5, 0.5)}


def curved_scene(n=4):
    """A quarter of a thick cylinder: an image's hexahedra bent around the z axis, the
    bend carried by an order-2 geometry."""
    image = td.uniform_grid((n + 1, 2 * n + 1, n + 1), spacing=(1 / n, 1 / (2 * n), 1 / n),
                            dtype=tack.f64)
    data = td.extract_cells(image, np.arange(image.num_cells))
    space = td.H1(data, order=2)
    x = td.project(data, data.geometry, space).values.numpy().reshape(-1, 3)
    r, theta = 1.0 + 0.6 * x[:, 0], 0.5 * np.pi * x[:, 1]
    bent = np.c_[r * np.cos(theta), r * np.sin(theta), 0.8 * x[:, 2]]
    data.geometry = td.Field(space, tack.array(bent, tack.vec3(tack.f64)))
    data.fields["wave"] = td.interpolate(data, wave, td.L2(data, order=4))
    return data, "wave", {"eye": (3.4, 2.6, 2.4), "target": (0.7, 0.7, 0.4)}


def adaptive_scene():
    """The Fides prototype's mesh, an order per cell from 1 to 5 across x."""
    data, _, camera = fides_scene()
    centres = td.cell_centers(data).values.numpy().reshape(-1, 3)
    orders = np.clip(1 + np.floor(5 * centres[:, 0]), 1, 5).astype(np.int32)
    space = td.L2(data, order=tack.array(orders))
    data.fields["wave"] = td.project(data, td.interpolate(data, wave, td.L2(data, order=5)),
                                     space)
    return data, "wave", camera


def check(name, data, field, records, probe, samples=3000, margin=0.03):
    """The shader's values against NumPy's evaluation of the records and against
    Tack's probe of the solid, at pixels away from face edges (where a probe could
    find the neighbouring cell, a DG field's other side)."""
    hit = np.argwhere(probe[..., 1] >= 0)
    at = probe[hit[:, 0], hit[:, 1]]
    face, u, v = at[:, 1].astype(np.int64), at[:, 2].astype(np.float64), at[:, 3].astype(np.float64)
    kinds = records.faces[face, 0] & 0xFF
    inside = np.minimum(u, v)
    inside = np.minimum(inside, np.where(kinds == 0, 1 - u - v, np.minimum(1 - u, 1 - v)))
    keep = np.flatnonzero(inside > margin)
    keep = np.random.default_rng(0).choice(keep, min(samples, len(keep)), replace=False)
    reference, points = evaluate(records, face[keep], u[keep], v[keep])
    shader = at[keep, 0]
    tack_values = td.probe(data, points).fields[field].values.numpy().reshape(-1)
    scale = max(abs(records.range[0]), abs(records.range[1]))
    print(f"  {name}: {len(hit)} pixels drawn; at {len(keep)} of them the shader is within "
          f"{np.abs(shader - reference).max() / scale:.1e} of NumPy's evaluation of the "
          f"records and {np.abs(shader - tack_values).max() / scale:.1e} of Tack's probe "
          "of the solid (relative to the field's largest magnitude)")


def main(out):
    tack.set_default_device("cpu")
    os.makedirs(out, exist_ok=True)
    renderer = Renderer()
    print(f"WebGPU on {renderer.adapter.info['device']} ({renderer.adapter.info['backend_type']})")
    scenes = [("fides", fides_scene), ("curved", curved_scene), ("adaptive", adaptive_scene)]
    for name, make in scenes:
        data, field, camera = make()
        start = time.perf_counter()
        records = surface_records(data, field)
        made = time.perf_counter() - start
        orders = sorted(set((records.faces[:, 0] >> 16).tolist()))
        uploaded = records.faces.nbytes + records.geometry.nbytes + records.field.nbytes
        solid = sum(tack.arrays.size_of(f.values) for f in (data.geometry, data.fields[field]))
        print(f"{name}: {data.num_cells} cells, {len(records.faces)} faces of field orders "
              f"{orders}, geometry order {(records.faces[0, 0] >> 8) & 0xFF}; records "
              f"{uploaded / 1024:.0f} KiB (the solid's geometry and field hold {solid} "
              f"values); made in {1e3 * made:.0f} ms")
        start = time.perf_counter()
        picture, probe = renderer.render(records, **camera)
        print(f"  rendered and read back in {1e3 * (time.perf_counter() - start):.0f} ms")
        write_png(os.path.join(out, f"{name}.png"), picture)
        check(name, data, field, records, probe)
        if name == "adaptive":
            picture, _ = renderer.render(records, mode=1, **camera)
            write_png(os.path.join(out, f"{name}-orders.png"), picture)
    print(f"wrote PNGs to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
