# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tack generating its own shaders: a surface's geometry and field evaluation, Tack's
``@tack.func`` views, compiled to Slang (``slang_gen``), then to WGSL by ``slangc``, and
drawn by WebGPU (wgpu-py). No Bernstein records, no basis written for the renderer:
each launch group (shape, order) is a pipeline running Tack's own evaluation.

Scenes, each a PNG and a check:

1. ``cells``: one curved hexahedron and one tetrahedron (order-2 geometry, an order-5
   field, Gauss-Lobatto and equispaced), drawn and with the template's triangles;
2. ``fides``: the Fides prototype's DG data (hexahedra and tetrahedra, order 2);
3. ``adaptive``: the same mesh's surface with an order per face, 1 to 5 (each
   (shape, order) a pipeline).

The check: at pixels away from face edges, the shader's value against the same
evaluation run by Tack on the CPU (the code generation), and against Tack's probe of
the solid (the whole path).

From the checkout, with wgpu-py on ``PYTHONPATH`` and ``SLANGC`` naming Slang's
compiler (and, for the Fides scenes, the Fides prototype and Conduit)::

    SLANGC=.../slangc .venv/bin/python packages/tack-vis/examples/slang_surfaces/demo.py OUTDIR
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "webgpu_surfaces"))

from renderer import write_png
from shaders import _geometry_at, _value_at, group_shaders
from slang_renderer import Renderer

import tack
import tack.dataset as td


def _evaluate(surface, name, groups, cells, uv):
    """Tack's own evaluation, on the CPU, of the field and position at surface ``cells``
    and parametric ``uv``: the same kernels the shaders were generated from."""
    field = surface.fields[name]
    values = np.zeros(len(cells))
    points = np.zeros((len(cells), 3))
    for g in groups:
        local = {int(e): k for k, e in enumerate(g.ids)}
        mine = np.flatnonzero(np.isin(cells, g.ids))
        if not len(mine):
            continue
        ids = tack.array(np.array([local[int(c)] for c in cells[mine]], np.int32))
        pcs = tack.array(np.c_[uv[mine], np.zeros(len(mine))].astype(np.float32), tack.vec3())
        out = tack.empty((len(mine),), tack.f32)
        _value_at(field.view(g.group), ids, pcs, out)
        values[mine] = out.numpy()
        geo = tack.empty((3 * len(mine),), tack.vec3())
        _geometry_at(surface.domain_view("cells", g.group), ids, pcs, geo)
        points[mine] = geo.numpy().reshape(-1, 3, 3)[:, 0]
    return values, points


def check(name, solid, surface, field, groups, probe, samples=3000, margin=0.03, exact=None):
    """``exact``: the surface cells where the solid's field is the surface's (all by
    default); the solid is compared there alone."""
    hit = np.argwhere(probe[..., 1] >= 0)
    at = probe[hit[:, 0], hit[:, 1]]
    cells = at[:, 1].astype(np.int64)
    uv = at[:, 2:4].astype(np.float64)
    types = surface.topology.types.numpy().reshape(-1)[cells]
    inside = np.minimum(uv[:, 0], uv[:, 1])
    inside = np.minimum(inside, np.where(types == 5, 1 - uv.sum(axis=1),
                                         np.minimum(1 - uv[:, 0], 1 - uv[:, 1])))
    keep = np.flatnonzero(inside > margin)
    keep = np.random.default_rng(0).choice(keep, min(samples, len(keep)), replace=False)
    tack_values, points = _evaluate(surface, field, groups, cells[keep], uv[keep])
    solid_values = td.probe(solid, points).fields[field].values.numpy().reshape(-1)
    shader = at[keep, 0]
    scale = np.abs(tack_values).max()
    same = np.ones(len(keep), bool) if exact is None else exact[cells[keep]]
    print(f"  {name}: {len(hit)} pixels; at {len(keep)}, the shader is within "
          f"{np.abs(shader - tack_values).max() / scale:.1e} of Tack's CPU evaluation of the "
          f"same code; at the {same.sum()} of them where the surface's field is the solid's, "
          f"within {np.abs(shader - solid_values)[same].max() / scale:.1e} of Tack's probe of "
          "the solid (relative)")


def show(renderer, solid, surface, name, out_dir, label, panels, exact=None):
    field = surface.fields[name]
    t0 = time.perf_counter()
    groups = group_shaders(surface, name, out_dir)
    made = time.perf_counter() - t0
    x = surface.geometry.values.numpy().reshape(-1, 3)
    bounds = (x.min(axis=0), x.max(axis=0))
    centre = 0.5 * (bounds[0] + bounds[1])
    values = field.values.numpy().reshape(-1)
    print(f"{label}: {surface.num_cells} faces in {len(groups)} launch groups "
          f"{[(g.shape.__name__, g.key) for g in groups]}, geometry {surface.geometry.space}, "
          f"field {field.space}; generated and compiled in {made:.1f} s; WGSL "
          f"{sum(len(g.wgsl.splitlines()) for g in groups)} lines")
    pictures = []
    for k, (offset, extra) in enumerate(panels):
        picture, probe = renderer.render(groups, bounds, centre + np.asarray(offset), centre,
                                         (float(values.min()), float(values.max())), **extra)
        pictures.append(picture)
        if k == 0:
            check(label, solid, surface, name, groups, probe, exact=exact)
    return np.concatenate(pictures, axis=1)


def single_cells(renderer, out_dir):
    from single_cell import cell

    rows = []
    for kind in ("hexahedron", "tetrahedron"):
        solid = cell(kind)
        surface = td.extract_surface(solid)
        rows.append(show(renderer, solid, surface, "f", out_dir, kind,
                         [((2.2, -2.6, 1.9), {"size": (640, 576), "level": 16, "edges": 1.5}),
                          ((2.2, -2.6, 1.9), {"size": (640, 576), "level": 4, "edges": 1.5,
                                              "tessellation": 0.8})]))
    return np.concatenate(rows, axis=0)


def fides_data():
    import importlib.util

    # The WebGPU prototype's demo (also named demo.py) builds it from the Fides IR.
    path = os.path.join(HERE, "..", "webgpu_surfaces", "demo.py")
    spec = importlib.util.spec_from_file_location("webgpu_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fides_scene()[0]


def fides(renderer, out_dir):
    solid = fides_data()
    surface = td.extract_surface(solid, fields=["pressure"])
    return show(renderer, solid, surface, "pressure", out_dir, "fides",
                [((2.1, -2.3, 1.7), {"level": 1})])


@tack.func
def _wave(x):
    return tack.sin(3.0 * x[0]) * tack.cos(2.0 * x[1]) + 0.5 * x[2]


def adaptive(renderer, out_dir):
    """The Fides mesh's surface, an order per face from 1 to 5 across x: the field
    interpolated at order 5 on the solid, traced, and projected to each face's order,
    all Tack; one pipeline a (shape, order)."""
    solid = fides_data()
    solid.fields["wave"] = td.interpolate(solid, _wave, td.L2(solid, order=5))
    surface = td.extract_surface(solid, fields=["wave"])
    centres = td.cell_centers(surface).values.numpy().reshape(-1, 3)
    orders = np.clip(1 + np.floor(5 * centres[:, 0]), 1, 5).astype(np.int32)
    space = td.L2(surface, order=tack.array(orders))
    surface.fields["wave"] = td.project(surface, surface.fields["wave"], space)
    # A face below order 5 holds a projection of the solid's trace, not the trace: the
    # solid is compared on the order-5 faces alone.
    return show(renderer, solid, surface, "wave", out_dir, "adaptive",
                [((2.1, -2.3, 1.7), {"level": 1})], exact=orders == 5)


def main(out):
    tack.set_default_device("cpu")
    os.makedirs(out, exist_ok=True)
    renderer = Renderer()
    print(f"WebGPU on {renderer.adapter.info['device']} ({renderer.adapter.info['backend_type']}), "
          "default limits")
    write_png(os.path.join(out, "cells.png"), single_cells(renderer, out))
    for name, scene in (("fides", fides), ("adaptive", adaptive)):
        try:
            write_png(os.path.join(out, f"{name}.png"), scene(renderer, out))
        except ImportError as e:
            print(f"{name}: skipped ({e})")
    print(f"wrote PNGs to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
