# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A surface dataset's launch groups as shaders: for each (shape, order) group, Tack's
own geometry and field evaluation exported to Slang (``slang_gen``), joined to the
renderer's hand-written ``surface.slang`` and compiled to WGSL by ``slangc``.

Nothing is converted on the way: the field is drawn in its own space and basis
(Gauss-Lobatto, equispaced, Bernstein, an order per cell, ...), the geometry as
Tack's domain view evaluates it (order 2 curved), from the dataset's own arrays."""

import os
import subprocess
from dataclasses import dataclass

import numpy as np
from slang_gen import export, joined

import tack

HERE = os.path.dirname(os.path.abspath(__file__))
SLANGC = os.environ.get("SLANGC", "slangc")


@tack.kernel
def _geometry_at(cells, ids, pcs, out):
    for k in range(ids.shape[0]):
        c = ids[k]
        pc = pcs[k]
        j = cells.geometry_jacobian(c, pc)
        out[3 * k] = cells.position(c, pc)
        out[3 * k + 1] = tack.vec3(j[0, 0], j[1, 0], j[2, 0])
        out[3 * k + 2] = tack.vec3(j[0, 1], j[1, 1], j[2, 1])


@tack.kernel
def _value_at(u, ids, pcs, out):
    for k in range(ids.shape[0]):
        out[k] = u.value(ids[k], pcs[k])


@tack.kernel
def _entity_ids(cells, out):
    for c in cells:
        out[c] = cells.entity_id(c)


@dataclass
class GroupShader:
    """One launch group's pipeline input: its WGSL, the buffers its generated functions
    read (group 0), its cells' ids in the surface, its shape and its key."""

    wgsl: str
    slang: str
    buffers: list
    ids: np.ndarray
    shape: type
    key: object
    group: object


def _one_cell_arrays():
    """The per-iteration arguments the export compiles against: one element each."""
    ids = tack.array(np.zeros(1, np.int32))
    pcs = tack.array(np.zeros((1, 3), np.float32), tack.vec3())
    return ids, pcs


def group_shaders(surface, name, out_dir):
    """Every launch group of ``surface`` for field ``name``, compiled."""
    field = surface.fields[name]
    with open(os.path.join(HERE, "surface.slang")) as f:
        scaffold = f.read()
    out = []
    for n, group in enumerate(surface.launch_groups("cells", [field])):
        if not group.count:
            continue
        cells = surface.domain_view("cells", group)
        ids, pcs = _one_cell_arrays()
        geometry = export(_geometry_at, (cells, ids, pcs, tack.empty((3,), tack.vec3())),
                          "tack_geometry",
                          {"ids": ("int", 1), "pcs": ("float", 3), "out": ("float", 9)},
                          ("float3x3", "float3x3("
                           + ", ".join(f"{{out}}[{k}]" for k in range(9)) + ")"),
                          binding=0)
        value = export(_value_at, (field.view(group), ids, pcs, tack.empty((1,), tack.f32)),
                       "tack_value",
                       {"ids": ("int", 1), "pcs": ("float", 3), "out": ("float", 1)},
                       ("float", "{out}[0]"), binding=geometry.next_binding)
        entity = tack.empty((group.count,), tack.i32)
        _entity_ids(cells, entity)
        triangle = int(group.shape.NUM_POINTS == 3)
        source = "\n".join([f"// {group.shape.__name__}, key {group.keys}",
                            f"#define TACK_TRIANGLE {triangle}",
                            joined(geometry, value), scaffold])
        stem = os.path.join(out_dir, f"group{n}_{group.shape.__name__}")
        with open(stem + ".slang", "w") as f:
            f.write(source)
        r = subprocess.run([SLANGC, stem + ".slang", "-target", "wgsl", "-o", stem + ".wgsl"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"slangc on {stem}.slang:\n{(r.stderr or r.stdout)[:3000]}")
        with open(stem + ".wgsl") as f:
            wgsl = f.read()
        out.append(GroupShader(wgsl, source, geometry.buffers + value.buffers,
                               entity.numpy().astype(np.uint32), group.shape, group.keys, group))
    return out
