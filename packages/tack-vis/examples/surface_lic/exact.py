# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact per-pixel evaluation for a slice: what the surface pass needs to evaluate the
volume's fields with the cells' own basis at every pixel, instead of interpolating the
slice's vertex values.

- Tack's evaluation as Slang (``slang_gen``, as the hooks): the cell's position and
  Jacobian at parametric coordinates (``tack_cell_geometry``), a vector field
  (``tack_vector``) and a scalar field (``tack_scalar``) there, each reading one packed
  buffer of Tack's arrays.
- The slice's triangles, each with its own three vertices, the cell it lies in (its
  centroid located by Tack's ``CellLocator``), and each vertex's parametric
  coordinates in that cell (the vertex located after a nudge towards the centroid, so
  it is found in that cell): the starting guess the surface pass interpolates and
  corrects by Newton's method on ``tack_cell_geometry``.
"""

import os
import sys

import numpy as np

import tack
import tack.dataset as td

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slang_surfaces"))

from slang_gen import export, joined  # noqa: E402
from shaders import _one_cell_arrays  # noqa: E402


@tack.kernel
def _cell_geometry_at(cells, ids, pcs, out):
    for k in range(ids.shape[0]):
        c = ids[k]
        pc = pcs[k]
        j = cells.geometry_jacobian(c, pc)
        out[4 * k] = cells.position(c, pc)
        out[4 * k + 1] = tack.vec3(j[0, 0], j[1, 0], j[2, 0])
        out[4 * k + 2] = tack.vec3(j[0, 1], j[1, 1], j[2, 1])
        out[4 * k + 3] = tack.vec3(j[0, 2], j[1, 2], j[2, 2])


@tack.kernel
def _field_at(u, ids, pcs, out):
    for k in range(ids.shape[0]):
        out[k] = u.value(ids[k], pcs[k])


def volume_functions(data, vector, scalar):
    """The generated Slang (``tack_cell_geometry``, ``tack_vector``, ``tack_scalar``,
    each taking its packed buffer first) and the three buffers' words. The volume must
    be one launch group (one cell shape and order)."""
    fields = [data.fields[vector], data.fields[scalar]]
    groups = [g for g in data.launch_groups("cells", fields) if g.count]
    if len(groups) != 1:
        raise ValueError(f"the volume has {len(groups)} launch groups; this takes one")
    group = groups[0]
    if group.count != data.num_cells:
        raise ValueError("the launch group is not every cell")
    cells = data.domain_view("cells", group)
    ids, pcs = _one_cell_arrays()
    geometry = export(_cell_geometry_at, (cells, ids, pcs, tack.empty((4,), tack.vec3())),
                      "tack_cell_geometry",
                      {"ids": ("int", 1), "pcs": ("float", 3), "out": ("float", 12)},
                      ("float4x3", "float4x3(" + ", ".join(f"{{out}}[{k}]" for k in range(12))
                       + ")"), binding=None)
    v = export(_field_at, (fields[0].view(group), ids, pcs, tack.empty((1,), tack.vec3())),
               "tack_vector", {"ids": ("int", 1), "pcs": ("float", 3), "out": ("float", 3)},
               ("float3", "float3({out}[0], {out}[1], {out}[2])"), binding=None)
    s = export(_field_at, (fields[1].view(group), ids, pcs, tack.empty((1,), tack.f32)),
               "tack_scalar", {"ids": ("int", 1), "pcs": ("float", 3), "out": ("float", 1)},
               ("float", "{out}[0]"), binding=None)
    return joined(geometry, v, s), [e.buffers[0][1] for e in (geometry, v, s)]


def slice_cells(data, points, triangles, nudge=1e-3):
    """Per triangle its cell, and per triangle vertex (unshared: ``(k, 3)``) its
    position and parametric coordinates in that cell."""
    locator = td.CellLocator(data)
    corners = points[triangles.astype(np.int64)].astype(np.float64)       # (k, 3, 3)
    centroid = corners.mean(axis=1)
    found = locator.find(centroid)
    cells = found.cells.numpy().astype(np.int64)
    if (cells < 0).any():
        raise ValueError(f"{int((cells < 0).sum())} triangles' centroids are in no cell")
    nudged = corners + nudge * (centroid[:, None, :] - corners)
    at = locator.find(nudged.reshape(-1, 3))
    vertex_cells = at.cells.numpy().reshape(-1, 3)
    pcs = at.pcs.numpy().reshape(-1, 3, 3)
    elsewhere = vertex_cells != cells[:, None]
    if elsewhere.any():
        # Found in a neighbour: start from the cell's centre instead.
        pcs[elsewhere] = 0.5
    return cells.astype(np.uint32), corners.astype(np.float32), pcs.astype(np.float32), \
        int(elsewhere.sum())
