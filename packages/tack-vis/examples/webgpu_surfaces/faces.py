# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A solid's surface as the records a WebGPU renderer draws: each face's geometry and
one field as Bernstein coefficients of the face's own order, in Tack's node order.

Tack computes everything: ``extract_surface`` gives the faces, the exact traces of
the solid's DG fields (#426) and its curved geometry (order 2), and ``project``
takes them to ``L2(surface, order=..., basis="bernstein")``, an order per face where
the solid has an order per cell. The coefficients of an ``L2`` field in blocks are
already one face after another at ``space.offsets``, so they go to the GPU as they
are; only the precision changes (f32, the geometry about an origin).

A face's coefficients, order ``p``:

- a quad's tensor lattice over [0, 1]^2, the first axis fastest; at order 1 its
  corners, in the corner order (0, 0), (1, 0), (1, 1), (0, 1);
- a triangle's lattice ``(i, j)``, ``i + j <= p``, ``i`` fastest, the coefficient of
  ``p! / (i! j! k!) r^i s^j (1 - r - s)^k`` with ``k = p - i - j``;
- at order 0, one value.
"""

from dataclasses import dataclass

import numpy as np

import tack
import tack.dataset as td
from tack import arrays

TRIANGLE, QUAD = 0, 1
_SHAPES = {5: TRIANGLE, 9: QUAD}          # VTK cell types


@dataclass
class Records:
    """What the renderer uploads: ``faces`` (``(n, 4)`` u32: shape | geometry order
    << 8 | field order << 16, geometry offset in points, field offset in values, 0),
    ``geometry`` (``(m, 3)`` f32 about ``origin``), ``field`` (f32), and the field's
    range over its coefficients (the convex hull property bounds it)."""

    faces: np.ndarray
    geometry: np.ndarray
    field: np.ndarray
    origin: np.ndarray
    range: tuple


def _host(values):
    return np.asarray(arrays.to_host(arrays.materialize(values)))


def _bernstein(surface, field, order):
    """``field`` on ``surface`` as Bernstein coefficients of ``order`` (an int, or an
    i32 array of one a face), in blocks: (values, offsets)."""
    space = td.L2(surface, order=order, basis="bernstein", layout="blocks")
    projected = td.project(surface, field, space)
    return _host(projected.values), _host(space.offsets).astype(np.int64)


def _per_face_orders(surface, field, orders):
    """``field`` as each face's Bernstein coefficients of its own order (``orders``):
    what ``L2(surface, order=orders, basis="bernstein")`` would hold, but Tack's
    Bernstein conversion takes one order, so one conversion an order present, each
    face's block gathered from its own (exact: the face's polynomial is of its
    order)."""
    present = sorted(set(orders.tolist()))
    blocks = {k: _bernstein(surface, field, k) for k in present}
    base, at = {}, 0
    for k in present:
        base[k] = at
        at += len(blocks[k][0])
    stacked = np.concatenate([blocks[k][0] for k in present])
    n = len(orders)
    starts = np.empty(n, np.int64)
    counts = np.empty(n, np.int64)
    for k in present:
        mine = orders == k
        offsets = blocks[k][1]
        starts[mine] = base[k] + offsets[:n][mine]
        counts[mine] = offsets[1:n + 1][mine] - offsets[:n][mine]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    gather = np.repeat(starts - offsets[:n], counts) + np.arange(offsets[-1])
    return stacked[gather], offsets


def surface_records(data, name):
    """The surface of ``data`` with field ``name`` (an ``L2`` field of one order or an
    order per cell, ``H1``, or one value a cell), as ``Records``."""
    field = data.fields[name]
    space = field.space
    orders = None
    if isinstance(space, td.L2) and space.varies:
        # extract_surface takes traces of one order (#423): raise the field to its
        # highest order, exactly, and carry each cell's order to its faces.
        orders = _host(space.cell_keys()).astype(np.float32)
        data = td.DataSet(data.topology, data.geometry,
                          fields={name: field})        # the caller's dataset stays as it is
        data.fields[name] = td.project(data, field, td.L2(data, order=space.max_order))
        data.fields["__order"] = td.Field(td.Constant(data), tack.array(orders))
    surface = td.extract_surface(data)
    types = _host(surface.topology.types).reshape(-1)
    shape = np.array([_SHAPES[int(t)] for t in types], np.uint32) if len(types) else \
        np.zeros(0, np.uint32)

    # The geometry: its own order on every face.
    g = surface.geometry.space.order
    geometry, goffsets = _bernstein(surface, surface.geometry, g)
    geometry = geometry.reshape(-1, 3).astype(np.float64)
    origin = 0.5 * (geometry.min(axis=0) + geometry.max(axis=0))

    # The field: the face's owner's order, or the field's one order.
    traced = surface.fields[name]
    if orders is not None:
        per_face = _host(surface.fields["__order"].values).reshape(-1).astype(np.int32)
        values, foffsets = _per_face_orders(surface, traced, per_face)
        forder = per_face.astype(np.uint32)
    elif isinstance(traced.space, td.L2):
        values, foffsets = _bernstein(surface, traced, traced.space.order)
        forder = np.full(len(types), traced.space.order, np.uint32)
    else:                                   # H1 or a value a cell: as L2 of its order
        p = getattr(traced.space, "order", 0)
        values, foffsets = _bernstein(surface, traced, p)
        forder = np.full(len(types), p, np.uint32)
    values = values.reshape(-1)

    faces = np.zeros((len(types), 4), np.uint32)
    faces[:, 0] = shape | (np.uint32(g) << 8) | (forder << 16)
    faces[:, 1] = goffsets[:len(types)]
    faces[:, 2] = foffsets[:len(types)]
    return Records(faces, (geometry - origin).astype(np.float32), values.astype(np.float32),
                   origin, (float(values.min()), float(values.max())))


def evaluate(records, face, u, v):
    """The field and the position at parametric ``(u, v)`` of faces ``face`` (arrays),
    from the records, in NumPy: the reference the shaders are checked against."""
    from math import comb, factorial

    def at(coefficients, offset, kind, p, uu, vv):
        if p == 0:
            return coefficients[offset]
        if kind == QUAD:
            bu = [comb(p, a) * uu ** a * (1 - uu) ** (p - a) for a in range(p + 1)]
            bv = [comb(p, b) * vv ** b * (1 - vv) ** (p - b) for b in range(p + 1)]
            out = 0.0
            for b in range(p + 1):
                for a in range(p + 1):
                    k = a + (p + 1) * b
                    if p == 1:
                        k = (0, 1, 3, 2)[k]
                    out = out + coefficients[offset + k] * bu[a] * bv[b]
            return out
        out, k = 0.0, 0
        for j in range(p + 1):
            for i in range(p + 1 - j):
                m = p - i - j
                c = factorial(p) // (factorial(i) * factorial(j) * factorial(m))
                out = out + coefficients[offset + k] * c * uu ** i * vv ** j * (1 - uu - vv) ** m
                k += 1
        return out

    values, points = [], []
    geometry = records.geometry.astype(np.float64)
    for f, uu, vv in zip(face, u, v, strict=True):
        word, goff, foff, _ = (int(x) for x in records.faces[f])
        kind, g, p = word & 0xFF, (word >> 8) & 0xFF, word >> 16
        values.append(at(records.field.astype(np.float64), foff, kind, p, uu, vv))
        points.append(at(geometry, goff, kind, g, uu, vv) + records.origin)
    return np.array(values), np.array(points)
