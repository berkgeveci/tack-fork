# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The benchmark's surfaces, made by Tack: the surface of an ``n``-cube of hexahedra
with a DG field, straight (order-2 field) or curved (order-2 geometry bent, an
order-5 field). ``surface(kind, n)`` is the surface dataset with field ``"f"``."""

import numpy as np

import tack
import tack.dataset as td


@tack.func
def wave(x):
    return tack.sin(6.0 * x[0]) * tack.cos(5.0 * x[1]) + 0.5 * tack.sin(4.0 * x[2])


def _bend(x):
    c = x - 0.5
    return x + 0.12 * np.c_[np.sin(np.pi * x[:, 1]) * c[:, 0], np.sin(np.pi * x[:, 2]) * c[:, 1],
                            np.sin(np.pi * x[:, 0]) * c[:, 2]]


#: kind -> (field order, curved)
KINDS = {"straight": (2, False), "curved": (5, True)}


def solid(kind, n):
    order, curved = KINDS[kind]
    image = td.uniform_grid((n + 1,) * 3, spacing=(1.0 / n,) * 3, dtype=tack.f64)
    data = td.extract_cells(image, np.arange(image.num_cells))
    if curved:
        space = td.H1(data, order=2)
        nodes = td.project(data, data.geometry, space).values.numpy().reshape(-1, 3)
        data.geometry = td.Field(space, tack.array(_bend(nodes), tack.vec3(tack.f64)))
    data.fields["f"] = td.interpolate(data, wave, td.L2(data, order=order))
    return data


def surface(kind, n):
    return td.extract_surface(solid(kind, n), fields=["f"])


def camera(frame, frames):
    """An orbit about the cube's centre: eye, target."""
    a = 2 * np.pi * frame / max(frames, 1) + 0.6
    centre = np.array([0.5, 0.5, 0.5])
    return centre + np.array([2.4 * np.cos(a), 2.4 * np.sin(a), 1.6]), centre


def indexed_template(quad, level):
    """A reference face subdivided ``level`` times a side as points (u, v) and the
    triangles' indices into them (the WebGPU renderers' ``template``
    triangles, points shared)."""
    n = level
    points, at = [], {}
    for j in range(n + 1):
        for i in range(n + 1 if quad else n + 1 - j):
            at[i, j] = len(points)
            points.append((i, j))
    tris = []
    for j in range(n):
        for i in range(n if quad else n - j):
            a, b, c, d = (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
            tris += [at[a], at[b], at[c]]
            if quad or i + j < n - 1:
                tris += [at[b], at[d], at[c]]
    return np.array(points, np.float32) / n, np.array(tris, np.uint32)
