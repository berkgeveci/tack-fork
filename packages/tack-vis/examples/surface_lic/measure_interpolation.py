# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How far per-vertex interpolation over a slice's triangles (what the surface pass
draws) is from the field itself, on disk_out_ref's middle slice: random points inside
every triangle, ``V`` and ``Temp`` interpolated from the triangle's vertices against
``td.probe`` at the same points (Tack locating each point in its hexahedron and
evaluating the cell's own trilinear basis).

    python measure_interpolation.py [path/to/disk_out_ref.ex2] [samples per triangle]
"""

import sys

import numpy as np
from disk import DISK, read

import scene
import tack
import tack.dataset as td


def probed(data, where, names):
    out = td.probe(data, np.ascontiguousarray(where, np.float64), fields=names)
    valid = out.fields["valid"].values.numpy().reshape(-1) > 0
    return {n: out.fields[n].values.numpy().reshape(len(where), -1) for n in names}, valid


def main(path=DISK, per_triangle=64, seed=1):
    tack.set_default_device("cpu")
    data, bounds = read(path)
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    centre = 0.5 * (lo + hi)
    points, triangles, vectors, normal, temp = scene.slice_triangles(
        data, tuple(centre), (1.0, 0.0, 0.0), vector="V", scalar="Temp")
    vmax = float(np.linalg.norm(vectors, axis=1).max())
    trange = float(temp.max() - temp.min())

    # The slice's own values at its vertices, against the probe: the cut is exact there.
    at, valid = probed(data, points, ["V", "Temp"])
    dv = np.linalg.norm(at["V"] - vectors, axis=1)[valid]
    print(f"slice: {len(triangles)} triangles, {len(points)} points; at the vertices "
          f"({valid.sum()} located): |dV| max {dv.max() / vmax:.1e} of max |V|, "
          f"dTemp max {np.abs(at['Temp'][:, 0] - temp)[valid].max() / trange:.1e} of its range")

    # Random points inside every triangle (uniform barycentric coordinates).
    rng = np.random.default_rng(seed)
    r = rng.random((len(triangles), per_triangle, 2))
    flip = r.sum(axis=2) > 1
    r[flip] = 1 - r[flip]
    w = np.concatenate([1 - r.sum(axis=2, keepdims=True), r], axis=2)    # (k, s, 3)
    corners = triangles.astype(np.int64)
    where = np.einsum("ksj,kjd->ksd", w, points[corners].astype(np.float64)).reshape(-1, 3)
    v_lin = np.einsum("ksj,kjd->ksd", w, vectors[corners].astype(np.float64)).reshape(-1, 3)
    t_lin = np.einsum("ksj,kj->ks", w, temp[corners].astype(np.float64)).reshape(-1)

    exact, valid = probed(data, where, ["V", "Temp"])
    v_ex, t_ex = exact["V"][valid], exact["Temp"][valid, 0]
    v_lin, t_lin = v_lin[valid], t_lin[valid]
    print(f"{valid.sum()} of {len(where)} sample points located "
          f"({per_triangle} a triangle)")

    dv = np.linalg.norm(v_lin - v_ex, axis=1)
    speed = np.linalg.norm(v_ex, axis=1)
    print(f"  |V_interpolated - V| / max|V|: median {np.median(dv) / vmax:.2e}, "
          f"99th percentile {np.percentile(dv, 99) / vmax:.2e}, max {dv.max() / vmax:.2e}")
    strong = speed > 0.05 * vmax
    rel = dv[strong] / speed[strong]
    print(f"  relative to the local |V| (where |V| > 5% of max, {strong.sum()} points): "
          f"median {np.median(rel):.2e}, 99th percentile {np.percentile(rel, 99):.2e}, "
          f"max {rel.max():.2e}")

    # What LIC draws: the direction of the field's part in the plane.
    def tangential(v):
        return v - (v @ normal)[:, None] * normal

    a, b = tangential(v_lin), tangential(v_ex)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = nb > 0.05 * vmax
    cos = np.clip(np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok]), -1, 1)
    angle = np.degrees(np.arccos(cos))
    print(f"  direction in the plane (where |V_t| > 5% of max, {ok.sum()} points): "
          f"median {np.median(angle):.2f} deg, 99th percentile {np.percentile(angle, 99):.2f} "
          f"deg, max {angle.max():.2f} deg")
    dt = np.abs(t_lin - t_ex) / trange
    print(f"  |Temp_interpolated - Temp| / its range: median {np.median(dt):.2e}, "
          f"99th percentile {np.percentile(dt, 99):.2e}, max {dt.max():.2e} "
          f"(one viridis step is {1 / 255:.1e})")
    for steps in (1, 5):
        print(f"    more than {steps} colour step{'s' if steps > 1 else ''} off: "
              f"{100 * np.mean(dt > steps / 255):.1f}% of the points")
    # Where the worst are: the triangles' share of them, against the triangles' size.
    where_ok = where[valid]
    worst = np.argsort(dt)[-max(1, len(dt) // 100):]
    print(f"    the worst 1% (Temp): y from {where_ok[worst, 1].min():.2f} to "
          f"{where_ok[worst, 1].max():.2f}, z from {where_ok[worst, 2].min():.2f} to "
          f"{where_ok[worst, 2].max():.2f}; median |y| {np.median(np.abs(where_ok[worst, 1])):.2f} "
          f"against {np.median(np.abs(where_ok[:, 1])):.2f} for all points")
    tri_of = np.repeat(np.arange(len(triangles)), per_triangle)[valid]
    area = 0.5 * np.linalg.norm(np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                                         points[triangles[:, 2]] - points[triangles[:, 0]]),
                                axis=1)
    print(f"    their triangles' median area {np.median(area[tri_of[worst]]):.3f} against "
          f"{np.median(area):.3f} for all triangles")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else DISK, int(a[1]) if len(a) > 1 else 64)
