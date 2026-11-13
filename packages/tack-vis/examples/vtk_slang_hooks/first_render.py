# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a fresh process pays before its first picture, by phase, for the same polydata
drawn three ways:

- ``plain``: VTK's WebGPU mapper, its WGSL shaders, the field at the points coloured
  through a lookup table (``MapScalars`` on the CPU), as VTK draws polygons today;
- ``slang``: the same, through VTK's Slang program with its default types;
- ``hooks``: Tack's hook module (``tack_hooks``), the field per pixel.

Phases (ms): ``tack`` (Tack's surface export and template evaluation, or the field at the
points for ``plain``/``slang``), ``device`` (``Initialize``), ``first`` (the first
``Render`` waited for: the mapper's CPU preparation, uploads, the topology pass,
shader compiles and pipelines, the frame), ``second`` (the next frame), ``new_actor``
(another actor of the same module added in the same window: what the caches save).

    python first_render.py [straight|curved|tets] n [plain|slang|hooks] [place]

``SAMPLE_FILE=path`` runs macOS's ``sample`` on the process from the first render on
(``SAMPLE_SECONDS``, 3).
"""

import os
import sys
import time

T0 = time.perf_counter()
import numpy as np  # noqa: E402
import tack_hooks  # noqa: E402
from tack_hooks import _evaluate, rr  # noqa: E402
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData  # noqa: E402

import tack  # noqa: E402
import tack.dataset as td  # noqa: E402
from scenes import camera, indexed_template, surface, wave  # noqa: E402

W, H = 1920, 1080


def tets_surface(n):
    image = td.uniform_grid((n + 1,) * 3, spacing=(1.0 / n,) * 3, dtype=tack.f64)
    data = td.tetrahedralize(image)
    data.fields["f"] = td.interpolate(data, wave, td.L2(data, order=3))
    return td.extract_surface(data, fields=["f"])


def polygons(surf, level, use_slang):
    """The same template triangles, the field evaluated at the points by Tack and
    coloured by VTK's lookup table."""
    from shaders import _value_at

    curved = surf.geometry.space.order > 1
    actors = []
    values = surf.fields["f"].values.numpy().reshape(-1)
    lo, hi = float(values.min()), float(values.max())
    for g in surf.launch_groups("cells", [surf.fields["f"]]):
        if not g.count:
            continue
        points, tris = indexed_template(g.shape.NUM_POINTS == 4, level if curved else 1)
        m = len(points)
        x, normals = _evaluate(surf.domain_view("cells", g), g.count, points)
        ids = tack.array(np.repeat(np.arange(g.count, dtype=np.int32), m))
        pcs = np.zeros((g.count * m, 3), np.float32)
        pcs[:, :2] = np.tile(points, (g.count, 1))
        val = tack.empty((g.count * m,), tack.f32)
        _value_at(surf.fields["f"].view(g), ids, tack.array(pcs, tack.vec3()), val)
        conn = (tris[None, :] + m * np.arange(g.count)[:, None]).reshape(-1).astype(np.int64)
        poly = vtkPolyData()
        poly.points = vtkPoints(data=vtkFloatArray(x.astype(np.float32)))
        poly.polys = vtkCellArray(offsets=np.arange(0, len(conn) + 1, 3), connectivity=conn)
        nv = vtkFloatArray(normals.astype(np.float32))
        nv.SetName("Normals")
        poly.GetPointData().SetNormals(nv)
        s = vtkFloatArray(val.numpy())
        s.SetName("f")
        poly.GetPointData().SetScalars(s)
        mapper = rr.vtkPolyDataMapper(input_data=poly, scalar_range=(lo, hi))
        actor = rr.vtkActor(mapper=mapper)
        if use_slang:
            actor.GetShaderProperty().SetUseSlang(True)
        actors.append(actor)
    return actors


def build(surf, mode, place):
    if mode == "hooks":
        return [a.actor for a in tack_hooks.surface_actors(surf, "f", place=place)[0]]
    return polygons(surf, 8, mode == "slang")


def frame(window, renderer, k=0, frames=30):
    eye, centre = camera(k, frames)
    cam = renderer.active_camera
    cam.position, cam.focal_point, cam.view_up = tuple(eye), tuple(centre), (0.0, 0.0, 1.0)
    cam.view_angle = 30.0
    cam.SetClippingRange(0.5, 6.0)
    window.Render()
    window.WaitForCompletion()


def main(kind, n, mode, place):
    tack.set_default_device("cpu")
    times = {"imports": time.perf_counter() - T0}
    surf = tets_surface(n) if kind == "tets" else surface(kind, n)
    start = time.perf_counter()
    actors = build(surf, mode, place)
    times["tack"] = time.perf_counter() - start
    start = time.perf_counter()
    window = rr.vtkRenderWindow(size=(W, H), off_screen_rendering=True)
    renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
    window.AddRenderer(renderer)
    window.Initialize()
    times["device"] = time.perf_counter() - start
    for a in actors:
        renderer.AddActor(a)
    if os.environ.get("SAMPLE_FILE") and os.environ.get("SAMPLE_AT", "first") == "first":
        # macOS's sampler on this process, from just before the first render.
        import subprocess
        subprocess.Popen(["sample", str(os.getpid()), os.environ.get("SAMPLE_SECONDS", "3"),
                          "1", "-file", os.environ["SAMPLE_FILE"]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(float(os.environ.get("SAMPLE_LEAD", "0.5")))
    start = time.perf_counter()
    frame(window, renderer)
    times["first"] = time.perf_counter() - start
    if os.environ.get("SAMPLE_FILE"):
        print(f"first render ended {time.perf_counter() - start:.3f} s after it began", flush=True)
    if os.environ.get("SAMPLE_FILE") and os.environ.get("SAMPLE_AT") == "second":
        import subprocess
        subprocess.Popen(["sample", str(os.getpid()), os.environ.get("SAMPLE_SECONDS", "3"),
                          "1", "-file", os.environ["SAMPLE_FILE"]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(float(os.environ.get("SAMPLE_LEAD", "0.5")))
    start = time.perf_counter()
    frame(window, renderer, 1)
    times["second"] = time.perf_counter() - start
    for k in (3, 4, 5, 6):
        start = time.perf_counter()
        frame(window, renderer, k - 1)
        times[f"frame{k}"] = time.perf_counter() - start
    # A timestep's worth of new values (the same numbers, marked modified): plain and
    # slang re-map the point scalars through the lookup table on the CPU and upload the
    # colours; hooks upload the field's words.
    steps = []
    for k in range(5):
        for a in actors:
            pd = a.GetMapper().GetInput().GetPointData()
            if pd.GetScalars() is not None:
                pd.GetScalars().Modified()
            sp = a.GetShaderProperty()
            if hasattr(sp, "GetSlangBuffer") and sp.GetSlangBuffer("hooks.field") is not None:
                sp.GetSlangBuffer("hooks.field").Modified()
        start = time.perf_counter()
        frame(window, renderer, 1)
        steps.append(time.perf_counter() - start)
    times["step"] = float(np.median(steps))
    # Another actor of the same kind in the same window: the Slang and pipeline caches.
    more = build(surf, mode, place)
    for a in actors:
        renderer.RemoveActor(a)
    for a in more:
        renderer.AddActor(a)
    start = time.perf_counter()
    frame(window, renderer, 2)
    times["new_actor"] = time.perf_counter() - start
    tris = sum(a.GetMapper().GetInput().GetNumberOfCells() for a in more)
    print(f"{kind} {n} {mode} place={int(place)} triangles={tris} "
          + " ".join(f"{k}={1e3 * v:.0f}" for k, v in times.items()), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2] if len(a) > 2 else "hooks", bool(int(a[3])) if len(a) > 3 else False)
