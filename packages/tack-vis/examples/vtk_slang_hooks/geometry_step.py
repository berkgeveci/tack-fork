# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""In situ: a timestep whose geometry changes, after the first frames (shaders compiled,
pipelines made). Each step a new surface arrives (``extract_surface`` of the
simulation's new solid, timed apart), and is drawn by the actors already there:

- ``hooks``: Tack's hook actors (``tack_hooks.update_surface``): Tack exports the group's
  buffers and evaluates positions and normals at the template points, a new polydata
  goes to the mapper, the buffers are refilled (or remade if their size changed);
- ``plain``: VTK's polygons: Tack evaluates positions, normals and the field at the
  template points, a new polydata with the field as point scalars goes to the mapper,
  which maps them through the lookup table.

``deform`` moves the points (the same mesh, bent by a factor that changes each step) and
the field with them; ``remesh`` alternates n and n - 4 (new topology, new counts).

Prints ms, medians over the steps: ``extract`` (the simulation's side: solid and
``extract_surface``), Tack's parts, ``frame`` (the first frame after the change:
VTK's uploads, topology, bounds, pipelines if any, the draw), ``steady`` (the next one).

    python geometry_step.py [straight|curved] n [hooks|plain] [deform|remesh] [steps]

``SAMPLE_FILE=path``: macOS's ``sample`` over the third step's frame (``SAMPLE_WHICH=steady``:
the frame after it). ``NO_CULLER=1`` takes the renderer's culler away. ``CHECK=1`` (hooks):
the pixels against Tack's evaluation after the last step (``demo.check``).
"""

import os
import subprocess
import sys
import time

import numpy as np
import tack_hooks  # noqa: F401  (paths)
from tack_hooks import _evaluate, rr, surface_actors, update_surface
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

import tack
import tack.dataset as td
from scenes import camera, indexed_template, solid
from shaders import _value_at

W, H = 1920, 1080


def step_surface(kind, n, mode, step):
    """The simulation's step: its solid (deformed, or remeshed) and the surface."""
    m = n - 4 if mode == "remesh" and step % 2 else n
    data = solid(kind, m)
    geometry = data.geometry
    x = geometry.values.numpy().reshape(-1, 3).copy()
    s = 0.04 * np.sin(0.7 * step + 0.3)
    x[:, 0] *= 1 + s
    x[:, 2] *= 1 - s
    x[:, 1] += s * x[:, 0] * x[:, 2]
    data.geometry = td.Field(geometry.space, tack.array(x, tack.vec3(tack.f64)))
    return td.extract_surface(data, fields=["f"])


class PlainActors:
    """VTK's polygons: positions, normals and the field at the template points."""

    def __init__(self, surf, lo, hi, level=8):
        self.level, self.lo, self.hi = level, lo, hi
        self.actors = []
        for _ in self._groups(surf):
            mapper = rr.vtkPolyDataMapper(scalar_range=(lo, hi))
            self.actors.append(rr.vtkActor(mapper=mapper))
        self.set_surface(surf)

    @staticmethod
    def _groups(surf):
        return [g for g in surf.launch_groups("cells", [surf.fields["f"]]) if g.count]

    def set_surface(self, surf):
        timings = {"evaluate": 0.0, "polydata": 0.0}
        curved = surf.geometry.space.order > 1
        for actor, g in zip(self.actors, self._groups(surf), strict=True):
            t = time.perf_counter()
            points, tris = indexed_template(g.shape.NUM_POINTS == 4, self.level if curved else 1)
            m = len(points)
            x, normals = _evaluate(surf.domain_view("cells", g), g.count, points)
            ids = tack.array(np.repeat(np.arange(g.count, dtype=np.int32), m))
            pcs = np.zeros((g.count * m, 3), np.float32)
            pcs[:, :2] = np.tile(points, (g.count, 1))
            val = tack.empty((g.count * m,), tack.f32)
            _value_at(surf.fields["f"].view(g), ids, tack.array(pcs, tack.vec3()), val)
            values = val.numpy()
            timings["evaluate"] += time.perf_counter() - t
            t = time.perf_counter()
            conn = (tris[None, :] + m * np.arange(g.count)[:, None]).reshape(-1).astype(np.int64)
            poly = vtkPolyData()
            poly.points = vtkPoints(data=vtkFloatArray(x.astype(np.float32)))
            poly.polys = vtkCellArray(offsets=np.arange(0, len(conn) + 1, 3), connectivity=conn)
            nv = vtkFloatArray(normals.astype(np.float32))
            nv.SetName("Normals")
            poly.GetPointData().SetNormals(nv)
            sv = vtkFloatArray(values)
            sv.SetName("f")
            poly.GetPointData().SetScalars(sv)
            actor.GetMapper().SetInputData(poly)
            timings["polydata"] += time.perf_counter() - t
        return timings


def frame(window, renderer, k):
    eye, centre = camera(k, 30)
    cam = renderer.active_camera
    cam.position, cam.focal_point, cam.view_up = tuple(eye), tuple(centre), (0.0, 0.0, 1.0)
    cam.view_angle = 30.0
    cam.SetClippingRange(0.5, 6.0)
    window.Render()
    window.WaitForCompletion()


def main(kind, n, how, mode, steps):
    tack.set_default_device("cpu")
    surf = step_surface(kind, n, mode, 0)
    values = surf.fields["f"].values.numpy().reshape(-1)
    lo, hi = float(values.min()), float(values.max())
    if how == "hooks":
        group_actors, (lo, hi) = surface_actors(surf, "f")
        actors = [a.actor for a in group_actors]
    else:
        plain = PlainActors(surf, lo, hi)
        actors = plain.actors
    window = rr.vtkRenderWindow(size=(W, H), off_screen_rendering=True)
    renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
    window.AddRenderer(renderer)
    if os.environ.get("NO_CULLER"):
        # vtkFrustumCoverageCuller asks each actor's bounds, which the polydata mapper
        # answers by walking every cell (GetCellsBounds) after each change.
        renderer.GetCullers().RemoveAllItems()
    for a in actors:
        renderer.AddActor(a)
    for k in range(4):                      # compiles, pipelines, the second-frame bounds
        frame(window, renderer, k)

    rows = []
    for step in range(1, steps + 1):
        t = time.perf_counter()
        new = step_surface(kind, n, mode, step)
        extract = time.perf_counter() - t
        t = time.perf_counter()
        if how == "hooks":
            parts = update_surface(group_actors, new, "f")
        else:
            parts = plain.set_surface(new)
        tack_total = time.perf_counter() - t
        which = os.environ.get("SAMPLE_WHICH", "frame")
        sampling = os.environ.get("SAMPLE_FILE") and step == 3

        def start_sampler():
            subprocess.Popen(["sample", str(os.getpid()), "1.5", "1", "-file",
                              os.environ["SAMPLE_FILE"]],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)

        if sampling and which == "frame":
            start_sampler()
        t = time.perf_counter()
        frame(window, renderer, step)
        first = time.perf_counter() - t
        if sampling:
            time.sleep(1.0)
            if which == "steady":
                start_sampler()
        t = time.perf_counter()
        frame(window, renderer, step)
        steady = time.perf_counter() - t
        rows.append({"extract": extract, **parts, "tack": tack_total, "frame": first,
                     "steady": steady})
    if how == "hooks" and os.environ.get("CHECK"):
        import importlib.util

        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("hooks_demo", os.path.join(here, "demo.py"))
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)       # this directory's, not slang_surfaces' demo
        check = demo.check

        pixels, faces, worst, median = check(new, group_actors, lo, hi, window, renderer)
        print(f"  check after the last step: {pixels} pixels on {faces} faces, max "
              f"{worst:.2e}, median {median:.2e}", flush=True)
    tris = sum(a.GetMapper().GetInput().GetNumberOfCells() for a in actors)
    med = {k: 1e3 * float(np.median([r[k] for r in rows[1:] or rows])) for k in rows[0]}
    print(f"{kind} {n} {how} {mode} ({tris} triangles, {steps} steps): "
          + " ".join(f"{k}={v:.0f}" for k, v in med.items())
          + f"   first step: frame={1e3 * rows[0]['frame']:.0f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2] if len(a) > 2 else "hooks", a[3] if len(a) > 3 else "deform",
         int(a[4]) if len(a) > 4 else 6)
