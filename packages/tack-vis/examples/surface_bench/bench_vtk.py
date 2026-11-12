# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VTK frame times on the benchmark's surfaces, through whatever rendering backend VTK
was built with and chooses (``VTK_FACTORY_PREFER=RenderingBackend=WebGPU`` for the
WebGPU one):

- ``cellgrid``: Tack's surface as a ``vtkCellGrid`` of 2D cells (``dataset_to_vtk(...,
  cell_grid=True)``), drawn by ``vtkCellGridMapper`` (GL: tessellation shaders, the
  field per pixel);
- ``polygons``: the same surface as triangles at the WebGPU renderers' template level
  (1 a straight face's side, 8 a curved one's), the field at their vertices and
  interpolated, drawn by ``vtkPolyDataMapper``.

Offscreen at 1920 x 1080, the camera orbiting; each frame waits for the GPU
(``WaitForCompletion``). Prints setup, first frame and the median steady frame, in ms.

    python bench_vtk.py [cellgrid|polygons] [straight|curved] [n ...]
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slang_surfaces"))
sys.path.insert(0, os.path.join(HERE, "..", "webgpu_surfaces"))

import importlib

import vtkmodules.vtkCommonDataModel  # noqa: F401
import vtkmodules.vtkRenderingCore as rr

# The backend's module registers its window and mappers (BENCH_VTK_BACKEND: OpenGL2 or
# WebGPU).
importlib.import_module("vtkmodules.vtkRendering" + os.environ.get("BENCH_VTK_BACKEND", "OpenGL2"))
from scenes import camera, surface

import tack

W, H = 1920, 1080


def _template(kind, level):
    """A reference face subdivided ``level`` times a side as a triangle list of (u, v),
    as the WebGPU renderers' (not imported: that would load wgpu-native beside VTK's
    Dawn, both exporting the wgpu* symbols)."""
    out = []
    for j in range(level):
        for i in range(level if kind == 1 else level - j):
            a, b, c, d = (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
            out += [a, b, c]
            if kind == 1 or i + j < level - 1:
                out += [b, d, c]
    return np.array(out, np.float32) / level


def polygons(surf, level):
    """``surf`` as a triangle soup at the templates' points, the field evaluated there by
    Tack (the same kernels the Slang shaders are made of), as a vtkPolyData."""
    from shaders import _geometry_at, _value_at
    from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    field = surf.fields["f"]
    curved = surf.geometry.space.order > 1
    xs, fs = [], []
    for g in surf.launch_groups("cells", [field]):
        if not g.count:
            continue
        t = _template(1 if g.shape.NUM_POINTS == 4 else 0, level if curved else 1)
        ids = tack.array(np.repeat(np.arange(g.count, dtype=np.int32), len(t)))
        pcs = np.zeros((g.count * len(t), 3), np.float32)
        pcs[:, :2] = np.tile(t, (g.count, 1))
        pcs = tack.array(pcs, tack.vec3())
        geo = tack.empty((3 * ids.shape[0],), tack.vec3())
        _geometry_at(surf.domain_view("cells", g), ids, pcs, geo)
        val = tack.empty((ids.shape[0],), tack.f32)
        _value_at(field.view(g), ids, pcs, val)
        xs.append(geo.numpy().reshape(-1, 3, 3)[:, 0])
        fs.append(val.numpy())
    x = np.concatenate(xs).astype(np.float32)
    n = len(x)
    poly = vtkPolyData()
    poly.points = vtkPoints(data=vtkFloatArray(x))
    poly.polys = vtkCellArray(offsets=np.arange(0, n + 1, 3), connectivity=np.arange(n))
    scalars = vtkFloatArray(np.concatenate(fs))
    scalars.SetName("f")
    poly.GetPointData().SetScalars(scalars)
    return poly, n // 3


def setup(mode, surf, level, window=None):
    renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
    values = surf.fields["f"].values.numpy().reshape(-1)
    lo, hi = float(values.min()), float(values.max())
    if mode == "cellgrid":
        from vtkmodules.vtkRenderingCellGrid import vtkRenderingCellGrid

        from tack.interop.vtk import dataset_to_vtk

        vtkRenderingCellGrid.RegisterCellsAndResponders()
        grid = dataset_to_vtk(surf, cell_grid=True)
        mapper = rr.vtkCellGridMapper()
        mapper.SetInputDataObject(grid)
        mapper.SetScalarMode(rr.VTK_SCALAR_MODE_USE_CELL_FIELD_DATA)
        mapper.SetArrayName("f")
        mapper.SetArrayComponent(0)
        triangles = None
    else:
        poly, triangles = polygons(surf, level)
        mapper = rr.vtkPolyDataMapper(input_data=poly, scalar_range=(lo, hi))
        mapper.SetInterpolateScalarsBeforeMapping(True)
    renderer.AddActor(rr.vtkActor(mapper=mapper))
    window = window or _window()
    window.AddRenderer(renderer)
    return window, renderer, triangles


def _window():
    window = rr.vtkRenderWindow(size=(W, H), off_screen_rendering=True)
    _webgpu_backend(window)
    return window


def first(mode, surf, level=8):
    """A fresh process's first picture: the window and its device (``Initialize``), the
    mapper's input and actor, and the first frame (rendered, waited for and read back),
    in ms. ``polygons`` includes Tack evaluating the triangles' points and values."""
    start = time.perf_counter()
    window = _window()
    window.Initialize()
    times = {"device": time.perf_counter() - start}
    start = time.perf_counter()
    window, renderer, _ = setup(mode, surf, level, window)
    times["setup"] = time.perf_counter() - start
    start = time.perf_counter()
    frame(window, renderer, 0, 30)
    grab = rr.vtkWindowToImageFilter(input=window)
    grab.Update()
    times["first"] = time.perf_counter() - start
    return {k: 1e3 * v for k, v in times.items()}


def _webgpu_backend(window):
    """A WebGPU window on Dawn's backend BENCH_WEBGPU_BACKEND (Metal, Vulkan: SwiftShader
    when the Vulkan driver is SwiftShader's)."""
    backend = os.environ.get("BENCH_WEBGPU_BACKEND")
    if not backend or not hasattr(window, "SetWGPUConfiguration"):
        return
    from vtkmodules.vtkRenderingWebGPU import vtkWebGPUConfiguration

    config = vtkWebGPUConfiguration()
    config.SetBackend(getattr(vtkWebGPUConfiguration.BackendType, backend))
    window.SetWGPUConfiguration(config)


def frame(window, renderer, k, frames):
    eye, centre = camera(k, frames)
    cam = renderer.active_camera
    cam.position, cam.focal_point, cam.view_up = tuple(eye), tuple(centre), (0.0, 0.0, 1.0)
    cam.view_angle = 30.0
    cam.SetClippingRange(0.5, 6.0)
    window.Render()
    window.WaitForCompletion()


def main(mode="cellgrid", kind="straight", sizes=(16, 32, 64, 128), frames=30, level=8):
    tack.set_default_device("cpu")
    probe_window = rr.vtkRenderWindow(off_screen_rendering=True)
    _webgpu_backend(probe_window)
    used = ""
    if hasattr(probe_window, "GetWGPUConfiguration") and probe_window.GetWGPUConfiguration():
        probe_window.Initialize()
        config = probe_window.GetWGPUConfiguration()
        used = f" ({config.GetBackendInUseAsString()}: {config.GetAdapterInfo() if hasattr(config, 'GetAdapterInfo') else ''})"
    print(f"# VTK {mode} through {probe_window.GetClassName()}{used}, {W}x{H}")
    print("renderer kind n faces triangles setup_ms first_ms frame_ms")
    for n in sizes:
        surf = surface(kind, n)
        start = time.perf_counter()
        window, renderer, triangles = setup(mode, surf, level)
        made = time.perf_counter() - start
        start = time.perf_counter()
        frame(window, renderer, 0, frames)
        first = time.perf_counter() - start
        times = []
        for k in range(1, frames + 1):
            start = time.perf_counter()
            frame(window, renderer, k, frames)
            times.append(time.perf_counter() - start)
        print(f"{mode} {kind} {n} {surf.num_cells} {triangles if triangles else '-'} "
              f"{1e3 * made:.0f} {1e3 * first:.1f} {1e3 * float(np.median(times)):.2f}",
              flush=True)
        if os.environ.get("BENCH_PNG"):
            from vtkmodules.vtkIOImage import vtkPNGWriter
            grab = rr.vtkWindowToImageFilter(input=window)
            w = vtkPNGWriter(file_name=f"{os.environ['BENCH_PNG']}/{mode}-{kind}-{n}.png")
            w.SetInputConnection(grab.output_port)
            w.Write()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "cellgrid", args[1] if len(args) > 1 else "straight",
         tuple(int(a) for a in args[2:]) or (16, 32, 64, 128))
