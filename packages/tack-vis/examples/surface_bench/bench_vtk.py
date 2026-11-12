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
from scenes import camera, indexed_template, surface

import tack
import tack.dataset as td

W, H = 1920, 1080


def polygons(surf, level):
    """``surf`` as triangles at the templates' points (shared within a face), the field
    evaluated there by Tack (the same kernels the Slang shaders are made of), as a
    vtkPolyData, and its update for a new field of the same space."""
    from shaders import _geometry_at, _value_at
    from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    curved = surf.geometry.space.order > 1
    xs, conn, groups, base = [], [], [], 0
    for g in surf.launch_groups("cells", [surf.fields["f"]]):
        if not g.count:
            continue
        points, tris = indexed_template(g.shape.NUM_POINTS == 4, level if curved else 1)
        m = len(points)
        ids = tack.array(np.repeat(np.arange(g.count, dtype=np.int32), m))
        pcs = np.zeros((g.count * m, 3), np.float32)
        pcs[:, :2] = np.tile(points, (g.count, 1))
        pcs = tack.array(pcs, tack.vec3())
        geo = tack.empty((3 * ids.shape[0],), tack.vec3())
        _geometry_at(surf.domain_view("cells", g), ids, pcs, geo)
        xs.append(geo.numpy().reshape(-1, 3, 3)[:, 0])
        conn.append((tris[None, :] + base + m * np.arange(g.count)[:, None]).reshape(-1))
        groups.append((g, ids, pcs, tack.empty((ids.shape[0],), tack.f32)))
        base += g.count * m
    x = np.concatenate(xs).astype(np.float32)
    conn = np.concatenate(conn).astype(np.int64)
    poly = vtkPolyData()
    poly.points = vtkPoints(data=vtkFloatArray(x))
    poly.polys = vtkCellArray(offsets=np.arange(0, len(conn) + 1, 3), connectivity=conn)
    scalars = vtkFloatArray(np.zeros(len(x), np.float32))
    scalars.SetName("f")
    poly.GetPointData().SetScalars(scalars)
    out = np.asarray(scalars)

    def update(field):
        at = 0
        for g, ids, pcs, val in groups:
            _value_at(field.view(g), ids, pcs, val)
            out[at:at + len(val)] = val.numpy()
            at += len(val)
        scalars.Modified()

    update(surf.fields["f"])
    return poly, len(conn) // 3, update


def _cell_grid_update(surf, grid):
    """The cell grid's field array refilled from a new field of the same space: Tack's
    projection to VTK's basis (equispaced, a block a cell) per step, as
    ``dataset_to_vtk`` does."""
    array = grid.GetAttributes("vtkDGQuad").GetArray("f")
    attribute = grid.GetCellAttributeByName("f")
    out = np.asarray(array)

    def update(field):
        space = td.L2(surf, order=field.space.order, basis="equispaced", layout="blocks")
        values = td.project(surf, field, space).values.numpy()
        out[...] = values.reshape(out.shape)
        array.Modified()
        # The attribute caches its range by its own time, not its arrays'.
        attribute.Modified()
        grid.Modified()

    return update


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
        update = _cell_grid_update(surf, grid)
    else:
        poly, triangles, update = polygons(surf, level)
        mapper = rr.vtkPolyDataMapper(input_data=poly, scalar_range=(lo, hi))
        mapper.SetInterpolateScalarsBeforeMapping(True)
    renderer.AddActor(rr.vtkActor(mapper=mapper))
    window = window or _window()
    window.AddRenderer(renderer)
    window.update = update
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
