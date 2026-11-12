# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VTK rendering a contour of millions of triangles, offscreen 1920x1080, through the
backend BENCH_VTK_BACKEND (OpenGL2, WebGPU) and Dawn's BENCH_WEBGPU_BACKEND (Metal,
Vulkan: SwiftShader). The wavelet's isosurface (flying edges, with normals), coloured
by a point field. Per size: triangles, first frame, then medians (ms) of:

- orbit: the camera moving, nothing else (interactive);
- step: a new point field (same mesh) uploaded, a frame, read back (in situ).

    python vtk_triangles.py N ...      (the wavelet's extent is -N..N on each axis)
"""

import importlib
import os
import sys
import time

import numpy as np
import vtkmodules.vtkCommonDataModel  # noqa: F401
import vtkmodules.vtkRenderingCore as rr
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource

importlib.import_module("vtkmodules.vtkRendering" + os.environ.get("BENCH_VTK_BACKEND", "OpenGL2"))

W, H = 1920, 1080
FRAMES = int(os.environ.get("BENCH_FRAMES", 10))


def window():
    w = rr.vtkRenderWindow(size=(W, H), off_screen_rendering=True)
    backend = os.environ.get("BENCH_WEBGPU_BACKEND")
    if backend and hasattr(w, "SetWGPUConfiguration"):
        from vtkmodules.vtkRenderingWebGPU import vtkWebGPUConfiguration

        config = vtkWebGPUConfiguration()
        config.SetBackend(getattr(vtkWebGPUConfiguration.BackendType, backend))
        w.SetWGPUConfiguration(config)
    return w


def surface(n):
    source = vtkRTAnalyticSource(whole_extent=(-n, n, -n, n, -n, n))
    contour = vtkFlyingEdges3D(compute_normals=True, compute_scalars=False)
    contour.SetValue(0, 160.0)
    poly = (source >> contour)()
    x = np.asarray(poly.GetPoints().GetData())
    field = vtkFloatArray(np.sin(x[:, 0] / n * 6) * np.cos(x[:, 1] / n * 5)
                          + 0.5 * np.sin(x[:, 2] / n * 4))
    field.SetName("f")
    poly.GetPointData().SetScalars(field)
    return poly


def main(*sizes):
    w = window()
    w.Initialize()
    if hasattr(w, "GetWGPUConfiguration") and w.GetWGPUConfiguration():
        c = w.GetWGPUConfiguration()
        print(f"# {w.GetClassName()} {c.GetBackendInUseAsString()}")
    else:
        print(f"# {w.GetClassName()}")
    print("n triangles points first_ms orbit_ms step_ms (update read)")
    for n in map(int, sizes):
        poly = surface(n)
        tris = poly.GetNumberOfPolys()
        renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
        mapper = rr.vtkPolyDataMapper(input_data=poly, scalar_range=(-1.5, 1.5))
        renderer.AddActor(rr.vtkActor(mapper=mapper))
        w = window()
        w.AddRenderer(renderer)
        renderer.ResetCamera()
        cam = renderer.active_camera
        t = time.perf_counter()
        w.Render()
        w.WaitForCompletion()
        first = time.perf_counter() - t
        orbit = []
        for _ in range(FRAMES):
            cam.Azimuth(360.0 / FRAMES)
            t = time.perf_counter()
            w.Render()
            w.WaitForCompletion()
            orbit.append(time.perf_counter() - t)
        grab = rr.vtkWindowToImageFilter(input=w)
        grab.ShouldRerenderOff()
        field = poly.GetPointData().GetScalars()
        values = np.asarray(field)
        base = values.copy()
        step, upd, rd = [], [], []
        for k in range(FRAMES):
            t0 = time.perf_counter()
            values[:] = base * np.cos(0.3 * k)
            field.Modified()
            t1 = time.perf_counter()
            w.Render()
            w.WaitForCompletion()
            t2 = time.perf_counter()
            grab.Modified()
            grab.Update()
            t3 = time.perf_counter()
            step.append(t3 - t0)
            upd.append(t1 - t0)
            rd.append(t3 - t2)
        m = lambda v: 1e3 * float(np.median(v))
        print(f"{n} {tris} {poly.GetNumberOfPoints()} {1e3 * first:.0f} {m(orbit):.1f} "
              f"{m(step):.1f} ({m(upd):.1f} {m(rd):.1f})", flush=True)
        if os.environ.get("BENCH_PNG"):
            from vtkmodules.vtkIOImage import vtkPNGWriter

            png = vtkPNGWriter(file_name=f"{os.environ['BENCH_PNG']}-{n}.png")
            png.SetInputConnection(grab.output_port)
            png.Write()
        del w


if __name__ == "__main__":
    main(*sys.argv[1:])
