# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Surface LIC on a slice of ``disk_out_ref.ex2`` that follows a plane widget.

A VTK window (OpenGL) holds a ``vtkImplicitPlaneWidget2`` placed on the dataset. Each
move of the plane cuts the volume again in Tack and evaluates ``V`` and ``Temp`` per
pixel with the hexahedra's own basis (``exact``: the cut cells in one buffer, the
generated Slang compiled once). Each frame renders the LIC with the window's camera in
WebGPU (``lic.LIC``, wgpu-py) and shows it as the window's background, the widget drawn
over it.

Mouse: the widget (drag its normal arrow to tilt the plane, its edges or the normal
line to move it), and VTK's trackball camera elsewhere. Keys: ``e`` toggles enhanced
LIC (two passes), ``t`` prints the last timings, ``q`` quits.

    python interactive_disk.py [path/to/disk_out_ref.ex2]
"""

import os
import sys
import tempfile
import time

import numpy as np

import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkRenderingCore as rr
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPlane
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import (vtkImplicitPlaneRepresentation,
                                              vtkImplicitPlaneWidget2)

import disk
import exact
import lic
import scene
import tack
import tack.dataset as td


class SliceLIC:
    """The volume, the current slice's exact inputs, and the LIC renderer."""

    def __init__(self, data, out_dir, size):
        self.data = data
        self.locator = td.CellLocator(data)
        temp = data.fields["Temp"].values.numpy().reshape(-1)
        self.range = (float(temp.min()), float(temp.max()))
        self.timings = {}
        self.renderer = None
        self.out_dir, self.size = out_dir, size

    def cut(self, origin, normal):
        """The slice through ``origin`` with ``normal``, into the renderer."""
        t0 = time.perf_counter()
        points, triangles, _, unit, _ = scene.slice_triangles(
            self.data, tuple(origin), tuple(normal), vector="V", scalar="Temp")
        t1 = time.perf_counter()
        k = len(triangles)
        if k:
            cells, corners, pcs, _ = exact.slice_cells(self.data, points, triangles,
                                                       locator=self.locator)
            cut, cells = exact.cut_cells(self.data, cells)
            source, pool = exact.volume_functions(cut, "V", "Temp")
        t2 = time.perf_counter()
        if self.renderer is None:
            if not k:
                raise ValueError("the first plane cuts nothing")
            self.renderer = lic.LIC(corners.reshape(-1, 3), np.arange(3 * k, dtype=np.uint32),
                                    None, self.out_dir, size=self.size, supersample=1,
                                    exact=dict(source=source, pool=pool, cells=cells, pcs=pcs))
        elif k:
            self.renderer.set_slice(corners.reshape(-1, 3), np.arange(3 * k, dtype=np.uint32),
                                    None, exact=dict(source=source, pool=pool, cells=cells,
                                                     pcs=pcs))
        else:
            self.renderer.set_slice(np.zeros((0, 3)), np.zeros(0, np.uint32), None,
                                    exact=dict(source=self.renderer.source, pool=[0], cells=[0],
                                               pcs=[0.0]))
        t3 = time.perf_counter()
        self.timings.update(slice=1000 * (t1 - t0), cells=1000 * (t2 - t1),
                            upload=1000 * (t3 - t2), triangles=k,
                            cut_cells=int(len(np.unique(cells))) if k else 0)

    def picture(self, camera, width, height, enhanced):
        """The LIC as the window's camera sees it: ``(h, w, 4)`` u8, ``w`` a multiple of
        64 (stretched onto the window, the projection is the window's)."""
        w64 = max(64, (width + 63) // 64 * 64)
        if self.renderer.size != (w64, height):
            self.renderer.resize((w64, height))
        near, far = camera.GetClippingRange()
        result = self.renderer.render(
            eye=camera.GetPosition(), target=camera.GetFocalPoint(), up=camera.GetViewUp(),
            field_range=self.range, by_scalar=True, fovy=camera.GetViewAngle(),
            near_far=(near, far), aspect=width / height, enhanced=enhanced,
            picture_only=True)
        self.timings.update(surface=result["timings"]["surface"], lic=result["timings"]["lic"])
        return result["picture"]


def main(path=disk.DISK):
    tack.set_default_device("cpu")
    data, bounds = disk.read(path)
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    centre = 0.5 * (lo + hi)

    window = rr.vtkRenderWindow(size=(1600, 1000), window_name="Tack surface LIC: disk_out_ref")
    renderer = rr.vtkRenderer()
    window.AddRenderer(renderer)
    interactor = rr.vtkRenderWindowInteractor(render_window=window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

    out_dir = tempfile.mkdtemp(prefix="tack-lic-")
    slicer = SliceLIC(data, out_dir, size=(1600, 1000))
    slicer.cut(centre, (1.0, 0.0, 0.0))
    print(f"{data.num_cells} hexahedra; first slice {slicer.timings['triangles']} triangles in "
          f"{slicer.timings['cut_cells']} cells", flush=True)

    # The LIC as the background, updated before each render.
    image = vtkImageData()
    pixels = vtkUnsignedCharArray()
    pixels.SetNumberOfComponents(4)
    image.GetPointData().SetScalars(pixels)
    texture = rr.vtkTexture()
    texture.SetInputData(image)
    texture.InterpolateOn()
    renderer.TexturedBackgroundOn()
    renderer.SetBackgroundTexture(texture)
    state = {"enhanced": True}

    def before_render(caller, event):
        width, height = window.GetSize()
        if width < 2 or height < 2:
            return
        renderer.ResetCameraClippingRange(bounds)
        rgba = slicer.picture(renderer.GetActiveCamera(), width, height, state["enhanced"])
        h, w, _ = rgba.shape
        if image.GetDimensions()[:2] != (w, h):
            image.SetDimensions(w, h, 1)
            pixels.SetNumberOfTuples(w * h)
        np.asarray(pixels)[:] = rgba[::-1].reshape(-1, 4)
        pixels.Modified()

    window.AddObserver("StartEvent", before_render)

    # The plane widget: the plane's polygon not drawn, so the slice shows.
    representation = vtkImplicitPlaneRepresentation()
    representation.SetPlaceFactor(1.0)
    representation.PlaceWidget(bounds)
    representation.SetOrigin(*centre)
    representation.SetNormal(1.0, 0.0, 0.0)
    representation.DrawPlaneOff()
    widget = vtkImplicitPlaneWidget2()
    widget.SetInteractor(interactor)
    widget.SetRepresentation(representation)
    plane = vtkPlane()

    def moved(caller, event):
        representation.GetPlane(plane)
        slicer.cut(plane.GetOrigin(), plane.GetNormal())

    widget.AddObserver("InteractionEvent", moved)
    widget.On()

    def key(caller, event):
        k = caller.GetKeySym()
        if k == "e":
            state["enhanced"] = not state["enhanced"]
            print(f"enhanced LIC {'on' if state['enhanced'] else 'off'}", flush=True)
            window.Render()
        elif k == "t":
            t = slicer.timings
            print(f"slice: {t['triangles']} triangles, {t['cut_cells']} cells; Tack slice "
                  f"{t['slice']:.0f} ms, cells and buffer {t['cells']:.0f} ms, upload "
                  f"{t['upload']:.0f} ms; frame: surface {t.get('surface', 0):.1f} ms, LIC "
                  f"{t.get('lic', 0):.1f} ms", flush=True)

    interactor.AddObserver("KeyPressEvent", key)

    camera = renderer.GetActiveCamera()
    renderer.ResetCamera(bounds)
    camera.Azimuth(60)
    camera.Elevation(20)
    camera.OrthogonalizeViewUp()
    renderer.ResetCamera(bounds)
    window.Render()
    print("move the plane with the widget; keys: e enhanced LIC, t timings, q quit", flush=True)
    interactor.Start()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else disk.DISK)
