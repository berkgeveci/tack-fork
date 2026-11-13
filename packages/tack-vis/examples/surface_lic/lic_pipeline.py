# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The surface LIC slice as a VTK pipeline: the mapper decides how the data is drawn.

    vtkExodusIIReader ─► TackSlice (Python algorithm: Tack cuts, packs, generates Slang)
                      ─► LICSliceMapper (a WebGPU mapper written in Python: its Slang
                         customization from the input) ─► actor
    renderer + vtkWebGPUSurfaceLIC, a plane widget

Nothing outside the pipeline touches the actor or its shader property: moving the plane
sets the filter's plane. ``lic_server.py``'s serving (the same page) on this scene, or with
``--check`` a desktop render compared with ``vtk_lic.py``'s (the push version) and the
orientation check.

    python lic_pipeline.py [--port 8766] [--refine N] [--check]
"""

import argparse
import os
import time

import numpy as np

import vtkmodules.vtkRenderingCore as rr
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import (vtkImplicitPlaneRepresentation,
                                              vtkImplicitPlaneWidget2)
from vtkmodules.vtkIOExodus import vtkExodusIIReader
from vtkmodules.vtkRenderingWebGPU import vtkWebGPUSurfaceLIC
from vtkmodules.vtkRenderingWebGPUPython import vtkPythonWebGPUMapper
from vtkmodules.vtkSerializationManager import vtkObjectManager

import disk
import lic_server
import tack
from slice_mapper import LICSliceMapper
from tack_slice import TackSlice


class PipelineScene(lic_server.Scene):
    """``lic_server.Scene``'s interface (update, plane, new_client) on a pipeline."""

    def __init__(self, refine=1, record=True, background=(0.32, 0.34, 0.43)):
        tack.set_default_device("cpu")
        self.reader = vtkExodusIIReader(file_name=disk.DISK)
        self.reader.UpdateInformation()
        self.reader.SetAllArrayStatus(vtkExodusIIReader.NODAL, 1)
        self.reader.Update()
        grid = self.reader.GetOutput().GetBlock(0).GetBlock(0)
        bounds = grid.GetBounds()
        lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
        self.centre, span = 0.5 * (lo + hi), float(np.max(hi - lo))
        temp = grid.GetPointData().GetArray("Temp").GetRange()

        self.slicer = TackSlice(refine=refine)
        self.slicer.SetInputConnection(self.reader.GetOutputPort())
        self.slicer.SetPlane(self.centre, (1.0, 0.0, 0.0))
        self.mapper = LICSliceMapper()
        self.mapper.SetInputConnection(self.slicer.GetOutputPort())
        self.mapper.SetScalarRange(*temp)
        self.mapper.slang.SetRecordSlangArtifacts(record)
        self.actor = rr.vtkActor(mapper=self.mapper)

        self.renderer = rr.vtkRenderer(background=background)
        self.renderer.AddActor(self.actor)
        self.lic = vtkWebGPUSurfaceLIC()
        self.lic.SetRecordSlangArtifacts(record)
        self.renderer.SetSurfaceLIC(self.lic)
        self.window = rr.vtkRenderWindow(size=(1280, 800), off_screen_rendering=True)
        self.window.AddRenderer(self.renderer)
        self.interactor = rr.vtkRenderWindowInteractor(render_window=self.window)
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

        self.representation = vtkImplicitPlaneRepresentation()
        self.representation.SetPlaceFactor(1.0)
        self.representation.PlaceWidget(bounds)
        self.representation.SetOrigin(*self.centre)
        self.representation.SetNormal(1.0, 0.0, 0.0)
        self.representation.DrawPlaneOff()
        self.representation.SetHandleSize(15.0)
        self.widget = vtkImplicitPlaneWidget2()
        self.widget.SetInteractor(self.interactor)
        self.widget.SetRepresentation(self.representation)
        self.widget.On()

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*(self.centre + np.array([1.5 * span, 0.4 * span, 0.6 * span])))
        camera.SetFocalPoint(*self.centre)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.SetViewAngle(30.0)
        self.renderer.ResetCameraClippingRange()
        t = time.perf_counter()
        self.window.Render()
        print(f"pipeline: first render {1000 * (time.perf_counter() - t):.0f} ms; slice "
              f"{self.slicer.timings['triangles']} triangles; RequestSlang built "
              f"{self.mapper.preparations}x; artifacts: hook "
              f"{self.mapper.slang.GetNumberOfSlangArtifacts()}, LIC "
              f"{self.lic.GetNumberOfSlangArtifacts()}", flush=True)

        self.manager = vtkObjectManager()
        self.manager.Initialize()
        # The Python mapper goes as a plain WebGPU mapper (the browser has no Python).
        vtkPythonWebGPUMapper.RegisterSerializationHandlers(self.manager.GetSerializer())
        self.window_id = self.manager.RegisterObject(self.window)
        self.widget_id = self.manager.RegisterObject(self.widget)
        self.new_client()

    def plane(self, origin, normal):
        t0 = time.perf_counter()
        self.slicer.SetPlane(origin, normal)
        self.representation.SetOrigin(*origin)
        self.representation.SetNormal(*normal)
        self.window.Render()            # the pipeline updates; RequestSlang runs
        update = self.update()
        update.update(triangles=self.slicer.timings["triangles"],
                      tack_ms=self.slicer.timings["tack_ms"],
                      server_ms=1000 * (time.perf_counter() - t0))
        return update


def check():
    """The pipeline's picture against the push version's, and the orientation check."""
    import vtk_lic

    scene = PipelineScene(record=False, background=(1.0, 1.0, 1.0))
    scene.widget.Off()
    scene.window.SetSize(1920, 1080)
    camera = scene.renderer.GetActiveCamera()
    span = 23.0
    camera.SetPosition(*(scene.centre + np.array([1.7 * span, 0.0, 0.0])))
    camera.SetFocalPoint(*scene.centre)
    camera.SetViewUp(0.0, 1.0, 0.0)
    scene.renderer.ResetCameraClippingRange()
    scene.window.Render()
    scene.window.Render()
    picture = vtk_lic.grab(scene.window)

    push = vtk_lic.SliceActor(vtk_lic.disk.read(disk.DISK)[0],
                              scene.mapper.GetScalarRange())
    push.cut(scene.centre, (1.0, 0.0, 0.0))
    renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
    renderer.AddActor(push.actor)
    renderer.SetSurfaceLIC(vtkWebGPUSurfaceLIC())
    window = rr.vtkRenderWindow(size=(1920, 1080), off_screen_rendering=True)
    window.AddRenderer(renderer)
    renderer.SetActiveCamera(camera)
    window.Render()
    window.Render()
    d = np.abs(picture.astype(int) - vtk_lic.grab(window).astype(int)).max(axis=2)
    print(f"pipeline vs push: {100 * float((d > 2).mean()):.3f}% of pixels differ by more "
          f"than 2/255 (max {int(d.max())}); RequestSlang built {scene.mapper.preparations}x",
          flush=True)
    vtk_lic.check(scene.slicer._data, picture, camera, scene.window.GetSize(), scene.centre,
                  np.array([1.0, 0.0, 0.0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--wasm", default=lic_server.WASM)
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    lic_server.serve(PipelineScene(args.refine), args.port, os.path.abspath(args.wasm))


if __name__ == "__main__":
    main()
