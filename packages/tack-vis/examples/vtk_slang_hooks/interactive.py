# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tack's surface in an interactive VTK WebGPU window, through the Slang hooks.

The instanced form with the placing pass by default (``INSTANCED=0`` for the template
polydata, ``COMPUTED=0`` for the vertex hook every frame). Mouse: VTK's trackball camera.
Keys:

- ``g``: a geometry step (the surface deformed further; Tack's geometry buffer, then
  the placing pass);
- ``f``: a field step (the field mirrored in its range; Tack's field buffer);
- ``r``: reset the camera; ``q``: quit.

    python interactive.py [straight|curved|tets] [n]
"""

import importlib.util
import os
import sys
import time

os.environ.setdefault("INSTANCED", "1")

import demo  # noqa: E402
from tack_hooks import rr, surface_actors, update_surface  # noqa: E402
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera  # noqa: E402

import tack  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def geometry_steps():
    spec = importlib.util.spec_from_file_location("hooks_geometry_step",
                                                  os.path.join(HERE, "geometry_step.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.step_surface


def main(kind="curved", n=32):
    tack.set_default_device("cpu")
    step_surface = geometry_steps()
    surf = demo.tets_surface(n) if kind == "tets" else step_surface(kind, n, "deform", 0)
    groups, (lo, hi) = surface_actors(surf, "f")
    renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
    for g in groups:
        renderer.AddActor(g.actor)
    window = rr.vtkRenderWindow(size=(1600, 1000), window_name=f"Tack {kind} {n}")
    window.AddRenderer(renderer)
    interactor = rr.vtkRenderWindowInteractor(render_window=window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    renderer.ResetCamera()
    renderer.active_camera.Elevation(25)
    renderer.active_camera.Azimuth(30)
    renderer.ResetCameraClippingRange()

    state = {"step": 0, "surf": surf}
    mode = ("instanced + placing pass" if groups[0].computed else
            "instanced, vertex hook" if groups[0].instanced else "template polydata")
    print(f"{kind} {n}: {surf.num_cells} faces, {sum(g.triangles for g in groups)} triangles "
          f"({mode}). Keys: g geometry step, f field step, r reset camera, q quit.",
          flush=True)

    def on_key(caller, event):
        key = caller.GetKeySym()
        start = time.perf_counter()
        if key == "g" and kind != "tets":
            state["step"] += 1
            state["surf"] = step_surface(kind, n, "deform", state["step"])
            update_surface(groups, state["surf"], "f")
            what = f"geometry step {state['step']}"
        elif key == "f":
            s = state["surf"]
            old = s.fields["f"]
            s.fields["f"] = demo.td.Field(old.space, tack.array((lo + hi) - old.values.numpy()))
            for g in groups:
                g.update_field(s, "f")
            what = "field step"
        else:
            return
        tack_ms = 1000 * (time.perf_counter() - start)
        start = time.perf_counter()
        window.Render()
        window.WaitForCompletion()
        print(f"{what}: Tack {tack_ms:.0f} ms, frame {1000 * (time.perf_counter() - start):.0f} ms",
              flush=True)

    interactor.AddObserver("KeyPressEvent", on_key)
    window.Render()
    interactor.Start()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "curved", int(a[1]) if len(a) > 1 else 32)
