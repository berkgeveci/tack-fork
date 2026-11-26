# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Vortex shedding behind a cylinder, live in a VTK WebGPU window: the lattice
Boltzmann prototype (``tack.dataset.octree_lbm``) on a quadtree of three
levels steps on the CPU while VTK shows the vorticity and the cells.

By default the mesh adapts (``octree_lbm_adapt.adapt``) every ``--every``
steps: cells refine where the vorticity turns the velocity across them and
merge where the flow is quiet, so the finest cells follow the shed vortices
downstream. The cylinder's surroundings stay at the finest level. ``--fixed``
uses a fixed mesh instead (finest around the cylinder and the near wake, the
middle level over the wake).

Inlet on the left (equilibrium at the free-stream speed), outlet on the right,
periodic top and bottom (a column of cylinders seven diameters apart, which
raises the shedding frequency over a lone cylinder's). The cylinder is a
staircase of solid cells. The Strouhal number is measured live from the
transverse velocity two diameters behind the cylinder.

    cylinder_view.py [--re 100] [--diameter 8] [--steps 8] [--every 16] [--fixed]
                     [--check OUTDIR]

Space pauses, r restarts, m shows or hides the cells, q quits; drag pans,
scroll zooms. Run it on VTK's WebGPU build (``cylinder_view.sh``)."""

import argparse
import os
import time

import numpy as np
import vtkmodules.vtkCommonDataModel  # pythonic constructors
import vtkmodules.vtkRenderingCore as rr
import vtkmodules.vtkRenderingWebGPU  # noqa: F401  (the WebGPU window)
from vortex_view import DIVERGING, Vorticity, cells_polydata
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkTextActor, vtkWindowToImageFilter

import tack
from tack.dataset import octree_lbm as lb
from tack.dataset.octree_lbm_adapt import adapt

U = 0.1                                    # free-stream speed, lattice units of level 0


def text(renderer, x, y, size, centred=False):
    actor = vtkTextActor()
    actor.GetTextProperty().SetColor(0.1, 0.1, 0.1)
    actor.GetTextProperty().SetFontSize(size)
    if centred:
        actor.GetTextProperty().SetJustificationToCentered()
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.GetPositionCoordinate().SetValue(x, y)
    renderer.AddViewProp(actor)
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--diameter", type=float, default=8.0, help="in root cells")
    parser.add_argument("--steps", type=int, default=8, help="level-0 steps a frame")
    parser.add_argument("--every", type=int, default=16, help="level-0 steps between remeshes")
    parser.add_argument("--refine", type=float, default=0.01,
                        help="refine where |vorticity| times the cell's side is above this")
    parser.add_argument("--coarsen", type=float, default=0.005,
                        help="merge where twice that is below this")
    parser.add_argument("--fixed", action="store_true", help="a fixed mesh, no adaptation")
    parser.add_argument("--check", default=None, metavar="OUTDIR")
    parser.add_argument("--view", type=float, nargs=4, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"), help="show only this box")
    args = parser.parse_args()
    tack.set_default_device("cpu")
    d = args.diameter
    nx, ny = int(20 * d), int(7 * d)
    xc, yc = 4.5 * d, ny / 2
    tau = 3 * U * d / args.re + 0.5
    region = (xc - 3 * d, nx - 6)

    def pinned(x, y):
        return np.hypot(x - xc, y - yc) < 1.25 * d

    def refine(x, y, h):
        if args.fixed:
            if h > 0.75:
                return xc - 2.5 * d < x < nx - 6 and abs(y - yc) < 2.75 * d
            return xc - 1.5 * d < x < xc + 3 * d and abs(y - yc) < 1.25 * d
        return bool(np.hypot(x - xc, y - yc) < 1.25 * d + 2 * h)

    def solid(x, y):
        return (x - xc) ** 2 + (y - yc) ** 2 < (d / 2) ** 2

    state = {"paused": False, "ms": [], "remesh_ms": [], "trace": [], "steps": []}

    def start_mesh():
        mesh = lb.lbm_mesh((nx, ny), refine, 2, periodic=(False, True), solid=solid)
        s = lb.LBM2D(mesh, tau, inflow=U)
        c, _ = s.centres()
        fluid = s.role != lb.SOLID
        uy = 0.02 * U * np.exp(-((c[:, 0] - xc - d) ** 2 + (c[:, 1] - yc - 0.3 * d) ** 2) / d ** 2)
        s.initialize(1.0, np.where(fluid, U, 0.0), np.where(fluid, uy, 0.0))
        return s

    begin = time.perf_counter()
    solver = start_mesh()
    print(f"{int(solver.computational().sum())} cells on 3 levels to start, Re {args.re:g}, "
          f"tau {tau:.3f} at level 0; {'fixed' if args.fixed else 'adaptive'} mesh "
          f"({time.perf_counter() - begin:.1f} s); Tack on the CPU")

    window = rr.vtkRenderWindow(size=(1600, 680), off_screen_rendering=bool(args.check))
    window.SetWindowName("Vortex shedding behind a cylinder (Tack LBM)")
    ren = rr.vtkRenderer(background=(1, 1, 1))
    mapper = rr.vtkPolyDataMapper()
    mapper.SetScalarModeToUseCellData()
    mapper.SetLookupTable(DIVERGING)
    mapper.SetScalarRange(-3, 3)
    ren.AddActor(rr.vtkActor(mapper=mapper))
    edges_mapper = rr.vtkPolyDataMapper()
    edges_mapper.ScalarVisibilityOff()
    edges = rr.vtkActor(mapper=edges_mapper)
    edges.GetProperty().SetRepresentationToWireframe()
    edges.GetProperty().SetColor(0.2, 0.2, 0.2)
    edges.GetProperty().SetOpacity(0.3)
    edges.SetVisibility(not args.fixed)
    ren.AddActor(edges)
    disk = vtkRegularPolygonSource(number_of_sides=96, radius=d / 2, center=(xc, yc, 0.01))
    disk_mapper = rr.vtkPolyDataMapper()
    disk_mapper.SetInputConnection(disk.GetOutputPort())
    cylinder = rr.vtkActor(mapper=disk_mapper)
    cylinder.GetProperty().SetColor(0.35, 0.35, 0.38)
    ren.AddActor(cylinder)
    bar = vtkScalarBarActor()
    bar.SetLookupTable(DIVERGING)
    bar.SetNumberOfLabels(3)
    bar.SetOrientationToHorizontal()
    bar.SetPosition(0.3, 0.86)
    bar.SetWidth(0.4)
    bar.SetHeight(0.08)
    bar.SetUnconstrainedFontSize(True)
    bar.GetLabelTextProperty().SetColor(0.1, 0.1, 0.1)
    bar.GetLabelTextProperty().SetFontSize(14)
    bar.GetLabelTextProperty().SetShadow(False)
    bar.GetLabelTextProperty().SetItalic(False)
    bar.GetLabelTextProperty().SetBold(False)
    ren.AddViewProp(bar)
    text(ren, 0.5, 0.955, 18, centred=True).SetInput(
        f"vorticity (free-stream speed / diameter), Re {args.re:g}, "
        + ("fixed mesh, three levels" if args.fixed else "adaptive mesh, three levels"))
    status = text(ren, 0.01, 0.015, 15)
    camera = ren.GetActiveCamera()
    camera.SetParallelProjection(True)
    camera.SetFocalPoint(nx / 2, 0.59 * ny, 0)
    camera.SetPosition(nx / 2, 0.59 * ny, 3 * nx)
    camera.SetParallelScale(0.62 * ny)
    if args.view:
        x0, y0, x1, y1 = args.view
        camera.SetFocalPoint((x0 + x1) / 2, (y0 + y1) / 2 + 0.1 * (y1 - y0), 0)
        camera.SetPosition((x0 + x1) / 2, (y0 + y1) / 2 + 0.1 * (y1 - y0), 3 * nx)
        camera.SetParallelScale(0.62 * (y1 - y0))
    window.AddRenderer(ren)

    def attach(s):
        """Show solver ``s``: its cells, their vorticity, the probe cell."""
        poly, values = cells_polydata(s)
        mapper.SetInputData(poly)
        edges_mapper.SetInputData(poly)
        c, _ = s.centres()
        state.update(solver=s, poly=poly, values=values, vorticity=Vorticity(s),
                     probe=int(np.argmin(np.where(s.computational(),
                                                  np.hypot(c[:, 0] - xc - 2 * d, c[:, 1] - yc),
                                                  np.inf))))

    def strouhal():
        p = np.array(state["trace"])
        up = np.flatnonzero((p[:-1] < 0) & (p[1:] >= 0)) if len(p) > 8 else []
        if len(up) < 3:
            return None
        steps = np.array(state["steps"])
        return d / (U * np.mean(np.diff(steps[up[-3:]])))

    def show():
        s = state["solver"]
        _, ux, uy = s.moments()
        state["trace"].append(uy[state["probe"]])
        state["steps"].append(s.steps)
        np.asarray(state["values"])[:] = (state["vorticity"](ux, uy) * d / U).astype(np.float32)
        state["values"].Modified()
        state["poly"].Modified()
        st = strouhal()
        remesh = (f"   remesh {np.median(state['remesh_ms'][-10:]):.0f} ms every {args.every} steps"
                  if state["remesh_ms"] else "")
        status.SetInput(f"step {s.steps}   {int(s.computational().sum())} cells   Strouhal "
                        f"{'...' if st is None else f'{st:.3f}'}   "
                        f"{np.median(state['ms'][-30:]) if state['ms'] else 0:.0f} ms a frame "
                        f"({args.steps} steps){remesh}" + ("   paused" if state["paused"] else ""))

    def advance():
        begin = time.perf_counter()
        s = state["solver"]
        s.step(args.steps)
        if not args.fixed and s.steps % args.every < args.steps:
            t = time.perf_counter()
            s, _ = adapt(s, args.refine, args.coarsen, finest=2, pinned=pinned, region=region)
            state["remesh_ms"].append(1000 * (time.perf_counter() - t))
            attach(s)
        show()
        state["ms"].append(1000 * (time.perf_counter() - begin))

    def restart():
        state["trace"], state["steps"] = [], []
        attach(start_mesh())

    restart()
    show()
    if args.check:
        os.makedirs(args.check, exist_ok=True)
        while state["solver"].steps < 6000:
            advance()
        window.Render()
        grab = vtkWindowToImageFilter()
        grab.SetInput(window)
        grab.Update()
        writer = vtkPNGWriter()
        name = ("cylinder_view_fixed" if args.fixed else "cylinder_view_adaptive") + \
            ("_zoom" if args.view else "") + ".png"
        writer.SetFileName(os.path.join(args.check, name))
        writer.SetInputConnection(grab.GetOutputPort())
        writer.Write()
        s = state["solver"]
        print(f"after {s.steps} steps: {int(s.computational().sum())} cells; a frame "
              f"{np.median(state['ms']):.1f} ms ({args.steps} steps"
              + (f", a remesh {np.median(state['remesh_ms']):.1f} ms" if state["remesh_ms"] else "")
              + f"); Strouhal {strouhal()}")
        return

    interactor = rr.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtkInteractorStyleImage())

    def on_timer(*_):
        if not state["paused"]:
            advance()
            window.Render()

    def on_key(caller, _):
        key = caller.GetKeySym()
        if key == "space":
            state["paused"] = not state["paused"]
        elif key in ("r", "R"):
            restart()
        elif key in ("m", "M"):
            edges.SetVisibility(not edges.GetVisibility())
        else:
            return
        show()
        window.Render()

    interactor.AddObserver("TimerEvent", on_timer)
    interactor.AddObserver("KeyPressEvent", on_key)
    window.Render()
    interactor.Initialize()
    interactor.CreateRepeatingTimer(10)
    print("space pauses, r restarts, m shows or hides the cells, q quits; drag pans, scroll zooms")
    interactor.Start()


if __name__ == "__main__":
    main()
