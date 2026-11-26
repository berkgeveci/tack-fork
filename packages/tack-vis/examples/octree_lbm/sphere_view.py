# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Flow past a sphere in 3D, live in a VTK WebGPU window: the lattice Boltzmann
prototype in D3Q27 (``tack.dataset.octree_lbm``) on an adaptive octree of three
levels, the finest around the sphere and wherever the wake's vorticity is,
steps on the CPU while VTK shows

- above, the wake: a surface of constant vorticity magnitude, contoured on the
  solver's own octree by the study's octree contour (``octree_contour``),
  ghosts and all (filled from their parents before every frame, so the
  surface does not change with where the frame falls in the solver's cycle);
- below, the mid-plane: the cells it cuts, coloured by vorticity magnitude,
  their edges showing the levels as they follow the wake.

Re 300 (a sphere sheds there; below about 210 its wake is steady and
symmetric), regularized BGK (level 0's ``tau`` is 0.508), an inlet at the low-x
side, a pressure outlet with a sponge before it at the high-x side, periodic
across. The wake first turns oblique, then starts to shed, after about 2000
steps; ``--start`` runs that many steps before the window opens.

    sphere_view.py [--re 300] [--diameter 8] [--steps 4] [--every 16] [--start 0]
                   [--level 1.5] [--check OUTDIR]

Space pauses, r restarts, m shows or hides the cells' edges, q quits; drag
rotates the wake (the plane below has its own camera). Run it on VTK's WebGPU
build (``sphere_view.sh``)."""

import argparse
import os
import time

import numpy as np
import vtkmodules.vtkCommonDataModel  # pythonic constructors
import vtkmodules.vtkRenderingCore as rr
import vtkmodules.vtkRenderingWebGPU  # noqa: F401  (the WebGPU window)
from vortex_view import SEQUENTIAL
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersSources import vtkOutlineSource, vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkTextActor, vtkWindowToImageFilter

import tack
from tack.dataset import octree_contour
from tack.dataset import octree_lbm as lb
from tack.dataset.octree_lbm_adapt import _vorticity, adapt

U = 0.1                                    # free-stream speed, lattice units of level 0


def text(renderer, x, y, size):
    actor = vtkTextActor()
    actor.GetTextProperty().SetColor(0.1, 0.1, 0.1)
    actor.GetTextProperty().SetFontSize(size)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.GetPositionCoordinate().SetValue(x, y)
    renderer.AddViewProp(actor)
    return actor


def polydata(points, connectivity, size):
    poly = vtkPolyData()
    poly.points = vtkPoints(data=vtkFloatArray(np.ascontiguousarray(points, np.float32)))
    count = len(connectivity) // size
    poly.polys = vtkCellArray(offsets=np.arange(0, size * count + 1, size, dtype=np.int64),
                              connectivity=np.ascontiguousarray(connectivity, np.int64))
    return poly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=300.0)
    parser.add_argument("--diameter", type=float, default=8.0, help="in root cells")
    parser.add_argument("--steps", type=int, default=4, help="level-0 steps a frame")
    parser.add_argument("--every", type=int, default=16, help="level-0 steps between remeshes")
    parser.add_argument("--start", type=int, default=0, help="steps to run before showing")
    parser.add_argument("--level", type=float, default=1.5,
                        help="the surface's vorticity, in free-stream speed / diameter")
    parser.add_argument("--check", default=None, metavar="OUTDIR")
    args = parser.parse_args()
    tack.set_default_device("cpu")
    d = args.diameter
    nx, ny, nz = int(10 * d), int(4 * d), int(4 * d)
    xc, yc, zc = 2.5 * d, ny / 2, nz / 2
    tau = 3 * U * d / args.re + 0.5
    region = (xc - 1.5 * d, nx - 14)

    def distance(x, y, z):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)

    def pinned(x, y, z):
        return distance(x, y, z) < 0.85 * d

    def solid(x, y, z):
        return distance(x, y, z) < 0.5 * d

    state = {"paused": False, "ms": [], "remesh_ms": [], "contour_ms": [], "trace": [],
             "steps": []}

    def remesh(s, **extra):
        return adapt(s, 0.01, 0.005, finest=2, pinned=pinned, region=region, **extra)

    def start_mesh():
        s = lb.LBM(lb.uniform_mesh((nx, ny, nz), periodic=(False, True, True)), tau, inflow=U,
                   regularized=True, sponge=(nx - 12, 0.1))
        for _ in range(2):
            s, _ = remesh(s)
        s, _ = remesh(s, solid=solid)
        c, _ = s.centres()
        fluid = s.role != lb.SOLID
        kick = 0.02 * U * np.exp(-((c[:, 0] - xc - d) ** 2 + (c[:, 1] - yc - 2) ** 2 +
                                   (c[:, 2] - zc) ** 2) / d ** 2)
        s.initialize(1.0, np.where(fluid, U, 0.0), np.where(fluid, kick, 0.0), 0.0)
        return s

    begin = time.perf_counter()
    solver = start_mesh()
    print(f"{int(solver.computational().sum())} cells on 3 levels to start, Re {args.re:g}, "
          f"tau {tau:.4f} at level 0 ({time.perf_counter() - begin:.1f} s); Tack on the CPU")

    window = rr.vtkRenderWindow(size=(1600, 940), off_screen_rendering=bool(args.check))
    window.SetWindowName("Flow past a sphere (Tack LBM, D3Q27, adaptive octree)")
    above = rr.vtkRenderer(background=(1, 1, 1))
    above.SetViewport(0, 0.36, 1, 1)
    below = rr.vtkRenderer(background=(1, 1, 1))
    below.SetViewport(0, 0, 1, 0.36)
    surface_mapper = rr.vtkPolyDataMapper()
    surface = rr.vtkActor(mapper=surface_mapper)
    surface.GetProperty().SetColor(0.35, 0.6, 0.85)
    surface.GetProperty().SetSpecular(0.3)
    above.AddActor(surface)
    ball = vtkSphereSource(center=(xc, yc, zc), radius=d / 2, theta_resolution=48,
                           phi_resolution=48)
    ball_mapper = rr.vtkPolyDataMapper()
    ball_mapper.SetInputConnection(ball.GetOutputPort())
    sphere = rr.vtkActor(mapper=ball_mapper)
    sphere.GetProperty().SetColor(0.45, 0.45, 0.48)
    above.AddActor(sphere)
    box = vtkOutlineSource(bounds=(0, nx, 0, ny, 0, nz))
    box_mapper = rr.vtkPolyDataMapper()
    box_mapper.SetInputConnection(box.GetOutputPort())
    outline = rr.vtkActor(mapper=box_mapper)
    outline.GetProperty().SetColor(0.3, 0.3, 0.3)
    above.AddActor(outline)
    text(above, 0.01, 0.95, 18).SetInput(
        f"vorticity magnitude {args.level:g} (free-stream speed / diameter), Re {args.re:g};"
        " below, the mid-plane's cells by vorticity magnitude, 0 to 3")
    status = text(above, 0.01, 0.02, 15)
    camera = above.GetActiveCamera()
    camera.SetFocalPoint(nx * 0.5, ny / 2 + 0.06 * nx, nz / 2)
    camera.SetPosition(nx * 0.5 - 0.22 * nx, ny / 2 + 0.45 * nx, nz / 2 + 0.68 * nx)
    camera.SetViewUp(0, 1, 0)
    camera.SetViewAngle(30)
    slice_mapper = rr.vtkPolyDataMapper()
    slice_mapper.SetScalarModeToUseCellData()
    slice_mapper.SetLookupTable(SEQUENTIAL)
    slice_mapper.SetScalarRange(0, 3)
    below.AddActor(rr.vtkActor(mapper=slice_mapper))
    edges_mapper = rr.vtkPolyDataMapper()
    edges_mapper.ScalarVisibilityOff()
    edges = rr.vtkActor(mapper=edges_mapper)
    edges.GetProperty().SetRepresentationToWireframe()
    edges.GetProperty().SetColor(0.2, 0.2, 0.2)
    edges.GetProperty().SetOpacity(0.35)
    below.AddActor(edges)
    flat = below.GetActiveCamera()
    flat.SetParallelProjection(True)
    flat.SetFocalPoint(nx / 2, ny / 2, zc)
    flat.SetPosition(nx / 2, ny / 2, zc + 3 * nx)
    flat.SetParallelScale(0.3 * ny)                 # the plane's middle, where the wake is
    window.AddRenderer(above)
    window.AddRenderer(below)

    def attach(s):
        """Show solver ``s``: its plane's cells, the probe, its solids hidden for
        the contour."""
        c, side = s.centres()
        cut = np.flatnonzero((np.isin(s.role, (lb.REAL, lb.INTERFACE, lb.SOLID))) &
                             (np.abs(c[:, 2] - zc) < side / 2 + 1e-9))
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.5
        xy = (c[cut, None, :2] + side[cut, None, None] * corners[None]).reshape(-1, 2)
        points = np.column_stack([xy, np.full(len(xy), zc)])
        plane = polydata(points, np.arange(4 * len(cut)), 4)
        values = vtkFloatArray(np.zeros(len(cut), np.float32))
        values.SetName("vorticity")
        plane.cell_data.AddArray(values)
        plane.cell_data.SetActiveScalars("vorticity")
        slice_mapper.SetInputData(plane)
        edges_mapper.SetInputData(plane)
        marks = s.octree.marks.numpy().copy()
        marks[s.role == lb.SOLID] |= 32                # HIDDENCELL: no surface around solids
        with s.octree.device:
            s.octree.marks = tack.array(marks, tack.u8)
        probe = int(np.argmin(np.where(s.computational(),
                                       np.hypot(c[:, 0] - xc - 2 * d, np.hypot(c[:, 1] - yc,
                                                                               c[:, 2] - zc)),
                                       np.inf)))
        state.update(solver=s, plane=plane, values=values, cut=cut, probe=probe,
                     solid=s.role[cut] == lb.SOLID,
                     nodes=[nodes for lv in range(s.finest + 1)
                            for nodes in (s.computed[lv], s.ghosts[lv]) if len(nodes)])

    def strouhal():
        p = np.array(state["trace"])
        if len(p) < 16:
            return None
        p = p - p[len(p) // 2:].mean()
        up = np.flatnonzero((p[:-1] < 0) & (p[1:] >= 0))
        if len(up) < 3:
            return None
        steps = np.array(state["steps"])
        return d / (U * np.mean(np.diff(steps[up[-3:]])))

    def show():
        s = state["solver"]
        s.fill_ghosts()           # ghosts from their parents, the same on every frame
        with s.octree.device:
            w = tack.zeros((s.octree.num_nodes,), s.dtype)
            for nodes in state["nodes"]:
                _vorticity(nodes, s.rows, s.octree.level, s.role_array, s.number, s.table,
                           s.side, s.q, s.dim, s.f, w)
        scaled = w.numpy() * d / U
        with s.octree.device:
            s.octree.fields["w"] = tack.array(scaled.astype(s.dtype.numpy_dtype), s.dtype)
        values = scaled[state["cut"]]
        values[state["solid"]] = 0.0
        np.asarray(state["values"])[:] = values.astype(np.float32)
        state["values"].Modified()
        state["plane"].Modified()
        begin = time.perf_counter()
        wake = octree_contour.contour(s.octree, "w", args.level)
        points = np.asarray(wake.positions(), np.float64).reshape(-1, 3)
        surface_mapper.SetInputData(polydata(points, wake.topology.connectivity.numpy(), 3))
        state["contour_ms"].append(1000 * (time.perf_counter() - begin))
        _, ux, uy, uz = s.moments()
        state["trace"].append(uy[state["probe"]])
        state["steps"].append(s.steps)
        st = strouhal()
        status.SetInput(
            f"step {s.steps}   {int(s.computational().sum())} cells   Strouhal "
            f"{'...' if st is None else f'{st:.3f}'}   "
            f"{np.median(state['ms'][-20:]) if state['ms'] else 0:.0f} ms a frame "
            f"({args.steps} steps; remesh "
            f"{np.median(state['remesh_ms'][-10:]) if state['remesh_ms'] else 0:.0f} ms every "
            f"{args.every}; surface {np.median(state['contour_ms'][-10:]):.0f} ms)"
            + ("   paused" if state["paused"] else ""))

    def advance(steps=None):
        begin = time.perf_counter()
        s = state["solver"]
        for _ in range((steps or args.steps) // args.steps):
            s.step(args.steps)
            if s.steps % args.every < args.steps:
                t = time.perf_counter()
                s, _ = remesh(s)
                state["remesh_ms"].append(1000 * (time.perf_counter() - t))
                attach(s)
        show()
        state["ms"].append(1000 * (time.perf_counter() - begin))

    def restart():
        state["trace"], state["steps"] = [], []
        attach(start_mesh())

    restart()
    if args.start:
        print(f"running {args.start} steps first...")
        while state["solver"].steps < args.start:
            advance(min(256, args.start - state["solver"].steps))
            print(f"  step {state['solver'].steps}")
        state["ms"] = []
    show()
    if args.check:
        os.makedirs(args.check, exist_ok=True)
        for _ in range(4):
            advance()
        window.Render()
        grab = vtkWindowToImageFilter()
        grab.SetInput(window)
        grab.Update()
        writer = vtkPNGWriter()
        writer.SetFileName(os.path.join(args.check, f"sphere_view_{state['solver'].steps}.png"))
        writer.SetInputConnection(grab.GetOutputPort())
        writer.Write()
        print(f"step {state['solver'].steps}: {int(state['solver'].computational().sum())} "
              f"cells; a frame {np.median(state['ms']):.0f} ms; a remesh "
              f"{np.median(state['remesh_ms']):.0f} ms; the surface "
              f"{np.median(state['contour_ms']):.0f} ms")
        return

    interactor = rr.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

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
    print("space pauses, r restarts, m shows or hides the cells' edges, q quits; drag rotates")
    interactor.Start()


if __name__ == "__main__":
    main()
