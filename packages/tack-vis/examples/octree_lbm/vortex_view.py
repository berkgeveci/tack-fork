# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A Taylor-Green vortex on a refined quadtree, live in a VTK WebGPU window: the
lattice Boltzmann prototype (``tack.dataset.octree_lbm``) steps on the CPU while
VTK shows, side by side,

1. the vorticity, with the cells (three levels: two nested squares);
2. the error of the corrected scheme (linear, time-centred explosion);
3. the error of Rohde's explosion as published, where the interfaces drive
   D2Q9's staggered momentum mode: the checkerboard in the fine cells.

Errors are the velocity's distance from the exact decaying vortex, as a
percentage of the vortex's speed at that time. The run starts over once the
vortex has decayed to 1 %.

    vortex_view.py [--roots 32] [--levels 3] [--tau 0.8] [--steps 2] [--check OUTDIR]

Space pauses, r restarts, q quits; drag pans, scroll zooms (the panels share a
camera). Run it on VTK's WebGPU build (``vortex_view.sh``)."""

import argparse
import os
import time

import numpy as np
import vtkmodules.vtkCommonDataModel  # pythonic constructors
import vtkmodules.vtkRenderingCore as rr
import vtkmodules.vtkRenderingWebGPU  # noqa: F401  (the WebGPU window)
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkTextActor, vtkWindowToImageFilter

import tack
from tack.dataset import octree_lbm as lb

U0 = 0.03


def exact(c, n, nu, t):
    k = 2 * np.pi / n
    x, y = c[:, 0], c[:, 1]
    d = np.exp(-2 * nu * k * k * t)
    return (1 - 3 * U0 * U0 / 4 * (np.cos(2 * k * x) + np.cos(2 * k * y)) * d * d,
            -U0 * np.cos(k * x) * np.sin(k * y) * d, U0 * np.sin(k * x) * np.cos(k * y) * d)


class Vorticity:
    """Vorticity at the computational cells from their neighbours at their level
    (central differences; a refined neighbour by its children's mean; one-sided
    beside a solid or a coarser cell)."""

    def __init__(self, solver):
        rows = solver.rows.numpy().reshape(-1, 9)
        number, table = (a.numpy() for a in (solver.number, solver.table))
        self.keep = np.flatnonzero(solver.computational())
        role, level = solver.role, solver.levels
        self.side = solver.centres()[1][self.keep]
        self.slots = []
        for o in (3, 5, 1, 7):                       # west, east, south, north
            q = rows[self.keep, o]
            qq = np.maximum(q, 0)
            at = (q >= 0) & (level[qq] == level[self.keep])
            direct = at & (role[qq] <= lb.GHOST)
            split = at & (role[qq] == lb.REFINED)
            first = np.where(split, (number[qq] - 1) * 4, 0)
            kids = table[first[:, None] + np.arange(4)[None, :]] if len(table) else \
                np.zeros((len(q), 4), np.int64)
            split &= (role[kids] <= lb.INTERFACE).all(axis=1)
            nodes = np.where(split[:, None], kids, qq[:, None])
            self.slots.append((nodes, direct | split))

    def __call__(self, ux, uy):
        def at(slot, v):
            nodes, ok = self.slots[slot]
            return v[nodes].mean(axis=1), ok

        def derivative(v, lo, hi):
            c = v[self.keep]
            a, use_a = at(lo, v)
            b, use_b = at(hi, v)
            return np.where(use_a & use_b, 0.5 * (b - a),
                            np.where(use_b, b - c, np.where(use_a, c - a, 0.0))) / self.side

        return derivative(uy, 0, 1) - derivative(ux, 2, 3)


def cells_polydata(solver):
    c, side = solver.centres()
    keep = np.flatnonzero(solver.computational())
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.5
    xy = (c[keep, None, :] + side[keep, None, None] * corners[None]).reshape(-1, 2)
    xyz = np.zeros((len(xy), 3), np.float32)
    xyz[:, :2] = xy
    poly = vtkPolyData()
    poly.points = vtkPoints(data=vtkFloatArray(xyz))
    poly.polys = vtkCellArray(offsets=np.arange(0, 4 * len(keep) + 1, 4, dtype=np.int64),
                              connectivity=np.arange(4 * len(keep), dtype=np.int64))
    values = vtkFloatArray(np.zeros(len(keep), np.float32))
    values.SetName("value")
    poly.cell_data.AddArray(values)
    poly.cell_data.SetActiveScalars("value")
    return poly, values


def table(points):
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    stops = np.array([p[0] for p in points])
    colours = np.array([p[1] for p in points])
    for i in range(256):
        t = i / 255
        lut.SetTableValue(i, *[float(np.interp(t, stops, colours[:, a])) for a in range(3)], 1.0)
    return lut


DIVERGING = table([(0.0, (0.23, 0.30, 0.75)), (0.5, (0.95, 0.95, 0.95)), (1.0, (0.71, 0.02, 0.15))])
SEQUENTIAL = table([(0.0, (0.98, 0.98, 0.96)), (0.35, (0.99, 0.75, 0.35)),
                    (0.7, (0.85, 0.25, 0.15)), (1.0, (0.35, 0.0, 0.2))])


def panel(window, index, count, poly, lut, low, high, title, unit, camera):
    ren = rr.vtkRenderer(background=(1, 1, 1))
    ren.SetViewport(index / count, 0, (index + 1) / count, 1)
    mapper = rr.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetScalarModeToUseCellData()
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(low, high)
    ren.AddActor(rr.vtkActor(mapper=mapper))
    edges_mapper = rr.vtkPolyDataMapper()
    edges_mapper.SetInputData(poly)
    edges_mapper.ScalarVisibilityOff()
    edges = rr.vtkActor(mapper=edges_mapper)
    edges.GetProperty().SetRepresentationToWireframe()
    edges.GetProperty().SetColor(0.25, 0.25, 0.25)
    edges.GetProperty().SetOpacity(0.35)
    ren.AddActor(edges)
    heading = vtkTextActor()
    heading.SetInput(f"{title} ({unit})")
    heading.GetTextProperty().SetColor(0.1, 0.1, 0.1)
    heading.GetTextProperty().SetFontSize(17)
    heading.GetTextProperty().SetJustificationToCentered()
    heading.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    heading.GetPositionCoordinate().SetValue(0.5, 0.955)
    ren.AddViewProp(heading)
    bar = vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetNumberOfLabels(3)
    bar.SetOrientationToHorizontal()
    bar.SetPosition(0.15, 0.855)
    bar.SetWidth(0.7)
    bar.SetHeight(0.085)
    bar.SetUnconstrainedFontSize(True)
    for p, size in ((bar.GetLabelTextProperty(), 14),):
        p.SetColor(0.1, 0.1, 0.1)
        p.SetShadow(False)
        p.SetItalic(False)
        p.SetBold(False)
        p.SetFontSize(size)
    ren.AddViewProp(bar)
    ren.SetActiveCamera(camera)
    window.AddRenderer(ren)
    return ren


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--levels", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=2, help="level-0 steps a frame")
    parser.add_argument("--check", default=None, metavar="OUTDIR")
    args = parser.parse_args()
    tack.set_default_device("cpu")
    n, tau = args.roots, args.tau
    nu = (tau - 0.5) / 3

    def refine(x, y, h):
        outer = abs(x - n / 2) < n / 4 and abs(y - n / 2) < n / 4
        return outer if h > 0.75 else abs(x - n / 2) < n / 8 and abs(y - n / 2) < n / 8

    octree = lb.lbm_mesh((n, n), refine, args.levels - 1)
    corrected = lb.LBM2D(octree, tau)
    published = lb.LBM2D(octree, tau, centred=False, linear=False)
    centres, _ = corrected.centres()
    keep = np.flatnonzero(corrected.computational())
    vorticity = Vorticity(corrected)
    k = 2 * np.pi / n
    print(f"{len(keep)} cells on {args.levels} levels, tau {tau}; Tack on the CPU")

    polys = [cells_polydata(corrected) for _ in range(3)]
    window = rr.vtkRenderWindow(size=(1650, 700), off_screen_rendering=bool(args.check))
    window.SetWindowName("Taylor-Green vortex on a refined quadtree (Tack LBM)")
    camera = rr.vtkCamera()
    camera.SetParallelProjection(True)
    camera.SetFocalPoint(n / 2, 0.6 * n, 0)
    camera.SetPosition(n / 2, 0.6 * n, 3 * n)
    camera.SetParallelScale(0.67 * n)
    panel(window, 0, 3, polys[0][0], DIVERGING, -1, 1, "vorticity", "of its peak", camera)
    panel(window, 1, 3, polys[1][0], SEQUENTIAL, 0, 3, "error, corrected explosion",
          "% of vortex speed", camera)
    panel(window, 2, 3, polys[2][0], SEQUENTIAL, 0, 3, "error, Rohde's explosion as published",
          "%", camera)
    clock = vtkTextActor()
    clock.GetTextProperty().SetColor(0.1, 0.1, 0.1)
    clock.GetTextProperty().SetFontSize(16)
    clock.SetPosition(12, 8)
    window.GetRenderers().GetFirstRenderer().AddViewProp(clock)

    state = {"t": 0, "paused": False, "ms": []}

    def restart():
        for s in (corrected, published):
            s.initialize(*exact(centres, n, nu, 0.0))
        state["t"] = 0

    def show():
        t = state["t"]
        scale = U0 * np.exp(-2 * nu * k * k * t)
        _, ex, ey = exact(centres, n, nu, t)
        _, ux, uy = corrected.moments()
        values = [vorticity(ux, uy) / (2 * U0 * k * np.exp(-2 * nu * k * k * t))]
        values.append(100 * np.hypot(ux - ex, uy - ey)[keep] / scale)
        _, px, py = published.moments()
        values.append(100 * np.hypot(px - ex, py - ey)[keep] / scale)
        for (poly, array), v in zip(polys, values):
            np.asarray(array)[:] = v.astype(np.float32)
            array.Modified()
            poly.Modified()
        clock.SetInput(f"step {t}, vortex at {100 * scale / U0:.0f} % of its start"
                       f"{'   (paused)' if state['paused'] else ''}")

    def advance():
        start = time.perf_counter()
        for s in (corrected, published):
            s.step(args.steps)
        state["t"] += args.steps
        if np.exp(-2 * nu * k * k * state["t"]) < 0.01:
            restart()
        show()
        state["ms"].append(1000 * (time.perf_counter() - start))

    restart()
    show()
    if args.check:
        os.makedirs(args.check, exist_ok=True)
        window.Render()
        for _ in range(100 // args.steps):
            advance()
            window.Render()
        grab = vtkWindowToImageFilter()
        grab.SetInput(window)
        grab.Update()
        writer = vtkPNGWriter()
        writer.SetFileName(os.path.join(args.check, "vortex_view.png"))
        writer.SetInputConnection(grab.GetOutputPort())
        writer.Write()
        print(f"a frame: {np.median(state['ms']):.1f} ms to step both runs and update "
              f"({args.steps} steps each)")
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
            show()
            window.Render()
        elif key in ("r", "R"):
            restart()
            show()
            window.Render()

    interactor.AddObserver("TimerEvent", on_timer)
    interactor.AddObserver("KeyPressEvent", on_key)
    window.Render()
    interactor.Initialize()
    interactor.CreateRepeatingTimer(15)
    print("space pauses, r restarts, q quits; drag pans, scroll zooms")
    interactor.Start()


if __name__ == "__main__":
    main()
