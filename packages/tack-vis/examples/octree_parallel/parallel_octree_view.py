# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An xRAGE octree forest volume rendered across MPI ranks, interactive in a VTK
WebGPU window on rank 0 (study; as the AMR session's ``parallel_amr_view`` is
for AMR, on ``tack.communication``).

Each rank holds consecutive xRAGE pieces (its trees, read from the file's blocks
holding them), the one-leaf ring by ``Routing``, and a renderer (``mpi_render``'s
setup). Rank 0's ``TackVolumeMapper`` is given ``Distributed`` for a renderer: each
frame (camera, size, transfer function, VTK's depth) goes to every rank, every
rank renders its fragments (a box each, rays clipped to it), and the fragments
composite by direct send in the kd-tree's order (``mpi_render.composite``), the
picture on rank 0. The other ranks serve frames until rank 0 quits. Moving frames
are the same path, the mapper's smaller picture and longer steps. The opacity's
unit is every rank's finest cell.

    parallel_octree_view.sh [-n 4] [PIECES.npz OCTREE.bp] [--field tev] [--size 1280x800]
                            [--check OUTDIR]

Drag rotates, scroll zooms, +/- change the opacity, q quits. ``--check`` renders a
still and a moving frame off screen and writes them."""

import argparse
import os
import resource
import sys
import time

import numpy as np
from check_fragments import AMR, ORIGIN, ROOTS, SIDE, piece_runs, rank_boxes, tev_functions
from check_ranks import rank_of_roots
from mpi4py import MPI
from mpi_render import camera_for, composite, my_blocks, unit_cell
from ring import own_forest, ring, ringed_forest

import tack
from tack.communication import Blocks, World
from tack.dataset import octree_io as oio
from tack.dataset import octree_volume as ov


def camera_state(camera):
    return {"position": camera.GetPosition(), "focal": camera.GetFocalPoint(),
            "up": camera.GetViewUp(), "angle": camera.GetViewAngle(),
            "clipping": camera.GetClippingRange(),
            "parallel": camera.GetParallelProjection(), "scale": camera.GetParallelScale()}


def camera_of(state):
    from vtkmodules.vtkRenderingCore import vtkCamera

    camera = vtkCamera()
    camera.SetPosition(*state["position"])
    camera.SetFocalPoint(*state["focal"])
    camera.SetViewUp(*state["up"])
    camera.SetViewAngle(state["angle"])
    camera.SetClippingRange(*state["clipping"])
    camera.SetParallelProjection(state["parallel"])
    camera.SetParallelScale(state["scale"])
    return camera


class Frames:
    """Every rank's part of a frame: rank 0's frame and depth to every rank
    (``Blocks.all_gather``, ``Blocks.exchange``), this rank's fragments, the
    composite (the picture on rank 0)."""

    def __init__(self, blocks, renderer, boxes, unit):
        self.blocks = blocks
        self.g = blocks.local[0]
        self.renderer = renderer
        self.boxes = boxes
        self.unit = unit
        self.mine = [i for i, (rank, _) in enumerate(boxes) if rank == self.g]

    def bcast(self, value):
        """Rank 0's ``value`` (small) on every rank."""
        return self.blocks.all_gather({self.g: value if self.g == 0 else None})[0]

    def depth(self, depth, shape):
        """Rank 0's depth (``(h, w)`` or None, as ``shape`` says) on every rank."""
        if shape is None:
            return None
        n = self.blocks.count
        device = self.blocks.device(self.g)
        outgoing = {self.g: {}}
        if self.g == 0:
            with device:
                values = tack.array(np.ascontiguousarray(depth, np.float32).reshape(-1),
                                    tack.f32)
            outgoing[0] = {r: {"depth": values} for r in range(1, n)}
        received = self.blocks.exchange(outgoing)
        if self.g == 0:
            return depth
        return received[self.g][0]["depth"].numpy().reshape(shape)

    def draw(self, frame, depth):
        """The frame's fragments, composited: ``(picture on rank 0 or None, this
        rank's march ms)``."""
        t = time.perf_counter()
        w, h = frame["size"]
        camera = camera_of(frame["camera"])
        renderer = self.renderer
        options = dict(frame["options"])
        options["unit_distance"] = frame["unit"] / renderer.cell
        fragments = {i: renderer.render(camera, (w, h), frame["table"], frame["lo"], frame["hi"],
                                        depth, aspect=frame["aspect"], region=self.boxes[i][1],
                                        **options)
                     for i in self.mine}
        ms = 1000 * (time.perf_counter() - t)
        picture, _ = composite(self.blocks, fragments, self.boxes, camera.GetPosition(), w, h)
        return picture, ms


class Distributed:
    """Rank 0's stand-in for a Tack renderer in ``TackVolumeMapper``: every rank's
    fragments of each frame, composited."""

    def __init__(self, frames, bounds):
        self.frames = frames
        self.bounds = bounds
        self.cell = frames.unit
        self.range = (0.0, 1.0)
        self.timings = {"march_ms": 0.0}
        #: The last frame's parts: every rank's march (ms) and the whole frame's.
        self.parts = {}

    def table(self, color, opacity, lo, hi, size=256):
        return self.frames.renderer.table(color, opacity, lo, hi, size)

    def render(self, camera, size, table, lo, hi, depth=None, unit_distance=1.0, aspect=None,
               **options):
        t = time.perf_counter()
        frame = {"op": "frame", "camera": camera_state(camera), "size": tuple(size),
                 "table": np.asarray(table, np.float32), "lo": float(lo), "hi": float(hi),
                 "unit": unit_distance * self.cell, "aspect": aspect, "options": options,
                 "depth": None if depth is None else tuple(np.shape(depth))}
        frames = self.frames
        frames.bcast(frame)
        depth = frames.depth(depth, frame["depth"])
        picture, ms = frames.draw(frame, depth)
        marches = frames.blocks.all_gather({frames.g: round(ms)})
        all_ms = 1000 * (time.perf_counter() - t)
        self.parts = {"ranks_ms": marches, "all_ms": all_ms}
        self.timings = {"march_ms": all_ms}
        return picture


def serve(frames):
    """A rank other than 0: frames until rank 0 says to quit."""
    while True:
        frame = frames.bcast(None)
        if frame["op"] == "quit":
            return
        depth = frames.depth(None, frame["depth"])
        _, ms = frames.draw(frame, depth)
        frames.blocks.all_gather({frames.g: round(ms)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pieces")
    parser.add_argument("octree")
    parser.add_argument("--field", default="tev")
    parser.add_argument("--range", type=float, nargs=2, default=(0.03, 1.0))
    parser.add_argument("--device", default="metal")
    parser.add_argument("--size", default="1280x800")
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--check", default=None, metavar="OUTDIR")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    n = comm.size
    tack.set_default_device(args.device)
    device = tack.device(args.device)
    blocks = Blocks([device], World(comm))
    g = blocks.local[0]
    say = (lambda *a: print(*a, flush=True)) if g == 0 else (lambda *a: None)

    # Setup, as mpi_render's: metadata, own trees, the ring, the forest, the renderer.
    t0 = time.perf_counter()
    laps = {}

    def lap(name, start):
        tack.sync()
        laps[name] = round(time.perf_counter() - start, 1)
        return time.perf_counter()

    firsts, lasts = piece_runs(args.pieces)
    owner = rank_of_roots(firsts, lasts, n)
    boxes = rank_boxes(firsts, lasts, n)
    part = oio.read_octree(args.octree, blocks=my_blocks(args.octree, np.flatnonzero(owner == g)),
                           fields=[args.field], device=device)
    with device:
        owner_d = tack.array(owner.astype(np.int32), tack.i32)
    own = own_forest(part, owner_d, g, args.field)
    del part
    own_nodes = own.num_nodes
    t = lap("load", t0)
    leaves, _ = ring(blocks, {g: own}, owner, owner_d, ROOTS, ORIGIN, SIDE, args.field)
    t = lap("ring", t)
    forest, _ = ringed_forest(own, leaves[g], args.field)
    del own, leaves
    t = lap("forest", t)
    renderer = ov.OctreeRenderer(forest, args.field, AMR, blend=2, macrocells=256, divide=-1,
                                 balanced=1)
    t = lap("renderer", t)
    frames = Frames(blocks, renderer, boxes, unit_cell(blocks, renderer))
    laps["setup"] = round(time.perf_counter() - t0, 1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30   # bytes on macOS
    setup = blocks.all_gather({g: (laps, own_nodes, forest.num_nodes, len(frames.mine), peak)})
    say(f"{n} ranks, Tack on {args.device} (one GPU for every rank)")
    for r in range(n):
        lp, on, fn, bx, pk = setup[r]
        say(f"  rank {r}: " + ", ".join(f"{k} {v} s" for k, v in lp.items()) +
            f"; own nodes {on:,}, held {fn / on:.2f}x, boxes {bx}, peak RSS {pk:.1f} GB")
    if g != 0:
        serve(frames)
        return
    try:
        _window(args, frames, renderer.bounds, say)
    finally:
        frames.bcast({"op": "quit"})


def _window(args, frames, bounds, say):
    import vtkmodules.vtkRenderingCore as rr
    import vtkmodules.vtkRenderingWebGPU  # noqa: F401  (the WebGPU window)
    from tack_view import BACKGROUND, grab, write_png
    from tack_volume_mapper import TackVolumeMapper
    from vtkmodules.vtkFiltersSources import vtkOutlineSource
    from vtkmodules.vtkRenderingCore import vtkTextActor, vtkVolume, vtkVolumeProperty

    w, h = (int(v) for v in args.size.split("x"))
    lo, hi = args.range
    distributed = Distributed(frames, bounds)
    mapper = TackVolumeMapper(None, renderer=distributed)
    mapper.sample_distance = args.step
    color, opacity = tev_functions(lo, hi)
    prop = vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.SetScalarOpacityUnitDistance(distributed.cell)
    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    view = rr.vtkRenderer(background=BACKGROUND)
    box = vtkOutlineSource(bounds=bounds)               # held: a port does not own its source
    outline_mapper = rr.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(box.GetOutputPort())
    outline = rr.vtkActor(mapper=outline_mapper)
    outline.GetProperty().SetColor(0.9, 0.9, 0.9)
    view.AddActor(outline)
    view.AddVolume(volume)
    label = vtkTextActor()
    label.GetTextProperty().SetFontSize(16)
    label.GetTextProperty().SetColor(0.9, 0.9, 0.9)
    label.SetPosition(12, 12)
    view.AddViewProp(label)
    window = rr.vtkRenderWindow(size=(w, h), off_screen_rendering=bool(args.check))
    window.SetWindowName(f"yA31 across {frames.blocks.count} ranks (Tack octree volume, WebGPU)")
    window.AddRenderer(view)
    _, camera = camera_for(bounds)
    view.SetActiveCamera(camera)
    view.ResetCameraClippingRange(*bounds)

    def report(*_):
        p = distributed.parts
        if p:
            label.SetInput(f"{frames.blocks.count} ranks   a frame {p['all_ms']:.0f} ms   "
                           f"ranks' marches {p['ranks_ms']} ms")

    view.AddObserver("StartEvent", report)
    if args.check:
        os.makedirs(args.check, exist_ok=True)
        for name, rate in (("still", 0.0001), ("moving", 15.0)):
            window.SetDesiredUpdateRate(rate)
            window.Render()
            window.Render()
            write_png(os.path.join(args.check, f"parallel_octree_{name}.png"), grab(window))
            say(f"{name}: a frame {distributed.parts['all_ms']:.0f} ms, ranks' marches "
                f"{distributed.parts['ranks_ms']} ms")
        return
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

    interactor = rr.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    base = []
    for i in range(opacity.GetSize()):
        node = [0.0] * 4
        opacity.GetNodeValue(i, node)
        base.append(node)
    scale = [1.0]

    def key(caller, _):
        k = caller.GetKeySym()
        if k in ("plus", "equal", "KP_Add", "minus", "KP_Subtract"):
            scale[0] *= 1.5 if k in ("plus", "equal", "KP_Add") else 1 / 1.5
            opacity.RemoveAllPoints()
            for x, a, midpoint, sharpness in base:
                opacity.AddPoint(x, min(a * scale[0], 1.0), midpoint, sharpness)
            window.Render()

    interactor.AddObserver("KeyPressEvent", key)
    say("drag rotates, scroll zooms, +/- opacity, q quits")
    window.Render()
    interactor.Start()


if __name__ == "__main__":
    sys.path.insert(0, AMR)
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
