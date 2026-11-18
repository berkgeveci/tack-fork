# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""AMR-Wind's vorticity volume rendered by Barney through pynari, straight from the
``vtkOverlappingAMR`` the AMReX reader gives: no Tack, no VTK ANARI module.

    AMReX plotfile ─► read_plotfile ─► vtkOverlappingAMR ─► ANARI "amr" spatial field
                   ─► transferFunction1D volume ─► Barney (CPU backend) ─► VTK window

The VTK window only shows Barney's frame (as the renderer's background texture) and
owns the camera: drag to rotate (right button or wheel to zoom, middle to pan). While
the camera moves the frame is 1/``--moving-scale`` the window's size; once it has
rested ``REST`` ms the full-size frame is drawn and refined, one more sample per
pixel per pass, ``--passes`` times (Barney accumulates until something changes).

Keys: ``r`` resets the camera to the turbines, ``q`` quits.

    python amrwind_barney.py [--plotfile P] [--max-level L] [--size WxH]
    python amrwind_barney.py --output out.png [--passes N]    # one frame, no window

Barney (CPU backend) and pynari are built from source, see env_barney.sh: the pip
wheel's Barney predates its AMR work, and the CPU backend needs a local patch (its
AMR sampler asks cuBQL's CPU builder for a wide BVH, which it cannot build).
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pynari as anari
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import (
    vtkDataSetAttributes,
    vtkImageData,
    vtkPiecewiseFunction,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkTexture,
)

sys.path.insert(0, os.path.expanduser("~/Work/amrex-reader"))

from amrex_reader import read_plotfile  # noqa: E402

PLOTFILE = os.path.expanduser("~/Data/amrwind/plt255900")

#: How long the camera rests before the full frame is drawn, in ms.
REST = 300

BACKGROUND = (0.32, 0.34, 0.43)


def transfer_functions(lo, hi):
    """Vorticity, as amrwind_demo.transfer_functions: transparent below ``lo``,
    warmer and more opaque up to ``hi``."""
    c = vtkColorTransferFunction()
    c.SetColorSpaceToDiverging()
    c.AddRGBPoint(lo, 0.23, 0.30, 0.75)
    c.AddRGBPoint(0.5 * (lo + hi), 0.86, 0.86, 0.86)
    c.AddRGBPoint(hi, 0.71, 0.02, 0.15)
    o = vtkPiecewiseFunction()
    o.AddPoint(0.0, 0.0)
    o.AddPoint(lo, 0.0)
    o.AddPoint(0.5 * (lo + hi), 0.05)
    o.AddPoint(hi, 0.4)
    return c, o


def amr_arrays(amr, field):
    """The hierarchy as the ANARI ``amr`` field's arrays: each block's lower corner in
    its level's cells, its cell counts, its level, and all blocks' cell values
    concatenated (x fastest, as VTK stores them). Barney takes level ``l``'s cells to
    be ``1 / 2^l`` level 0's, so every refinement ratio must be 2.

    Barney reconstructs a point from every block that contains it, weighting each
    level's cells alike, so the coarse cells under finer ones would be averaged into
    the fine values. Cells the hierarchy marks refined (``vtkGhostType``, as the reader
    marks them with ``blank=True``) are left out: each block goes in as the boxes of
    its unrefined cells (``unrefined_boxes``)."""
    levels = amr.GetNumberOfLevels()
    for level in range(levels - 1):
        ratio = amr.GetRefinementRatio(level)
        if ratio != 2:
            raise ValueError(f"level {level}'s refinement ratio is {ratio}; Barney needs 2")
    refined_bit = vtkDataSetAttributes.REFINEDCELL
    origins, dims, block_levels, values = [], [], [], []
    for level in range(levels):
        for lo, cells, grid in level_blocks(amr, level):
            data = vtk_to_numpy(grid.GetCellData().GetArray(field))
            ghosts = grid.GetCellData().GetArray("vtkGhostType")
            if ghosts is None:
                pieces = [(np.zeros(3, np.int64), cells)]
            else:
                refined = (vtk_to_numpy(ghosts) & refined_bit) != 0
                pieces = unrefined_boxes(refined.reshape(cells[::-1]))
                if len(pieces) == 1 and np.array_equal(pieces[0][1], cells):
                    pieces = [(np.zeros(3, np.int64), cells)]
            grid_values = data.reshape(cells[::-1])
            for start, size in pieces:
                origins.append(lo + start)
                dims.append(size)
                block_levels.append(level)
                (x0, y0, z0), (nx, ny, nz) = start, size
                values.append(grid_values[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx].reshape(-1))
    return (np.array(origins, np.int32), np.array(dims, np.int32),
            np.array(block_levels, np.int32),
            np.concatenate(values).astype(np.float32, copy=False))


def unrefined_boxes(refined):
    """Boxes covering the cells where ``refined`` ((nz, ny, nx) bool) is False, as
    (start, size) in (x, y, z): split where the refined region begins or ends along
    the longest axis, until each box is all unrefined (kept) or all refined (dropped)."""
    out = []

    def split(z0, z1, y0, y1, x0, x1):
        sub = refined[z0:z1, y0:y1, x0:x1]
        if not sub.any():
            out.append((np.array([x0, y0, z0]), np.array([x1 - x0, y1 - y0, z1 - z0])))
            return
        if sub.all():
            return
        bounds = [(x0, x1), (y0, y1), (z0, z1)]
        # numpy axis for x, y, z in (z, y, x) order
        for axis in sorted(range(3), key=lambda a: bounds[a][0] - bounds[a][1]):
            other = tuple(a for a in (0, 1, 2) if a != 2 - axis)
            profile_any = sub.any(axis=other)
            profile_all = sub.all(axis=other)
            changes = np.flatnonzero((profile_any[1:] != profile_any[:-1])
                                     | (profile_all[1:] != profile_all[:-1])) + 1
            if len(changes):
                n = bounds[axis][1] - bounds[axis][0]
                cut = bounds[axis][0] + int(changes[np.argmin(np.abs(changes - n / 2))])
                break
        else:
            # Mixed in every slab along every axis: halve the longest.
            axis = max(range(3), key=lambda a: bounds[a][1] - bounds[a][0])
            cut = (bounds[axis][0] + bounds[axis][1]) // 2
        lo_hi = [list(b) for b in bounds]
        first, second = [list(b) for b in lo_hi], [list(b) for b in lo_hi]
        first[axis][1], second[axis][0] = cut, cut
        for (a0, a1), (b0, b1), (c0, c1) in (first, second):
            split(c0, c1, b0, b1, a0, a1)

    nz, ny, nx = refined.shape
    split(0, nz, 0, ny, 0, nx)
    return out


def level_blocks(amr, level):
    """Each of ``level``'s blocks as (lower corner in the level's cells, cell counts,
    vtkUniformGrid). The index box comes from the grid's origin and dimensions:
    vtkAMRBox's corners, like the hierarchy's origin, come back from Python as
    pointers."""
    origin, h = domain_origin(amr), level_spacing(amr, level)
    for grid in level_grids(amr, level):
        lo = np.rint((np.array(grid.GetOrigin()) - origin) / h).astype(np.int64)
        cells = np.maximum(np.array(grid.GetDimensions()) - 1, 1)
        yield lo, cells, grid


def level_grids(amr, level):
    count = amr.GetNumberOfBlocks if hasattr(amr, "GetNumberOfBlocks") else amr.GetNumberOfDataSets
    get = amr.GetDataSetAsImageData if hasattr(amr, "GetDataSetAsImageData") else amr.GetDataSet
    for i in range(count(level)):
        grid = get(level, i)
        if grid is not None:
            yield grid


def domain_origin(amr):
    """The lowest corner of level 0's blocks (which cover the domain)."""
    return np.min([grid.GetOrigin() for grid in level_grids(amr, 0)], axis=0)


def level_spacing(amr, level):
    spacing = [0.0, 0.0, 0.0]
    amr.GetSpacing(level, spacing)
    return np.array(spacing)


def finest_region(amr):
    """The box around the finest level's blocks, in world coordinates."""
    finest = amr.GetNumberOfLevels() - 1
    blocks = list(level_blocks(amr, finest))
    lo = np.min([b[0] for b in blocks], axis=0)
    hi = np.max([b[0] + b[1] for b in blocks], axis=0)
    origin, h = domain_origin(amr), level_spacing(amr, finest)
    return origin + lo * h, origin + hi * h


class BarneyScene:
    """The ANARI objects: the AMR field, its volume, a camera, a renderer, and a frame
    per size (moving and still) sharing them."""

    def __init__(self, amr, field, lo, hi, unit_distance, ambient=1.0, denoise=False,
                 shading="emission"):
        """``shading``: "emission" is Tack's model (emission-absorption: the transfer
        function's colour emitted, no lighting, colours written as they are);
        "scatter" is Barney's path-traced medium (colour is albedo, lit by
        ``ambient``, sRGB output)."""
        self.shading = shading
        t = time.perf_counter()
        origins, dims, levels, values = amr_arrays(amr, field)
        self.cells = values.size
        print(f"VTK -> arrays: {len(levels)} blocks, {values.size / 1e6:.0f}M cells, "
              f"{time.perf_counter() - t:.1f} s", flush=True)

        d = self.device = anari.newDevice("default")
        sf = d.newSpatialField("amr")
        sf.setParameter("block.origin", anari.ARRAY1D, d.newArray1D(anari.INT32_VEC3, origins))
        sf.setParameter("block.dimensions", anari.ARRAY1D, d.newArray1D(anari.INT32_VEC3, dims))
        sf.setParameter("block.level", anari.ARRAY1D, d.newArray1D(anari.INT32, levels))
        sf.setParameter("data", anari.ARRAY1D, d.newArray1D(anari.FLOAT32, values))
        sf.setParameter("origin", anari.FLOAT32_VEC3, tuple(float(v) for v in domain_origin(amr)))
        sf.setParameter("spacing", anari.FLOAT32_VEC3,
                        tuple(float(v) for v in level_spacing(amr, 0)))
        sf.commitParameters()

        color, opacity = transfer_functions(lo, hi)
        n = 256
        xs = np.linspace(0.0, hi, n)
        colors = np.array([color.GetColor(x) for x in xs], np.float32)
        opacities = np.array([opacity.GetValue(x) for x in xs], np.float32)
        if shading == "emission":
            # Tack composites alpha a per unit distance, an extinction of -ln(1 - a);
            # Barney takes the table as the extinction itself.
            opacities = -np.log1p(-np.minimum(opacities, 0.999)).astype(np.float32)
            # Emission at each collision (strength x emissionMap(value)), apart from
            # scattering; with no ambient light and no lights it is all there is.
            vol = d.newVolume("principledVolume")
            vol.setParameter("emissionMap", anari.ARRAY1D,
                             d.newArray1D(anari.FLOAT32_VEC3, colors))
            vol.setParameter("emissionStrength", anari.FLOAT32, 1.0)
            ambient = 0.0
        else:
            vol = d.newVolume("transferFunction1D")
        vol.setParameter("value", anari.SPATIAL_FIELD, sf)
        vol.setParameter("color", anari.ARRAY1D, d.newArray1D(anari.FLOAT32_VEC3, colors))
        vol.setParameter("opacity", anari.ARRAY1D, d.newArray1D(anari.FLOAT32, opacities))
        vol.setParameter("valueRange", anari.FLOAT32_BOX1, (0.0, float(hi)))
        vol.setParameter("unitDistance", anari.FLOAT32, float(unit_distance))
        vol.commitParameters()

        world = self.world = d.newWorld()
        world.setParameterArray1D("volume", anari.VOLUME, [vol])
        world.commitParameters()

        self.camera = d.newCamera("perspective")
        r = self.renderer = d.newRenderer("default")
        r.setParameter("pixelSamples", anari.INT32, 1)
        r.setParameter("ambientRadiance", anari.FLOAT32, float(ambient))
        # Barney's OIDN pass over the accumulated frame (needs Barney built with
        # BARNEY_OIDN_CPU; otherwise ignored).
        r.setParameter("denoise", anari.BOOL, bool(denoise))
        # In scatter mode Barney writes sRGB and the background is linear; in emission
        # mode the frame is written as is, as Tack does.
        bg = tuple(c ** 2.2 if shading == "scatter" else c for c in BACKGROUND) + (1.0,)
        r.setParameter("background", anari.FLOAT32_VEC4, bg)
        r.commitParameters()
        self.frames = {}

    def set_camera(self, vtk_camera, aspect):
        c = self.camera
        c.setParameter("aspect", anari.FLOAT32, float(aspect))
        c.setParameter("position", anari.FLOAT32_VEC3, vtk_camera.GetPosition())
        c.setParameter("direction", anari.FLOAT32_VEC3, vtk_camera.GetDirectionOfProjection())
        c.setParameter("up", anari.FLOAT32_VEC3, vtk_camera.GetViewUp())
        c.setParameter("fovy", anari.FLOAT32, math.radians(vtk_camera.GetViewAngle()))
        c.commitParameters()

    def frame(self, size):
        f = self.frames.get(size)
        if f is None:
            f = self.frames[size] = self.device.newFrame()
            f.setParameter("size", anari.uint2, size)
            f.setParameter("channel.color", anari.DATA_TYPE,
                           anari.UFIXED8_RGBA_SRGB if self.shading == "scatter"
                           else anari.UFIXED8_VEC4)
            f.setParameter("renderer", anari.RENDERER, self.renderer)
            f.setParameter("camera", anari.CAMERA, self.camera)
            f.setParameter("world", anari.WORLD, self.world)
            f.commitParameters()
        return f

    def render(self, size):
        """One more sample per pixel at ``size``; the (h, w, 4) uint8 pixels, first
        row at the bottom."""
        f = self.frame(size)
        f.render()
        return np.asarray(f.get("channel.color"))


def default_camera(camera, amr):
    """Looking at the turbines (the finest level's blocks), as amrwind_view does."""
    lo, hi = finest_region(amr)
    centre, span = 0.5 * (lo + hi), 0.9 * float(np.max(hi - lo))
    camera.SetFocalPoint(*centre)
    camera.SetPosition(*(centre + np.array([-0.9, -1.3, 0.9]) * span))
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetViewAngle(30.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotfile", default=PLOTFILE)
    parser.add_argument("--field", default="mag_vorticity")
    parser.add_argument("--max-level", type=int, default=None)
    parser.add_argument("--size", default="2400x1500",
                        help="in pixels; points are half that on a Retina screen")
    parser.add_argument("--opaque-from", type=float, default=0.4)
    parser.add_argument("--opaque-to", type=float, default=2.0)
    parser.add_argument("--unit-distance", type=float, default=1.0,
                        help="the opacity's unit distance, in the finest level's cells")
    parser.add_argument("--ambient", type=float, default=3.0,
                        help="--shading scatter: the ambient radiance that lights the medium")
    parser.add_argument("--shading", choices=("emission", "scatter"), default="emission",
                        help="emission: Tack's emission-absorption model; scatter: "
                             "Barney's path-traced medium lit by --ambient")
    parser.add_argument("--denoise", action="store_true",
                        help="denoise each frame with Barney's OIDN pass")
    parser.add_argument("--moving-scale", type=int, default=4,
                        help="while moving, the frame is the window's size over this")
    parser.add_argument("--passes", type=int, default=16,
                        help="samples per pixel accumulated once the camera rests")
    parser.add_argument("--output", help="render one frame to this PNG and exit")
    args = parser.parse_args()

    t = time.perf_counter()
    # blank: refined cells marked, so that amr_arrays leaves them out.
    amr = read_plotfile(args.plotfile, fields=[args.field], max_level=args.max_level,
                        blank=True)
    print(f"read: {time.perf_counter() - t:.1f} s", flush=True)

    finest = amr.GetNumberOfLevels() - 1
    unit = args.unit_distance * float(level_spacing(amr, finest).min())
    t = time.perf_counter()
    scene = BarneyScene(amr, args.field, args.opaque_from, args.opaque_to, unit,
                        args.ambient, args.denoise, args.shading)
    print(f"ANARI scene: {time.perf_counter() - t:.1f} s", flush=True)

    size = tuple(int(v) for v in args.size.split("x"))
    renderer = vtkRenderer()
    default_camera(renderer.GetActiveCamera(), amr)

    if args.output:
        from PIL import Image

        scene.set_camera(renderer.GetActiveCamera(), size[0] / size[1])
        for k in range(args.passes):
            t = time.perf_counter()
            pixels = scene.render(size)
            print(f"pass {k + 1}: {1000 * (time.perf_counter() - t):.0f} ms", flush=True)
        Image.fromarray(pixels[::-1]).convert("RGB").save(args.output)
        print(f"wrote {args.output}")
        return

    # The window shows Barney's frame as the renderer's background texture.
    image = vtkImageData()
    texture = vtkTexture(interpolate=True)
    texture.SetInputData(image)
    renderer.TexturedBackgroundOn()
    renderer.SetBackgroundTexture(texture)
    window = vtkRenderWindow(size=size, window_name="Barney AMR volume: AMR-Wind vorticity")
    window.AddRenderer(renderer)
    interactor = vtkRenderWindowInteractor(render_window=window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    camera = renderer.GetActiveCamera()
    state = {"still": True, "passes": 0, "camera": None, "timer": None}

    def show(pixels):
        h, w = pixels.shape[:2]
        if image.GetDimensions()[:2] != (w, h):
            image.SetDimensions(w, h, 1)
        array = numpy_to_vtk(pixels.reshape(-1, 4), deep=True)
        image.GetPointData().SetScalars(array)
        image.Modified()

    def frame_size():
        w, h = window.GetSize()
        if state["still"]:
            return (w, h)
        s = args.moving_scale
        return (max(1, w // s), max(1, h // s))

    def before_render(*_):
        w, h = window.GetSize()
        key = (camera.GetMTime(), w, h)
        if key != state["camera"]:
            state["camera"] = key
            scene.set_camera(camera, w / h)
        t = time.perf_counter()
        show(scene.render(frame_size()))
        if state["still"]:
            state["passes"] += 1
            print(f"pass {state['passes']}: {1000 * (time.perf_counter() - t):.0f} ms",
                  flush=True)

    def cancel():
        timer = state.pop("timer", None)
        if timer is not None:
            interactor.DestroyTimer(timer)

    def interaction_started(*_):
        cancel()
        state["still"] = False

    def interaction_ended(*_):
        cancel()
        state["timer"] = interactor.CreateOneShotTimer(REST)

    def timer(*_):
        # The rest timer, then one per refining pass.
        state.pop("timer", None)
        if not state["still"]:
            state["still"], state["passes"] = True, 0
        window.Render()
        if state["passes"] < args.passes:
            state["timer"] = interactor.CreateOneShotTimer(1)

    def key(obj, _):
        if obj.GetKeySym() == "r":
            default_camera(camera, amr)
            interaction_started()
            interaction_ended()

    renderer.AddObserver("StartEvent", before_render)
    interactor.AddObserver("StartInteractionEvent", interaction_started)
    interactor.AddObserver("EndInteractionEvent", interaction_ended)
    interactor.AddObserver("TimerEvent", timer)
    interactor.AddObserver("KeyPressEvent", key)
    # A small frame first (Initialize renders on macOS), then the full one at rest.
    state["still"] = False
    interactor.Initialize()
    window.Render()
    interaction_ended()
    print("drag to rotate; r: reset, q: quit", flush=True)
    interactor.Start()


if __name__ == "__main__":
    main()
