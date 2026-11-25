# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A volume rendered by Tack, blended with opaque geometry VTK's WebGPU renderer drew.

    VTK: the opaque scene ─► the frame's colour and depth (host arrays)
    Tack: rays marched to that depth (``tack_composite.TackRenderer``) ─► RGBA
    the RGBA over the frame's colour

The test hierarchy (``amr_case``) with a sphere inside a blob, a slab through the ring
and the boxes' outlines; or AMR-Wind with stand-ins for the turbines' towers
(``--amrwind --close``). Checked: the depth buffer against the sphere's known distance,
and no volume in front of geometry that lies before the box.

    python composite_demo.py [OUTDIR] [--size WxH] [--amrwind] [--close] [--device D]
"""

import argparse
import os
import sys
import time

import numpy as np
import vtkmodules.vtkRenderingCore as rr
from tack_composite import TackRenderer, composite, depth_distances, frame_depth
from volume_demo import grab, write_png
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkCylinderSource, vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction

import tack

BACKGROUND = (0.32, 0.34, 0.43)


def actor_of(source, colour):
    mapper = rr.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = rr.vtkActor(mapper=mapper)
    actor.GetProperty().SetColor(*colour)
    return source, actor


def functions(points, opacities):
    color = vtkColorTransferFunction()
    color.SetColorSpaceToDiverging()
    for x, rgb in points:
        color.AddRGBPoint(x, *rgb)
    opacity = vtkPiecewiseFunction()
    for x, a in opacities:
        opacity.AddPoint(x, a)
    return color, opacity


def test_case():
    import amr_case
    import amr_demo

    amr = amr_case.hierarchy()
    data = amr_case.as_vtk(amr)
    sphere = vtkSphereSource(center=(0.35, 0.25, -0.1), radius=0.22, theta_resolution=64,
                             phi_resolution=64)
    slab = vtkCubeSource(bounds=(-0.9, 0.1, -0.75, -0.7, -0.6, 0.6))
    props = [actor_of(sphere, (0.95, 0.8, 0.3)), actor_of(slab, (0.4, 0.75, 0.5))]
    props += amr_demo.outlines(amr)
    view = {"focal": (0.0, 0.0, 0.0), "position": (1.6, -3.6, 2.4)}
    return data, "values", props, view, None, sphere


def amrwind_case(args):
    sys.path.insert(0, os.path.expanduser("~/Work/amrex-reader"))
    from amrex_reader import read_plotfile

    t = time.perf_counter()
    data = read_plotfile(os.path.expanduser(args.plotfile), fields=["mag_vorticity"],
                         max_level=args.max_level)
    print(f"read {time.perf_counter() - t:.1f} s", flush=True)
    return data, "mag_vorticity", [], None, (0.4, 2.0), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=".")
    parser.add_argument("--size", default="1280x800")
    parser.add_argument("--amrwind", action="store_true")
    parser.add_argument("--plotfile", default="~/Data/amrwind/plt255900")
    parser.add_argument("--max-level", type=int, default=None)
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-distance", type=float, default=0.5)
    args = parser.parse_args()
    tack.set_default_device(args.device)
    size = tuple(int(v) for v in args.size.split("x"))
    os.makedirs(args.out, exist_ok=True)

    data, array, props, view, opaque, sphere = (amrwind_case(args) if args.amrwind
                                                else test_case())
    t = time.perf_counter()
    tack_renderer = TackRenderer(data, array, macrocells=128 if args.amrwind else 64)
    print(f"Tack's volume {time.perf_counter() - t:.1f} s", flush=True)
    b = tack_renderer.bounds
    centre = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    if args.amrwind:
        if args.close:
            amr, finest = tack_renderer.amr, tack_renderer.amr.num_levels - 1
            boxes = amr.boxes(finest)
            h = np.array(tack_renderer.box.spacing.numpy()[3 * finest:3 * finest + 3],
                         np.float64)
            lo = np.array(b[::2]) + boxes[:, 0].min(axis=0) * h
            hi = np.array(b[::2]) + boxes[:, 1].max(axis=0) * h
            centre, span = 0.5 * (lo + hi), 0.9 * float(np.max(hi - lo))
            # Stand-ins for the turbines: towers under the finest boxes, and the ground.
            ground = vtkCubeSource(bounds=(b[0], b[1], b[2], b[3], b[4], b[4] + 0.004 * span))
            props.append(actor_of(ground, (0.35, 0.3, 0.25)))
            for fx in (0.3, 0.7):
                x = lo[0] + fx * (hi[0] - lo[0])
                y = 0.5 * (lo[1] + hi[1])
                height = 0.5 * (hi[2] - b[4])
                tower = vtkCylinderSource(radius=0.012 * span, height=height, resolution=32,
                                          center=(x, b[4] + height / 2, -y))
                source, actor = actor_of(tower, (0.9, 0.9, 0.9))
                actor.RotateX(90.0)     # the cylinder's axis is y: stand it up along z
                props.append((source, actor))
        view = {"focal": centre, "position": centre + np.array([-0.9, -1.3, 0.9]) * span}
        lo_v, hi_v = opaque
        color, opacity = functions(
            [(lo_v, (0.23, 0.30, 0.75)), (0.5 * (lo_v + hi_v), (0.86, 0.86, 0.86)),
             (hi_v, (0.71, 0.02, 0.15))],
            [(0.0, 0.0), (lo_v, 0.0), (0.5 * (lo_v + hi_v), 0.05), (hi_v, 0.4)])
        lo_t, hi_t = tack_renderer.range
    else:
        lo_t, hi_t = tack_renderer.range
        r = hi_t - lo_t
        color, opacity = functions(
            [(lo_t, (0.23, 0.30, 0.75)), (lo_t + 0.5 * r, (0.86, 0.86, 0.86)),
             (hi_t, (0.71, 0.02, 0.15))],
            [(lo_t, 0.0), (lo_t + 0.3 * r, 0.0), (lo_t + 0.55 * r, 0.08), (hi_t, 0.6)])
    table = tack_renderer.table(color, opacity, lo_t, hi_t)

    renderer = rr.vtkRenderer(background=BACKGROUND)
    for _, actor in props:
        renderer.AddActor(actor)
    window = rr.vtkRenderWindow(size=size, off_screen_rendering=True)
    window.AddRenderer(renderer)
    print(f"render window: {window.GetClassName()}", flush=True)
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*view["focal"])
    camera.SetPosition(*view["position"])
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetViewAngle(30.0)
    renderer.ResetCameraClippingRange(*b)

    def frame():
        """One composited frame and its parts' times (ms)."""
        t0 = time.perf_counter()
        window.Render()
        window.WaitForCompletion()
        t1 = time.perf_counter()
        depth = frame_depth(window)
        colour = grab(window)
        t2 = time.perf_counter()
        volume = tack_renderer.render(camera, size, table, lo_t, hi_t, depth,
                                      args.sample_distance)
        t3 = time.perf_counter()
        picture = composite(colour, volume)
        t4 = time.perf_counter()
        times = [1000 * (b - a) for a, b in ((t0, t1), (t1, t2), (t2, t3), (t3, t4))]
        return colour, depth, volume, picture, times

    frame()
    colour, depth, volume, picture, times = frame()
    print(f"frame at {size[0]}x{size[1]} on {args.device}: opaque render {times[0]:.1f} ms, "
          f"depth and colour read {times[1]:.1f} ms, Tack march {times[2]:.1f} ms, "
          f"blend {times[3]:.1f} ms", flush=True)
    parts = tack_renderer.timings
    print(f"  of Tack's {times[2]:.1f} ms: the rays' kernel and uploads "
          f"{parts['rays_ms']:.1f}, the march kernel {parts['march_ms']:.1f}, read "
          f"{parts['read_ms']:.1f}", flush=True)
    for _ in range(3):
        _, _, _, _, again = frame()
        print(f"  again: Tack {again[2]:.1f} ms, the march kernel "
              f"{tack_renderer.timings['march_ms']:.1f}", flush=True)
    tag = "composite_amrwind" + ("_close" if args.close else "") if args.amrwind \
        else "composite"
    write_png(os.path.join(args.out, f"{tag}_frame.png"), colour)
    alone = composite(np.zeros_like(colour), volume)
    alone[..., 3] = 255
    write_png(os.path.join(args.out, f"{tag}_volume.png"), alone)
    write_png(os.path.join(args.out, f"{tag}.png"), picture)

    # Checks.
    distances = depth_distances(camera, size, depth).reshape(size[1], size[0])
    drawn = depth < 1.0
    print(f"depth: {100 * drawn.mean():.1f}% of pixels hold geometry, values "
          f"{depth.min():.4f} to {depth.max():.4f}", flush=True)
    if sphere is not None:
        # The sphere's nearest point to the camera, and the pixel it projects to.
        c = np.array(sphere.center)
        eye = np.array(camera.GetPosition())
        towards = (eye - c) / np.linalg.norm(eye - c)
        nearest = c + sphere.radius * towards
        near_plane = camera.GetClippingRange()[0]
        m = camera.GetCompositeProjectionTransformMatrix(size[0] / size[1], 0, 1)
        clip = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)]) \
            @ np.append(nearest, 1.0)
        px = int((clip[0] / clip[3] + 1) / 2 * size[0])
        py = int((1 - clip[1] / clip[3]) / 2 * size[1])
        # Along the pixel's ray, from the near plane (the ray is nearly the view axis).
        forward = np.array(camera.GetDirectionOfProjection())
        ray = (nearest - eye) / np.linalg.norm(nearest - eye)
        expected = np.linalg.norm(nearest - eye) - near_plane / float(ray @ forward)
        print(f"depth at the sphere's nearest point (pixel {px}, {py}): "
              f"{distances[py, px]:.4f} from the near plane, expected {expected:.4f}",
              flush=True)
    near, d, tmin, tmax = __import__("volume_tack").rays(camera, size, b)
    before = (distances.reshape(-1) <= tmin) & (tmax > tmin)
    alpha = volume[..., 3].reshape(-1)
    print(f"geometry before the box on {int(before.sum())} pixels: largest volume alpha "
          f"there {float(alpha[before].max()) if before.any() else 0.0:.4f}", flush=True)
    hidden = volume[..., 3] > 0
    print(f"volume on {100 * hidden.mean():.1f}% of pixels; over geometry on "
          f"{100 * (hidden & drawn).mean():.1f}%", flush=True)
    print(f"  pictures in {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
