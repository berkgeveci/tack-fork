# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A volume rendered by Tack inside a VTK WebGPU scene, through ``TackVolumeMapper``.

    VTK draws the opaque geometry ─► TackVolumeMapper (depth in, Tack's picture out)
                                  ─► VTK blends it over the frame

The test hierarchy with a sphere, a slab and the boxes' outlines, or AMR-Wind with a
ground and a tower as stand-ins (``--amrwind``, ``--close``). With ``--check OUTDIR``
it renders off screen, times still and moving frames, and compares the frame with the
same picture blended on the host (``tack_composite.composite``); otherwise it opens an
interactive window.

    python tack_view.py [--amrwind] [--close] [--device D] [--size WxH] [--check OUTDIR]
                        [--backend Vulkan]
"""

import argparse
import os
import sys
import time

import numpy as np
import vtkmodules.vtkRenderingCore as rr
from tack_volume_mapper import TackVolumeMapper
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkCylinderSource, vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction, vtkVolume, vtkVolumeProperty

import tack

BACKGROUND = (0.32, 0.34, 0.43)


def actor_of(source, colour):
    mapper = rr.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = rr.vtkActor(mapper=mapper)
    actor.GetProperty().SetColor(*colour)
    return source, actor


def outlines(amr):
    """The test hierarchy's boxes as outlines, coloured by level."""
    import amr_case
    from vtkmodules.vtkFiltersSources import vtkOutlineSource

    colours = [(0.9, 0.9, 0.9), (1.0, 0.85, 0.3), (0.3, 1.0, 0.5)]
    made = []
    for level in range(amr.num_levels):
        h = amr_case.spacing(level)
        for lo, hi in amr.boxes(level):
            a = amr_case.ORIGIN + lo * h
            b = amr_case.ORIGIN + hi * h
            source = vtkOutlineSource(bounds=(a[0], b[0], a[1], b[1], a[2], b[2]))
            made.append(actor_of(source, colours[level % len(colours)]))
    return made


def volume_property(points, opacities, unit):
    color = vtkColorTransferFunction()
    color.SetColorSpaceToDiverging()
    for x, rgb in points:
        color.AddRGBPoint(x, *rgb)
    opacity = vtkPiecewiseFunction()
    for x, a in opacities:
        opacity.AddPoint(x, a)
    prop = vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.SetScalarOpacityUnitDistance(unit)
    return prop


def grab(window):
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

    g = vtkWindowToImageFilter(input=window, input_buffer_type=3)
    g.Update()
    img = g.GetOutput()
    w, h, _ = img.GetDimensions()
    rgb = np.asarray(img.GetPointData().GetScalars()).reshape(h, w, -1)[..., :3]
    return np.ascontiguousarray(np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)],
                                               axis=2)[::-1])


def write_png(path, rgba):
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkIOImage import vtkPNGWriter

    h, w, _ = rgba.shape
    image = vtkImageData(dimensions=(w, h, 1))
    array = vtkUnsignedCharArray(np.ascontiguousarray(rgba[::-1]).reshape(-1, 4))
    array.SetName("rgba")
    image.GetPointData().SetScalars(array)
    writer = vtkPNGWriter(file_name=path)
    writer.SetInputData(image)
    writer.Write()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amrwind", action="store_true")
    parser.add_argument("--plotfile", default="~/Data/amrwind/plt255900")
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", default="1280x800")
    parser.add_argument("--check", default=None, metavar="OUTDIR")
    parser.add_argument("--backend", default=None,
                        help="the WebGPU backend VTK draws with (Metal, Vulkan; Vulkan is "
                             "SwiftShader, on the CPU, when that is the Vulkan driver)")
    args = parser.parse_args()
    tack.set_default_device(args.device)
    size = tuple(int(v) for v in args.size.split("x"))

    props = []
    if args.amrwind:
        sys.path.insert(0, os.path.expanduser("~/Work/amrex-reader"))
        from amrex_reader import read_plotfile

        data = read_plotfile(os.path.expanduser(args.plotfile), fields=["mag_vorticity"])
        t = time.perf_counter()
        mapper = TackVolumeMapper(data, "mag_vorticity")
        print(f"Tack's volume {time.perf_counter() - t:.1f} s", flush=True)
        b = mapper.tack.bounds
        centre = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        if args.close:
            amr, finest = mapper.tack.amr, mapper.tack.amr.num_levels - 1
            boxes = amr.boxes(finest)
            h = np.array(mapper.tack.box.spacing.numpy()[3 * finest:3 * finest + 3], float)
            lo = np.array(b[::2]) + boxes[:, 0].min(axis=0) * h
            hi = np.array(b[::2]) + boxes[:, 1].max(axis=0) * h
            centre, span = 0.5 * (lo + hi), 0.9 * float(np.max(hi - lo))
            ground = vtkCubeSource(bounds=(b[0], b[1], b[2], b[3], b[4], b[4] + 0.004 * span))
            props.append(actor_of(ground, (0.35, 0.3, 0.25)))
            height = 0.5 * (hi[2] - b[4])
            tower = vtkCylinderSource(radius=0.012 * span, height=height, resolution=32,
                                      center=(lo[0] + 0.3 * (hi[0] - lo[0]), b[4] + height / 2,
                                              -0.5 * (lo[1] + hi[1])))
            source, actor = actor_of(tower, (0.9, 0.9, 0.9))
            actor.RotateX(90.0)         # the cylinder's axis is y: stand it up along z
            props.append((source, actor))
        view = (centre, centre + np.array([-0.9, -1.3, 0.9]) * span)
        prop = volume_property(
            [(0.4, (0.23, 0.30, 0.75)), (1.2, (0.86, 0.86, 0.86)), (2.0, (0.71, 0.02, 0.15))],
            [(0.0, 0.0), (0.4, 0.0), (1.2, 0.05), (2.0, 0.4)], mapper.tack.cell)
    else:
        import amr_case

        amr = amr_case.hierarchy()
        mapper = TackVolumeMapper(amr_case.as_vtk(amr), "values", macrocells=64)
        b = mapper.tack.bounds
        sphere = vtkSphereSource(center=(0.35, 0.25, -0.1), radius=0.22, theta_resolution=64,
                                 phi_resolution=64)
        slab = vtkCubeSource(bounds=(-0.9, 0.1, -0.75, -0.7, -0.6, 0.6))
        props = [actor_of(sphere, (0.95, 0.8, 0.3)), actor_of(slab, (0.4, 0.75, 0.5))]
        props += outlines(amr)
        view = ((0.0, 0.0, 0.0), (1.6, -3.6, 2.4))
        lo, hi = mapper.tack.range
        r = hi - lo
        prop = volume_property(
            [(lo, (0.23, 0.30, 0.75)), (lo + 0.5 * r, (0.86, 0.86, 0.86)),
             (hi, (0.71, 0.02, 0.15))],
            [(lo, 0.0), (lo + 0.3 * r, 0.0), (lo + 0.55 * r, 0.08), (hi, 0.6)],
            mapper.tack.cell)

    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    renderer = rr.vtkRenderer(background=BACKGROUND)
    for _, actor in props:
        renderer.AddActor(actor)
    renderer.AddVolume(volume)
    window = rr.vtkRenderWindow(size=size, off_screen_rendering=bool(args.check))
    window.AddRenderer(renderer)
    drawn_by = ""
    if args.backend:
        from vtkmodules.vtkRenderingWebGPU import vtkWebGPUConfiguration

        configuration = vtkWebGPUConfiguration()
        configuration.SetBackend(getattr(vtkWebGPUConfiguration.BackendType, args.backend))
        window.SetWGPUConfiguration(configuration)
        window.Initialize()
        drawn_by = (f" on {configuration.GetBackendInUseAsString()} "
                    f"({' '.join(configuration.ReportCapabilities().split())[:160]})")
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*view[0])
    camera.SetPosition(*view[1])
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetViewAngle(30.0)
    renderer.ResetCameraClippingRange(*b)
    print(f"render window: {window.GetClassName()}{drawn_by}, Tack on {args.device}", flush=True)

    if not args.check:
        interactor = rr.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        window.Render()
        interactor.Start()
        return

    os.makedirs(args.check, exist_ok=True)

    def frames(count=5):
        times, marches = [], []
        for _ in range(count):
            t = time.perf_counter()
            window.Render()
            window.WaitForCompletion()
            times.append(1000 * (time.perf_counter() - t))
            marches.append(mapper.timings["march_ms"])
        return float(np.median(times)), float(np.median(marches))

    window.Render()
    window.Render()
    still, march = frames()
    picture = grab(window)
    inside = mapper.timings["render_image_ms"]
    print(f"still at {size[0]}x{size[1]}: frame {still:.1f} ms (Tack's march {march:.1f}, all of "
          f"RenderImage {inside:.1f})", flush=True)
    tag = "view_amrwind" + ("_close" if args.close else "") if args.amrwind else "view"
    write_png(os.path.join(args.check, f"{tag}.png"), picture)
    window.SetDesiredUpdateRate(15.0)
    window.Render()
    moving, march = frames()
    write_png(os.path.join(args.check, f"{tag}_moving.png"), grab(window))
    print(f"moving: frame {moving:.1f} ms (Tack's march {march:.1f}, picture "
          f"{mapper.timings['size'][0]}x{mapper.timings['size'][1]})", flush=True)
    window.SetDesiredUpdateRate(0.0001)

    # The same picture blended on the host: the frame without the volume, its depth,
    # Tack's picture to that depth, composited.
    from tack_composite import composite, frame_depth

    volume.SetVisibility(False)
    window.Render()
    window.WaitForCompletion()
    plain = grab(window)
    depth = frame_depth(window)
    table, lo, hi = mapper._functions(volume)
    unit = prop.GetScalarOpacityUnitDistance(0) / mapper.tack.cell
    expected = composite(plain, mapper.tack.render(camera, size, table, lo, hi, depth,
                                                   mapper.sample_distance, unit))
    d = np.abs(picture[..., :3].astype(int) - expected[..., :3].astype(int)).max(axis=2)
    print(f"the mapper's frame vs the host's blend: {100 * float((d > 2).mean()):.3f}% of "
          f"pixels differ by more than 2/255 (max {int(d.max())}); the volume changes "
          f"{100 * float((np.abs(picture[..., :3].astype(int) - plain[..., :3].astype(int)).max(axis=2) > 2).mean()):.1f}% "
          f"of the frame", flush=True)
    write_png(os.path.join(args.check, f"{tag}_expected.png"), expected)
    print(f"  pictures in {os.path.abspath(args.check)}", flush=True)


if __name__ == "__main__":
    main()
