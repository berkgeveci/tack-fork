# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""AMR-Wind's vorticity in an interactive window: the plotfile through the AMReX
reader, ``TackVolume`` and ``VolumeMapper``, on VTK's WebGPU renderer.

Drag to rotate (right button or wheel to zoom, middle to pan): while the camera moves
the mapper's interactive uniforms apply (longer, jittered steps); once it has rested
``REST`` ms the frame is drawn in full. The frame times are printed then.

Keys: ``l`` turns the interactive steps off and on (to compare), ``r`` resets the
camera to the turbines, ``q`` quits.

    python amrwind_view.py [--plotfile P] [--max-level L] [--size WxH]
"""

import argparse
import os
import sys
import time

import numpy as np
import vtkmodules.vtkRenderingCore as rr
from amrwind_demo import PLOTFILE, transfer_functions
from tack_volume import TackVolume
from volume_mapper import VolumeMapper
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

import tack

sys.path.insert(0, os.path.expanduser("~/Work/amrex-reader"))

from amrex_reader import read_plotfile

#: How long the camera rests before the full frame is drawn, in ms.
REST = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotfile", default=PLOTFILE)
    parser.add_argument("--field", default="mag_vorticity")
    parser.add_argument("--max-level", type=int, default=None)
    parser.add_argument("--size", default="1600x1000")
    args = parser.parse_args()
    tack.set_default_device("cpu")

    t = time.perf_counter()
    source = read_plotfile(args.plotfile, fields=[args.field], max_level=args.max_level)
    volume = TackVolume(args.field, macrocells=128)
    volume.SetInputDataObject(source)
    volume.Update()
    print(f"read and Tack: {time.perf_counter() - t:.1f} s; "
          f"{len(volume.box.values) / 1e6:.0f}M cells, pool "
          f"{volume.timings['pool_bytes'] / 2**20:.0f} MiB", flush=True)

    mapper = VolumeMapper()
    mapper.SetInputConnection(volume.GetOutputPort())
    transfer_functions(mapper, 0.4, 2.0)
    actor = rr.vtkActor(mapper=mapper)
    actor.ForceTranslucentOn()
    renderer = rr.vtkRenderer(background=(0.32, 0.34, 0.43))
    renderer.AddActor(actor)
    size = tuple(int(v) for v in args.size.split("x"))
    window = rr.vtkRenderWindow(size=size, window_name="Tack AMR volume: AMR-Wind vorticity")
    window.AddRenderer(renderer)
    interactor = rr.vtkRenderWindowInteractor(render_window=window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

    # The finest level's boxes, from nearer: the turbines.
    b = volume.GetOutputDataObject(0).GetFieldData().GetArray("TackBounds").GetTuple(0)
    finest = volume.amr.num_levels - 1
    boxes = volume.amr.boxes(finest)
    h = np.array(volume.box.spacing.numpy()[3 * finest:3 * finest + 3], np.float64)
    lo = np.array(b[::2]) + boxes[:, 0].min(axis=0) * h
    hi = np.array(b[::2]) + boxes[:, 1].max(axis=0) * h
    centre, span = 0.5 * (lo + hi), 0.9 * float(np.max(hi - lo))
    camera = renderer.GetActiveCamera()

    def reset():
        camera.SetFocalPoint(*centre)
        camera.SetPosition(*(centre + np.array([-0.9, -1.3, 0.9]) * span))
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.SetViewAngle(30.0)
        renderer.ResetCameraClippingRange()

    reset()
    prop = mapper.slang
    times = {"moving": [], "still": []}
    clock = {}

    def started(*_):
        clock["t"] = time.perf_counter()

    def ended(*_):
        window.WaitForCompletion()
        moving = window.GetDesiredUpdateRate() >= prop.GetInteractiveUpdateRate()
        times["moving" if moving else "still"].append(1000 * (time.perf_counter() - clock["t"]))

    def report():
        moving, still = times["moving"], times["still"]
        if moving and still:
            lod = "on" if prop.GetInteractiveUpdateRate() < 1e9 else "off"
            print(f"moving ({len(moving)} frames, interactive steps {lod}): median "
                  f"{np.median(moving):.0f} ms; then still {still[-1]:.0f} ms", flush=True)
        moving.clear()

    # VTK's interactor style draws a full frame at the end of every interaction, and a
    # wheel tick is one: zooming would draw one per tick. Instead every interaction
    # ends at the moving frames' rate, and one full frame is drawn once the camera has
    # rested ``REST`` ms.
    interactor.SetStillUpdateRate(interactor.GetDesiredUpdateRate())
    pending = {}

    def rested(*_):
        pending.pop("timer", None)
        window.SetDesiredUpdateRate(0.0001)
        window.Render()
        window.SetDesiredUpdateRate(interactor.GetDesiredUpdateRate())
        report()

    def interaction_started(*_):
        timer = pending.pop("timer", None)
        if timer is not None:
            interactor.DestroyTimer(timer)

    def interaction_ended(*_):
        interaction_started()
        pending["timer"] = interactor.CreateOneShotTimer(REST)

    def key(obj, _):
        k = obj.GetKeySym()
        if k == "l":
            off = prop.GetInteractiveUpdateRate() < 1e9
            prop.SetInteractiveUpdateRate(1e30 if off else 1.0)
            print(f"interactive steps {'off' if off else 'on'}", flush=True)
        elif k == "r":
            reset()
            rested()

    # The window's events: it submits the frame after the renderer's EndEvent.
    window.AddObserver("StartEvent", started)
    window.AddObserver("EndEvent", ended)
    interactor.AddObserver("StartInteractionEvent", interaction_started)
    interactor.AddObserver("EndInteractionEvent", interaction_ended)
    interactor.AddObserver("TimerEvent", rested)
    interactor.AddObserver("KeyPressEvent", key)
    window.Render()
    print("drag to rotate; l: interactive steps off/on, r: reset, q: quit", flush=True)
    interactor.Start()


if __name__ == "__main__":
    main()
