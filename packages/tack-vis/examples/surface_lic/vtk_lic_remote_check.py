# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The VTK surface LIC (``vtk_lic``) through VTK's serialization to a client without the
Slang compiler, as a browser is: the server records the compiled Slang (the hook's
program and the LIC's passes), writes the states and blobs, and a second process with
``SLANGKIT_DISABLE_SLANG=1`` builds the scene from them, renders it, and compares its
picture with the server's. Then a second plane: only the changed states and new blobs.

    python vtk_lic_remote_check.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import vtkmodules.vtkRenderingCore as rr
from vtkmodules.vtkCommonCore import vtkTypeUInt8Array
from vtkmodules.vtkRenderingWebGPU import vtkWebGPUSurfaceLIC
from vtkmodules.vtkSerializationManager import vtkObjectManager

import disk
import tack
import vtk_lic


def write(directory, states, blobs, picture):
    os.makedirs(os.path.join(directory, "blobs"))
    with open(os.path.join(directory, "states.json"), "w") as f:
        json.dump(states, f)
    for h, blob in blobs.items():
        np.save(os.path.join(directory, "blobs", h + ".npy"), np.asarray(blob))
    np.save(os.path.join(directory, "picture.npy"), picture)


def camera_of(renderer, centre, span):
    camera = renderer.GetActiveCamera()
    camera.SetPosition(*(centre + np.array([1.4 * span, 0.35 * span, 0.4 * span])))
    camera.SetFocalPoint(*centre)
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetViewAngle(30.0)
    camera.SetClippingRange(0.2 * span, 4.0 * span)


def server(out):
    tack.set_default_device("cpu")
    data, bounds = disk.read(disk.DISK)
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    centre, span = 0.5 * (lo + hi), float(np.max(hi - lo))
    temp = data.fields["Temp"].values.numpy().reshape(-1)
    actor = vtk_lic.SliceActor(data, (float(temp.min()), float(temp.max())))
    actor.property.SetRecordSlangArtifacts(True)
    actor.cut(centre, (1.0, 0.0, 0.0))
    renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
    renderer.AddActor(actor.actor)
    lic = vtkWebGPUSurfaceLIC()
    lic.SetRecordSlangArtifacts(True)
    renderer.SetSurfaceLIC(lic)
    window = rr.vtkRenderWindow(size=(1280, 800), off_screen_rendering=True)
    window.AddRenderer(renderer)
    camera_of(renderer, centre, span)
    window.Render()
    window.Render()

    manager = vtkObjectManager()
    manager.Initialize()
    window_id = manager.RegisterObject(window)
    manager.UpdateStatesFromObjects()
    ids = manager.GetAllDependencies(0)
    states = [manager.GetState(i) for i in ids]
    blobs = {h: manager.GetBlob(h, True) for h in manager.GetBlobHashes(ids)}
    print(f"server: {len(states)} states ({sum(map(len, states)) / 1024:.0f} KiB), "
          f"{len(blobs)} blobs ({sum(b.GetNumberOfValues() for b in blobs.values()) / 1024:.0f}"
          f" KiB); artifacts: hook {actor.property.GetNumberOfSlangArtifacts()}, LIC "
          f"{lic.GetNumberOfSlangArtifacts()}", flush=True)
    write(os.path.join(out, "0"), states, blobs, vtk_lic.grab(window))
    with open(os.path.join(out, "window"), "w") as f:
        f.write(str(window_id))

    # Another plane: what changes.
    sent_states = {json.loads(s)["Id"]: s for s in states}
    sent_blobs = set(blobs)
    actor.cut(centre + np.array([0.0, 0.0, 1.0]), (1.0, 0.25, 0.15))
    window.Render()
    window.Render()
    manager.UpdateStatesFromObjects()
    ids = manager.GetAllDependencies(0)
    changed = [manager.GetState(i) for i in ids if manager.GetState(i) != sent_states.get(i)]
    new = {h: manager.GetBlob(h, True) for h in manager.GetBlobHashes(ids) if h not in sent_blobs}
    names = sorted({json.loads(s)["ClassName"] for s in changed})
    print(f"  a new plane: {len(changed)} states ({', '.join(names)}), {len(new)} blobs "
          f"({sum(b.GetNumberOfValues() for b in new.values()) / 1024:.0f} KiB); Tack: slice "
          f"{actor.timings['slice']:.0f} ms, pack {actor.timings['pack']:.0f} ms", flush=True)
    write(os.path.join(out, "1"), changed, new, vtk_lic.grab(window))


def client(out):
    from vtkmodules.vtkRenderingSlang import vtkSlangShaderManager

    print(f"client: Slang available: {vtkSlangShaderManager.IsSlangAvailable()}", flush=True)
    manager = vtkObjectManager()
    manager.Initialize()
    with open(os.path.join(out, "window")) as f:
        window_id = int(f.read())
    window = None
    for k in (0, 1):
        directory = os.path.join(out, str(k))
        with open(os.path.join(directory, "states.json")) as f:
            for s in json.load(f):
                manager.RegisterState(s)
        for name in os.listdir(os.path.join(directory, "blobs")):
            data = np.load(os.path.join(directory, "blobs", name))
            blob = vtkTypeUInt8Array()
            blob.SetNumberOfValues(len(data))
            np.asarray(blob)[:] = data
            manager.RegisterBlob(name[:-4], blob)
        manager.UpdateObjectsFromStates()
        if window is None:
            window = manager.GetObjectAtId(window_id)
            window.SetOffScreenRendering(True)
        window.Render()
        window.Render()
        d = np.abs(np.load(os.path.join(directory, "picture.npy")).astype(int)
                   - vtk_lic.grab(window).astype(int)).max(axis=2)
        print(f"  message {k}: {100 * float((d > 2).mean()):.3f}% of pixels differ by more "
              f"than 2/255 from the server's (max {int(d.max())})", flush=True)


def main():
    if "--client" in sys.argv:
        client(os.environ["LIC_REMOTE_DIR"])
        return
    with tempfile.TemporaryDirectory() as out:
        server(out)
        env = dict(os.environ, SLANGKIT_DISABLE_SLANG="1", LIC_REMOTE_DIR=out)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--client"], env=env,
                       check=True)


if __name__ == "__main__":
    main()
