# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The hooks' scene for a client without the Slang compiler, as a browser's WASM VTK is.

The server (this process) turns on ``RecordSlangArtifacts`` on each group's shader
property, so its mapper keeps every Slang compile (WGSL and reflection) on the property;
the artifacts travel with the property's state, as arrays with their blobs. The server
writes what a remote session would send, one directory per message (the first picture,
a field step, a geometry step): the states that changed, the blobs the client has not
seen, and the server's picture. A second process, the client, run with
``SLANGKIT_DISABLE_SLANG=1``, applies them in order, renders, compares its picture with
the server's and checks its pixels against Tack's evaluation (``demo.check``).

``--no-artifacts`` strips the artifacts from the states: the control, a client that
cannot draw the hooks.

    python precompiled.py [straight|curved|tets] [n] [--no-artifacts]
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from roundtrip import ClientGroup, demo, serialize
from tack_hooks import surface_actors, update_surface
from vtkmodules.vtkCommonCore import vtkTypeUInt8Array
from vtkmodules.vtkSerializationManager import vtkObjectManager

import tack

HERE = os.path.dirname(os.path.abspath(__file__))


def geometry_step(kind, n):
    import importlib.util

    spec = importlib.util.spec_from_file_location("hooks_geometry_step",
                                                  os.path.join(HERE, "geometry_step.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.step_surface(kind, n, "deform", 1)


def new_field(surf, lo, hi):
    old = surf.fields["f"]
    surf.fields["f"] = demo.td.Field(old.space, demo.tack.array((lo + hi) - old.values.numpy()))
    return surf


def write(directory, states, blobs, picture):
    os.makedirs(os.path.join(directory, "blobs"))
    with open(os.path.join(directory, "states.json"), "w") as f:
        json.dump(states, f)
    for h, blob in blobs.items():
        np.save(os.path.join(directory, "blobs", h + ".npy"), np.asarray(blob))
    np.save(os.path.join(directory, "picture.npy"), picture)


def artifact_ids(states):
    """The ids of the artifact arrays the shader properties reference."""
    ids = set()
    for s in states:
        for ref in json.loads(s).get("SlangArtifacts", []):
            ids.add(ref["Id"])
    return ids


def server(kind, n, out):
    tack.set_default_device("cpu")
    surf = demo.tets_surface(n) if kind == "tets" else demo.surface(kind, n)
    groups, (lo, hi) = surface_actors(surf, "f")
    for g in groups:
        g.actor.GetShaderProperty().SetRecordSlangArtifacts(True)
    window, renderer = demo.window_for(groups)
    start = time.perf_counter()
    demo.frame(window, renderer, 0, 1)
    print(f"server: first frame {1000 * (time.perf_counter() - start):.0f} ms")
    demo.frame(window, renderer, 0, 30)

    manager, window_id, states, blobs = serialize(window)
    ids = artifact_ids(states)
    by_id = {json.loads(s)["Id"]: json.loads(s) for s in states}
    artifact_bytes = sum(blobs[by_id[i]["Hash"]].GetNumberOfValues() for i in ids
                         if "Hash" in by_id[i])
    print(f"server, {kind} {n}: {len(states)} states, {len(blobs)} blobs "
          f"({sum(b.GetNumberOfValues() for b in blobs.values()) / 2**20:.1f} MB); "
          f"{len(ids)} Slang artifacts, {artifact_bytes / 1024:.0f} KiB")
    write(os.path.join(out, "0"), states, blobs, demo.grab(window))
    with open(os.path.join(out, "window"), "w") as f:
        f.write(str(window_id))

    sent_states = {json.loads(s)["Id"]: s for s in states}
    sent_blobs = set(blobs)
    steps = []
    if kind != "tets":
        def field():
            new_field(surf, lo, hi)
            for g in groups:
                g.update_field(surf, "f")

        def geometry():
            update_surface(groups, geometry_step(kind, n), "f")

        steps = [field, geometry]
    for k, change in enumerate(steps, 1):
        change()
        demo.frame(window, renderer, 0, 30)
        manager.UpdateStatesFromObjects()
        ids_now = manager.GetAllDependencies(0)
        changed = [manager.GetState(i) for i in ids_now
                   if manager.GetState(i) != sent_states.get(i)]
        new = {h: manager.GetBlob(h, True) for h in manager.GetBlobHashes(ids_now)
               if h not in sent_blobs}
        if os.environ.get("DIFF"):
            for st in changed:
                state = json.loads(st)
                if state["Id"] in sent_states and "UInt8" in state["ClassName"]:
                    before = json.loads(sent_states[state["Id"]])
                    print("   ", state["ClassName"], {key: (before.get(key), state.get(key))
                                                     for key in set(state) | set(before)
                                                     if before.get(key) != state.get(key)})
        sent_states.update({json.loads(s)["Id"]: s for s in changed})
        sent_blobs |= set(new)
        write(os.path.join(out, str(k)), changed, new, demo.grab(window))
        names = sorted({json.loads(st)["ClassName"] for st in changed})
        print(f"  step {k}: {len(changed)} states ({', '.join(names)}), {len(new)} blobs "
              f"({sum(b.GetNumberOfValues() for b in new.values()) / 2**20:.2f} MB)")
    return len(steps) + 1


def client(kind, n, out, strip):
    from vtkmodules.vtkRenderingSlang import vtkSlangShaderManager

    tack.set_default_device("cpu")
    print(f"client: Slang available: {vtkSlangShaderManager.IsSlangAvailable()}"
          f"{', artifacts stripped' if strip else ''}")
    # What the check needs from the server's groups (their place in the surface), made
    # here as the server made them; nothing of it is drawn.
    surf = demo.tets_surface(n) if kind == "tets" else demo.surface(kind, n)
    groups, (lo, hi) = surface_actors(surf, "f")
    with open(os.path.join(out, "window")) as f:
        window_id = int(f.read())

    manager = vtkObjectManager()
    manager.Initialize()
    window = renderer = actors = None
    k = 0
    while os.path.isdir(os.path.join(out, str(k))):
        directory = os.path.join(out, str(k))
        with open(os.path.join(directory, "states.json")) as f:
            states = json.load(f)
        if strip:
            stripped = []
            for s in states:
                state = json.loads(s)
                if "SlangArtifacts" in state:
                    state["SlangArtifacts"] = []
                stripped.append(json.dumps(state))
            states = stripped
        for s in states:
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
            renderer = window.GetRenderers().GetFirstRenderer()
            actors = list(renderer.GetActors())
            recorded = [a.GetShaderProperty().GetNumberOfSlangArtifacts() for a in actors]
            print(f"  {len(actors)} actors, artifacts per shader property {recorded}")
        if k == 1:
            new_field(surf, lo, hi)
        elif k == 2:
            surf = geometry_step(kind, n)
            update_surface(groups, surf, "f")
        start = time.perf_counter()
        demo.frame(window, renderer, 0, 1)
        first = time.perf_counter() - start
        demo.frame(window, renderer, 0, 30)
        d = np.abs(np.load(os.path.join(directory, "picture.npy")) - demo.grab(window)).max(axis=2)
        print(f"  message {k}: first frame {1000 * first:.0f} ms; vs server picture "
              f"{100 * float((d > 2).mean()):.3f}% of pixels differ by more than 2/255 "
              f"(max {int(d.max())})")
        pixels, faces, worst, median = demo.check(
            surf, [ClientGroup(a, g) for a, g in zip(actors, groups, strict=True)],
            lo, hi, window, renderer)
        print(f"    check: {pixels} pixels on {faces} faces; vs Tack max {worst:.2e}, "
              f"median {median:.2e}")
        k += 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    kind = args[0] if args else "curved"
    n = int(args[1]) if len(args) > 1 else 16
    if "--client" in flags:
        client(kind, n, os.environ["PRECOMPILED_DIR"], "--no-artifacts" in flags)
        return
    with tempfile.TemporaryDirectory() as out:
        server(kind, n, out)
        env = dict(os.environ, SLANGKIT_DISABLE_SLANG="1", PRECOMPILED_DIR=out)
        command = [sys.executable, os.path.abspath(__file__), kind, str(n), "--client"]
        command += [f for f in flags if f == "--no-artifacts"]
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
