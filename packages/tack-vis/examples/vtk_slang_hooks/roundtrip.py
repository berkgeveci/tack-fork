# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The hooks' scene (``tack_hooks``) through VTK's serialization, in one process: what a
remote WebGPU renderer (VTK in WASM, through ``vtkRemoteSession``) would receive.

A "server" builds and renders the scene; one ``vtkObjectManager`` serializes its render
window and everything it depends on (states as JSON, arrays as blobs by hash); a second
manager, the "client", registers those states and blobs and builds its own window from
them, as the browser would. Then:

- what of the Slang customization is in the shader properties' states;
- the client's picture against the server's;
- the probe check (``demo.check``) on the client: its pixels against Tack's evaluation
  on the CPU, so the client draws Tack's field, not only a similar picture.

    python roundtrip.py [straight|curved|tets] [n]
"""

import importlib.util
import json
import os
import sys

import numpy as np
from tack_hooks import rr, surface_actors, update_surface
from vtkmodules.vtkSerializationManager import vtkObjectManager

import tack

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("hooks_demo", os.path.join(HERE, "demo.py"))
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)          # this directory's demo, not slang_surfaces'


def serialize(window):
    manager = vtkObjectManager()
    manager.Initialize()
    window_id = manager.RegisterObject(window)
    manager.UpdateStatesFromObjects()
    ids = manager.GetAllDependencies(0)
    states = [manager.GetState(i) for i in ids]
    blobs = {h: manager.GetBlob(h, True) for h in manager.GetBlobHashes(ids)}
    return manager, window_id, states, blobs


def deserialize(window_id, states, blobs):
    manager = vtkObjectManager()
    manager.Initialize()
    for state in states:
        manager.RegisterState(state)
    for h, blob in blobs.items():
        manager.RegisterBlob(h, blob)
    manager.UpdateObjectsFromStates()
    return manager, manager.GetObjectAtId(window_id)


class ClientGroup:
    """A client actor, with the server group's place in the surface (for the check)."""

    def __init__(self, actor, server):
        self.actor = actor
        self.group, self.base, self.count = server.group, server.base, server.count

    def set_probe(self, probe):
        self.actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformi("hooks.probe",
                                                                              probe)


def describe(states):
    """The shader properties' states: which Slang settings arrived."""
    out = []
    for s in states:
        state = json.loads(s)
        if "ShaderProperty" in state.get("ClassName", ""):
            keys = {k: (len(v) if isinstance(v, (str, list, dict)) else v)
                    for k, v in state.items()
                    if k not in ("Id", "SuperClassNames", "ClassName", "ObjectName")}
            out.append((state["ClassName"], keys))
    return out


def main(kind="curved", n=16):
    tack.set_default_device("cpu")
    surf = demo.tets_surface(n) if kind == "tets" else demo.surface(kind, n)
    groups, (lo, hi) = surface_actors(surf, "f")
    window, renderer = demo.window_for(groups)
    demo.frame(window, renderer, 0, 30)
    server_picture = demo.grab(window)

    server, window_id, states, blobs = serialize(window)
    state_bytes = sum(len(s) for s in states)
    blob_bytes = sum(b.GetNumberOfValues() for b in blobs.values())
    print(f"{kind} {n}: {len(states)} states ({state_bytes / 1024:.0f} KiB), {len(blobs)} blobs "
          f"({blob_bytes / 2**20:.1f} MB)")
    for cls, keys in describe(states):
        print(f"  {cls}: {keys}")

    client, client_window = deserialize(window_id, states, blobs)
    client_window.SetOffScreenRendering(True)
    client_renderer = client_window.GetRenderers().GetFirstRenderer()
    client_actors = list(client_renderer.GetActors())
    print(f"  client: {client_window.GetClassName()}, {len(client_actors)} actors, shader "
          f"properties {[a.GetShaderProperty().GetClassName() for a in client_actors]}")
    demo.frame(client_window, client_renderer, 0, 30)
    client_picture = demo.grab(client_window)
    d = np.abs(server_picture - client_picture).max(axis=2)
    print(f"  client vs server picture: {100 * float((d > 2).mean()):.3f}% of pixels differ "
          f"by more than 2/255 (max {int(d.max())})")

    def client_check(label, surface):
        pixels, faces, worst, median = demo.check(
            surface, [ClientGroup(a, g) for a, g in zip(client_actors, groups, strict=True)],
            lo, hi, client_window, client_renderer)
        print(f"  {label}: {pixels} pixels on {faces} faces; vs Tack max {worst:.2e}, "
              f"median {median:.2e}")

    if len(client_actors) == len(groups):
        client_check("client check", surf)

    # Timesteps as a remote session sends them: only the states that changed and the
    # blobs the client has not seen.
    sent_states = {json.loads(s)["Id"]: s for s in states}
    sent_blobs = set(blobs)

    def step(label, change):
        nonlocal surf
        surf = change()
        demo.frame(window, renderer, 0, 30)
        server.UpdateStatesFromObjects()
        ids = server.GetAllDependencies(0)
        changed = [server.GetState(i) for i in ids if server.GetState(i) != sent_states.get(i)]
        new_blobs = {h: server.GetBlob(h, True) for h in server.GetBlobHashes(ids)
                     if h not in sent_blobs}
        for st in changed:
            client.RegisterState(st)
            sent_states[json.loads(st)["Id"]] = st
        for h, blob in new_blobs.items():
            client.RegisterBlob(h, blob)
            sent_blobs.add(h)
        client.UpdateObjectsFromStates()
        names = sorted({json.loads(st)["ClassName"] for st in changed})
        print(f"  {label}: {len(changed)} states sent ({sum(len(c) for c in changed) / 1024:.0f} "
              f"KiB: {', '.join(names)}), {len(new_blobs)} blobs "
              f"({sum(b.GetNumberOfValues() for b in new_blobs.values()) / 2**20:.2f} MB)")
        demo.frame(client_window, client_renderer, 0, 30)
        d = np.abs(demo.grab(window) - demo.grab(client_window)).max(axis=2)
        print(f"    client vs server picture: {100 * float((d > 2).mean()):.3f}% of pixels "
              f"differ by more than 2/255 (max {int(d.max())})")
        client_check("client check", surf)

    def new_field():
        old = surf.fields["f"]
        surf.fields["f"] = demo.td.Field(old.space,
                                         demo.tack.array((lo + hi) - old.values.numpy()))
        for g in groups:
            g.update_field(surf, "f")
        return surf

    def new_geometry():
        _spec = importlib.util.spec_from_file_location(
            "hooks_geometry_step", os.path.join(HERE, "geometry_step.py"))
        geometry_step = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(geometry_step)
        new = geometry_step.step_surface(kind, n, "deform", 1)
        update_surface(groups, new, "f")
        return new

    if len(client_actors) == len(groups) and kind != "tets":
        step("a field step", new_field)
        step("a geometry step", new_geometry)
    return server, client


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "curved", int(a[1]) if len(a) > 1 else 16)
