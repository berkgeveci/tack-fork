# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tack's surface in a browser: VTK's WebGPU renderer compiled to WebAssembly, fed by
this Python process through VTK's serialization.

The server builds the hooks' scene (instanced templates, the placing pass,
``RecordSlangArtifacts`` on), renders it offscreen once so its mapper compiles the Slang
and records the artifacts, and serves:

- ``/``: the page; ``/vtk/...``: the WebAssembly build of VTK (``vtkWebAssembly.mjs``);
- ``/update``: the states that changed since the browser's last update and the hashes of
  the blobs it has not had (JSON);
- ``/blob/<hash>``: a blob's bytes;
- ``POST /step/geometry`` and ``/step/field``: a timestep (Tack computes it here), then
  the update it makes.

The browser (``vtkRemoteSession``) registers states and blobs, builds the scene and draws
it with the browser's WebGPU; the shaders are the server's compiled Slang (the browser
has no Slang compiler), the positions the browser's GPU computes (the placing pass). The
camera is the browser's own after the first update.

    python remote.py [straight|curved|tets] [n] [--port 8765] [--wasm DIR]
"""

import argparse
import http.server
import json
import mimetypes
import os
import time

os.environ.setdefault("INSTANCED", "1")

import demo  # noqa: E402
from precompiled import new_field  # noqa: E402
from tack_hooks import rr, surface_actors, update_surface  # noqa: E402
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera  # noqa: E402
from vtkmodules.vtkSerializationManager import vtkObjectManager  # noqa: E402

import tack  # noqa: E402

WASM = "/usr/local/scratch/builds/webgpu-perf/vtk-wasm-build/bin"

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tack surface, VTK WebAssembly</title>
<style>
  html, body { margin: 0; height: 100%; background: #fff; font: 13px system-ui, sans-serif; }
  #bar { position: absolute; top: 8px; left: 8px; z-index: 2; background: #ffffffdd;
         padding: 6px 8px; border-radius: 6px; }
  #bar button { margin-right: 6px; }
  #log { white-space: pre; margin-top: 6px; color: #333; }
  canvas { width: 100vw; height: 100vh; display: block; }
</style>
</head>
<body>
<div id="bar">
  <button id="geometry">geometry step</button><button id="field">field step</button>
  <div id="log">loading VTK...</div>
</div>
<canvas id="vtk-canvas" tabindex="0"></canvas>
<script type="module">
import "/vtk/vtkWebAssembly.mjs";
const log = (line) => { document.getElementById("log").textContent += "\n" + line; };
const vtk = await globalThis.createVTKWASM({});
const session = new vtk.vtkRemoteSession();
let windowId = null;

async function apply(update, label, t0) {
  const t1 = performance.now();
  let bytes = 0;
  for (const hash of update.blobs) {
    const blob = new Uint8Array(await (await fetch("/blob/" + hash)).arrayBuffer());
    bytes += blob.length;
    session.registerBlob(hash, blob);
  }
  for (const state of update.states) {
    session.registerState(JSON.parse(state));
  }
  const t2 = performance.now();
  session.updateObjectsFromStates();
  if (windowId === null) {
    // render() is the async entry point (JSPI): the first one initializes WebGPU, which
    // the synchronous calls must not be the first to reach.
    windowId = update.window;
    const canvas = document.getElementById("vtk-canvas");
    session.bindRenderWindow(windowId, "#vtk-canvas");
    await session.render(windowId);
    session.setSize(windowId, canvas.clientWidth * devicePixelRatio,
                    canvas.clientHeight * devicePixelRatio);
    session.startEventLoop(windowId);
  }
  await session.render(windowId);
  const t3 = performance.now();
  const states = update.states.reduce((n, s) => n + s.length, 0);
  log(`${label}: ${update.states.length} states (${(states / 1024).toFixed(0)} KiB), ` +
      `${update.blobs.length} blobs (${(bytes / 2 ** 20).toFixed(2)} MB); ` +
      (t0 === undefined ? "" : `server ${(t1 - t0).toFixed(0)} ms, `) +
      `transfer ${(t2 - t1).toFixed(0)} ms, build + render ${(t3 - t2).toFixed(0)} ms`);
}

async function step(kind) {
  const t0 = performance.now();
  const update = await (await fetch("/step/" + kind, { method: "POST" })).json();
  await apply(update, kind + " step", t0);
}

document.getElementById("geometry").onclick = () => step("geometry");
document.getElementById("field").onclick = () => step("field");
document.getElementById("log").textContent = "WebGPU: " + ("gpu" in navigator);
await apply(await (await fetch("/update")).json(), "first update");
</script>
</body>
</html>
"""


class Scene:
    """The server's scene and what the browser has of it."""

    def __init__(self, kind, n):
        tack.set_default_device("cpu")
        self.kind, self.n, self.steps = kind, n, 0
        self.surf = demo.tets_surface(n) if kind == "tets" else demo.surface(kind, n)
        self.groups, (self.lo, self.hi) = surface_actors(self.surf, "f")
        for g in self.groups:
            g.actor.GetShaderProperty().SetRecordSlangArtifacts(True)
        self.window, self.renderer = demo.window_for(self.groups)
        # An interactor for the browser's window (never started here).
        self.interactor = rr.vtkRenderWindowInteractor(render_window=self.window)
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        demo.frame(self.window, self.renderer, 0, 30)
        self.manager = vtkObjectManager()
        self.manager.Initialize()
        self.window_id = self.manager.RegisterObject(self.window)
        self.sent_states, self.sent_blobs = {}, set()
        self.blobs = {}
        self.camera_sent = False

    def new_client(self):
        """A page loaded: the browser has nothing yet."""
        self.sent_states, self.sent_blobs, self.blobs = {}, set(), {}
        self.camera_sent = False

    def update(self):
        """The states that changed since the last update, and the new blobs' hashes."""
        self.manager.UpdateStatesFromObjects()
        ids = self.manager.GetAllDependencies(0)
        states = []
        for i in ids:
            text = self.manager.GetState(i)
            if text == self.sent_states.get(i):
                continue
            state = json.loads(text)
            name = state.get("ClassName", "")
            if "Camera" in name:
                if self.camera_sent:
                    continue        # the browser's camera is its own
                self.camera_sent = True
            if "RenderWindow" in name and "Interactor" not in name:
                state["OffScreenRendering"] = 0     # the browser's window is on screen
                state.pop("Size", None)
            self.sent_states[i] = text
            states.append(json.dumps(state))
        hashes = [h for h in self.manager.GetBlobHashes(ids) if h not in self.sent_blobs]
        for h in hashes:
            self.blobs[h] = bytes(memoryview(self.manager.GetBlob(h, True)))
            self.sent_blobs.add(h)
        return {"window": self.window_id, "states": states, "blobs": hashes}

    def step(self, what):
        start = time.perf_counter()
        if what == "geometry" and self.kind != "tets":
            self.steps += 1
            self.surf = _step_surface()(self.kind, self.n, "deform", self.steps)
            update_surface(self.groups, self.surf, "f")
        elif what == "field":
            new_field(self.surf, self.lo, self.hi)
            for g in self.groups:
                g.update_field(self.surf, "f")
        demo.frame(self.window, self.renderer, 0, 30)
        print(f"{what} step: {1000 * (time.perf_counter() - start):.0f} ms on the server",
              flush=True)
        return self.update()


def _step_surface():
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("hooks_geometry_step",
                                                  os.path.join(here, "geometry_step.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.step_surface


def serve(scene, port, wasm):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body, kind, status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                scene.new_client()
                self._send(PAGE.encode(), "text/html")
            elif self.path == "/update":
                self._send(json.dumps(scene.update()).encode(), "application/json")
            elif self.path.startswith("/blob/"):
                blob = scene.blobs.get(self.path[len("/blob/"):])
                if blob is None:
                    self._send(b"no such blob", "text/plain", 404)
                else:
                    self._send(blob, "application/octet-stream")
            elif self.path.startswith("/vtk/"):
                path = os.path.normpath(os.path.join(wasm, self.path[len("/vtk/"):]))
                if not path.startswith(wasm) or not os.path.isfile(path):
                    self._send(b"not found", "text/plain", 404)
                    return
                kind = ("application/wasm" if path.endswith(".wasm") else
                        "text/javascript" if path.endswith((".js", ".mjs")) else
                        mimetypes.guess_type(path)[0] or "application/octet-stream")
                with open(path, "rb") as f:
                    self._send(f.read(), kind)
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path in ("/step/geometry", "/step/field"):
                update = scene.step(self.path.rsplit("/", 1)[1])
                self._send(json.dumps(update).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def log_message(self, *args):
            pass

    # One thread: VTK and Tack are used from the thread that made them.
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"serving {scene.kind} {scene.n} on http://127.0.0.1:{port}/ "
          f"(VTK WebAssembly from {wasm})", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", nargs="?", default="curved")
    parser.add_argument("n", nargs="?", type=int, default=32)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--wasm", default=WASM)
    args = parser.parse_args()
    serve(Scene(args.kind, args.n), args.port, os.path.abspath(args.wasm))


if __name__ == "__main__":
    main()
