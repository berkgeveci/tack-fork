# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Surface LIC on a slice of ``disk_out_ref`` in a browser, following a plane widget:
VTK WebAssembly fed by this Python server through VTK's serialization (as
``vtk_slang_hooks/remote.py``).

The server builds the scene (``vtk_lic.SliceActor`` through Tack's hook,
``vtkWebGPUSurfaceLIC`` on the renderer, a ``vtkImplicitPlaneWidget2``), renders it once
offscreen so the Slang compiles and its artifacts are recorded, and serves:

- ``/``: the page; ``/vtk/...``: the WebAssembly build of VTK;
- ``/update``: the states that changed and the hashes of blobs not yet sent;
- ``/blob/<hash>``;
- ``POST /plane`` (``{"origin": [...], "normal": [...]}``): Tack cuts the volume again
  and packs the cut cells; the update that makes.

The browser draws everything (the hook evaluates Tack's fields per pixel, the LIC runs on
its GPU) with the server's compiled Slang; it moves the widget and the camera itself, and
posts the plane as the widget moves (its InteractionEvent: the latest only, one at a time). The camera's, the
widget's and its representation's states go once: the browser's are its own after.

    python lic_server.py [--port 8766] [--wasm DIR] [--refine N]

``--refine N`` cuts every hexahedron into N x N x N (Tack's ``subdivide``: the fields
evaluated at the new lattice points, exactly for these trilinear fields): a bigger mesh
of the same field.
"""

import argparse
import http.server
import json
import mimetypes
import os
import time

import numpy as np

import vtkmodules.vtkRenderingCore as rr
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import (vtkImplicitPlaneRepresentation,
                                              vtkImplicitPlaneWidget2)
from vtkmodules.vtkRenderingWebGPU import vtkWebGPUSurfaceLIC
from vtkmodules.vtkSerializationManager import vtkObjectManager

import disk
import tack
import tack.dataset as td
import vtk_lic

WASM = "/usr/local/scratch/builds/webgpu-perf/vtk-wasm-build/bin"

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tack surface LIC, VTK WebAssembly</title>
<style>
  html, body { margin: 0; height: 100%; background: #fff; font: 13px system-ui, sans-serif; }
  #bar { position: absolute; top: 8px; left: 8px; z-index: 2; background: #ffffffdd;
         padding: 6px 8px; border-radius: 6px; max-width: 46em; }
  #log { white-space: pre; margin-top: 4px; color: #333; }
  canvas { width: 100vw; height: 100vh; display: block; }
</style>
</head>
<body>
<div id="bar">Drag the widget to move the plane; drag elsewhere to turn the view.
  <div id="log">loading VTK...</div>
</div>
<canvas id="vtk-canvas" tabindex="0"></canvas>
<script type="module">
import "/vtk/vtkWebAssembly.mjs";
const logEl = document.getElementById("log");
const log = (line) => { logEl.textContent = line; };
const vtk = await globalThis.createVTKWASM({});
const session = new vtk.vtkRemoteSession();
let windowId = null, representation = null, owned = [];

async function apply(update) {
  let bytes = 0;
  for (const hash of update.blobs) {
    const blob = new Uint8Array(await (await fetch("/blob/" + hash)).arrayBuffer());
    bytes += blob.length;
    session.registerBlob(hash, blob);
  }
  // The objects this page owns (window, interaction, widget, camera): their states as
  // they are now, so that applying the update leaves them be.
  for (const id of owned) session.updateStateFromObject(id);
  for (const state of update.states) session.registerState(JSON.parse(state));
  session.updateObjectsFromStates();
  if (windowId === null) {
    windowId = update.window;
    representation = update.representation;
    owned = update.owned;
    const canvas = document.getElementById("vtk-canvas");
    session.bindRenderWindow(windowId, "#vtk-canvas");
    await session.render(windowId);   // the first render initializes WebGPU (JSPI)
    session.setSize(windowId, canvas.clientWidth * devicePixelRatio,
                    canvas.clientHeight * devicePixelRatio);
    session.startEventLoop(windowId);
    // The widget fires the interaction events (its representation holds the plane).
    session.observe(update.widget, "InteractionEvent", () => planeMoved());
    session.observe(update.widget, "EndInteractionEvent", () => planeMoved());
  }
  await session.render(windowId);
  return bytes;
}

// The plane as the widget leaves it: the latest posted, one post at a time.
let busy = false, again = false;
async function planeMoved() {
  if (busy) { again = true; return; }
  busy = true;
  do {
    again = false;
    session.updateStateFromObject(representation);
    const state = session.get(representation);
    const t0 = performance.now();
    const response = await fetch("/plane", { method: "POST",
      body: JSON.stringify({ origin: state.Origin, normal: state.Normal }) });
    const update = await response.json();
    const t1 = performance.now();
    const bytes = await apply(update);
    const t2 = performance.now();
    log(`plane: ${update.triangles} triangles; server ${update.server_ms.toFixed(0)} ms ` +
        `(Tack ${update.tack_ms.toFixed(0)}), round trip ${(t1 - t0).toFixed(0)} ms, ` +
        `${update.states.length} states, ${(bytes / 1024).toFixed(0)} KiB; ` +
        `apply + render ${(t2 - t1).toFixed(0)} ms`);
  } while (again);
  busy = false;
}

// For tests and the console: the session, the plane handler and the representation.
globalThis.lic = { session, planeMoved, representation: () => representation };

const t0 = performance.now();
const first = await (await fetch("/update")).json();
const bytes = await apply(first);
log(`first update: ${first.states.length} states, ${(bytes / 2 ** 10).toFixed(0)} KiB of ` +
    `blobs; ${(performance.now() - t0).toFixed(0)} ms. WebGPU: ${"gpu" in navigator}`);
</script>
</body>
</html>
"""


class Scene:
    def __init__(self, refine=1):
        tack.set_default_device("cpu")
        self.data, bounds = disk.read(disk.DISK)
        if refine > 1:
            t = time.perf_counter()
            coarse = self.data.num_cells
            self.data = td.subdivide(self.data, level=refine, fields=["V", "Temp"])
            print(f"refined {coarse} hexahedra {refine}x{refine}x{refine}: "
                  f"{self.data.num_cells} hexahedra, {self.data.num_points} points, in "
                  f"{time.perf_counter() - t:.1f} s", flush=True)
        lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
        self.centre, span = 0.5 * (lo + hi), float(np.max(hi - lo))
        temp = self.data.fields["Temp"].values.numpy().reshape(-1)
        self.slice = vtk_lic.SliceActor(self.data, (float(temp.min()), float(temp.max())))
        self.slice.property.SetRecordSlangArtifacts(True)
        self.slice.cut(self.centre, (1.0, 0.0, 0.0))

        # ParaView's default background: the widget's white handles show.
        self.renderer = rr.vtkRenderer(background=(0.32, 0.34, 0.43))
        self.renderer.AddActor(self.slice.actor)
        self.lic = vtkWebGPUSurfaceLIC()
        self.lic.SetRecordSlangArtifacts(True)
        self.renderer.SetSurfaceLIC(self.lic)
        self.window = rr.vtkRenderWindow(size=(1280, 800), off_screen_rendering=True)
        self.window.AddRenderer(self.renderer)
        self.interactor = rr.vtkRenderWindowInteractor(render_window=self.window)
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

        self.representation = vtkImplicitPlaneRepresentation()
        self.representation.SetPlaceFactor(1.0)
        self.representation.PlaceWidget(bounds)
        self.representation.SetOrigin(*self.centre)
        self.representation.SetNormal(1.0, 0.0, 0.0)
        self.representation.DrawPlaneOff()
        self.representation.SetHandleSize(15.0)     # pixels (5 by default: hard to grab)
        self.widget = vtkImplicitPlaneWidget2()
        self.widget.SetInteractor(self.interactor)
        self.widget.SetRepresentation(self.representation)
        self.widget.On()

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*(self.centre + np.array([1.5 * span, 0.4 * span, 0.6 * span])))
        camera.SetFocalPoint(*self.centre)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.SetViewAngle(30.0)
        self.renderer.ResetCameraClippingRange()
        t = time.perf_counter()
        self.window.Render()
        print(f"server: first render {1000 * (time.perf_counter() - t):.0f} ms; artifacts: "
              f"hook {self.slice.property.GetNumberOfSlangArtifacts()}, LIC "
              f"{self.lic.GetNumberOfSlangArtifacts()}", flush=True)

        self.manager = vtkObjectManager()
        self.manager.Initialize()
        self.window_id = self.manager.RegisterObject(self.window)
        # The widget too: nothing in the window's graph references it (it observes the
        # interactor), so without this the browser draws its representation but nothing
        # handles the mouse for it.
        self.widget_id = self.manager.RegisterObject(self.widget)
        self.new_client()

    def new_client(self):
        self.sent_states, self.sent_blobs, self.blobs = {}, set(), {}
        self.once_sent = set()

    def update(self):
        self.manager.UpdateStatesFromObjects()
        ids = self.manager.GetAllDependencies(0)
        states = []
        for i in ids:
            text = self.manager.GetState(i)
            if text == self.sent_states.get(i):
                continue
            state = json.loads(text)
            name = state.get("ClassName", "")
            if any(k in name for k in ("Camera", "Widget", "Representation", "Interactor")):
                # The browser's own after the first: never sent again.
                if i in self.once_sent:
                    continue
                self.once_sent.add(i)
            if "RenderWindow" in name and "Interactor" not in name:
                state["OffScreenRendering"] = 0
                state.pop("Size", None)
            self.sent_states[i] = text
            states.append(json.dumps(state))
        hashes = [h for h in self.manager.GetBlobHashes(ids) if h not in self.sent_blobs]
        for h in hashes:
            self.blobs[h] = bytes(memoryview(self.manager.GetBlob(h, True)))
            self.sent_blobs.add(h)
        # The objects the browser owns after the first update (its window, interaction,
        # widget and camera): it refreshes their states from them before applying an
        # update, so re-applying a state the server sent once does not undo what it did.
        owned = [self.window_id, self.widget_id] + [
            self.manager.GetId(o) for o in (self.representation, self.interactor,
                                            self.interactor.GetInteractorStyle(),
                                            self.renderer.GetActiveCamera())]
        return {"window": self.window_id, "widget": self.widget_id, "owned": owned,
                "representation": self.manager.GetId(self.representation),
                "states": states, "blobs": hashes}

    def plane(self, origin, normal):
        t0 = time.perf_counter()
        self.slice.cut(origin, normal)
        t1 = time.perf_counter()
        self.representation.SetOrigin(*origin)
        self.representation.SetNormal(*normal)
        update = self.update()
        update.update(triangles=self.slice.timings["triangles"], tack_ms=1000 * (t1 - t0),
                      server_ms=1000 * (time.perf_counter() - t0))
        return update


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
            if self.path == "/plane":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                update = scene.plane(np.array(body["origin"], float),
                                     np.array(body["normal"], float))
                self._send(json.dumps(update).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"serving the disk's surface LIC on http://127.0.0.1:{port}/ "
          f"(VTK WebAssembly from {wasm})", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--wasm", default=WASM)
    parser.add_argument("--refine", type=int, default=1)
    args = parser.parse_args()
    serve(Scene(args.refine), args.port, os.path.abspath(args.wasm))


if __name__ == "__main__":
    main()
