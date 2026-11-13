# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Step 4 of the AMR volume rendering plan: AMR-Wind volume rendered in a browser, by
VTK WebAssembly, from what this Python server sends through VTK's serialization (as
``surface_lic/lic_server.py``).

The server reads the plotfile, makes what the client gets (``TackVolume``: levels 0 to
``--base`` and, with ``--histograms``, histograms of the finer data under them; all
levels with ``--base 4``), renders it once offscreen so the Slang compiles and its
artifacts are recorded, and serves:

- ``/``: the page; ``/vtk/...``: the WebAssembly build of VTK;
- ``/update``: the states and the hashes of blobs not yet sent;
- ``/blob/<hash>``.

The browser draws everything with the server's compiled Slang and turns the camera
itself, one canvas pixel per CSS pixel (``?scale=2`` for a Retina screen's full
resolution: four times the rays). While it moves, the mapper's interactive uniforms apply (the client's own
interactor raises the window's desired update rate); every interaction ends at that
rate (the interactor's still rate is set to it), and the page draws the full frame once
the camera has rested ``REST`` ms, as ``amrwind_view.py`` does.

For measuring: ``vol.spin(n, moving)`` in the page turns the camera ``n`` steps and
times each frame (to the next animation frame after the render). A render returns once
its work is submitted, so only frames in a row measure the GPU: the queue fills and each
waits for an earlier one (take the median of 20 or more).

    python volume_server.py [--port 8767] [--base 1] [--histograms 8x4] [--floor 0.4]
"""

import argparse
import http.server
import json
import mimetypes
import os
import sys
import time
import urllib.parse

import numpy as np
import vtkmodules.vtkRenderingCore as rr
from amrwind_demo import PLOTFILE, transfer_functions
from lod_demo import place, views
from tack_volume import TackVolume
from volume_mapper import VolumeMapper
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingWebGPUPython import vtkPythonWebGPUMapper
from vtkmodules.vtkSerializationManager import vtkObjectManager

import tack

sys.path.insert(0, os.path.expanduser("~/Work/amrex-reader"))

from amrex_reader import read_plotfile

WASM = "/usr/local/scratch/builds/webgpu-perf/vtk-wasm-build/bin"

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tack AMR volume, VTK WebAssembly</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #525770;
               font: 13px system-ui, sans-serif; }
  #bar { position: absolute; top: 8px; left: 8px; z-index: 2; background: #ffffffdd;
         padding: 6px 8px; border-radius: 6px; max-width: 52em; }
  #log { white-space: pre; margin-top: 4px; color: #333; }
  /* Fixed to the window: VTK sizes the canvas in device pixels, which on a screen of
     pixel ratio 2 would lay it out twice the window's size (the volume off screen, and
     scrollbars that shrink the render area under VTK's scissor rectangle). */
  canvas { position: fixed; inset: 0; width: 100vw !important; height: 100vh !important;
           display: block; }
</style>
</head>
<body>
<div id="bar">AMR-Wind vorticity: drag to turn, wheel or right-drag to zoom. The full frame
  is drawn once the camera rests.
  <div id="log">loading VTK...</div>
  <div id="moves"></div>
</div>
<canvas id="vtk-canvas" tabindex="0"></canvas>
<script>
// Canvas pixels per CSS pixel: 1 by default (a Retina screen's ratio of 2 draws four times
// the rays, about 4x slower), the browser scaling the picture up; ?scale=2 for the
// screen's full resolution. VTK's interactor sizes the window and maps the mouse by
// window.devicePixelRatio, so the page sets what it reads, before VTK loads.
const screenRatio = window.devicePixelRatio;
const scale = Number(new URLSearchParams(location.search).get("scale") ?? 1);
Object.defineProperty(window, "devicePixelRatio", { get: () => scale, configurable: true });
// Anything that fails shows in the status box (a module that fails to load says nothing
// otherwise), with what the browser offers.
const report = (what) => {
  const el = document.getElementById("log");
  el.textContent = `${el.textContent}\n${what}`;
};
addEventListener("error", (e) => report(`error: ${e.message}`));
addEventListener("unhandledrejection", (e) => report(`error: ${e.reason?.message ?? e.reason}`));
const origError = console.error;
console.error = (...args) => { report(args.join(" ").slice(0, 300)); origError(...args); };
report(`WebGPU: ${"gpu" in navigator}; JSPI: ${typeof WebAssembly.Suspending === "function"}; ` +
       `screen pixel ratio ${screenRatio}, drawing at ${scale}`);
</script>
<script type="module">
import "/vtk/vtkWebAssembly.mjs";
const logEl = document.getElementById("log");
const movesEl = document.getElementById("moves");
const log = (line) => { logEl.textContent = line; };
log(logEl.textContent + "\nloading the data...");
const vtk = await globalThis.createVTKWASM({});
const session = new vtk.vtkRemoteSession();
const REST = 300;
let windowId = null, cameraId = null, interactorId = null, owned = [];
const moving = 15.0, still = 0.0001;

async function apply(update) {
  let bytes = 0;
  for (const hash of update.blobs) {
    const blob = new Uint8Array(await (await fetch("/blob/" + hash)).arrayBuffer());
    bytes += blob.length;
    session.registerBlob(hash, blob);
  }
  for (const id of owned) session.updateStateFromObject(id);
  for (const state of update.states) session.registerState(JSON.parse(state));
  session.updateObjectsFromStates();
  if (windowId === null) {
    windowId = update.window;
    cameraId = update.camera;
    interactorId = update.interactor;
    owned = update.owned;
    const canvas = document.getElementById("vtk-canvas");
    session.bindRenderWindow(windowId, "#vtk-canvas");
    await session.render(windowId);   // the first render initializes WebGPU (JSPI)
    const fit = () => session.setSize(windowId,
      Math.round(canvas.clientWidth * scale), Math.round(canvas.clientHeight * scale));
    fit();
    addEventListener("resize", () => { fit(); session.render(windowId); });
    session.startEventLoop(windowId);
    let timer = null, frames = 0, started = 0;
    session.observe(interactorId, "StartInteractionEvent", () => {
      if (timer !== null) { clearTimeout(timer); timer = null; }
      frames = 0; started = performance.now();
    });
    session.observe(windowId, "EndEvent", () => { frames += 1; });
    session.observe(interactorId, "EndInteractionEvent", () => {
      if (timer !== null) clearTimeout(timer);
      const seconds = (performance.now() - started) / 1000;
      timer = setTimeout(async () => {
        timer = null;
        await drawStill();
        // A render returns once its work is submitted: the moving frames' rate holds
        // (each waits for the queue), one full frame's time would not.
        movesEl.textContent = `last interaction: ${frames} frames in ${seconds.toFixed(1)} s` +
          ` (${(frames / Math.max(seconds, 1e-3)).toFixed(0)} a second); full frame drawn`;
      }, REST);
    });
  }
  await session.render(windowId);
  return bytes;
}

const frame = () => new Promise((r) => requestAnimationFrame(r));

async function drawStill() {
  session.invoke(windowId, "SetDesiredUpdateRate", [still]);
  await session.render(windowId);
  await frame();
  session.invoke(windowId, "SetDesiredUpdateRate", [moving]);
}

// Turn the camera n steps of `degrees`, each frame at the moving (or still) rate, and
// time them: the render and the animation frame after it.
async function spin(n, isMoving = true, degrees = 3) {
  session.invoke(windowId, "SetDesiredUpdateRate", [isMoving ? moving : still]);
  await session.render(windowId);
  await frame();
  const times = [];
  for (let i = 0; i < n; ++i) {
    const t0 = performance.now();
    session.invoke(cameraId, "Azimuth", [degrees]);
    await session.render(windowId);
    await frame();
    times.push(performance.now() - t0);
  }
  session.invoke(windowId, "SetDesiredUpdateRate", [moving]);
  times.sort((a, b) => a - b);
  return { frames: n, median: times[Math.floor(n / 2)], min: times[0], max: times[n - 1] };
}

globalThis.vol = { session, spin, drawStill };

const t0 = performance.now();
const first = await (await fetch("/update")).json();
const bytes = await apply(first);
await drawStill();
log(`first update: ${first.states.length} states, ${(bytes / 2 ** 20).toFixed(1)} MiB of ` +
    `blobs; ${((performance.now() - t0) / 1000).toFixed(1)} s to the first full frame. ` +
    `${first.what}`);
</script>
</body>
</html>
"""


class VolumeScene:
    """AMR-Wind as the client gets it, in a scene the server renders once."""

    def __init__(self, plotfile, field, base, histograms, floor_value, macrocells=128):
        tack.set_default_device("cpu")
        t = time.perf_counter()
        source = read_plotfile(plotfile, fields=[field])
        self.volume = TackVolume(field, macrocells=macrocells)
        self.volume.SetInputDataObject(source)
        self.volume.Update()
        amr = self.volume.amr
        if base < amr.num_levels - 1:
            keep = [np.arange(len(amr.boxes(level))) if level <= base else np.array([], int)
                    for level in range(amr.num_levels)]
            self.volume.SetKeep(keep)
            if histograms:
                bins, bits = histograms
                self.volume.SetHistograms(True, bins, bits, floor_value)
        self.volume.Update()
        timings = self.volume.timings
        self.what = (f"levels 0-{min(base, amr.num_levels - 1)}"
                     + (f" + histograms {histograms[0]}x{histograms[1]}"
                        + (f", floor {floor_value:g}" if floor_value is not None else "")
                        if histograms and base < amr.num_levels - 1 else "")
                     + f": pool {timings['pool_bytes'] / 2**20:.0f} MiB")
        print(f"read and Tack {time.perf_counter() - t:.1f} s; {self.what}", flush=True)

        self.mapper = VolumeMapper()
        self.mapper.SetInputConnection(self.volume.GetOutputPort())
        self.mapper.slang.SetRecordSlangArtifacts(True)
        transfer_functions(self.mapper, 0.4, 2.0)
        self.actor = rr.vtkActor(mapper=self.mapper)
        self.actor.ForceTranslucentOn()
        self.renderer = rr.vtkRenderer(background=(0.32, 0.34, 0.43))
        self.renderer.AddActor(self.actor)
        self.window = rr.vtkRenderWindow(size=(1280, 800), off_screen_rendering=True)
        self.window.AddRenderer(self.renderer)
        self.interactor = rr.vtkRenderWindowInteractor(render_window=self.window)
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        # Every interaction ends at the moving rate; the page draws the full frame once
        # the camera rests (a wheel tick is a whole interaction).
        self.interactor.SetStillUpdateRate(self.interactor.GetDesiredUpdateRate())
        centre, span = views(self.volume)["close"]
        place(self.renderer.GetActiveCamera(), self.renderer, centre, span)
        t = time.perf_counter()
        self.window.Render()
        print(f"server: first render {1000 * (time.perf_counter() - t):.0f} ms; artifacts "
              f"{self.mapper.slang.GetNumberOfSlangArtifacts()}", flush=True)

        self.manager = vtkObjectManager()
        self.manager.Initialize()
        # The Python mapper goes as a plain WebGPU mapper (the browser has no Python).
        vtkPythonWebGPUMapper.RegisterSerializationHandlers(self.manager.GetSerializer())
        self.window_id = self.manager.RegisterObject(self.window)
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
            if any(k in name for k in ("Camera", "Interactor")):
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
        camera = self.manager.GetId(self.renderer.GetActiveCamera())
        interactor = self.manager.GetId(self.interactor)
        owned = [self.window_id, interactor, camera,
                 self.manager.GetId(self.interactor.GetInteractorStyle())]
        return {"window": self.window_id, "camera": camera, "interactor": interactor,
                "owned": owned, "states": states, "blobs": hashes, "what": self.what}


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
            path = urllib.parse.urlsplit(self.path).path      # without ?scale=...
            if path in ("/", "/index.html"):
                scene.new_client()
                self._send(PAGE.encode(), "text/html")
            elif path == "/update":
                self._send(json.dumps(scene.update()).encode(), "application/json")
            elif path.startswith("/blob/"):
                blob = scene.blobs.get(path[len("/blob/"):])
                if blob is None:
                    self._send(b"no such blob", "text/plain", 404)
                else:
                    self._send(blob, "application/octet-stream")
            elif path.startswith("/vtk/"):
                path = os.path.normpath(os.path.join(wasm, path[len("/vtk/"):]))
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

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"serving AMR-Wind's volume on http://127.0.0.1:{port}/ "
          f"(VTK WebAssembly from {wasm})", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--wasm", default=WASM)
    parser.add_argument("--plotfile", default=PLOTFILE)
    parser.add_argument("--field", default="mag_vorticity")
    parser.add_argument("--base", type=int, default=1, help="levels sent whole: 0 to this")
    parser.add_argument("--histograms", default="8x4",
                        help="bins x bits of the stand-ins' histograms, or 'none'")
    parser.add_argument("--floor", type=float, default=0.4)
    args = parser.parse_args()
    histograms = None if args.histograms == "none" else tuple(
        int(v) for v in args.histograms.split("x"))
    scene = VolumeScene(args.plotfile, args.field, args.base, histograms, args.floor)
    serve(scene, args.port, os.path.abspath(args.wasm))


if __name__ == "__main__":
    main()
