# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The first picture of a fresh process, as in situ: one renderer, one surface, timed by
phase (ms):

- ``imports``: the renderer's Python modules (Tack, wgpu-py or VTK);
- ``surface``: Tack extracting the surface from the solid (made before, untimed: the
  simulation's data);
- ``device``: the WebGPU adapter and device, or VTK's window and context;
- ``setup``: the renderer's preparation (Tack evaluating, shaders generated and compiled,
  pipelines, uploads);
- ``first``: the first frame, rendered, waited for and read back.

Run one per process (``TACK_CACHE=off`` for Tack's compilation cold)::

    python first_render.py wgpu {records|slang|cached} {straight|curved} n OUTDIR
    python first_render.py vtk {cellgrid|polygons} {straight|curved} n
"""

import time

START = time.perf_counter()

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(api, renderer, kind, n, out=None):
    n = int(n)
    if api == "wgpu":
        import bench_wgpu as bench
    else:
        import bench_vtk as bench
    import tack
    import tack.dataset as td

    imports = time.perf_counter() - START
    tack.set_default_device("cpu")
    from scenes import solid

    data = solid(kind, n)
    start = time.perf_counter()
    surf = td.extract_surface(data, fields=["f"])
    surface = time.perf_counter() - start
    times = {"imports": 1e3 * imports, "surface": 1e3 * surface}
    times.update(bench.first(renderer, surf, out) if api == "wgpu" else bench.first(renderer, surf))
    total = sum(times.values())
    print(f"{api} {renderer} {kind} {n} " + " ".join(f"{v:.0f}" for v in times.values())
          + f" {total:.0f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
