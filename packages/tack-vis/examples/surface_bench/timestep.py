# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A timestep in situ: the geometry stays, the field changes. Set up once (untimed:
imports, device, shaders, pipelines, the geometry); then per step, with a fixed camera,
the median in ms of:

- ``update``: a new field of the same space onto the device, in the renderer's form
  (Tack + Slang: the values as they are; records: Tack's Bernstein conversion; VTK cell
  grid: Tack's projection to VTK's equispaced basis; VTK polygons: Tack evaluating the
  field at the triangles' points);
- ``draw``: the frame, waited for;
- ``read``: the picture read back.

Not timed: Tack tracing the solid's new field onto the surface (the same for all).
``BENCH_CHECK=1`` compares a step's picture with a renderer made afresh with that field.

    python timestep.py wgpu {records|slang|cached} {straight|curved} OUTDIR n ...
    python timestep.py vtk {cellgrid|polygons} {straight|curved} - n ...
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEPS, WARM = 10, 2


def fields(surf, count):
    """``count`` new fields of the surface field's space (a simulation's steps): its
    values permuted, so the colour range stays the same and a step's picture can be
    compared with a renderer made afresh."""
    import tack
    import tack.dataset as td

    f = surf.fields["f"]
    base = f.values.numpy()
    rng = np.random.default_rng(0)
    return [td.Field(f.space, tack.array(rng.permutation(base.reshape(-1)).reshape(base.shape),
                                         f.values.dtype))
            for _ in range(count)]


class WGPU:
    def __init__(self, renderer, surf, out):
        import bench_wgpu as b

        self.b, self.out, self.renderer = b, out, renderer
        _, self.device = b.open_device()
        self.target = b._Target(self.device)
        self.r = b.makers(self.device, out)[renderer](surf)

    def update(self, field):
        self.r.update(field)

    def draw(self):
        self.r.draw(self.target, *self.b.camera(0, 30))
        self.device.queue.on_submitted_work_done_sync()

    def read(self):
        data = self.device.queue.read_texture(
            {"texture": self.target.color, "origin": (0, 0, 0), "mip_level": 0},
            {"offset": 0, "bytes_per_row": self.b.W * 4}, (self.b.W, self.b.H, 1))
        return np.frombuffer(data, np.uint8).reshape(self.b.H, self.b.W, 4)

    def fresh(self, surf):
        return WGPU(self.renderer, surf, self.out)


class VTK:
    def __init__(self, renderer, surf, out=None):
        import bench_vtk as b
        import vtkmodules.vtkRenderingCore as rr

        self.b, self.renderer = b, renderer
        self.window, self.ren, _ = b.setup(renderer, surf, 8)
        self.grab = rr.vtkWindowToImageFilter(input=self.window)
        self.grab.ShouldRerenderOff()

    def update(self, field):
        self.window.update(field)

    def draw(self):
        self.b.frame(self.window, self.ren, 0, 30)

    def read(self):
        self.grab.Modified()
        self.grab.Update()
        return np.asarray(self.grab.GetOutput().GetPointData().GetScalars())

    def fresh(self, surf):
        return VTK(self.renderer, surf)


def main(api, renderer, kind, out, *sizes):
    from scenes import surface

    import tack
    import tack.dataset as td

    tack.set_default_device("cpu")
    make = WGPU if api == "wgpu" else VTK
    print("renderer kind n faces update_ms draw_ms read_ms step_ms")
    for n in map(int, sizes):
        surf = surface(kind, n)
        steps = fields(surf, WARM + STEPS)
        r = make(renderer, surf, out)
        r.draw()
        r.read()
        times = []
        for k, field in enumerate(steps):
            t0 = time.perf_counter()
            r.update(field)
            t1 = time.perf_counter()
            r.draw()
            t2 = time.perf_counter()
            picture = r.read()
            t3 = time.perf_counter()
            if k >= WARM:
                times.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0))
        med = 1e3 * np.median(np.array(times), axis=0)
        print(f"{api}-{renderer} {kind} {n} {surf.num_cells} "
              + " ".join(f"{v:.2f}" for v in med), flush=True)
        if os.environ.get("BENCH_CHECK"):
            fresh_surf = td.DataSet(surf.topology, surf.geometry, fields={"f": steps[-1]})
            again = r.fresh(fresh_surf)
            again.draw()
            other = again.read()
            first = make(renderer, surf, out)
            first.draw()
            diff = np.abs(picture.astype(int) - other.astype(int)).reshape(-1, picture.shape[-1])
            d = diff.max()
            print(f"#   pixels differing by more than 8: {(diff.max(axis=1) > 8).sum()} "
                  f"of {len(diff)}")
            moved = np.abs(picture.astype(int) - first.read().astype(int)).max()
            print(f"#   check: updated vs fresh picture max diff {d}; vs the first field {moved}")


if __name__ == "__main__":
    main(*sys.argv[1:])
