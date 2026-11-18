"""The batched pipeline: read -> one Tack conversion -> batched projection -> batched
flying edges (amr_flying_edges), timed; the surface written as an .npz like
tack_isosurface.py's, for same_surface.py.

    python fe_hierarchy_run.py OUT.npz [--iso V] [--max-level L] [--device cpu|metal]
                                       [--layout joined|table] [--separate]

``--layout``: how the kernels read the boxes' fields (``amr_fields``): one array
(``joined``) or each box's own through a table of addresses (``table``, CPU).
``--separate``: copy each box's input fields into their own allocations first (not
timed), as separate datasets would have them; only ``table`` can then read them.
"""
import argparse
import os
import sys
import time

import numpy as np

import tack

sys.path.insert(0, os.path.expanduser("~/Work/amrex-reader"))
from amrex_reader import read_plotfile  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--iso", type=float, default=0.5)
parser.add_argument("--max-level", type=int, default=None)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--device", default="cpu")
parser.add_argument("--f32", action="store_true", help="the field and positions in f32")
parser.add_argument("--layout", default="joined", choices=("joined", "table"))
parser.add_argument("--separate", action="store_true")
args = parser.parse_args()
tack.set_default_device("cpu")

import amr_flying_edges as fe  # noqa: E402
from amr_project import project_to_points  # noqa: E402
from tack.dataset.dataset import Field  # noqa: E402
from tack.interop.vtk import algorithms as ta  # noqa: E402

T = time.perf_counter
t = T()
amr = read_plotfile(os.path.expanduser("~/Data/amrwind/plt255900"), fields=["mag_vorticity"],
                    max_level=args.max_level, blank=False)
print(f"read: {T() - t:.1f} s", flush=True)
t = T()
data = ta._as_dataset(amr)
print(f"to Tack: {T() - t:.2f} s", flush=True)


def separate(name):
    """Each box's field ``name`` copied into its own allocation (not timed)."""
    t = T()
    for _, box in data:
        f = box.fields[name]
        box.fields[name] = Field(f.space, tack.array(f.values.numpy().copy()), f.unit)
    print(f"  (each box's {name!r} copied into its own allocation: {T() - t:.2f} s)",
          flush=True)


if args.separate:
    separate("mag_vorticity")
for r in range(args.repeats):
    if r:
        for _, box in data:
            del box.fields["point_values"]
    t = T()
    project_to_points(data, "mag_vorticity", "point_values", layout=args.layout)
    print(f"project (batched, {args.layout}): {T() - t:.2f} s", flush=True)
if args.separate:
    separate("point_values")
t = T()
device = tack.device(args.device)
prepared = fe.prepare(data, "point_values", device=device,
                      float_dtype=tack.f32 if args.f32 else None, layout=args.layout)
device.synchronize()
print(f"tables + inputs on {args.device}{' (f32)' if args.f32 else ''}: {T() - t:.2f} s",
      flush=True)
for r in range(args.repeats):
    t = T()
    positions, rows, boxes, count, ntri = fe.hierarchy_surface(prepared, args.iso)
    device.synchronize()
    print(f"flying edges over the hierarchy ({args.layout}): {T() - t:.2f} s; "
          f"{ntri / 1e6:.2f}M triangles, {count / 1e6:.2f}M points", flush=True)
np.savez(args.out, points=positions.numpy().astype(np.float32),
         triangles=rows.numpy().reshape(-1, 3).astype(np.uint32),
         level=np.zeros(ntri, np.uint8))
