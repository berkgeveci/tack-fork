"""End to end on the CPU: plotfile -> vtkOverlappingAMR -> Tack (copy=False) -> batched
projection -> batched flying edges, with the blocks' arrays either views of one buffer
(the reader's, joined) or each block's own allocation (as FABs or Blueprint domains
are), read through address tables.

    python e2e_layouts.py OUT.npz --mode joined|separate [--iso V] [--repeats N]

``separate``: after the read each block's cell arrays are deep-copied in VTK (not
timed), so the conversion cannot join them and shares each block's array as is.
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

import tack

from amrex_reader import read_plotfile  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--mode", choices=("joined", "separate"), required=True)
parser.add_argument("--iso", type=float, default=0.5)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--max-level", type=int, default=None)
args = parser.parse_args()
tack.set_default_device("cpu")

import amr_flying_edges as fe  # noqa: E402
from amr_project import project_to_points  # noqa: E402
from tack import arrays  # noqa: E402
from tack.interop.vtk import algorithms as ta  # noqa: E402

T = time.perf_counter
FIELD = "mag_vorticity"
total = 0.0


def stage(label, seconds):
    global total
    total += seconds
    print(f"{label}: {seconds:.2f} s", flush=True)


t = T()
amr = read_plotfile(os.environ.get("AMRWIND_PLOTFILE", os.path.expanduser("~/Data/amrwind/plt255900")), fields=[FIELD],
                    max_level=args.max_level, blank=False)
stage("read", T() - t)

if args.mode == "separate":
    t = T()
    n = 0
    it = amr.NewIterator()
    it.InitTraversal()
    while not it.IsDoneWithTraversal():
        cells = it.GetCurrentDataObject().GetCellData()
        for i in range(cells.GetNumberOfArrays()):
            a = cells.GetAbstractArray(i)
            own = a.NewInstance()
            own.DeepCopy(a)
            cells.AddArray(own)                 # replaces the view by name
            n += 1
        it.GoToNextItem()
    print(f"  ({n} block arrays given their own allocations: {T() - t:.2f} s, not timed)",
          flush=True)

t = T()
data = ta._as_dataset(amr)
stage("to Tack (copy=False)", T() - t)
parts = [box.fields[FIELD].values for _, box in data]
joined = arrays._joined_view(parts) is not None
from vtkmodules.util.numpy_support import vtk_to_numpy  # noqa: E402

vtk_first = vtk_to_numpy(amr.GetDataSetAsCartesianGrid(0, 0).GetCellData().GetArray(FIELD))
shared = vtk_first.ctypes.data == parts[0]._buffer._data.ctypes.data
print(f"  {len(parts)} boxes; one array end to end: {joined}; box 0 shares VTK's "
      f"memory: {shared}", flush=True)
layout = "joined" if joined else "table"

for r in range(args.repeats):
    if r:
        for _, box in data:
            del box.fields["point_values"]
    t = T()
    project_to_points(data, FIELD, "point_values", layout=layout)
    seconds = T() - t
    if r == args.repeats - 1:
        stage(f"project ({layout})", seconds)
    else:
        print(f"  project ({layout}), run {r + 1}: {seconds:.2f} s")

# The flying edges read the projection's output: one array Tack made, either way.
points = [box.fields["point_values"].values for _, box in data]
layout = "joined" if arrays._joined_view(points) is not None else "table"
t = T()
prepared = fe.prepare(data, "point_values", layout=layout)
stage(f"tables ({layout})", T() - t)
for r in range(args.repeats):
    t = T()
    positions, rows, boxes, count, ntri = fe.hierarchy_surface(prepared, args.iso)
    seconds = T() - t
    if r == args.repeats - 1:
        stage(f"flying edges ({layout})", seconds)
    else:
        print(f"  flying edges ({layout}), run {r + 1}: {seconds:.2f} s")
print(f"total: {total:.2f} s; {ntri / 1e6:.2f}M triangles, {count / 1e6:.2f}M points; "
      f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GiB",
      flush=True)
np.savez(args.out, points=positions.numpy().astype(np.float32),
         triangles=rows.numpy().reshape(-1, 3).astype(np.uint32),
         level=np.zeros(ntri, np.uint8))
