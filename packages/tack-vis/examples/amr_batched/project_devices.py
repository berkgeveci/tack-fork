"""AMR-Wind's cells to points, three ways of splitting the work, on the CPU and Metal:
an iteration a point (box search + div/mod), a box, or a point row (box, k, j) from a
precomputed table. Kernel time only (warm, synchronized, best of 5); every result
compared with the CPU's per-box one."""
import os
import time

import numpy as np

import tack
from amrex_reader import read_plotfile
from tack import arrays
from tack.interop.vtk import algorithms as ta


T = time.perf_counter


@tack.kernel
def by_point(cells, cell_start, point_start, dims, nboxes, out):
    for p in range(out.shape[0]):
        lo = 0
        hi = nboxes
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if point_start[mid] <= p:
                lo = mid
            else:
                hi = mid
        b = lo
        nx = dims[3 * b]
        ny = dims[3 * b + 1]
        nz = dims[3 * b + 2]
        local = p - point_start[b]
        i = local % (nx + 1)
        j = (local // (nx + 1)) % (ny + 1)
        k = local // ((nx + 1) * (ny + 1))
        first = cell_start[b]
        total = cells[first] * 0
        count = 0
        for dk in range(2):
            kk = k - 1 + dk
            if kk >= 0 and kk < nz:
                for dj in range(2):
                    jj = j - 1 + dj
                    if jj >= 0 and jj < ny:
                        for di in range(2):
                            ii = i - 1 + di
                            if ii >= 0 and ii < nx:
                                total = total + cells[first + ii + nx * (jj + ny * kk)]
                                count = count + 1
        out[p] = total / count


@tack.kernel
def by_box(cells, cell_start, point_start, dims, out):
    for b in range(dims.shape[0] // 3):
        nx = dims[3 * b]
        ny = dims[3 * b + 1]
        nz = dims[3 * b + 2]
        first = cell_start[b]
        p = point_start[b]
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    total = cells[first] * 0
                    count = 0
                    for dk in range(2):
                        kk = k - 1 + dk
                        if kk >= 0 and kk < nz:
                            for dj in range(2):
                                jj = j - 1 + dj
                                if jj >= 0 and jj < ny:
                                    for di in range(2):
                                        ii = i - 1 + di
                                        if ii >= 0 and ii < nx:
                                            total = total + cells[first + ii + nx * (jj + ny * kk)]
                                            count = count + 1
                    out[p] = total / count
                    p = p + 1


@tack.kernel
def by_row(cells, cell_start, point_start, dims, row_box, row_j, row_k, out):
    # A point row (box, k, j) an iteration, from the table: its points along i.
    for r in range(row_box.shape[0]):
        b = row_box[r]
        j = row_j[r]
        k = row_k[r]
        nx = dims[3 * b]
        ny = dims[3 * b + 1]
        nz = dims[3 * b + 2]
        first = cell_start[b]
        p = point_start[b] + (nx + 1) * (j + (ny + 1) * k)
        for i in range(nx + 1):
            total = cells[first] * 0
            count = 0
            for dk in range(2):
                kk = k - 1 + dk
                if kk >= 0 and kk < nz:
                    for dj in range(2):
                        jj = j - 1 + dj
                        if jj >= 0 and jj < ny:
                            for di in range(2):
                                ii = i - 1 + di
                                if ii >= 0 and ii < nx:
                                    total = total + cells[first + ii + nx * (jj + ny * kk)]
                                    count = count + 1
            out[p + i] = total / count


@tack.kernel
def by_row_point(cells, cell_start, point_start, dims, row_box, row_j, row_k, width, out):
    # A 2D launch: (row, i), i fastest, so neighbouring threads take neighbouring
    # points of one row and read neighbouring cells. Rows shorter than ``width``
    # (the longest) leave their extra threads idle.
    for r, i in tack.ndindex(row_box.shape[0], width):
        b = row_box[r]
        nx = dims[3 * b]
        if i <= nx:
            ny = dims[3 * b + 1]
            nz = dims[3 * b + 2]
            j = row_j[r]
            k = row_k[r]
            first = cell_start[b]
            total = cells[first] * 0
            count = 0
            for dk in range(2):
                kk = k - 1 + dk
                if kk >= 0 and kk < nz:
                    for dj in range(2):
                        jj = j - 1 + dj
                        if jj >= 0 and jj < ny:
                            for di in range(2):
                                ii = i - 1 + di
                                if ii >= 0 and ii < nx:
                                    total = total + cells[first + ii + nx * (jj + ny * kk)]
                                    count = count + 1
            out[point_start[b] + (nx + 1) * (j + (ny + 1) * k) + i] = total / count


@tack.func
def point_value(cells, first, nx, ny, nz, i, j, k):
    total = cells[first] * 0
    count = 0
    for dk in range(2):
        kk = k - 1 + dk
        if kk >= 0 and kk < nz:
            for dj in range(2):
                jj = j - 1 + dj
                if jj >= 0 and jj < ny:
                    for di in range(2):
                        ii = i - 1 + di
                        if ii >= 0 and ii < nx:
                            total = total + cells[first + ii + nx * (jj + ny * kk)]
                            count = count + 1
    return total / count


@tack.kernel
def by_row_point_32(cells, cell_start, point_start, dims, row_box, row_j, row_k, lanes, out):
    # (row, i) over ``lanes`` threads a row (32: a SIMD group): thread i takes point i,
    # and in a row of lanes + 1 points the last thread takes the last one too.
    for r, i in tack.ndindex(row_box.shape[0], lanes):
        b = row_box[r]
        nx = dims[3 * b]
        if i <= nx:
            ny = dims[3 * b + 1]
            nz = dims[3 * b + 2]
            j = row_j[r]
            k = row_k[r]
            first = cell_start[b]
            row = point_start[b] + (nx + 1) * (j + (ny + 1) * k)
            out[row + i] = point_value(cells, first, nx, ny, nz, i, j, k)
            if i == lanes - 1 and nx == lanes:
                out[row + nx] = point_value(cells, first, nx, ny, nz, nx, j, k)


@tack.kernel
def rows_first_lanes(cells, cell_start, point_start, dims, row_box, row_j, row_k, lanes, out):
    # Launch 1: (row, i) over ``lanes`` threads a row, each one point, no other
    # work: every point of a row but those at i >= lanes.
    for r, i in tack.ndindex(row_box.shape[0], lanes):
        b = row_box[r]
        nx = dims[3 * b]
        if i <= nx:
            ny = dims[3 * b + 1]
            nz = dims[3 * b + 2]
            j = row_j[r]
            k = row_k[r]
            out[point_start[b] + (nx + 1) * (j + (ny + 1) * k) + i] = point_value(
                cells, cell_start[b], nx, ny, nz, i, j, k)


@tack.kernel
def rows_rest(cells, cell_start, point_start, dims, row_box, row_j, row_k, lanes, out):
    # Launch 2: a row a thread, its points at i >= lanes (AMR-Wind: the 33rd).
    for r in range(row_box.shape[0]):
        b = row_box[r]
        nx = dims[3 * b]
        if nx >= lanes:
            ny = dims[3 * b + 1]
            nz = dims[3 * b + 2]
            j = row_j[r]
            k = row_k[r]
            row = point_start[b] + (nx + 1) * (j + (ny + 1) * k)
            for i in range(lanes, nx + 1):
                out[row + i] = point_value(cells, cell_start[b], nx, ny, nz, i, j, k)


tack.set_default_device("cpu")
amr = read_plotfile(os.environ.get("AMRWIND_PLOTFILE", os.path.expanduser("~/Data/amrwind/plt255900")), fields=["mag_vorticity"],
                    blank=False)
data = ta._as_dataset(amr)
boxes = [box for _, box in data]
whole = arrays._joined_view([b.fields["mag_vorticity"].values for b in boxes])
point_dims = np.array([b.topology.point_dims for b in boxes], np.int64)
cell_dims = point_dims - 1
cell_start = np.concatenate([[0], np.cumsum(cell_dims.prod(axis=1))])
point_start = np.concatenate([[0], np.cumsum(point_dims.prod(axis=1))])
n = int(point_start[-1])

t = T()
rows_per_box = point_dims[:, 1] * point_dims[:, 2]
row_box = np.repeat(np.arange(len(boxes), dtype=np.int32), rows_per_box)
first_row = np.concatenate([[0], np.cumsum(rows_per_box)])[:-1]
local = np.arange(len(row_box), dtype=np.int64) - np.repeat(first_row, rows_per_box)
py = np.repeat(point_dims[:, 1], rows_per_box)
row_j = (local % py).astype(np.int32)
row_k = (local // py).astype(np.int32)
print(f"{len(boxes)} boxes, {n / 1e6:.0f}M points, {len(row_box) / 1e6:.2f}M rows "
      f"(table made in {1000 * (T() - t):.0f} ms, {3 * 4 * len(row_box) / 2**20:.0f} MiB)",
      flush=True)

width = int(point_dims[:, 0].max())
print(f"2D launch: {len(row_box) / 1e6:.2f}M rows x {width} = "
      f"{len(row_box) * width / 1e6:.0f}M threads, {100 * n / (len(row_box) * width):.0f}% "
      f"with a point (row lengths {sorted(set(point_dims[:, 0].tolist()))[:12]}...)",
      flush=True)
cells64 = whole.numpy()
print(f"field: {cells64.dtype}, {cells64.nbytes / 2**30:.2f} GiB", flush=True)
tables64 = {"cell_start": cell_start, "point_start": point_start,
          "dims": cell_dims.reshape(-1).astype(np.int64), "row_box": row_box,
          "row_j": row_j, "row_k": row_k}
references = {}


def best(fn, device, repeats=5):
    fn()
    device.synchronize()
    times = []
    for _ in range(repeats):
        t = T()
        fn()
        device.synchronize()
        times.append(T() - t)
    return 1000 * min(times)


import os
CONFIGS = [tuple(c.split(":")) for c in os.environ.get(
    "CONFIGS", "cpu:f64:i64,cpu:f32:i64,metal:f32:i64").split(",")]
for name, fdt, idt in CONFIGS:
    dtype = np.float64 if fdt == "f64" else np.float32
    itype = np.int64 if idt == "i64" else np.int32
    tables = {k: v.astype(itype) for k, v in tables64.items()}
    device = tack.device(name)
    cells = cells64 if dtype is np.float64 else cells64.astype(np.float32)
    with device:
        t = T()
        a = {k: tack.array(v) for k, v in tables.items()}
        a["cells"] = whole if (name == "cpu" and dtype is np.float64) else tack.array(cells)
        device.synchronize()
        moved = T() - t
        out = tack.empty((n,), a["cells"].dtype)
        runs = {
            "a point": lambda: by_point(a["cells"], a["cell_start"], a["point_start"],
                                        a["dims"], len(boxes), out),
            "a box": lambda: by_box(a["cells"], a["cell_start"], a["point_start"],
                                    a["dims"], out),
            "a row": lambda: by_row(a["cells"], a["cell_start"], a["point_start"],
                                    a["dims"], a["row_box"], a["row_j"], a["row_k"], out),
            "a (row, point), 2D": lambda: by_row_point(
                a["cells"], a["cell_start"], a["point_start"], a["dims"], a["row_box"],
                a["row_j"], a["row_k"], width, out),
            "a (row, point), 2D, 32 lanes": lambda: by_row_point_32(
                a["cells"], a["cell_start"], a["point_start"], a["dims"], a["row_box"],
                a["row_j"], a["row_k"], width - 1, out),
            "two launches: (row, 32 lanes) + the rest a row": lambda: (
                rows_first_lanes(a["cells"], a["cell_start"], a["point_start"], a["dims"],
                                 a["row_box"], a["row_j"], a["row_k"], width - 1, out),
                rows_rest(a["cells"], a["cell_start"], a["point_start"], a["dims"],
                          a["row_box"], a["row_j"], a["row_k"], width - 1, out)),
            "  launch 1 alone": lambda: rows_first_lanes(
                a["cells"], a["cell_start"], a["point_start"], a["dims"], a["row_box"],
                a["row_j"], a["row_k"], width - 1, out),
            "  launch 2 alone": lambda: rows_rest(
                a["cells"], a["cell_start"], a["point_start"], a["dims"], a["row_box"],
                a["row_j"], a["row_k"], width - 1, out),
        }
        print(f"{name} {np.dtype(dtype).name}, {np.dtype(itype).name} indices: inputs to the device {1000 * moved:.0f} ms",
              flush=True)
        for label, fn in runs.items():
            ms = best(fn, device)
            got = out.numpy()
            key = np.dtype(dtype).name
            if key not in references:
                references[key] = got.copy()
            ref = references[key]
            same = np.array_equal(got, ref)
            diff = "" if same else f", max rel diff {np.abs(got - ref).max() / np.abs(ref).max():.1e}"
            print(f"  an iteration {label}: {ms:.0f} ms (identical to the CPU's {key}: "
                  f"{same}{diff})", flush=True)
