"""Where amr_project's 1.4 s goes: setup, allocation, the kernel, and variants."""
import time

import numpy as np

import tack

tack.set_default_device("cpu")
from amrex_reader import read_plotfile  # noqa: E402
from tack import arrays  # noqa: E402
from tack.interop.vtk import algorithms as ta  # noqa: E402

import amr_project as ap  # noqa: E402

T = time.perf_counter


@tack.kernel
def _search_only(point_start, nboxes, out):
    for p in range(out.shape[0]):
        lo = 0
        hi = nboxes
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if point_start[mid] <= p:
                lo = mid
            else:
                hi = mid
        out[p] = lo


@tack.kernel
def _by_box(cells, cell_start, point_start, dims, out):
    # One iteration a box, its points in order: no search, no division.
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


amr = read_plotfile("/Users/berk.geveci/Data/amrwind/plt255900", fields=["mag_vorticity"],
                    blank=False)
data = ta._as_dataset(amr)
print("fields per box:", list(next(iter(data))[1].fields.keys()))

t = T()
boxes = [box for _, box in data]
whole = arrays._joined_view([b.fields["mag_vorticity"].values for b in boxes])
point_dims = np.array([b.topology.point_dims for b in boxes], np.int64)
cell_dims = point_dims - 1
cell_start = np.concatenate([[0], np.cumsum(cell_dims.prod(axis=1))])
point_start = np.concatenate([[0], np.cumsum(point_dims.prod(axis=1))])
cs, ps = tack.array(cell_start), tack.array(point_start)
dm = tack.array(cell_dims.reshape(-1).astype(np.int64))
print(f"setup (Python over the boxes, small arrays): {T() - t:.3f} s")
n = int(point_start[-1])


def run(label, fn, repeats=3):
    fn()                                       # compile, first touch
    times = []
    for _ in range(repeats):
        t = T()
        fn()
        times.append(T() - t)
    print(f"{label}: {1000 * min(times):.0f} ms (best of {repeats})", flush=True)


t = T()
out = tack.empty((n,), whole.dtype)
print(f"allocate {n / 1e6:.0f}M f32: {1000 * (T() - t):.0f} ms")
run("kernel, as in amr_project (search + div/mod + stencil)",
    lambda: ap._cells_to_points(whole, cs, ps, dm, len(boxes), out))
ids = tack.empty((n,), tack.i64)
run("search only (box id of each point)", lambda: _search_only(ps, len(boxes), ids))
out2 = tack.empty((n,), whole.dtype)
run("a box an iteration (no search, no div/mod)", lambda: _by_box(whole, cs, ps, dm, out2))
print("same values:", np.array_equal(out.numpy(), out2.numpy()))
