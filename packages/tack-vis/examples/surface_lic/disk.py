# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Surface LIC of the velocity on the middle slice of ``disk_out_ref.ex2`` (VTK's and
ParaView's example: flow in a cylinder around a spinning disk; 7,472 linear hexahedra,
fields at the points). VTK reads the file, Tack converts it and cuts it through the
centre (normal x, ParaView's default slice, so the plane holds the cylinder's axis), and
``lic.LIC`` draws ``V`` as LIC over ``Temp``.

By default the surface pass evaluates ``V`` and ``Temp`` per pixel with the hexahedra's
own basis (``exact``: Tack's evaluation as Slang, each pixel located in its triangle's
cell by Newton's method); ``EXACT=0`` interpolates the slice's vertex values instead. The
check renders the values themselves and compares them, pixel by pixel, with ``td.probe``
at the pixels' points on the plane (both ways, for comparison).

    python disk.py [path/to/disk_out_ref.ex2] [OUTDIR]
"""

import os
import sys

import numpy as np

import exact
import lic
import scene

import tack
import tack.dataset as td

DISK = os.path.expanduser(
    "~/.ExternalData/SHA512/d66c52c6b7ad3766ac2693bab2cc773ac43d18d753742764498ea7fab4c3e3cb06c58d0d922977c46a6ed7c59868700b7a2faba9f73c40b71caff62d1f9eb0a9")


def read(path):
    from vtkmodules.vtkIOExodus import vtkExodusIIReader

    from tack.interop.vtk import vtk_to_dataset

    reader = vtkExodusIIReader(file_name=path)
    reader.UpdateInformation()
    reader.SetAllArrayStatus(vtkExodusIIReader.NODAL, 1)
    reader.Update()
    grid = reader.GetOutput().GetBlock(0).GetBlock(0)
    return vtk_to_dataset(grid), grid.GetBounds()


def main(path=DISK, out="."):
    tack.set_default_device("cpu")
    data, bounds = read(path)
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    centre = 0.5 * (lo + hi)
    points, triangles, vectors, normal, temp = scene.slice_triangles(
        data, tuple(centre), (1.0, 0.0, 0.0), vector="V", scalar="Temp")
    print(f"{data.num_cells} hexahedra; the middle slice: {len(triangles)} triangles, "
          f"{len(points)} points; |V| up to {np.linalg.norm(vectors, axis=1).max():.3g}, "
          f"Temp {temp.min():.4g} to {temp.max():.4g}", flush=True)

    def renderer_for(per_pixel):
        size = dict(size=(1920, 1080), supersample=int(os.environ.get("SUPERSAMPLE", "1")))
        if not per_pixel:
            return lic.LIC(points, triangles, vectors, os.path.join(out, "wgsl-vertex"),
                           scalars=temp, **size)
        source, pools = exact.volume_functions(data, "V", "Temp")
        cells, corners, pcs, moved = exact.slice_cells(data, points, triangles)
        print(f"  exact: {len(np.unique(cells))} cells under the slice; "
              f"generated Slang {len(source) // 1024} KiB; buffers "
              f"{sum(len(p) for p in pools) * 4 / 2**20:.2f} MB; {moved} vertices started "
              f"from a cell centre", flush=True)
        k = len(triangles)
        return lic.LIC(corners.reshape(-1, 3), np.arange(3 * k, dtype=np.uint32).reshape(-1, 3),
                       np.zeros((3 * k, 3), np.float32), os.path.join(out, "wgsl-exact"),
                       exact=dict(source=source, pools=pools, cells=cells, pcs=pcs), **size)

    renderers = {"exact": renderer_for(True), "per vertex": renderer_for(False)}
    renderer = renderers["exact" if os.environ.get("EXACT", "1") != "0" else "per vertex"]
    # From +x, the cylinder's axis (z) across the picture.
    span = float(np.max(hi - lo))
    view = dict(eye=centre + np.array([1.7 * span, 0.0, 0.0]), target=centre,
                up=(0.0, 1.0, 0.0), field_range=(float(temp.min()), float(temp.max())),
                by_scalar=True)
    renderer.render(**view)
    runs = [renderer.render(**view) for _ in range(3)]
    result = min(runs, key=lambda r: r["timings"]["lic"])
    print(f"{renderer.w} x {renderer.h}: surface pass {result['timings']['surface']:.1f} ms, "
          f"LIC passes {result['timings']['lic']:.1f} ms", flush=True)
    lic.write_png(os.path.join(out, "disk_lic.png"), result["picture"])
    mask = result["field"][..., 2] >= 0
    lic.write_png(os.path.join(out, "disk_lic_gray.png"), lic.gray(result["lic"], mask))
    err = lic.orientation_error(result["lic"], result["field"])
    print(f"  orientation vs V's screen direction over {err.size} pixels: median "
          f"{np.median(err):.1f} deg, 90th percentile {np.percentile(err, 90):.1f} deg",
          flush=True)
    for label, r in renderers.items():
        check(data, r, view, centre, np.array([1.0, 0.0, 0.0]), label)
    print(f"  pictures in {os.path.abspath(out)}", flush=True)


def check(data, renderer, view, origin, normal, label):
    """The rendered values at every covered pixel against ``td.probe`` at that pixel's
    point on the plane (its ray through the pixel's centre)."""
    result = renderer.render(**view, check=True)
    values, rows = result["values"], result["rows"]
    h, w, _ = values.shape
    covered = values[..., 3] > 0          # Temp is positive; 0 where no surface
    j, i = np.nonzero(covered)
    ndc = np.stack([(i + 0.5) / w * 2 - 1, 1 - (j + 0.5) / h * 2], axis=1)
    inverse = np.linalg.inv(rows)

    def unproject(z):
        clip = np.concatenate([ndc, np.full((len(ndc), 1), z), np.ones((len(ndc), 1))], axis=1)
        p = clip @ inverse.T
        return p[:, :3] / p[:, 3:]

    near, far = unproject(0.0), unproject(1.0)
    d = far - near
    t = ((origin - near) @ normal) / (d @ normal)
    where = near + t[:, None] * d
    out = td.probe(data, where, fields=["V", "Temp"])
    valid = out.fields["valid"].values.numpy().reshape(-1) > 0
    v_exact = out.fields["V"].values.numpy().reshape(-1, 3)[valid]
    t_exact = out.fields["Temp"].values.numpy().reshape(-1)[valid]
    v_drawn, t_drawn = values[j, i, :3][valid], values[j, i, 3][valid]
    vmax = float(np.linalg.norm(v_exact, axis=1).max())
    trange = float(t_exact.max() - t_exact.min())
    dv = np.linalg.norm(v_drawn - v_exact, axis=1) / vmax
    dt = np.abs(t_drawn - t_exact) / trange
    print(f"  check, {label}: {valid.sum()} pixels; |dV| / max|V| median {np.median(dv):.1e}, "
          f"99th percentile {np.percentile(dv, 99):.1e}, max {dv.max():.1e}; dTemp / range "
          f"median {np.median(dt):.1e}, max {dt.max():.1e}; more than one colour step off: "
          f"{100 * np.mean(dt > 1 / 255):.2f}%", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else DISK, a[1] if len(a) > 1 else ".")
