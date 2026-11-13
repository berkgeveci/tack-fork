# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Surface LIC of the velocity on the middle slice of ``disk_out_ref.ex2`` (VTK's and
ParaView's example: flow in a cylinder around a spinning disk; 7,472 linear hexahedra,
fields at the points). VTK reads the file, Tack converts it and cuts it through the
centre (normal x, ParaView's default slice, so the plane holds the cylinder's axis), and
``lic.LIC`` draws ``V`` as LIC over ``Temp``.

    python disk.py [path/to/disk_out_ref.ex2] [OUTDIR]
"""

import os
import sys

import numpy as np

import lic
import scene

import tack

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

    renderer = lic.LIC(points, triangles, vectors, os.path.join(out, "wgsl"),
                       size=(1920, 1080), supersample=int(os.environ.get("SUPERSAMPLE", "1")),
                       scalars=temp)
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
    print(f"  pictures in {os.path.abspath(out)}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else DISK, a[1] if len(a) > 1 else ".")
