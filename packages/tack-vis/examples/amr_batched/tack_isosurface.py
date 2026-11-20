"""AMR-Wind's vorticity isosurface through Tack's VTK filters, box by box, timed:

    read_plotfile -> [GhostCells(1)] -> Project(to points) -> [RemoveGhosts] -> Contour

``extract`` gives the triangles, all levels joined into one mesh: ``points`` (n, 3)
f32, ``triangles`` (m, 3) u32, ``level`` (m,) u8. Run as a script it writes them to an
.npz for the renderers.

    python tack_isosurface.py OUT.npz [--iso V] [--max-level L] [--no-ghosts]
"""
import argparse
import os
import sys
import time

import numpy as np

import tack

from amrex_reader import read_plotfile  # noqa: E402

PLOTFILE = os.environ.get("AMRWIND_PLOTFILE", os.path.expanduser("~/Data/amrwind/plt255900"))


def timed(label, fn):
    t = time.perf_counter()
    out = fn()
    print(f"{label}: {time.perf_counter() - t:.1f} s", flush=True)
    return out


def extract(iso, field="mag_vorticity", max_level=None, ghosts=False, batched=False):
    tack.set_default_device("cpu")
    from tack.interop.vtk import algorithms as ta
    from vtkmodules.util.numpy_support import vtk_to_numpy

    amr = timed("read", lambda: read_plotfile(PLOTFILE, fields=[field],
                                              max_level=max_level, blank=False))
    total = time.perf_counter()
    source = amr
    if ghosts:
        g = ta.GhostCells(layers=1)
        g.SetInputDataObject(source)
        timed("ghost cells", g.Update)
        source = g.GetOutputDataObject(0)
    if batched:
        from amr_project import ProjectToPointsAMR
        project = ProjectToPointsAMR(field=field, name="point_values")
    else:
        project = ta.Project(field=field, to="points", name="point_values")
    project.SetInputDataObject(source)
    timed("project to points", project.Update)
    source = project.GetOutputDataObject(0)
    if ghosts:
        strip = ta.RemoveGhosts()
        strip.SetInputDataObject(source)
        timed("remove ghosts", strip.Update)
        source = strip.GetOutputDataObject(0)
    contour = ta.Contour(field="point_values", value=iso)
    contour.SetInputDataObject(source)
    timed("contour", contour.Update)
    print(f"Tack total (after the read): {time.perf_counter() - total:.1f} s", flush=True)

    # The vtkPartitionedDataSetCollection: one partitioned dataset per level.
    t = time.perf_counter()
    out = contour.GetOutputDataObject(0)
    points, triangles, levels, offset = [], [], [], 0
    for level in range(out.GetNumberOfPartitionedDataSets()):
        pds = out.GetPartitionedDataSet(level)
        for i in range(pds.GetNumberOfPartitions()):
            poly = pds.GetPartition(i)
            if poly is None or poly.GetNumberOfCells() == 0:
                continue
            p = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)
            polys = poly.GetPolys()
            conn = vtk_to_numpy(polys.GetConnectivityArray())
            offs = vtk_to_numpy(polys.GetOffsetsArray())
            if not np.all(np.diff(offs) == 3):
                raise ValueError("expected triangles only")
            points.append(p)
            triangles.append(conn.reshape(-1, 3).astype(np.int64) + offset)
            levels.append(np.full(len(conn) // 3, level, np.uint8))
            offset += len(p)
    points = np.concatenate(points)
    triangles = np.concatenate(triangles).astype(np.uint32)
    levels = np.concatenate(levels)
    print(f"gather: {time.perf_counter() - t:.1f} s; {len(triangles) / 1e6:.1f}M triangles, "
          f"{len(points) / 1e6:.1f}M points; by level {np.bincount(levels).tolist()}",
          flush=True)
    return points, triangles, levels


def vertex_normals(points, triangles):
    """Area-weighted vertex normals (as barney_iso.py computes them)."""
    p = points.astype(np.float64)
    face = np.cross(p[triangles[:, 1]] - p[triangles[:, 0]],
                    p[triangles[:, 2]] - p[triangles[:, 0]])
    normals = np.zeros_like(p)
    for k in range(3):
        for c in range(3):
            normals[:, c] += np.bincount(triangles[:, k], weights=face[:, c], minlength=len(p))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    return normals.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--iso", type=float, default=2.0)
    parser.add_argument("--field", default="mag_vorticity")
    parser.add_argument("--max-level", type=int, default=None)
    parser.add_argument("--no-ghosts", action="store_true")
    parser.add_argument("--batched", action="store_true",
                        help="project with amr_project's one launch over the hierarchy")
    args = parser.parse_args()
    points, triangles, levels = extract(args.iso, args.field, args.max_level,
                                        not args.no_ghosts, args.batched)
    np.savez(args.out, points=points, triangles=triangles, level=levels)
