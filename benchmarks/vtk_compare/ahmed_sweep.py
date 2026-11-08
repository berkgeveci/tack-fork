# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A wide sweep of filters on one AhmedML volume, Tack against VTK.

Each run is one system on one device, in a process of its own (both fix their
threading, and Tack its backend, at start-up), and writes a JSON file of every
step's first and warm times and a summary of its output, for checking the two
agree:

    python ahmed_sweep.py tack --device cpu   --output tack-cpu.json
    python ahmed_sweep.py tack --device cuda  --output tack-cuda.json
    python ahmed_sweep.py vtk                 --output vtk.json
    python ahmed_sweep.py compare vtk.json tack-cpu.json tack-cuda.json

The data is an AhmedML ``volume_N.vtu`` (about 20M cells: hexahedra, wedges,
pyramids, tetrahedra and polyhedra), given with ``--data`` or
``TACK_AHMED_VTU``. Both systems read it with VTK's XML reader. Tack converts
it with ``vtk_to_dataset(orient=True, merge_points=True)`` (timed as
``convert``), then copies it to the device (``to_device``); VTK merges its
points with ``vtkStaticCleanUnstructuredGrid`` (``convert``). Both then run the
same steps on the same fields (pMean and UMean on points, pMean on cells).

``--steps`` takes a comma-separated list of step names (as printed) to run
alone; ``--warm`` the number of runs after the first, whose median is the
warm time. Tack is timed through ``tack.sync()``, since GPU backends queue
launches. VTK runs on its SMP backend's threads (``VTK_SMP_BACKEND_IN_USE``,
``VTK_SMP_MAX_THREADS``); use a nightly wheel or a development build, not the
9.7.1 wheel.
"""

import argparse
import json
import os
import resource
import statistics
import sys
import time
import warnings

import numpy as np


def _peak_rss_gb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**30 if sys.platform == "darwin" else 2**20)   # bytes on macOS, KiB on Linux


def log(*args):
    print(*args, flush=True)


class Sweep:
    """The inputs every step shares, read off the grid, and the timer."""

    def __init__(self, grid, only, warm, sync):
        self.only, self.warm, self.sync = only, warm, sync
        self.steps = {}
        x = np.asarray(grid.GetPoints().GetData())
        lo, hi = x.min(0).astype(float), x.max(0).astype(float)
        self.mid = (lo + hi) / 2
        # The slice and clip plane, normal to x. Not through the middle: AhmedML's
        # x = 1.0 is a plane of 31,570 mesh points, where clips differ only in the
        # cells of no volume they keep (VTK's table clip keeps a piece of a cell
        # that touches the plane from below; Tack's clip of polyhedra beside shapes
        # does not), so the cell counts would not compare. A thousandth of the
        # length further on, no point is within a millionth of it.
        self.origin = self.mid + np.array([1e-3 * (hi[0] - lo[0]), 0.0, 0.0])
        near = np.abs(x[:, 0].astype(np.float64) - self.origin[0]) < 1e-6 * (hi[0] - lo[0])
        if near.any():
            warnings.warn(f"{int(near.sum())} points lie on the slice and clip plane, "
                          f"x = {self.origin[0]}", stacklevel=2)
        gx, gz = np.meshgrid(np.linspace(lo[0], hi[0], 800), np.linspace(lo[2], hi[2], 300))
        self.probes = np.c_[gx.ravel(), np.full(gx.size, self.mid[1]), gz.ravel()]
        self.seeds = np.c_[np.full(200, lo[0] + 0.05 * (hi[0] - lo[0])),
                           np.full(200, self.mid[1]), np.linspace(lo[2], hi[2], 202)[1:-1]]
        self.step = 0.001 * float(np.linalg.norm(hi - lo))
        self.nsteps = 4000
        p_cells = np.asarray(grid.GetCellData().GetArray("pMean"))
        self.level = float(np.median(np.asarray(grid.GetPointData().GetArray("pMean"))))
        self.low, self.high = (float(v) for v in np.quantile(p_cells, [0.25, 0.75]))

    def timed(self, name, run, summarize):
        """``run()`` once and ``warm`` more times; returns its last output."""
        if self.only is not None and name not in self.only:
            return None
        times = []
        out = None
        for _ in range(1 + self.warm):
            out = None
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = run()
            self.sync()
            times.append(time.perf_counter() - t0)
        warm = statistics.median(times[1:]) if len(times) > 1 else float("nan")
        summary = summarize(out)
        self.steps[name] = {"first": times[0], "warm": warm, "summary": summary}
        log(f"{name:28s} first {times[0]:8.3f} s  warm {warm:8.3f} s  {summary}")
        return out


def run_tack(grid, records, args):
    import tack
    import tack.dataset as td
    from tack.interop.vtk import vtk_to_dataset

    device = args.device
    sweep = Sweep(grid, args.only, args.warm, tack.sync)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = vtk_to_dataset(grid, orient=True, merge_points=True)
    records["convert"] = time.perf_counter() - t0
    del grid
    if device != "cpu":
        t0 = time.perf_counter()
        data = data.to_device(tack.device(device))
        tack.sync()
        records["to_device"] = time.perf_counter() - t0
    tack.set_default_device(device)
    log(f"tack {device}: read {records['read']:.2f} s, convert {records['convert']:.2f} s"
        + (f", to {device} {records['to_device']:.2f} s" if device != "cpu" else "")
        + f"; {data.num_cells} cells, {data.num_points} points")
    P = ("pMean", data.fields["pMean", td.H1].space)
    U = ("UMean", data.fields["UMean", td.H1].space)
    PC = ("pMean", data.fields["pMean", td.Constant].space)
    origin, timed = tuple(sweep.origin), sweep.timed

    def dataset(d):
        return [int(d.num_cells), int(d.num_points)]

    def norm(f):
        v = np.asarray(tack.arrays.to_host(f.values), np.float64)
        return [list(v.shape), float(np.sqrt(np.nansum(v * v)))]

    timed("cell volume", lambda: td.cell_geometry(data).measure,
          lambda f: float(np.asarray(f.values.numpy(), np.float64).sum()))
    timed("cell centers", lambda: td.cell_centers(data), norm)
    timed("cell gradient", lambda: td.gradients(data, P), norm)
    timed("point gradient", lambda: td.gradients(data, P, at="points"), norm)
    timed("flow quantities (U)", lambda: td.flow_quantities(
        data, td.gradients(data, U, at="points")),
        lambda r: sorted(r) if isinstance(r, dict) else type(r).__name__)
    timed("point to cell", lambda: td.project(data, P, td.Constant(data)), norm)
    timed("cell to point", lambda: td.project(data, PC, td.H1(data)), norm)
    timed("cell quality", lambda: td.cell_quality(data, "scaled_jacobian")["scaled_jacobian"],
          lambda f: float(np.nanmin(f.values.numpy())))
    timed("integrate", lambda: td.integrate(data), lambda r: float(r.measure))
    timed("magnitude (U)", lambda: td.magnitude(data, U), norm)
    surface = timed("extract surface", lambda: td.extract_surface(data), dataset)
    timed("slice", lambda: td.slice_plane(data, origin, (1.0, 0.0, 0.0)), dataset)
    timed("contour", lambda: td.contour(data, P, sweep.level), dataset)
    timed("clip", lambda: td.clip(data, td.Plane(origin, (1.0, 0.0, 0.0)), 0.0), dataset)
    thr = timed("threshold (cells)", lambda: td.threshold(data, PC, sweep.low, sweep.high),
                dataset)
    if thr is not None:
        timed("connected components", lambda: td.connected_components(thr),
              lambda r: int(np.max(r.fields["region"].values.numpy())) + 1)
        del thr
    timed("extract edges", lambda: td.extract_edges(data), dataset)
    timed("tetrahedralize", lambda: td.tetrahedralize(data), dataset)
    timed("cell locator", lambda: td.CellLocator(data), lambda r: "built")
    # Point fields only: a cell field of the same name would land on the points too.
    point_fields = [k for k in data.fields if isinstance(data.fields[k].space, td.H1)]
    timed("probe (240K)", lambda: td.probe(data, sweep.probes, fields=point_fields),
          lambda r: int(r.fields["valid"].values.numpy().sum()))
    timed("streamlines (200)", lambda: td.streamlines(
        data, U, sweep.seeds, sweep.step, sweep.nsteps, step_unit="length"), dataset)
    if surface is not None:
        timed("surface normals", lambda: td.surface_normals(surface), dataset)
        timed("feature edges", lambda: td.feature_edges(surface), dataset)
    return sweep.steps


def run_vtk(grid, records, args):
    from vtkmodules.vtkCommonCore import vtkPoints, vtkSMPTools
    from vtkmodules.vtkCommonDataModel import (
        vtkDataObject,
        vtkPlane,
        vtkPolyData,
        vtkStaticCellLocator,
    )
    from vtkmodules.vtkFiltersCore import (
        vtkArrayCalculator,
        vtkCellCenters,
        vtkCellDataToPointData,
        vtkConnectivityFilter,
        vtkContourFilter,
        vtkExtractEdges,
        vtkFeatureEdges,
        vtkPlaneCutter,
        vtkPointDataToCellData,
        vtkPolyDataNormals,
        vtkProbeFilter,
        vtkStaticCleanUnstructuredGrid,
        vtkThreshold,
    )
    from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
    from vtkmodules.vtkFiltersGeneral import (
        vtkCellDerivatives,
        vtkDataSetTriangleFilter,
        vtkGradientFilter,
        vtkTableBasedClipDataSet,
    )
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from vtkmodules.vtkFiltersParallel import vtkIntegrateAttributes
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter, vtkMeshQuality

    smp = vtkSMPTools()
    records["vtk_smp"] = [smp.GetBackend(), smp.GetEstimatedNumberOfThreads()]
    log(f"VTK SMP {smp.GetBackend()}, {smp.GetEstimatedNumberOfThreads()} threads")
    sweep = Sweep(grid, args.only, args.warm, lambda: None)
    t0 = time.perf_counter()
    grid = vtkStaticCleanUnstructuredGrid(tolerance=0.0, tolerance_is_absolute=True,
                                          remove_unused_points=False)(grid)
    records["convert"] = time.perf_counter() - t0
    log(f"vtk: read {records['read']:.2f} s, merge {records['convert']:.2f} s; "
        f"{grid.number_of_cells} cells, {grid.number_of_points} points")
    POINTS, CELLS = vtkDataObject.FIELD_ASSOCIATION_POINTS, vtkDataObject.FIELD_ASSOCIATION_CELLS
    origin, timed = tuple(sweep.origin), sweep.timed

    def run(alg):
        def go():
            alg.Modified()
            alg.Update()
            return alg.GetOutputDataObject(0)
        return go

    def dataset(d):
        return [int(d.GetNumberOfCells()), int(d.GetNumberOfPoints())]

    def array(name, kind):
        def get(d):
            a = (d.GetCellData() if kind == "cell" else d.GetPointData()).GetArray(name)
            v = np.asarray(a, np.float64)
            return [list(v.shape), float(np.sqrt(np.nansum(v * v)))]
        return get

    def processing(alg, name, association=POINTS):
        alg.SetInputArrayToProcess(0, 0, 0, association, name)
        return alg

    timed("cell volume", run(vtkCellSizeFilter(
        input_data=grid, compute_vertex_count=False, compute_length=False,
        compute_area=False, compute_volume=True, compute_sum=False)),
        lambda d: float(np.asarray(d.GetCellData().GetArray("Volume"), np.float64).sum()))
    timed("cell centers", run(vtkCellCenters(input_data=grid, vertex_cells=False,
                                             copy_arrays=False)),
          lambda d: [d.GetNumberOfPoints()])
    derivatives = vtkCellDerivatives(input_data=grid)
    derivatives.SetVectorModeToComputeGradient()
    derivatives.SetTensorModeToPassTensors()
    processing(derivatives, "pMean")
    timed("cell gradient", run(derivatives), dataset)
    timed("point gradient", run(processing(vtkGradientFilter(
        input_data=grid, result_array_name="g"), "pMean")), array("g", "point"))
    timed("flow quantities (U)", run(processing(vtkGradientFilter(
        input_data=grid, result_array_name="gU", compute_vorticity=True,
        compute_q_criterion=True), "UMean")), array("Q Criterion", "point"))
    p2c = vtkPointDataToCellData(input_data=grid, process_all_arrays=False,
                                 pass_point_data=False)
    p2c.AddPointDataArray("pMean")
    timed("point to cell", run(p2c), array("pMean", "cell"))
    c2p = vtkCellDataToPointData(input_data=grid, process_all_arrays=False,
                                 pass_cell_data=False)
    c2p.AddCellDataArray("pMean")
    timed("cell to point", run(c2p), array("pMean", "point"))
    quality = vtkMeshQuality(input_data=grid)
    quality.SetHexQualityMeasureToScaledJacobian()
    quality.SetWedgeQualityMeasureToScaledJacobian()
    timed("cell quality", run(quality),
          lambda d: float(np.nanmin(np.asarray(d.GetCellData().GetArray("Quality")))))
    timed("integrate", run(vtkIntegrateAttributes(input_data=grid)),
          lambda d: float(np.asarray(d.GetCellData().GetArray("Volume"))[0]))
    calc = vtkArrayCalculator(input_data=grid, function="mag(UMean)", result_array_name="m")
    calc.AddVectorArrayName("UMean")
    timed("magnitude (U)", run(calc), array("m", "point"))
    surface = timed("extract surface", run(vtkGeometryFilter(
        input_data=grid, merging=False, pass_through_cell_ids=False,
        pass_through_point_ids=False)), dataset)
    plane = vtkPlane(origin=origin, normal=(1.0, 0.0, 0.0))
    timed("slice", run(vtkPlaneCutter(input_data=grid, plane=plane, merge_points=True,
                                      interpolate_attributes=True, compute_normals=False,
                                      generate_polygons=True, build_tree=False,
                                      build_hierarchy=False)), dataset)
    contour = processing(vtkContourFilter(input_data=grid, compute_normals=False,
                                          compute_gradients=False, compute_scalars=False,
                                          use_scalar_tree=False), "pMean")
    contour.SetValue(0, sweep.level)
    timed("contour", run(contour), dataset)
    timed("clip", run(vtkTableBasedClipDataSet(input_data=grid, clip_function=plane,
                                               value=0.0)), dataset)
    threshold = processing(vtkThreshold(input_data=grid, lower_threshold=sweep.low,
                                        upper_threshold=sweep.high,
                                        threshold_function=vtkThreshold.THRESHOLD_BETWEEN),
                           "pMean", CELLS)
    thr = timed("threshold (cells)", run(threshold), dataset)
    if thr is not None:
        thr = thr.NewInstance()
        thr.ShallowCopy(threshold.GetOutput())
        connectivity = vtkConnectivityFilter(input_data=thr, color_regions=True)
        connectivity.SetExtractionModeToAllRegions()
        timed("connected components", run(connectivity),
              lambda d: int(connectivity.GetNumberOfExtractedRegions()))
    timed("extract edges", run(vtkExtractEdges(input_data=grid, use_all_points=True)), dataset)
    timed("tetrahedralize", run(vtkDataSetTriangleFilter(input_data=grid,
                                                         tetrahedra_only=True)), dataset)

    def locate():
        locator = vtkStaticCellLocator(data_set=grid)
        locator.BuildLocator()
        return locator
    timed("cell locator", locate, lambda r: "built")
    points = vtkPolyData()
    points.points = vtkPoints(data=np.ascontiguousarray(sweep.probes))
    timed("probe (240K)", run(vtkProbeFilter(source_data=grid, input_data=points)),
          lambda d: int(np.asarray(d.GetPointData().GetArray("vtkValidPointMask"))
                        .view(np.int8).sum()))
    seeds = vtkPolyData()
    seeds.points = vtkPoints(data=np.ascontiguousarray(sweep.seeds))
    tracer = vtkStreamTracer(input_data=grid, source_data=seeds, integrator_type=1,
                             integration_direction=0, initial_integration_step=sweep.step,
                             maximum_number_of_steps=sweep.nsteps,
                             maximum_propagation=sweep.step * sweep.nsteps,
                             compute_vorticity=False)
    tracer.SetIntegrationStepUnit(1)                        # length
    processing(tracer, "UMean")
    timed("streamlines (200)", run(tracer), dataset)
    if surface is not None:
        surface = vtkGeometryFilter(merging=False)(grid)
        timed("surface normals", run(vtkPolyDataNormals(input_data=surface, splitting=False,
                                                        consistency=True)),
              array("Normals", "point"))
        timed("feature edges", run(vtkFeatureEdges(input_data=surface, feature_angle=30.0)),
              dataset)
    return sweep.steps


def compare(paths, which):
    """Print the runs' times side by side, in seconds, the first run's steps in order."""
    runs = [json.load(open(p)) for p in paths]
    names = [f"{r['system']} {r['device']}" if r["system"] == "tack" else "vtk" for r in runs]
    width = max(12, *(len(n) + 2 for n in names))
    print(f"{which} times, in seconds")
    print(f"{'':28s}" + "".join(f"{n:>{width}s}" for n in names))
    for key in ("read", "convert", "to_device"):
        if any(key in r for r in runs):
            print(f"{key:28s}" + "".join(
                f"{r[key]:>{width}.3f}" if key in r else f"{'-':>{width}s}" for r in runs))
    steps = list(dict.fromkeys(s for r in runs for s in r["steps"]))
    for step in steps:
        print(f"{step:28s}" + "".join(
            f"{r['steps'][step][which]:>{width}.3f}" if step in r["steps"]
            else f"{'-':>{width}s}" for r in runs))
    print(f"{'peak RSS (GB)':28s}" + "".join(f"{r['peak_rss_gb']:>{width}.1f}" for r in runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("system", choices=("tack", "vtk", "compare"))
    parser.add_argument("runs", nargs="*", help="compare: the JSON files of earlier runs")
    parser.add_argument("--device", default="cpu",
                        help="tack: cpu, metal, cuda, hip or level_zero (default cpu)")
    parser.add_argument("--data", default=os.environ.get("TACK_AHMED_VTU"),
                        help="an AhmedML volume_N.vtu (default $TACK_AHMED_VTU)")
    parser.add_argument("--output", help="the JSON file to write")
    parser.add_argument("--steps", help="comma-separated step names to run alone")
    parser.add_argument("--warm", type=int, default=3, help="runs after the first (default 3)")
    parser.add_argument("--first", action="store_true",
                        help="compare: first-call times, not warm ones")
    args = parser.parse_args()
    if args.system == "compare":
        if not args.runs:
            parser.error("compare takes the JSON files of earlier runs")
        compare(args.runs, "first" if args.first else "warm")
        return
    if not args.data:
        parser.error("give the AhmedML volume with --data or TACK_AHMED_VTU")
    args.only = set(args.steps.split(",")) if args.steps else None

    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic constructors)
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    records = {"system": args.system, "device": args.device if args.system == "tack" else "-",
               "data": args.data}
    t = time.perf_counter()
    grid = vtkXMLUnstructuredGridReader(file_name=args.data)()
    records["read"] = time.perf_counter() - t
    records["steps"] = (run_tack if args.system == "tack" else run_vtk)(grid, records, args)
    records["peak_rss_gb"] = _peak_rss_gb()
    log(f"peak RSS {records['peak_rss_gb']:.1f} GB")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(records, f, indent=1)


if __name__ == "__main__":
    main()
