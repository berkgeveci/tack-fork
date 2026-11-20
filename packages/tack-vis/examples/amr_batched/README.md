# Batched AMR filters (prototypes)

Scratch prototypes of AMR filters that run every box of a hierarchy in one launch per
pass instead of box by box, measured on AMR-Wind (`plt255900`, 15,598 boxes, 5 levels,
183M cells). Not for review: they live here so the notes that report them can link to
running code. The notes are in Berk's shared AMR notes vault ("Batched AMR filters -
findings").

This branch is PR #401 (flying edges) merged into PR #458 (AMR level buffers), plus
`tack.indirect_load` (a load through a table of addresses, CPU only).

## Files

- `amr_fields.py`: a field over many boxes as kernels read it, `field.at(b, k)`.
  `Joined` is one array plus starts (any device); `Table` is each box's own array
  through a table of addresses (`tack.indirect_load`, CPU). One kernel source, a
  variant compiled for each.
- `amr_project.py`: cell values to point values over the whole hierarchy in one launch
  (`project_to_points(amr, field, name, layout=)`), bit-identical to box-by-box
  `td.project`; also the VTK filter `ProjectToPointsAMR`.
- `amr_flying_edges.py`: PR #401's flying-edges contour over the hierarchy's rows, one
  scan, one surface out (`tables`, `prepare(..., layout=)`, `hierarchy_surface`).
- `fe_hierarchy_run.py`: read, convert, project, contour, timed
  (`--device cpu|metal --f32 --layout joined|table --separate --repeats N`).
- `e2e_layouts.py`: plotfile to surface on the CPU with the blocks' arrays in one buffer
  or each its own allocation (`--mode joined|separate`).
- `tack_isosurface.py`: the box-by-box pipeline through Tack's VTK filters, for
  comparison (`--batched` uses `ProjectToPointsAMR`).
- `project_kernel_timing.py`, `project_devices.py`: the projection kernel's layouts (per
  point, per box, per row, 2D), timed on the CPU and Metal
  (`CONFIGS=cpu:f32:i32,metal:f32:i32`).
- `same_surface.py`: whether two meshes (.npz) hold the same triangles, by vertex
  positions, up to order.
- `audit_kernels.py`: a static scan of `tack.dataset.algorithms`' kernels (loop kind,
  grid-size parameters, view methods).
- `measure.py`: runs a script in-process and reports its peak footprint.

## Running

What you need:

- **This branch** on `PYTHONPATH`: `packages/tack-core/src` and `packages/tack-vis/src`
  (Python 3.13; NumPy, llvmlite; `pyobjc-framework-Metal` for `--device metal`).
- **A VTK newer than the 9.7.1 wheel**: a nightly wheel or a build of master. The
  9.7.1 wheel's Python wrappers slow down as objects pile up (fixed in master), so it
  reads AMR-Wind in 6-8 s and converts it in 25-48 s instead of about 1 s each.
- **The AMReX reader**, `amrex_reader.py` here (a copy of Berk's standalone reader:
  NumPy, threads, optional MPI; boxes to a `vtkOverlappingAMR`). The scripts import it
  from this folder, so run them from here.
- **An AMR-Wind plotfile**: `AMRWIND_PLOTFILE=/path/to/plt255900` (by default
  `~/Data/amrwind/plt255900`; 22 GB, of which only `mag_vorticity` is read).

```sh
cd packages/tack-vis/examples/amr_batched
export PYTHONPATH=$REPO/packages/tack-core/src:$REPO/packages/tack-vis/src:$VTK_SITE_PACKAGES
python fe_hierarchy_run.py OUT --layout joined --repeats 3        # one buffer
python fe_hierarchy_run.py OUT --layout table --separate --repeats 3  # each box its own array
python fe_hierarchy_run.py OUT --device metal --f32 --repeats 3   # Metal
python e2e_layouts.py OUT --mode separate                          # plotfile to surface
```

The first repetition compiles the kernels (flying edges about 5 s instead of 0.6 s);
the numbers below are warm.

## Results (Apple M-series, 10 cores)

| Full AMR-Wind, isovalue 0.5 | Time |
|---|---|
| Read (AMReX reader, 16 threads, warm) | 0.8 s |
| To Tack, no copy | 1.2 s |
| Projection, batched (box by box: 17-22 s) | 0.15-0.3 s |
| Flying edges, batched, CPU f64 (box by box: 22-27 s) | 0.52-0.67 s |
| Flying edges, batched, Metal f32 | 0.21-0.27 s |
| Read to surface, CPU | 2.9-3.0 s (box by box: 30-49 s) |
| The same from separately allocated blocks (`Table`) | 3.1-3.5 s |

The batched runs give the same 21,829,723 triangles as box by box.
