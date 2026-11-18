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

The scripts import each other, so run them from this folder. They read the plotfile at
`~/Data/amrwind/plt255900` with the AMReX reader in `~/Work/amrex-reader`
(`read_plotfile`). Timings in the notes used Python 3.13, this branch on `PYTHONPATH`
(`packages/tack-core/src`, `packages/tack-vis/src`), and a VTK build with the bulk
`SetAMRBoxes` path. VTK 9.7.1 wheels read and convert many times slower.

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
