### Changed

- An AMR hierarchy keeps its values in one array per device, levels in order
  and each box's a view of it (`array[i:j]`): the marks `AMRDataSet` makes, the
  fields `vtk_to_dataset` copies from a `vtkOverlappingAMR` and
  `amr_gaussian_pulse` makes (`tack.dataset.amr.in_device_buffers`), and the
  grown boxes `ghost_cells` makes. Writes through a box land in that array. `vtk_to_dataset(..., copy=False)`
  still shares VTK's arrays box by box.

### Fixed

- `statistics` of an AMR hierarchy reduces each level as one array, the
  level's own where its boxes are views of it, instead of box by box: on a
  hierarchy of 2,592 boxes (10.6M cells, CPU) volume-weighted statistics went
  from 0.68 to 0.05 s, plain ones from 0.37 to 0.03 s.
- `ghost_cells`, `update_ghosts` and `average_down` of an AMR hierarchy fill a
  device's boxes in place where they are one stretch of memory, instead of
  concatenating them and copying back, and `average_down` weighs and divides
  a device's cells in one launch each, not three a box: on 2,592 boxes (CPU)
  `average_down` went from 0.58 to 0.37 s.
