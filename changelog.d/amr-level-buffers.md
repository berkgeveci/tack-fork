### Changed

- An AMR hierarchy keeps each level's values in one array per device, each
  box's a view of it (`array[i:j]`): the marks `AMRDataSet` makes, and the
  fields `vtk_to_dataset` copies from a `vtkOverlappingAMR` and
  `amr_gaussian_pulse` makes (`tack.dataset.amr.in_level_buffers`). Writes
  through a box land in its level's array. `vtk_to_dataset(..., copy=False)`
  still shares VTK's arrays box by box.

### Fixed

- `statistics` of an AMR hierarchy reduces each level as one array, the
  level's own where its boxes are views of it, instead of box by box: on a
  hierarchy of 2,592 boxes (10.6M cells, CPU) volume-weighted statistics went
  from 0.68 to 0.05 s, plain ones from 0.37 to 0.03 s.
