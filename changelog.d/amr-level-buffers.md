### Changed

- An AMR hierarchy keeps each level's values in one array per device, each
  box's a view of it (`array[i:j]`): the marks `AMRDataSet` makes, and the
  fields `vtk_to_dataset` copies from a `vtkOverlappingAMR` and
  `amr_gaussian_pulse` makes (`tack.dataset.amr.in_level_buffers`). Writes
  through a box land in its level's array. `vtk_to_dataset(..., copy=False)`
  still shares VTK's arrays box by box.
