### Changed

- `vtk_to_dataset(amr, copy=False)` of a `vtkOverlappingAMR` whose blocks' arrays
  lie end to end in memory -- a field read into one array, each block a view of
  it, as `dataset_to_vtk(amr, copy=False)` and a NumPy reader lay them out --
  shares each such field as one array, every box's values a view of it, over the
  same memory: the single-array form the AMR algorithms take as it is, with no
  copy. It is found from each block's address (about 1 us a block); a field
  whose blocks do not abut is shared box by box as before.
