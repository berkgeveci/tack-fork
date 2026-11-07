### Fixed

- An AMR hierarchy's boxes cost less again: `vtk_to_dataset` of a
  `vtkOverlappingAMR` makes each image box's topology, coordinates, fields and
  marks without their constructors' checks (as `ghost_cells` makes its grown
  boxes and the `AMRDataSet` constructor its marks), interns a space without
  parameters by a key worked out once, and reads VTK's information keys
  through the key's own `Has`/`Get` (3 us a call to `vtkInformation`'s). On
  AMR-Wind's five levels (15,598 boxes, CPU), `vtk_to_dataset(copy=False)` of
  the hierarchy a NumPy reader makes went from 1.65 to 0.90 s, copying VTK's
  reader's from 1.9 to 1.25 s, and `ghost_cells` from 2.2 to 2.0 s, the boxes
  the same.
