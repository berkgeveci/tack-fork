### Fixed

- Making an AMR hierarchy's boxes costs less a box: `vtk_to_dataset` of a
  `vtkOverlappingAMR` and `ghost_cells` of an `AMRDataSet` assemble each box
  from the parts they just made, without the `DataSet` constructor's checks of
  each (about 12 us a box), and every dataset, field and image is cheaper to
  make (a unit's spelling kept rather than printed by unyt each time, origins
  and image parameters of plain numbers read without NumPy, one identity
  matrix for every unoriented image). On AMR-Wind's five levels (15,598 boxes,
  CPU) `vtk_to_dataset` went from 2.1 to 1.9 s and `ghost_cells` from 2.8 to
  2.2 s, the boxes the same.
