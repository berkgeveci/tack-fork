### Fixed

- `vtk_to_dataset` of a `vtkOverlappingAMR` copies each box's arrays straight
  into its view of one array a field and a kind of ghost marks, instead of an
  array a box concatenated afterwards; boxes read no direction matrix (AMR
  boxes are axis-aligned), unoriented images share one identity matrix on
  their device instead of storing one each, and a topology's spaces are made
  without a `WeakValueDictionary` a call: on AMR-Wind's five levels (15,598
  boxes, CPU) the conversion went from 4.5 to 2.9 s.
