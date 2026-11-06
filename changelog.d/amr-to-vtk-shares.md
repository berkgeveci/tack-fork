### Fixed

- `dataset_to_vtk(amr, copy=False)` shares an `AMRDataSet`'s memory as it does
  any dataset's: each block's arrays point at the box's own values (views of one
  array a field, at their offsets in it). It copied them, as it did a dict of
  datasets and the collection of levels a hierarchy VTK's AMR cannot hold
  becomes.
