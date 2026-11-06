### Fixed

- `AMRDataSet` makes its marks (`REFINEDCELL`, `REFINEDPOINT`, siblings'
  `DUPLICATEPOINT`) on the device, finding the boxes that meet by binning them
  on the device instead of on the host by a sweep along x, and checks overlaps
  and nesting the same way: on AMR-Wind's five levels (15,598 boxes, CPU) the
  constructor went from 4.0 to 0.8 s, the marks bit for bit the same.
- `ghost_cells` of an `AMRDataSet` makes the grown boxes' marks on the device
  too, one array a device as the constructor's, instead of on the host box by
  box: on AMR-Wind's five levels the marks went from 2.2 to 0.5 s and
  `ghost_cells` from 11.5 to 9.0 s, the marks bit for bit the same.
- `ghost_cells` and `average_down` of an `AMRDataSet` keep their fills' rows
  (targets, sources, weights) on the device, compacted there, instead of
  reading them back and assembling them in NumPy; only sources another device
  or rank holds come to the host, to make the exchange. On AMR-Wind's five
  levels `ghost_cells` went from 9.0 to 5.8 s, the values bit for bit the
  same.
- `ghost_cells` of an `AMRDataSet` places a device's boxes' own values in their
  grown arrays by one launch where they are views of one array (as
  `vtk_to_dataset` makes them), not one launch a box: on AMR-Wind's five
  levels 1.3 to 0.09 s, `ghost_cells` 5.8 to 4.4 s.
