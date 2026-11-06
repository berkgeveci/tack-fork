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
