### Fixed

- `AMRDataSet` makes its marks (`REFINEDCELL`, `REFINEDPOINT`, siblings'
  `DUPLICATEPOINT`) on the device, finding the boxes that meet by binning them
  on the device instead of on the host by a sweep along x, and checks overlaps
  and nesting the same way: on AMR-Wind's five levels (15,598 boxes, CPU) the
  constructor went from 4.0 to 0.8 s, the marks bit for bit the same.
