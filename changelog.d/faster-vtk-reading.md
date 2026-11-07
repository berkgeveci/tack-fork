### Changed

- `vtk_to_dataset` reads AhmedML's 21.5M cells (`orient=True`,
  `merge_points=True`) in 3.4-3.7 s, from 9.1: points of equal coordinates
  are merged by sorting their bits (no locator and no atomics; `merge_points`
  with no tolerance too, the same sets and numbering), the merge's gathers of
  point data and connectivity, the inside-out check of polyhedra and the
  screen for collapsed hexahedra are Tack kernels over VTK's arrays where they
  lie, not NumPy on one thread.
