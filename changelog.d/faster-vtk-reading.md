### Changed

- `vtk_to_dataset` reads AhmedML's 21.5M cells (`orient=True`,
  `merge_points=True`) in 2.7-3.1 s, from 9.1: points of equal coordinates
  are merged by sorting their bits (no locator and no atomics; `merge_points`
  with no tolerance too, the same sets and numbering), the merge's gathers of
  point data and connectivity, the inside-out check of polyhedra and the
  screen for collapsed hexahedra are Tack kernels over VTK's arrays where they
  lie, not NumPy on one thread.
- Polyhedra's faces are matched by Tack kernels (each face keyed, radix sorted
  and compared point for point), 0.12 s for AhmedML's 4.5M polyhedral faces
  against 0.75 in NumPy; reading, `from_cell_faces`, `merge_points`,
  `ghost_cells` and `merge_pieces` of polyhedra all match faces this way.
