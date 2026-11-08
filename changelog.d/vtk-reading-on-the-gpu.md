### Changed

- `vtk_to_dataset` with CUDA, HIP or Level Zero the default device converts an
  unstructured grid of linear shapes and polyhedra there: VTK's arrays go to the
  device once, and the merge of equal points, the screen and repair of inside-out
  cells and the building and matching of polyhedra's faces run on it, with the same dataset as
  the host conversion bit for bit. AhmedML's 21.5M cells (`orient=True`,
  `merge_points=True`) reach CUDA in 2.1 s from a fresh process, against 6.7 s
  converting on the host and copying (12.8 before this release's other changes),
  at a host peak of 6.1 GB, from 12.4 (jura). The CPU and Metal convert on the
  host as before; what the device path does not do falls back to it.
