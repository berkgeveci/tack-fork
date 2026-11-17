### Changed

- `contour` of a structured grid of solid cells -- an image, a rectilinear or a
  curvilinear grid -- with a point field goes by flying edges, as VTK's
  `vtkFlyingEdges3D`: crossings counted along the grid's rows, numbered by a
  scan and written once, with no per-corner arrays and no sort. The triangles,
  points and carried values are the general path's, bit for bit, up to their
  order. On an 11M-cell image at ten values (CPU, M1 Max) the peak above the
  input fell from 2.24 to 0.32 GB and the time from 259 to 219 ms; on an
  80M-cell image at two values, from 1.32 to 0.30 GB and 346 to 263 ms (#316).
  Hidden and refined cells (an AMR box's) are left out in place: before, the
  rest were extracted into unstructured cells first. A 2D structured grid's
  contour lines go by flying edges too, the same lines as the general path,
  their orientation included: on a 4096^2 image at ten values the peak above
  the input fell from 14.1 to 0.18 GB, at one value from 2.01 to 0.02 GB and
  from 53 to 33 ms.
- `slice` of an image or a rectilinear grid by a `Plane` (`slice_plane`) goes
  by a plane cutter, as VTK's `vtkFlyingEdgesPlaneCutter`: flying edges over
  each row's stretch near the plane, found from the plane's equation, the
  distance computed where it is read. Only the cells near the plane are
  visited and nothing is stored for the others; the output is the general
  path's, bit for bit, up to its order. A plane through a 432^3 image (80M
  cells, CPU, M1 Max) took 177 ms and 0.71 GB above the input; it takes 14 ms
  and 0.03 GB. A curvilinear grid's planes visit every row, the distance still
  computed where read (a 256^3 grid: 47 to 7 ms), and hidden and refined cells
  are left out in place (an image of 256^3 with a tenth refined: 116 ms and
  1.49 GB to 12 ms and 0.01 GB).
