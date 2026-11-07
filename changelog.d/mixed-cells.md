### Added

- Polyhedra beside shaped cells in one `UnstructuredTopology` (#307, phase 1):
  type 42 among the types, a polyhedron's connectivity its distinct points as
  VTK's is, and its faces in `polyhedra=(face_offsets, face_points,
  cell_face_offsets, cell_faces, cell_face_sides)`, each face stored once.
  `faces()` gives a face a polyhedron shares with a shaped cell once, with
  both as its sides; the polyhedral faces keep their ids, faces of more than
  four points are a `Polygon` group. `edges()`, `groups()` (the polyhedra a
  group of their own), `validate()`, `to_device`, `extract_cells`,
  `threshold`, `extract_geometry` and `mask` take such meshes, and
  `dataset_to_vtk` writes them. Every algorithm that does not take them yet
  refuses them by name. AhmedML's 21.5M cells (20.9M hexahedra, 508K
  polyhedra) read in 10 s with `merge_points` and `orient`, against 40-54 s
  as polyhedra.
- `Shape.REFERENCE`: whether a cell group has a reference element, so an
  algorithm can take the polyhedral path per group.
- Per-cell work on polyhedra beside shaped cells (#307, phase 2): `gradients`
  (at cells and points), `cell_centers`, `values_at_centers`, `integrate`,
  `project` between point and cell data, `cell_quality`, `traces`, `jump`,
  `upwind_flux`, `divergence`, `flow_quantities` and `interpolate` of point
  values take a mixed mesh, each shape group through its basis and the
  polyhedra as on a `PolyhedralTopology`; `point_links` covers the
  polyhedra. A mixed topology's `reference_cells` is false, so `H1` above
  order 1 and `L2` refuse it by name.

- Polygons beside triangles and quads in one `UnstructuredTopology` (#307):
  type 7 among the types, a polygon's connectivity its loop. Its edges are
  derived from the loops (`has_polygons`; `mixed` is true with polyhedra or
  polygons), and every algorithm that takes polyhedra beside shapes takes
  them. `dataset_to_vtk` writes them.
- Filters that build a topology take polyhedra beside shapes (#307, phase 3):
  `contour` and `slice` give the shapes' triangles beside the polyhedra's
  polygons, one surface whose crossings both share; `clip` keeps whole shapes
  as they are and makes every other kept cell the polyhedron of its kept part;
  `extract_surface` gives a polyhedron's faces of five points or more as
  polygons beside the triangles and quads; `tetrahedralize` cones each
  polyhedron from a point it adds, its quads split as a shape's are (so they
  conform) and its larger faces fanned from points they add; `merge_points`
  matches the polyhedra's faces again; `as_polyhedra`, `check_winding`,
  `as_polygons`, `triangulate`, `surface_normals` and `feature_edges` take
  them too.

### Fixed

- Hiding cells by their hidden points (`hidden_by_points`, which
  `vtk_to_dataset` calls) compiles for polyhedra beside shaped cells.

### Changed

- **Breaking:** `vtk_to_dataset` of a grid with polyhedra gives an
  `UnstructuredTopology` with the polyhedra beside the shaped cells, not a
  `PolyhedralTopology`. `polyhedral=True` (or `"all"`) reads every cell as a
  polyhedron as before; the VTK algorithm wrappers do so for grids with
  polyhedra, so their filters keep their inputs.
