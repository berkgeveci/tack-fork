### Changed

- Faster analysis of large meshes, measured on AhmedML's 21.5M cells (warm,
  CPU; Metal faster again):
  - `integrate` of point and cell data is a weighted sum of their values,
    the weights (each point's shape function's integral, each cell's measure)
    made once for every field: 14 fields in 0.32 s, from 13.6. Quadrature that
    remains (higher orders) takes the fewest Gauss points exact for each shape,
    8 on a hexahedron rather than 64.
  - Point gradients: a straight hexahedron's corner Jacobians are its edges,
    and a 3-vector's (a velocity's) take one Jacobian inverse a corner for all
    three components: 0.47 s from 1.0 for a scalar, 0.9 s from 2.9 for
    `flow_quantities`.
  - `cell_quality` computes only the measures asked for, and a straight cell's
    corner edges from its points: the scaled Jacobian in 0.23 s, from 1.0.
  - `connected_components` and `surface_normals`' orientation set their
    "changed" flag once a pass, not by an atomic from every thread: 0.17 s
    from 0.96 for 10.7M cells (4.2 s the first time).
  - `clip` of polyhedra beside shapes skips polyhedra with nothing to keep:
    0.17 s from 0.30.
  - Deriving faces and edges: the bucket sort's scatter moves each key beside
    its index (`buckets.sorted_key_order`), so the sort and the runs read a
    bucket's keys together rather than at random, and the edges of polyhedra
    beside shapes go into arrays of their own rather than one split by copies:
    edges 0.81 s from 1.47, faces 0.93 from 1.06, the first time they are
    needed.
  - Corner gradients of straight hexahedra take the Jacobian's inverse
    transpose from its columns' cross products: a scalar's point gradient
    0.35 s from 0.40.
