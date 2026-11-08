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
    0.17 s from 0.30. The input's point and cell ids it carries through are
    counting arrays, not ranges made on the host and copied to the device:
    on CUDA 0.10 s from 0.33 (jura), on the CPU 0.83 from 1.07.
  - Deriving faces and edges: the bucket sort's scatter moves each key beside
    its index (`buckets.sorted_key_order`), so the sort and the runs read a
    bucket's keys together rather than at random, and the edges of polyhedra
    beside shapes go into arrays of their own rather than one split by copies:
    edges 0.81 s from 1.47, faces 0.93 from 1.06, the first time they are
    needed.
  - Corner gradients of straight hexahedra take the Jacobian's inverse
    transpose from its columns' cross products: a scalar's point gradient
    0.35 s from 0.40.
  - `integrate`'s measure is summed as its fields are, by chunks in f64 on the
    device, rather than every owned cell's measure copied to the host: on CUDA
    0.06 s from 0.17 for AhmedML (jura). A device without f64 sums it on the
    host as before.
  - A pure number's unit (`"dimensionless"`, `"1"`), which filters give
    their results (`cell_quality`, `surface_normals`, `voxelize`), is
    answered without unyt: a program that gives no unit no longer imports
    unyt and sympy at its first such filter, half a second
    (`cell_quality`'s first call on AhmedML 0.58 s to 0.25).
  - A mixed mesh's faces are matched by `_derive_bucketed`'s counting sort
    of the records themselves (each (cell, face) and each polyhedral face a
    shaped cell may share placed under its lowest point), not by a sort of
    keys whose runs were read through the permutation: the same faces, array
    for array. AhmedML's first `faces()` 1.19 s to 1.03, its memory
    6.7 GB to 2.0; crowded buckets keep the sorted path.
