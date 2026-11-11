### Added

- `extract_surface` gives `L2` (DG) fields to the surface in the same space,
  each face holding its cell's trace, on every shape with a reference element
  and every basis (equispaced, Gauss-Lobatto, Gauss-Legendre, Bernstein); an
  order-0 field takes its cell's value. They were left behind, so a DG
  solution's surface had none of its fields (#418).
