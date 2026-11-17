### Added

- `L2` with an order per cell from 0 to 15 (p-adaptive DG fields), in any `L2`
  basis (Gauss-Lobatto, equispaced, Gauss-Legendre, Bernstein) on every shape
  that has that order: `L2(data, order=orders, basis=...)`. Each cell is read
  as the space of its order (`space.at_order(p)`), its values in blocks; the
  cuts subdivide to the highest order present, with their crossings at the
  field's roots, and `integrate`, gradients, `project`, `jump`, `subdivide`,
  `probe` and the cell filters take such fields (#423). Orders were limited to 0
  and 1.

### Fixed

- `clip` of an `L2` field of order 0 or 1, or of orders 0 and 1 per cell, was
  refused ("clip takes point data"); it clips through subdivision, as higher
  orders do.
- With orders 0 and 1 per cell: `integrate` raised a `TypeError`, and the cuts
  of a curved mesh (`contour`, `clip`, `threshold`) lost the field in
  subdivision (`KeyError`) (#423).
