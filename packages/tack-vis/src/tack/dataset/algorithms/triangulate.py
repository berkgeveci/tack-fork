# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``triangulate``: every 2D cell as triangles: a quad or pixel in 2, a polygon by ear cutting; other cells are left out. The points are the same; cell fields go to each cell's triangles."""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.arrays import materialize, size_of
from tack.dataset import ids, shapes
from tack.dataset.algorithms._refine import _ears, _fan_counts, _pieces_topology, _split
from tack.runtime.dispatch import on_inputs_device

__all__ = ['triangulate']


@on_inputs_device
def triangulate(data, fields=None):
    """Every 2D cell as triangles: a quad or pixel in 2, a polygon of ``n`` corners in
    ``n - 2`` by ear cutting in its plane (as VTK's vtkTriangleFilter), so a concave
    one is covered once, each triangle wound as the polygon; other cells are left
    out. The points are the same; cell fields go to each cell's triangles."""
    from tack.dataset.ghosts import visible

    data = visible(data)
    t = data.topology
    if getattr(t, "reference_cells", True):
        return _split(data, "triangles", fields)
    if t.mixed:
        return _mixed_triangles(data, fields)
    if t.dimension != 2:
        raise NotImplementedError(f"triangulate takes 2D cells, not {t.dimension}D ones")
    loop_offsets, loop_points = t.loops()
    n = data.num_cells
    odt = ids.for_output(data, 3 * size_of(loop_points))
    counts = tack.empty((n,), tack.i32)
    starts = tack.empty((n,), odt)
    if n:
        _fan_counts(loop_offsets, counts)
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.empty((3 * count,), odt)
    sources = tack.empty((count,), data.id_dtype)
    if count:
        # The corners left of each polygon as it is cut, linked within its range.
        after = tack.empty((size_of(loop_points),), tack.i32)
        before = tack.empty((size_of(loop_points),), tack.i32)
        _ears(loop_offsets, loop_points, materialize(data.geometry.values), starts, after,
              before, connectivity, sources)
    return _pieces_topology(data, connectivity, sources, count, shapes.TRIANGLE, 3, fields)


@tack.kernel
def _polygon_sizes(types, offsets, sizes):
    for c in range(sizes.shape[0]):
        sizes[c] = offsets[c + 1] - offsets[c] if types[c] == shapes.POLYGON else 0


@tack.kernel
def _polygon_loops(types, offsets, connectivity, loop_offsets, loop_points, n_cells):
    for c in range(n_cells):
        if types[c] == shapes.POLYGON:
            for k in range(offsets[c + 1] - offsets[c]):
                loop_points[loop_offsets[c] + k] = connectivity[offsets[c] + k]


def _mixed_triangles(data, fields):
    """Triangles of shapes beside polygons: each shape split by its table, as
    ``triangulate`` splits a shape-based topology's, each polygon by ear cutting;
    polyhedra and other 3D cells left out."""
    from tack.dataset.algorithms._common import _table
    from tack.dataset.algorithms._refine import _PIECES, _piece_counts, _split_cells

    t = data.topology
    n = data.num_cells
    tables = _PIECES["triangles"]
    groups = [g for g in t.groups() if g.count and g.shape.REFERENCE
              and int(g.shape.ID) in tables]
    counts = tack.zeros((n,), tack.i32)
    for group in groups:
        _piece_counts(group.view(), len(tables[int(group.shape.ID)]), counts)
    # Each polygon's loop, the other cells' empty, so ear cutting skips them.
    sizes = tack.empty((n,), data.id_dtype)
    loop_offsets = tack.empty((n + 1,), data.id_dtype)
    loop_count = 0
    if n:
        _polygon_sizes(t.types, t.offsets, sizes)
        loop_count = exclusive_scan(sizes, loop_offsets, n, close=True)
    loop_points = tack.empty((loop_count,), data.id_dtype)
    fans = tack.empty((n,), tack.i32)
    if n:
        _polygon_loops(t.types, t.offsets, t.connectivity, loop_offsets, loop_points, n)
        _fan_counts(loop_offsets, fans)
        _add(fans, counts)
    odt = ids.for_output(data, 3 * (2 * n + loop_count))
    starts = tack.empty((n,), odt)
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.empty((3 * count,), odt)
    sources = tack.empty((count,), data.id_dtype)
    for group in groups:
        table = np.asarray(tables[int(group.shape.ID)], np.int32)
        _split_cells(group.view(), _table(("pieces", "triangles", int(group.shape.ID)), table),
                     len(table), 3, starts, connectivity, sources)
    if loop_count:
        after = tack.empty((loop_count,), tack.i32)
        before = tack.empty((loop_count,), tack.i32)
        _ears(loop_offsets, loop_points, materialize(data.geometry.values), starts, after,
              before, connectivity, sources)
    return _pieces_topology(data, connectivity, sources, count, shapes.TRIANGLE, 3, fields)


@tack.kernel
def _add(more, counts):
    for c in range(counts.shape[0]):
        counts[c] += more[c]
