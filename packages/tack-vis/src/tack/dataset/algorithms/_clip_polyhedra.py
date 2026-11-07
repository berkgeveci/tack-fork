# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``clip`` of a polyhedral topology: each cut cell becomes the polyhedron of its kept
part -- its faces clipped, plus its iso-polygons (López's, as ``contour`` traces
them) as caps -- so faces two cells share stay one face, and the output is a
conforming ``PolyhedralTopology``."""

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.arrays import size_of
from tack.dataset import ids
from tack.dataset.algorithms._common import _close, _crossing, _first_end
from tack.dataset.buckets import key_order, run_offsets
from tack.dataset.carry import Interpolate, Take, carry


@tack.kernel
def _face_sizes(face_offsets, face_points, values, iso, sizes, kept):
    # Each face's clipped polygon: its kept points, and a crossing on each edge
    # strictly between the sides (a point at the value is kept, so an edge from
    # it has no other crossing). Below is not at or above, so a NaN is below,
    # as contour's tracing and the shape path's cases have it. Under three
    # points, the face is gone.
    for f in range(sizes.shape[0]):
        start = face_offsets[f]
        n = face_offsets[f + 1] - start
        k = 0
        for j in range(n):
            va = values[face_points[start + j]]
            vb = values[face_points[start + (j + 1 if j + 1 < n else 0)]]
            if va >= iso:
                k += 1
            if (va > iso and not vb >= iso) or (not va >= iso and vb > iso):
                k += 1
        sizes[f] = k if k >= 3 else 0
        kept[f] = 1 if k >= 3 else 0


@tack.kernel
def _face_keys(face_offsets, face_points, values, iso, sizes, starts, keys):
    # The clipped polygon's points, named as contour names crossings: (a, a) for a
    # kept point, (low, high) for an edge's crossing.
    for f in range(sizes.shape[0]):
        if sizes[f] > 0:
            start = face_offsets[f]
            n = face_offsets[f + 1] - start
            at = starts[f]
            for j in range(n):
                a = face_points[start + j]
                b = face_points[start + (j + 1 if j + 1 < n else 0)]
                va = values[a]
                vb = values[b]
                if va >= iso:
                    keys[at] = [a, a]
                    at += 1
                if (va > iso and not vb >= iso) or (not va >= iso and vb > iso):
                    keys[at] = [min(a, b), max(a, b)]
                    at += 1


@tack.kernel
def _cap_ranges(poly_cells, polygons, first, last):
    # Polygons come in cell order: each cell's run of them.
    for p in range(polygons):
        c = poly_cells[p]
        if p == 0 or poly_cells[p - 1] != c:
            first[c] = p
        if p == polygons - 1 or poly_cells[p + 1] != c:
            last[c] = p + 1


@tack.kernel
def _cell_counts(cell_offsets, cell_faces, face_offsets, face_points, values, iso, sizes,
                 first, last, counts, kept):
    # A cell is kept when one of its points is strictly on the kept side (a cell
    # only touching the value has no volume there); its faces are its clipped
    # faces left and its caps.
    for c in range(counts.shape[0]):
        inside = 0
        faces = 0
        for i in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[i]
            if sizes[f] > 0:
                faces += 1
            for j in range(face_offsets[f], face_offsets[f + 1]):
                if values[face_points[j]] > iso:
                    inside = 1
        counts[c] = faces + last[c] - first[c] if inside == 1 else 0
        kept[c] = inside


@tack.kernel
def _point_runs(keys, order, offsets, values, positions, iso, ends, weights, point_of, count):
    # One output point per run of equal names: a kept point (weight 0) or an
    # edge's crossing, interpolated from its end first by position as contour's.
    for r in range(count):
        key = keys[order[offsets[r]]]
        a = key[0]
        b = key[1]
        if a != b and not _first_end(positions[a], positions[b], a, b):
            a, b = b, a
        ends[r] = [a, b]
        weights[r] = 0.0 if a == b else _crossing(iso, values[a], values[b])
        for i in range(offsets[r], offsets[r + 1]):
            point_of[order[i]] = r


@tack.kernel
def _side_zero_kept(cell_offsets, cell_faces, cell_face_sides, kept, out, n_cells):
    # Whether each face's side-0 cell (the one it is wound out of) is kept.
    for c in range(n_cells):
        for i in range(cell_offsets[c], cell_offsets[c + 1]):
            if cell_face_sides[i] == 0:
                out[cell_faces[i]] = kept[c]


@tack.kernel
def _faces_out(sizes, kept, starts, slots, point_of, side_zero, out_offsets, out_points,
               n_faces):
    # The clipped faces left, in face order, their points renumbered; a face whose
    # side-0 cell is gone is walked backwards, out of its other cell, which now
    # has it on side 0.
    for f in range(n_faces):
        if kept[f] == 1:
            s = slots[f]
            out_offsets[s] = starts[f]
            n = sizes[f]
            for j in range(n):
                k = j if side_zero[f] == 1 else n - 1 - j
                out_points[starts[f] + j] = point_of[starts[f] + k]


@tack.kernel
def _caps_out(poly_starts, vertices, point_of, face_keys, first_face, first_point,
              out_offsets, out_points, polygons):
    # Each iso-polygon as a cap face after the clipped faces, wound as López traced
    # it: toward the lower values, out of the kept part.
    for p in range(polygons):
        out_offsets[first_face + p] = first_point + poly_starts[p]
        for j in range(poly_starts[p], poly_starts[p + 1]):
            out_points[first_point + j] = point_of[face_keys + vertices[j]]


@tack.kernel
def _cells_out(cell_offsets, cell_faces, cell_face_sides, sizes, face_slots, side_zero, counts,
               kept, cell_slots, starts, first, last, first_cap, out_faces, out_sides, sources,
               n_cells):
    for c in range(n_cells):
        if kept[c] == 1:
            s = cell_slots[c]
            sources[s] = c
            at = starts[c]
            for i in range(cell_offsets[c], cell_offsets[c + 1]):
                f = cell_faces[i]
                if sizes[f] > 0:
                    out_faces[at] = face_slots[f]
                    out_sides[at] = cell_face_sides[i] if side_zero[f] == 1 else 0
                    at += 1
            for p in range(first[c], last[c]):
                out_faces[at] = first_cap + p
                out_sides[at] = 0
                at += 1


@tack.kernel
def _kept_offsets(counts, kept, starts, slots, out, n_cells):
    for c in range(n_cells):
        if kept[c] == 1:
            out[slots[c]] = starts[c]


def clip_polyhedra(data, values, iso, inside, fields):
    """``clip`` of ``data``'s polyhedral topology by point ``values`` (a materialized
    scalar array) at ``iso``: the part at or above it, or with ``inside`` at or
    below it."""
    topology, ends, weights, sources = clip_parts(data, values, iso, inside)
    return carry(data, topology, points=Interpolate(ends, weights), cells=Take(sources),
                 fields=fields)


def clip_parts(data, values, iso, inside):
    """`clip_polyhedra`'s output before its fields are carried: ``(topology, ends,
    weights, sources)``, its ``PolyhedralTopology``, each point's input ends and
    weight (``Interpolate``'s; a kept point's ends are itself twice, weight 0) and
    each cell's input cell."""
    from tack.dataset.algorithms.contour import _lopez_polygons
    from tack.dataset.polyhedra import PolyhedralTopology

    t = data.topology
    if inside:
        negated = tack.empty_like(values)
        if arrays.size_of(values):
            _negate(values, negated, float("inf"))
        values, iso = negated, -iso
    idt = data.id_dtype
    weight_dtype = arrays.dtype_of(values)
    levels = tack.array([iso], weight_dtype)
    (poly_starts, vertices, poly_cells, _, polygons, cap_ends, _, _,
     cap_count) = _lopez_polygons(data, values, levels, 1, weight_dtype, snap=False)

    n_faces, n_cells = size_of(t.face_offsets) - 1, data.num_cells
    sizes = tack.empty((n_faces,), tack.i32)
    face_kept = tack.empty((n_faces,), tack.i32)
    if n_faces:
        _face_sizes(t.face_offsets, t.face_points, values, iso, sizes, face_kept)
    odt = ids.for_output(data, 2 * size_of(t.face_points) + cap_count + n_faces)
    starts = tack.empty((n_faces,), odt)
    face_keys = exclusive_scan(sizes, starts, n_faces) if n_faces else 0
    face_slots = tack.empty((n_faces,), odt)
    kept_faces = exclusive_scan(face_kept, face_slots, n_faces) if n_faces else 0

    # Every point's name: the clipped faces' points, then the caps' iso-vertices.
    keys = tack.empty((face_keys + cap_count,), tack.vec(2, idt))
    if face_keys:
        _face_keys(t.face_offsets, t.face_points, values, iso, sizes, starts, keys)
    if cap_count:
        _cap_names(cap_ends, keys, face_keys, cap_count)
    total = face_keys + cap_count
    point_of = tack.empty((total,), odt)
    count = 0
    if total:
        order = key_order(keys, data.num_points, odt)
        runs, count = run_offsets(keys, order)
    ends = tack.empty((count,), tack.vec(2, idt))
    weights = tack.empty((count,), weight_dtype)
    if count:
        _point_runs(keys, order, runs, values, data.geometry.values, iso,
                    ends, weights, point_of, count)

    # Cells: each kept one's faces left and caps.
    first = tack.zeros((n_cells,), odt)
    last = tack.zeros((n_cells,), odt)
    if polygons:
        _cap_ranges(poly_cells, polygons, first, last)
    counts = tack.empty((n_cells,), tack.i32)
    kept = tack.empty((n_cells,), tack.i32)
    if n_cells:
        _cell_counts(t.cell_offsets, t.cell_faces, t.face_offsets, t.face_points, values, iso,
                     sizes, first, last, counts, kept)
    side_zero = tack.zeros((n_faces,), tack.i32)
    if n_cells:
        _side_zero_kept(t.cell_offsets, t.cell_faces, t.cell_face_sides, kept, side_zero,
                        n_cells)

    # Faces: the clipped ones left, then the caps.
    cap_points = int(poly_starts.numpy()[-1]) if polygons else 0
    out_offsets = tack.empty((kept_faces + polygons + 1,), odt)
    out_points = tack.empty((face_keys + cap_points,), odt)
    if n_faces:
        _faces_out(sizes, face_kept, starts, face_slots, point_of, side_zero, out_offsets,
                   out_points, n_faces)
    if polygons:
        _caps_out(poly_starts, vertices, point_of, face_keys, kept_faces, face_keys,
                  out_offsets, out_points, polygons)
    _close(out_offsets, kept_faces + polygons, face_keys + cap_points)

    cell_starts = tack.empty((n_cells,), odt)
    entries = exclusive_scan(counts, cell_starts, n_cells) if n_cells else 0
    cell_slots = tack.empty((n_cells,), odt)
    kept_cells = exclusive_scan(kept, cell_slots, n_cells) if n_cells else 0
    out_cell_offsets = tack.empty((kept_cells + 1,), odt)
    out_faces = tack.empty((entries,), odt)
    out_sides = tack.empty((entries,), tack.u8)
    sources = tack.empty((kept_cells,), idt)
    if n_cells:
        _kept_offsets(counts, kept, cell_starts, cell_slots, out_cell_offsets, n_cells)
        _cells_out(t.cell_offsets, t.cell_faces, t.cell_face_sides, sizes, face_slots,
                   side_zero, counts, kept, cell_slots, cell_starts, first, last, kept_faces,
                   out_faces, out_sides, sources, n_cells)
    _close(out_cell_offsets, kept_cells, entries)
    topology = PolyhedralTopology(out_offsets, out_points, out_cell_offsets, out_faces,
                                  out_sides, num_points=count, id_dtype=odt)
    return topology, ends, weights, sources


@tack.kernel
def _negate(values, out, infinity):
    # A NaN is below every level, so after negating it must be above them all:
    # the inside part keeps it, as the shape path's inside clip does.
    for i in range(out.shape[0]):
        v = values[i]
        out[i] = -v if v == v else infinity


@tack.kernel
def _cap_names(ends, keys, offset, n):
    # A cap's iso-vertex named as a face's crossing is, (low id, high id): contour
    # gives its ends first by position, which only interpolation reads.
    for i in range(n):
        e = ends[i]
        keys[offset + i] = [min(e[0], e[1]), max(e[0], e[1])]


# ── Polyhedra beside shapes ─────────────────────────────────────────

@tack.kernel
def _whole_or_cut(cells, values, iso, whole, cut):
    # A shape is whole when every point is kept (at or above) and one is strictly
    # above (a cell flat on the value has no volume there); cut when one is above
    # and one below; else gone.
    for c in cells:
        above = 0
        below = 0
        for j in range(cells.NUM_POINTS):
            v = values[cells.point_id(c, j)]
            if v > iso:
                above = 1
            if not v >= iso:
                below = 1
        e = cells.entity_id(c)
        whole[e] = 1 if above == 1 and below == 0 else 0
        cut[e] = 1 if above == 1 and below == 1 else 0


@tack.kernel
def _every_cell(cells, whole, cut):
    for c in cells:
        e = cells.entity_id(c)
        whole[e] = 0
        cut[e] = 1


@tack.kernel
def _own_keys(ids, keys):
    for i in range(ids.shape[0]):
        keys[i] = [ids[i], ids[i]]


@tack.kernel
def _clipped_keys(ends, ids, first, keys):
    # A clipped point's name in the input's ids: (p, p) kept, (low, high) a crossing.
    for r in range(ends.shape[0]):
        a = ids[ends[r][0]]
        b = ids[ends[r][1]]
        keys[first + r] = [min(a, b), max(a, b)]


@tack.kernel
def _joined_points(keys, order, runs, kept_ids, clipped_ids, ends, weights, first, out_ends,
                   out_weights, point_of, count):
    # One output point per run of equal names, its ends and weight from its first
    # entry: a whole shape's point (itself, weight 0) or a clipped one.
    for r in range(count):
        i = order[runs[r]]
        if i < first:
            p = kept_ids[i]
            out_ends[r] = [p, p]
            out_weights[r] = 0.0
        else:
            e = ends[i - first]
            out_ends[r] = [clipped_ids[e[0]], clipped_ids[e[1]]]
            out_weights[r] = weights[i - first]
        for k in range(runs[r], runs[r + 1]):
            point_of[order[k]] = r


@tack.kernel
def _renumbered(points, point_of, first, out):
    for i in range(out.shape[0]):
        out[i] = point_of[first + points[i]]


@tack.kernel
def _joined_offsets(shape_offsets, poly_offsets, shape_entries, n_shapes, out):
    # The shapes' offsets, then the polyhedra's after the shapes' entries.
    for i in range(out.shape[0]):
        out[i] = (shape_offsets[i] if i <= n_shapes
                  else shape_entries + poly_offsets[i - n_shapes])


@tack.kernel
def _joined_face_offsets(poly_offsets, n_shapes, out):
    # Per cell: a shape's range empty, then the polyhedra's.
    for i in range(out.shape[0]):
        out[i] = 0 if i <= n_shapes else poly_offsets[i - n_shapes]


def clip_mixed(data, values, iso, inside, fields):
    """``clip`` of a topology with polyhedra beside its shapes. A shape whose points
    are all kept stays as it is; every other kept cell -- each polyhedron and each
    cut shape -- becomes the polyhedron of its kept part, as a polyhedral
    topology's cells do (`clip_parts`). A cut shape cannot keep a shape's pieces
    from the tables: beside a polyhedron, its pieces would split a clipped face
    the polyhedron holds whole. The two parts share the points named alike (an
    input point, or the crossing on an edge), and the faces between them are
    matched by ``faces()``. The output is a mixed ``UnstructuredTopology``, the
    whole shapes first."""
    from tack.dataset.algorithms._extract import _keep_cells
    from tack.dataset.algorithms.as_polyhedra import as_polyhedra
    from tack.dataset.dataset import DataSet, Field
    from tack.dataset.shapes import POLYHEDRON
    from tack.dataset.spaces import Values
    from tack.dataset.topology import UnstructuredTopology

    if inside:                          # the part below, as the part above of -values
        negated = tack.empty_like(values)
        if size_of(values):
            _negate(values, negated, float("inf"))
        values, iso = negated, -iso
    t = data.topology
    n, idt = data.num_cells, data.id_dtype
    whole = tack.empty((n,), tack.i32)
    cut = tack.empty((n,), tack.i32)
    for group in t.groups():
        if group.count:
            if group.shape.REFERENCE:
                _whole_or_cut(group.view(), values, iso, whole, cut)
            else:
                _every_cell(group.view(), whole, cut)
    # The input's point and cell ids ride along as fields, through both parts.
    marked = DataSet(t, data.geometry, fields={
        "point": Field(Values(data, "points"), tack.arange(data.num_points, dtype=idt)),
        "cell": Field(Values(data, "cells"), tack.arange(n, dtype=idt)),
        "value": Field(Values(data, "points"), values)}, **data._frame())
    shaped = _keep_cells(marked, whole)
    kept_ids = arrays.materialize(shaped.fields["point"].values)
    n_kept, n_shapes = shaped.num_points, shaped.num_cells
    if int(cut.numpy().max(initial=0)):
        part = as_polyhedra(_keep_cells(marked, cut))
        clipped, ends, weights, sources = clip_parts(
            part, arrays.materialize(part.fields["value"].values), iso, False)
        clipped_ids = arrays.materialize(part.fields["point"].values)
        sources = _picked(arrays.materialize(part.fields["cell"].values), sources)
    else:
        clipped, ends = None, tack.empty((0,), tack.vec(2, idt))
        weights = tack.empty((0,), arrays.dtype_of(values))
        clipped_ids = tack.empty((0,), idt)
        sources = tack.empty((0,), idt)
    n_clipped = clipped.num_points if clipped is not None else 0
    n_poly = clipped.num_cells if clipped is not None else 0

    # The points, named in the input's ids and made one per name.
    keys = tack.empty((n_kept + n_clipped,), tack.vec(2, idt))
    if n_kept:
        _own_keys(kept_ids, keys)
    if n_clipped:
        _clipped_keys(ends, clipped_ids, n_kept, keys)
    total = n_kept + n_clipped
    count = 0
    odt = ids.for_output(data, 8 * n_shapes + 2 * total
                         + (size_of(clipped.face_points) if clipped is not None else 0))
    point_of = tack.empty((total,), odt)
    if total:
        order = key_order(keys, data.num_points, odt)
        runs, count = run_offsets(keys, order)
    out_ends = tack.empty((count,), tack.vec(2, idt))
    out_weights = tack.empty((count,), arrays.dtype_of(weights))
    if count:
        _joined_points(keys, order, runs, kept_ids, clipped_ids, ends, weights, n_kept,
                       out_ends, out_weights, point_of, count)

    # The cells: the whole shapes, then the polyhedra, each on the joined points.
    s = shaped.topology
    shape_conn = arrays.materialize(s.connectivity)
    shape_offsets = arrays.materialize(s.offsets)
    shape_entries = size_of(shape_conn)
    pieces = [_renumbered_ids(shape_conn, point_of, 0, odt)]
    types = [arrays.materialize(s.types)] if n_shapes else []
    cell_sources = [arrays.materialize(shaped.fields["cell"].values)]
    polyhedra = None
    poly_offsets = tack.zeros((1,), odt)
    if n_poly:
        point_offsets, poly_points = clipped.cell_points()
        poly_offsets = point_offsets
        pieces.append(_renumbered_ids(poly_points, point_of, n_kept, odt))
        poly_types = tack.empty((n_poly,), tack.u8)
        poly_types.fill(int(POLYHEDRON))
        types.append(poly_types)
        cell_sources.append(sources)
        cell_face_offsets = tack.empty((n_shapes + n_poly + 1,), odt)
        _joined_face_offsets(clipped.cell_offsets, n_shapes, cell_face_offsets)
        polyhedra = (clipped.face_offsets,
                     _renumbered_ids(clipped.face_points, point_of, n_kept, odt),
                     cell_face_offsets, clipped.cell_faces, clipped.cell_face_sides)
    offsets = tack.empty((n_shapes + n_poly + 1,), odt)
    _joined_offsets(shape_offsets, poly_offsets, shape_entries, n_shapes, offsets)
    connectivity = tack.concat([p for p in pieces if size_of(p)]) \
        if any(size_of(p) for p in pieces) else tack.empty((0,), odt)
    out = UnstructuredTopology(
        tack.concat(types) if types else tack.empty((0,), tack.u8), offsets, connectivity,
        num_points=count, polyhedra=polyhedra)
    return carry(data, out, points=Interpolate(out_ends, out_weights),
                 cells=Take(tack.concat([c for c in cell_sources if size_of(c)])
                            if n_shapes + n_poly else tack.empty((0,), idt)),
                 fields=fields)


@tack.kernel
def _pick(values, at, out):
    for i in range(out.shape[0]):
        out[i] = values[at[i]]


def _picked(values, at):
    out = tack.empty((size_of(at),), arrays.dtype_of(values))
    if size_of(at):
        _pick(values, at, out)
    return out


def _renumbered_ids(points, point_of, first, dtype):
    out = tack.empty((size_of(points),), dtype)
    if size_of(points):
        _renumbered(points, point_of, first, out)
    return out
