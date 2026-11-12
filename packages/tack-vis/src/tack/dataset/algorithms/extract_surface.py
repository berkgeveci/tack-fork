# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``extract_surface``: the faces of set ``name`` as a surface dataset: triangles and quads on the points they use, each in its side 0's outward order. Fields on faces become the surface's cell fields, and so do cell fields (``Constant``, values on cells), each face taking its side 0 cell's value; point fields (``H1``, values on points) come with the points, and ``L2`` fields as each face's trace. A solid's order-2 field or geometry stays order 2 on its surface, gathered by entity: the faces' points, edges and quad centers are the surface's. The geometry must be ``H1``: an ``L2`` one has no shared points for the faces to stand on."""

import itertools

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.arrays import CountingArray
from tack.dataset import ids as id_types
from tack.dataset import shapes
from tack.dataset.algorithms._common import _close
from tack.dataset.carry import Pieces, Same, Take, _like, _take, carry, selected
from tack.dataset.dataset import Field
from tack.dataset.sets import Set
from tack.dataset.spaces import H1, L2
from tack.dataset.topology import UnstructuredTopology, _face_row, _same_key
from tack.runtime.dispatch import on_inputs_device

__all__ = ['extract_surface']


@tack.kernel
def _surface_sizes(ids, kinds, face_offsets, sizes):
    # A polygon face (a mixed topology's polyhedra's, of five points or more) has
    # its points in the topology's face arrays, by its own id.
    for i in range(ids.shape[0]):
        f = ids[i]
        kind = kinds[f]
        sizes[i] = (2 if kind == 3 else (3 if kind == 5 else (
            4 if kind != shapes.POLYGON else face_offsets[f + 1] - face_offsets[f])))


@tack.kernel
def _surface_cells(ids, kinds, rows, face_offsets, face_points, starts, types, offsets,
                   connectivity, length, n):
    for i in range(n):
        f = ids[i]
        at = starts[i]
        types[i] = tack.u8(kinds[f])
        offsets[i] = at
        if kinds[f] == shapes.POLYGON:
            for j in range(face_offsets[f + 1] - face_offsets[f]):
                connectivity[at + j] = face_points[face_offsets[f] + j]
        else:
            row = rows[f]
            for j in range(2 if kinds[f] == 3 else (3 if kinds[f] == 5 else 4)):
                connectivity[at + j] = row[j]
        if i == 0:
            offsets[n] = length


@tack.func
def _open_side(cells, c, k, sides, marks):
    """Whether cell ``c``'s local face ``k`` bounds the cells left -- the cell is not
    hidden or refined (asked by the caller) and the face's other side is none,
    or is."""
    s = sides[cells.face_id(c, k)]
    other = s[2] if cells.face_side(c, k) == 0 else s[0]
    return other < 0 or (tack.i32(marks[other]) & 40) != 0


@tack.kernel
def _open_face_counts(cells, sides, marks, faces, points):
    # Each cell's faces on the boundary of the cells left, and their points.
    for c in cells:
        e = cells.entity_id(c)
        n = 0
        m = 0
        if (tack.i32(marks[e]) & 40) == 0:
            for k in range(cells.NUM_FACES):
                if _open_side(cells, c, k, sides, marks):
                    n += 1
                    m += cells.side_num_points(c, k)
        faces[e] = n
        points[e] = m


@tack.kernel
def _open_faces(cells, sides, marks, kinds, face_starts, point_starts, types, offsets,
                connectivity, owners, ids):
    # Those faces, each going round outward for its cell -- the cell's own local
    # face, as the faces of the cells left once extracted would have it.
    for c in cells:
        e = cells.entity_id(c)
        if (tack.i32(marks[e]) & 40) == 0:
            at = face_starts[e]
            p = point_starts[e]
            for k in range(cells.NUM_FACES):
                if _open_side(cells, c, k, sides, marks):
                    f = cells.face_id(c, k)
                    n = cells.side_num_points(c, k)
                    types[at] = tack.u8(kinds[f])
                    offsets[at] = p
                    for j in range(n):
                        connectivity[p + j] = cells.side_point_id(c, k, j)
                    owners[at] = e
                    ids[at] = f
                    at += 1
                    p += n


def _masked_surface(data, marks, fields):
    """``extract_surface``'s default on a 3D mesh of shape cells with hidden or
    refined cells: the faces bounding the cells left, from the topology's own
    faces (kept by it, where extracting the rest derived another mesh's every
    call), each going round outward for its cell -- the extracted cells'
    surface, on the points it uses, numbered in the same order."""
    faces = data.topology.faces()
    n = data.num_cells
    idt = data.id_dtype
    face_counts = tack.zeros((n,), tack.i32)
    point_counts = tack.zeros((n,), tack.i32)
    groups = [g for g in data.launch_groups("cells", []) if g.count]
    for group in groups:
        _open_face_counts(data.domain_view("cells", group, incidence="faces"), faces.sides, marks, face_counts,
                          point_counts)
    odt = id_types.for_output(data, 24 * n)
    face_starts = tack.empty((n,), odt)
    point_starts = tack.empty((n,), odt)
    count = exclusive_scan(face_counts, face_starts, n) if n else 0
    length = exclusive_scan(point_counts, point_starts, n) if n else 0
    types = tack.empty((count,), tack.u8)
    offsets = tack.empty((count + 1,), odt)
    connectivity = tack.empty((length,), odt)
    owners = tack.empty((count,), idt)
    chosen = tack.empty((count,), idt)
    if count:
        for group in groups:
            _open_faces(data.domain_view("cells", group, incidence="faces"), faces.sides, marks, faces.kinds,
                        face_starts, point_starts, types, offsets, connectivity, owners,
                        chosen)
    _close(offsets, count, length)
    used, points = _used_points(connectivity, data.num_points, idt)
    surface = UnstructuredTopology(types, offsets, connectivity, num_points=used)
    out = carry(data, surface, points=Take(points), cells=Pieces(owners),
                faces_to_cells=Take(chosen), fields=fields)
    return _keep_traces(data, out, owners, points, fields)


@tack.kernel
def _face_cells(ids, sides, out):
    for i in range(ids.shape[0]):
        out[i] = sides[ids[i]][0]


@tack.func
def _alone(records, i, begin, end):
    # 1 for a record whose key no neighbour in its bucket shares: a face of one
    # cell. Records of another bucket have another lowest point, so another key.
    alone = 1
    if i > begin and _same_key(records[i], records[i - 1]) == 1:
        alone = 0
    if i + 1 < end and _same_key(records[i], records[i + 1]) == 1:
        alone = 0
    return alone


@tack.kernel
def _bucket_boundary_counts(starts, records, counts, lo, hi, base):
    # Each bucket's faces of one cell, for the buckets [lo, hi) whose records
    # start at base, into counts from lo.
    for b in range(lo, hi):
        k = 0
        begin = starts[b] - base
        end = starts[b + 1] - base
        for i in range(begin, end):
            k += _alone(records, i, begin, end)
        counts[b - lo] = k


@tack.kernel
def _boundary_slots(starts, records, first, slot_entry, sizes, lo, hi, base):
    # Each boundary face's place in the surface (the records' order: by bucket,
    # then by key): its (cell, face) entry, and its size from its key's pads.
    for b in range(lo, hi):
        slot = first[b - lo]
        begin = starts[b] - base
        end = starts[b + 1] - base
        for i in range(begin, end):
            if _alone(records, i, begin, end) == 1:
                r = records[i]
                slot_entry[slot] = r[3]
                sizes[slot] = 2 if r[1] == -1 else (3 if r[2] == -1 else 4)
                slot += 1


@tack.kernel
def _boundary_rows(cells, start, slot_entry, starts, types, connectivity, sources):
    # Each boundary face of this group's cells from its one cell, in the cell's
    # walk (its side 0's): the entry is the cell's index times its faces, plus
    # the face, from the group's start.
    for s in range(slot_entry.shape[0]):
        e = slot_entry[s] - start
        if e >= 0 and e < cells.num_cells * cells.NUM_FACES:
            c = e // cells.NUM_FACES
            f = e % cells.NUM_FACES
            row = _face_row(cells, c, f)
            n = cells.face_num_points(f)
            at = starts[s]
            for j in range(n):
                connectivity[at + j] = row[j]
            types[s] = tack.u8(3 if n == 2 else (5 if n == 3 else 9))
            sources[s] = cells.entity_id(c)


#: The most bytes of face records the fresh boundary holds at once: 20M
#: hexahedra in one pass. Larger meshes ran faster in more, smaller passes
#: (100M hexahedra on a 64 GB M1 Max with other work: one pass 3.9 s, 4 GB
#: passes 2.8 s, 2 GB passes 1.4 s, VTK 1.3), each of a fresh allocation.
_RECORDS_BYTES = 2 << 30


def _fresh_boundary(data, fields):
    """``extract_surface``'s default on a mesh whose faces have not been derived:
    the same surface, from a counting sort of the (cell, face) records keeping
    the faces of one cell, without the whole face incidence (rows, sides and
    orientations of every face) nobody asked for -- VTK's external faces
    hashes faces the same way. ``None`` where the full path is needed: faces
    derived already (then they are free), ghost marks (placed through the
    faces), fields or a geometry above order 1 (whose surface nodes come
    through the faces), or buckets too crowded to sort a thread each
    (``topology._face_buckets_affordable``). The topology keeps what it found
    (``_boundary``), so a repeated call costs only the carry, as it did when
    the first call derived the faces. The surface is on the points it uses
    (``_used_points``), renumbered once."""
    from tack.dataset.topology import (
        _face_bucket_counts,
        _face_buckets_affordable,
        _place_faces,
        _sort_face_buckets,
    )

    topology = data.topology
    if (not isinstance(topology, UnstructuredTopology) or topology._faces is not None
            or topology.dimension != 3 or data.ghosts is not None or topology.mixed
            or any(isinstance(f.space, H1) and f.space.order != 1
                   for f in (data.geometry, *data.fields.values()))):
        return None
    if topology._boundary is not None:
        return _on_boundary(data, fields)
    groups = [g for g in topology.groups() if g.shape.NUM_FACES and g.shape.DIMENSION == 3]
    starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_FACES
                                             for g in groups])]).astype(int)
    total, np_, idt = int(starts[-1]), topology.num_points, topology.id_dtype
    if not total:
        return None
    counts = tack.zeros((np_,), tack.i32)
    for group in groups:
        if group.count:
            _face_bucket_counts(group.view(), counts)
    if not _face_buckets_affordable(counts, np_, total):
        return None
    bucket_starts = tack.empty((np_ + 1,), idt)
    exclusive_scan(counts, bucket_starts, np_, total=False, close=True)
    # The records of at most _RECORDS_BYTES at a time: the points in ranges,
    # each range's faces placed, sorted and numbered on from the last range's.
    # The surface's order is the buckets', so it is the same in any number of
    # passes. At 100M hexahedra the records were 9.6 GB at once.
    itemsize = 4 * np.dtype(idt.numpy_dtype).itemsize
    passes = max(1, -(-total * itemsize // _RECORDS_BYTES))
    bounds = [np_ * k // passes for k in range(passes + 1)]
    entries, sizes = [], []
    for lo, hi in itertools.pairwise(bounds):
        base, end = (int(bucket_starts[b]) for b in (lo, hi))
        counts.fill(0)
        records = tack.empty((end - base,), tack.vec(4, idt))
        for group, start in zip(groups, starts[:-1]):
            if group.count:
                _place_faces(group.view(), int(start), bucket_starts, counts, records, lo, hi,
                             base)
        _sort_face_buckets(bucket_starts, records, lo, hi, base)
        # Each bucket's boundary faces counted, then numbered in their order: two
        # walks of the buckets in place of per-record flags and slots.
        first = (counts if idt == tack.i32 and passes == 1
                 else tack.empty((hi - lo,), idt))
        _bucket_boundary_counts(bucket_starts, records, first, lo, hi, base)
        m = exclusive_scan(first, first, hi - lo) if hi > lo else 0
        entries.append(tack.empty((m,), idt))
        sizes.append(tack.empty((m,), tack.i32))
        if m:
            _boundary_slots(bucket_starts, records, first, entries[-1], sizes[-1], lo, hi,
                            base)
        del records, first
    del counts, bucket_starts
    slot_entry, sizes = (parts[0] if len(parts) == 1 else tack.concat(parts)
                         for parts in (entries, sizes))
    n = int(slot_entry.shape[0])
    odt = id_types.for_output(data, 4 * n)
    offsets = tack.empty((n + 1,), odt)
    length = exclusive_scan(sizes, offsets, n, close=True)
    types = tack.empty((n,), tack.u8)
    connectivity = tack.empty((length,), odt)
    cells = tack.empty((n,), idt)
    if n:
        for group, start in zip(groups, starts[:-1]):
            if group.count:
                _boundary_rows(group.view(), int(start), slot_entry, offsets, types,
                               connectivity, cells)
    count, points = _used_points(connectivity, data.num_points, idt)
    surface = UnstructuredTopology(types, offsets, connectivity, num_points=count)
    topology._boundary = (surface, cells, points, None)
    return _on_boundary(data, fields)


def _on_boundary(data, fields):
    """The boundary ``data.topology._boundary`` holds, with ``data``'s fields carried
    to it. The surface's geometry is kept there too, with the geometry and the
    positions' revision it was taken from, so a repeated call on the same mesh
    gives the same surface geometry, and what is kept with that (``mesh_kept``:
    its normals) is computed once."""
    topology = data.topology
    surface, cells, points, kept = topology._boundary
    source = (data.geometry, arrays._revision(data.geometry.values))
    geometry = None
    if kept is not None and kept[0][0] is source[0] and kept[0][1] == source[1]:
        geometry = kept[1]
    out = carry(data, surface, points=Take(points), cells=Pieces(cells), geometry=geometry,
                fields=fields)
    topology._boundary = (surface, cells, points, (source, out.geometry))
    return _keep_traces(data, out, cells, points, fields)


@on_inputs_device
def extract_surface(data, faces=None, fields=None):
    """The ``faces`` -- a face ``Set`` (as ``boundary_faces`` returns), a set's name
    in ``data.sets``, or face ids (a field or an array); by default the boundary
    (``boundary_faces``), VTK's external faces -- as a dataset on the points they use: a 3D
    mesh's faces as triangles, quads or polygons, each in its side 0's outward
    order; a 2D mesh's edges as line cells; a polyline mesh's ends as vertex
    cells. By default a 2D mesh is its own surface: its cells pass through (its
    hidden ones left out), as VTK's geometry filter passes polygons; its
    boundary edges are ``extract_surface(data, boundary_faces(data))``. Fields on faces become the surface's
    cell fields, and so do cell fields (``Constant``, values on cells), each face
    taking its side 0 cell's value; point fields (``H1``, values on points) come with
    the points, which are the ones the faces use, renumbered in order, as VTK's
    geometry filter keeps them (so a surface's point statistics are the
    surface's). A 3D mesh's order-2 ``H1`` fields and geometry stay order 2
    on the surface, the trace of the solid's: each face's corners, edge middles
    and (a quad's) center are the surface cell's, so their values are gathered,
    not evaluated. A 2D mesh's boundary lines take an order-2 field's values at
    the points. ``L2`` (DG) fields of one order stay in the same space on the
    surface, each face holding its side 0 cell's trace (the cell's field at the
    face's nodes, through the cell's parametric coordinates, exact for every
    basis); an order-0 one takes its cell's value, as cell fields do, and so
    does every ``L2`` field on polyhedra. The geometry must be ``H1``: an
    ``L2`` one has no shared points for the faces to stand on.

    Given a ``PartitionedDataSet``, the whole's surface, a piece each: each
    piece's boundary, less the faces two pieces share. Pieces without ghost
    cells take the faces between them for boundary; the faces of every piece's
    surface are matched over the pieces by their points (by position, as
    ``generate_global_ids`` matches them), and a face held unmarked by two
    pieces is between them and dropped. On pieces with ghost cells no such face
    is on a boundary, and each piece's surface is its own, its ghost faces
    marked. The surface's points are owned once (``generate_global_ids`` marks
    them where the pieces had no ghost cells)."""
    from tack.dataset.ghosts import refuse_hidden, visible
    from tack.dataset.partitioned import PartitionedDataSet

    if isinstance(data, PartitionedDataSet):
        if faces is not None:
            raise TypeError("extract_surface of pieces takes their boundary; choose faces "
                            "of each piece instead")
        return _surface_of_pieces(data, fields)
    from tack.dataset.algorithms._refine import _masks

    marks, hidden = _masks(data)
    if (faces is None and hidden and isinstance(data.geometry.space, H1)
            and data.topology.dimension == 3 and getattr(data.topology, "reference_cells", True)
            and not _has_order_two(data, fields)):
        # Hidden and refined cells left out in place, the topology's faces kept.
        return _masked_surface(data, marks, fields)
    if faces is None:
        data = visible(data)
    else:
        refuse_hidden(data, "extract_surface of chosen faces")
    if not isinstance(data.geometry.space, H1):
        raise TypeError("extract_surface keeps the points, so needs an H1 geometry")
    if faces is None and data.topology.dimension == 2:
        # A surface mesh is its own surface: its polygons pass through, as VTK's
        # geometry filter passes them.
        return _on_used_points(carry(data, data.topology, points=Same(), cells=Same(),
                                     fields=fields))
    if faces is None:
        fresh = _fresh_boundary(data, fields)
        if fresh is not None:
            return fresh
        from tack.dataset.algorithms.boundary_faces import boundary_faces
        faces = boundary_faces(data)
    ids = _face_ids(data, faces)
    topology = data.topology
    if not getattr(topology, "reference_cells", True) and not topology.mixed:
        return _polygon_surface(data, ids, fields)
    faces = topology.faces()
    idt = data.id_dtype
    n = ids.shape[0]
    # Polyhedra beside shapes: their faces of five points or more become polygons
    # beside the triangles and quads (a mixed surface).
    face_offsets, face_points = ((topology.face_offsets, topology.face_points)
                                 if topology.mixed else (ids, ids))
    # Up to 4 points a face: past the topology's own bound (12 a cell) for a
    # surface of faces that share no points; a polygon's as many as it has.
    odt = id_types.for_output(data, 4 * n + (arrays.size_of(face_points) if topology.mixed else 0))
    sizes = tack.empty((n,), tack.i32)
    starts = tack.empty((n,), odt)
    if n:
        _surface_sizes(ids, faces.kinds, face_offsets, sizes)
    length = exclusive_scan(sizes, starts, n) if n else 0
    types = tack.empty((n,), tack.u8)
    offsets = tack.zeros((n + 1,), odt)
    connectivity = tack.empty((length,), odt)
    if n:
        _surface_cells(ids, faces.kinds, faces.rows, face_offsets, face_points, starts, types,
                       offsets, connectivity, length, n)
    cells = tack.empty((n,), idt)
    if n:
        _face_cells(ids, faces.sides, cells)
    if data.topology.dimension == 3 and _has_order_two(data, fields):
        # The order-2 values are gathered from the solid's nodes by its point
        # numbering, so the surface is put on its points after.
        surface = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
        out = carry(data, surface, points=Same(), cells=Pieces(cells), faces_to_cells=Take(ids),
                    fields=fields)
        _keep_order_two(data, out, ids, fields)
        return _on_used_points(_keep_traces(data, out, cells, None, fields))
    count, points = _used_points(connectivity, data.num_points, idt)
    surface = UnstructuredTopology(types, offsets, connectivity, num_points=count)
    out = carry(data, surface, points=Take(points), cells=Pieces(cells),
                faces_to_cells=Take(ids), fields=fields)
    return _keep_traces(data, out, cells, points, fields)


@tack.kernel
def _mark_points(points, used):
    for k in range(points.shape[0]):
        tack.atomic_max(used, points[k], 1)


def _used_points(connectivity, num_points, idt):
    """``(count, ids)`` of the points ``connectivity`` uses, in order, renumbering it
    in place to them: a surface is on the points its faces use, as VTK's
    geometry filter makes it. On the input's points, a surface's point
    statistics counted the volume's interior (the AhmedML mesh's 22.8M points
    for a surface of 1.23M), and matching pieces' surfaces matched every
    point."""
    from tack.dataset.algorithms._common import _compact
    from tack.dataset.algorithms._extract import _renumber

    used = tack.zeros((num_points,), tack.i32)
    length = connectivity.shape[0]
    if length:
        _mark_points(connectivity, used)
    new_ids, points, count = _compact(used, idt)
    if length:
        _renumber(connectivity, new_ids)
    return count, points


def _on_used_points(surface):
    """A 2D mesh passed through as it is if its cells use every point, else on the
    points they use (``_keep_cells``, which carries fields, marks and global
    ids). A structured grid uses them all."""
    from tack.dataset.algorithms._extract import _keep_cells

    t = surface.topology
    if isinstance(t, UnstructuredTopology):
        points = t.connectivity
    elif hasattr(t, "loops"):
        _, points = t.loops()
    else:
        return surface
    used = tack.zeros((surface.num_points,), tack.i32)
    if points.shape[0]:
        _mark_points(points, used)
    if int(used.sum()) == surface.num_points:
        return surface
    keep = tack.empty((surface.num_cells,), tack.i32)
    keep.fill(1)
    return _keep_cells(surface, keep)


def _has_order_two(data, fields):
    return _order_two(data.geometry) or any(_order_two(data.fields[name])
                                            for name in selected(data, fields))


# ── Order 2 ─────────────────────────────────────────────────────────

@tack.kernel
def _point_sources(sources, n):
    for p in range(n):
        sources[p] = p


@tack.kernel
def _edge_sources(rows, solid_rows, sources, base, count):
    # A surface edge's middle is the solid's edge of the same (low, high) points.
    # Edges are numbered in increasing (low, high) order, so bisection finds it.
    for e in range(rows.shape[0]):
        key = rows[e]
        lo = 0
        hi = count
        while lo < hi:
            mid = (lo + hi) // 2
            r = solid_rows[mid]
            if r[0] < key[0] or (r[0] == key[0] and r[1] < key[1]):
                lo = mid + 1
            else:
                hi = mid
        sources[base + e] = base + lo


@tack.kernel
def _center_sources(ids, kinds, face_dofs, cell_dofs, sources):
    # A quad face's center is the surface quad's.
    for c in range(ids.shape[0]):
        f = ids[c]
        if kinds[f] == shapes.QUAD:
            sources[cell_dofs[c]] = face_dofs[f]


def _order_two(field):
    return isinstance(field.space, H1) and field.space.order == 2


# ── L2 fields: each face's trace ───────────────────────────────────

@tack.kernel
def _solid_keys(surface, cells, keys):
    for s in surface:
        keys[s] = cells[surface.entity_id(s)]


@tack.func
def _solid_corner(solid, c, point):
    # The parametric point of solid cell c's corner at input point `point`: a
    # face's corners are its cell's.
    pc = solid.parametric_point(0)
    for m in range(1, solid.NUM_POINTS):
        if solid.point_id(c, m) == point:
            pc = solid.parametric_point(m)
    return pc


@tack.kernel
def _face_traces(solid, u, surface, t, points, starts, faces, out):
    # Each surface cell's L2 nodes from the solid cell it is a face of: a node's
    # parametric point on the face, taken to the solid's by the face's corners
    # (the surface shape's corner functions weighing the solid corners'
    # parametric points), where the solid's field is evaluated. The trace of an
    # order-p polynomial on a face is in the face's order-p space, so the
    # surface field is the solid's there.
    for c in solid:
        e = solid.entity_id(c)
        for k in range(starts[e], starts[e + 1]):
            s = faces[k]
            p0 = _solid_corner(solid, c, points[surface.point_id(s, 0)])
            p1 = _solid_corner(solid, c, points[surface.point_id(s, 1)])
            p2 = _solid_corner(solid, c, points[surface.point_id(s, 2)])
            p3 = p2
            if surface.NUM_POINTS == 4:
                p3 = _solid_corner(solid, c, points[surface.point_id(s, 3)])
            for n in range(t.NODES):
                f = t.node_parametric(n)
                pc = (surface.shape_function(0, f) * p0 + surface.shape_function(1, f) * p1
                      + surface.shape_function(2, f) * p2)
                if surface.NUM_POINTS == 4:
                    pc = pc + surface.shape_function(3, f) * p3
                out[t.dof_index(s, n)] = u.value(c, pc)


def _traced(data, field):
    """Whether ``extract_surface`` gives ``field`` its faces' trace: an ``L2``
    field of one order, stored, of floating-point values (an order-0 one of any
    type, taken by cell), with no gradient of its own to carry."""
    space = field.space
    return (isinstance(space, L2) and not space.varies and not space.evaluated
            and field.gradient is None
            and (space.order == 0 or arrays.dtype_of(field.values) in (tack.f32, tack.f64)))


def _keep_traces(data, out, cells, points, fields):
    """``out``, the surface of ``data`` whose cell ``s`` is a face of cell
    ``cells[s]`` and whose point ``p`` is ``data``'s ``points[p]`` (``None``: the
    same), with ``data``'s ``L2`` fields as ``L2`` fields of the same order and
    basis on it, each face holding its cell's trace. ``carry`` leaves them
    behind, since a face is a piece of its cell."""
    from tack.dataset import _bernstein
    from tack.dataset.buckets import bucket_groups

    # Without reference cells (polyhedra, polygons) a face has no parametric
    # point in its cell: only order 0, a value a cell, comes along.
    shaped = all(getattr(t, "reference_cells", True) for t in (data.topology, out.topology))
    names = [name for name in selected(data, fields) if _traced(data, data.fields[name])
             and (shaped or data.fields[name].space.order == 0)]
    if not names or not out.num_cells:
        return out
    if points is None:
        points = CountingArray(out.num_points, dtype=data.id_dtype)
    links = {}
    for name in names:
        field = data.fields[name]
        space = field.space
        if space.order == 0:
            out.fields[name] = Field(L2(out, order=0), _take(field.values, cells), field.unit)
            continue
        target_space = L2(out, order=space.order,
                          **({} if space.basis is None else {"basis": space.basis}))
        nodal_space = (_bernstein.lagrange_companion(target_space)
                       if _bernstein.is_bernstein(target_space) else target_space)
        target = Field(nodal_space, _like(field.values, nodal_space.size))
        for h in out.launch_groups("cells", [target]):
            if not h.count:
                continue
            surface = out.domain_view("cells", h)
            if h.shape not in links:
                # This group's faces by the solid cell they bound.
                keys = tack.empty((h.count,), cells.dtype)
                _solid_keys(surface, cells, keys)
                links[h.shape] = bucket_groups(keys, data.num_cells,
                                               id_types.at_least(data.id_dtype, h.count))
            faces, starts = links[h.shape]
            for g in data.launch_groups("cells", [field]):
                if g.count:
                    _face_traces(data.domain_view("cells", g), field.view(g), surface,
                                 target.view(h), points, starts, faces, target.values)
        traced = Field(nodal_space, target.values, field.unit)
        if nodal_space is not target_space:
            traced = _bernstein.from_lagrange(out, traced, target_space)
        out.fields[name] = Field(target_space, traced.values, field.unit)
    return out


def _keep_order_two(data, out, ids, fields):
    """``out``'s fields and geometry that are order 2 in ``data``, made order 2 on the
    surface ``out`` of ``data``'s faces ``ids``: the surface's nodes are the
    solid's, so each value is the solid's at the same node."""
    names = [name for name in selected(data, fields) if _order_two(data.fields[name])]
    curved = _order_two(data.geometry)
    if not (names or curved):
        return
    solid = H1(data, order=2)
    space = H1(out, order=2)
    solid_face_dofs, _, _ = solid._layout()
    _, cell_dofs, size = space._layout()
    surface_edges, solid_edges = out.topology.edges(), data.topology.edges()
    base = data.num_points
    sources = tack.empty((size,), id_types.at_least(data.id_dtype, solid.size))
    if base:
        _point_sources(sources, base)
    if surface_edges.num_edges:
        _edge_sources(surface_edges.rows, solid_edges.rows, sources, base,
                      solid_edges.num_edges)
    if ids.shape[0]:
        _center_sources(ids, data.topology.faces().kinds, solid_face_dofs, cell_dofs, sources)
    for name in names:
        out.fields[name] = Field(space, _take(data.fields[name].values, sources),
                                 data.fields[name].unit)
    if curved:
        out.geometry = Field(space, _take(data.geometry.values, sources), data.unit)


@tack.kernel
def _surface_loop_sizes(ids, face_offsets, sizes):
    for i in range(ids.shape[0]):
        f = ids[i]
        sizes[i] = face_offsets[f + 1] - face_offsets[f]


@tack.kernel
def _surface_loops(ids, face_offsets, face_points, starts, loops):
    for i in range(ids.shape[0]):
        f = ids[i]
        for j in range(face_offsets[f + 1] - face_offsets[f]):
            loops[starts[i] + j] = face_points[face_offsets[f] + j]


def _face_ids(data, faces):
    """``faces`` -- a face set, its name, or face ids -- as a field of the topology's
    ids: a set of this topology's faces, or ids within its face count."""
    if isinstance(faces, str):
        faces = data.sets[faces]
    if isinstance(faces, Set):
        if faces.on != "faces":
            raise ValueError(f"extract_surface takes faces, not a set of {faces.on}")
        if faces.topology is not data.topology:
            raise ValueError("extract_surface takes a set of this dataset's faces, not "
                             "one of another topology")
        return faces.ids
    ids = id_types.as_ids(faces, data.id_dtype)
    id_types.check_bounds(ids, data.topology.faces().num_faces, "face")
    return ids


@tack.kernel
def _facet_types(offsets, types):
    # A polygonal mesh's facets are lines, a polyline mesh's vertices.
    for i in range(types.shape[0]):
        types[i] = tack.u8(shapes.LINE if offsets[i + 1] - offsets[i] == 2 else shapes.VERTEX)


def _polygon_surface(data, ids, fields=None):
    """``extract_surface`` of a polyhedral topology: the set's faces, each a polygon in
    its side 0's outward order, as a ``PolygonalTopology`` on the points they use; of
    a polygonal or polyline one, its edges or ends as line or vertex cells."""
    from tack.dataset.polyhedra import PolygonalTopology

    topology = data.topology
    faces = topology.faces()
    idt = data.id_dtype
    n = ids.shape[0]
    sizes = tack.empty((n,), tack.i32)
    offsets = tack.empty((n + 1,), idt)
    if n:
        _surface_loop_sizes(ids, topology.face_offsets, sizes)
    total = exclusive_scan(sizes, offsets, n, close=True)
    loops = tack.empty((total,), idt)
    if n:
        _surface_loops(ids, topology.face_offsets, topology.face_points, offsets, loops)
    count, points = _used_points(loops, data.num_points, idt)
    if topology.dimension == 3:
        surface = PolygonalTopology(offsets, loops, num_points=count)
    else:
        types = tack.empty((n,), tack.u8)
        if n:
            _facet_types(offsets, types)
        surface = UnstructuredTopology(types, offsets, loops, num_points=count)
    cells = tack.empty((n,), idt)
    if n:
        _face_cells(ids, faces.sides, cells)
    out = carry(data, surface, points=Take(points), cells=Pieces(cells),
                faces_to_cells=Take(ids), fields=fields)
    return _keep_traces(data, out, cells, points, fields)


def _boundary_of(piece):
    """A piece's boundary faces on the host: ``(ids, ring offsets, ring points, marks)``,
    each face's points in its ring and the ghost marks of the one cell it bounds."""
    import numpy as np

    from tack import arrays

    topology = piece.topology
    faces = topology.faces()
    ids = np.asarray(arrays.to_host(faces.boundary()), np.int64).reshape(-1)
    n = piece.num_cells
    marks = piece.ghosts.cells.numpy() if piece.ghosts is not None and \
        piece.ghosts.cells is not None else np.zeros(n, np.uint8)
    if hasattr(faces, "rows"):
        rows = np.asarray(arrays.to_host(faces.rows), np.int64).reshape(-1, 4)[ids]
        sizes = (rows >= 0).sum(axis=1)
        points = rows[rows >= 0]
        cells = np.asarray(arrays.to_host(faces.sides), np.int64).reshape(-1, 4)[ids, 0]
    else:
        face_offsets, face_points, cell_offsets, cell_faces, sides = (
            np.asarray(a, np.int64) for a in topology.arrays())
        sizes = face_offsets[ids + 1] - face_offsets[ids]
        starts = np.concatenate(([0], np.cumsum(sizes)))
        at = (np.arange(starts[-1], dtype=np.int64) - np.repeat(starts[:-1], sizes)
              + np.repeat(face_offsets[ids], sizes))
        points = face_points[at]
        cell_of = np.full(len(face_offsets) - 1, -1, np.int64)
        first = sides == 0
        cell_of[cell_faces[first]] = np.repeat(np.arange(n), np.diff(cell_offsets))[first]
        cells = cell_of[ids]
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    return ids, offsets, points, marks[np.maximum(cells, 0)]


def _surface_of_pieces(pds, fields):
    """``extract_surface`` over a ``PartitionedDataSet``: each piece's boundary faces,
    matched over the pieces by their points' positions before any surface is made,
    less the faces two pieces hold unmarked (those between pieces without ghost
    cells); then each piece's surface of the rest -- part of the whole's surface,
    so manifold where the whole's is, though a piece's own boundary need not be
    (a sheet of polyhedra one cell thick meets itself along edges) -- and, where
    a surface lost its marks, the rest marked by matching."""
    import numpy as np

    import tack
    from tack import communication as tc
    from tack.dataset import _matching
    from tack.dataset.algorithms.generate_global_ids import generate_global_ids
    from tack.dataset.carry import selected
    from tack.dataset.ghosts import Ghosts, visible
    from tack.dataset.partitioned import PartitionedDataSet, agree
    from tack.dataset.topology import lines_only

    world = pds.world
    pieces = [visible(piece) for piece in pds]

    def check():
        # Refused on every rank, not only where a piece fails, as each piece's
        # surface (or its faces) would refuse it.
        for p in pieces:
            if not isinstance(p.geometry.space, H1):
                raise TypeError("extract_surface keeps the points, so needs an H1 geometry")
            if lines_only(p.topology):
                raise NotImplementedError("extract_surface of pieces matches their faces, "
                                          "and the faces of a mesh of lines are not derived")
            selected(p, fields)
        return all(p.topology.dimension == 2 for p in pieces)

    # Every rank's pieces' say: a rank without pieces must not take the shortcut
    # while the others match faces.
    if all(agree(world, check)):
        # A surface's pieces are their own surfaces, as one surface is.
        return PartitionedDataSet([extract_surface(p, fields=fields) for p in pieces],
                                  world=world)
    blocks = tc.Blocks([p.device for p in pieces], world)
    bounds, entries = {}, {}
    for g, piece in zip(blocks.local, pieces):
        ids, offsets, points, marks = _boundary_of(piece)
        bounds[g] = (ids, marks)
        # A face by its points' positions: count, a hash of them sorted, the least, the most.
        point_keys = _matching.hashed(list(_matching.position_bits(piece.positions())),
                                      seed=2)
        sizes = np.diff(offsets)
        columns = [np.zeros(len(ids), np.uint64) for _ in range(4)]
        for size in np.unique(sizes).tolist():
            which = np.flatnonzero(sizes == size)
            rows = points[offsets[which][:, None] + np.arange(size)]
            keys = np.sort(point_keys[rows], axis=1)
            columns[0][which] = np.uint64(size)
            columns[1][which] = _matching.hashed(list(keys.T), seed=1)
            columns[2][which] = keys[:, 0]
            columns[3][which] = keys[:, -1]
        flags = ((marks & Ghosts.DUPLICATECELL) != 0).astype(np.uint8)
        entries[g] = (np.arange(len(ids), dtype=np.int64), columns, flags)
    partners = _matching.match(blocks, entries)
    surfaces = []
    for g, piece in zip(blocks.local, pieces):
        ids, marks = bounds[g]
        index, _, _, other_flag = partners[g]
        between = np.zeros(len(ids), bool)
        between[index[other_flag == 0]] = True
        between &= (marks & Ghosts.DUPLICATECELL) == 0
        with blocks.device(g):
            kept = tack.array(ids[~between].astype(np.int64), tack.i64)
            surfaces.append(extract_surface(piece, kept, fields=fields))
    out = PartitionedDataSet(surfaces, world=world)
    lost = bool(blocks.all_reduce({g: int(bool(s._ghosts_lost))
                                   for g, s in zip(blocks.local, surfaces)}, "max"))
    return generate_global_ids(out) if lost else out
