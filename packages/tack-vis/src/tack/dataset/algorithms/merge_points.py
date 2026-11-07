# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``merge_points``: a dataset's coincident points made one, as VTK's
vtkStaticCleanUnstructuredGrid and Viskores' CleanGrid merge them."""

from types import SimpleNamespace

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids
from tack.dataset.algorithms._common import _length
from tack.dataset.algorithms.connected_components import _hook_near, _jump
from tack.dataset.carry import Same, Take, carry
from tack.dataset.ghosts import visible
from tack.dataset.point_locator import cached_point_locator
from tack.dataset.polyhedra import (
    PolygonalTopology,
    PolyhedralTopology,
    PolylineTopology,
    _match_rings,
    _ranges,
    _side_counts,
)
from tack.dataset.spaces import H1
from tack.dataset.topology import StructuredTopology, UnstructuredTopology
from tack.runtime.dispatch import get_backend, on_inputs_device

__all__ = ["merge_points"]


@tack.kernel
def _roots(labels, flags):
    for p in range(flags.shape[0]):
        flags[p] = 1 if labels[p] == p else 0


@tack.kernel
def _numbered(labels, index, new_id, kept):
    # Each point's merged point is its root's number; each root is kept there.
    for p in range(labels.shape[0]):
        r = labels[p]
        new_id[p] = index[r]
        if r == p:
            kept[index[p]] = p


@tack.kernel
def _run_starts(positions, order, flags):
    # 1 where a point, in the order of its coordinates' bits, starts a run of equal
    # coordinates: compared as numbers, so a NaN starts its own.
    for i in range(flags.shape[0]):
        start = 1
        if i > 0:
            a = positions[order[i]]
            b = positions[order[i - 1]]
            if a[0] == b[0] and a[1] == b[1] and a[2] == b[2]:
                start = 0
        flags[i] = start


@tack.kernel
def _run_firsts(order, flags, runs, firsts):
    # Each run's first point: its smallest, the sort being stable.
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            firsts[runs[i]] = order[i]


@tack.kernel
def _run_labels(order, flags, runs, firsts, labels):
    for i in range(flags.shape[0]):
        labels[order[i]] = firsts[runs[i] + flags[i] - 1]


@tack.kernel
def _renumbered(connectivity, new_id, out):
    for e in range(out.shape[0]):
        out[e] = new_id[connectivity[e]]


def _renumber(point_ids, new_id, dtype):
    out = tack.empty((max(arrays.size_of(point_ids), 1),), dtype)
    if arrays.size_of(point_ids):
        _renumbered(point_ids, new_id, out)
    return out if arrays.size_of(point_ids) else tack.empty((0,), dtype)


@tack.kernel
def _distinct_points(face_offsets, face_points, out):
    # How many different points each face has once merged: under three, it has no area.
    for f in range(out.shape[0]):
        n = 0
        for i in range(face_offsets[f], face_offsets[f + 1]):
            new = 1
            for j in range(face_offsets[f], i):
                if face_points[j] == face_points[i]:
                    new = 0
            n += new
        out[f] = n


def _merged_polyhedra(topology, new_id, count):
    """``topology`` on the merged points, and the faces it keeps: faces renumbered, then
    matched by point set (``_match_rings``, as ``from_cell_faces`` matches its rings),
    so the copies of a face that pieces each stored once, with one side, become one
    face with two. The first copy is the face, in its winding; a cell on a later copy,
    which runs the other way, sees the face from its other side. A face left with
    fewer than three different points has no area and is matched with nothing: it
    stays as it is, as an unstructured cell's merged corners do."""
    renumbered = _renumber(topology.face_points, new_id, topology.id_dtype)
    nf = topology.num_faces
    distinct = tack.empty((max(nf, 1),), tack.i32)
    if nf:
        _distinct_points(topology.face_offsets, renumbered, distinct)
    face_points = renumbered.numpy()
    face_offsets = arrays.to_host(topology.face_offsets).astype(np.int64)
    # Match the faces with an area among themselves; each face's representative is its
    # first copy, or itself.
    flat = np.flatnonzero(distinct.numpy()[:nf] >= 3)
    if flat.size == nf:                                   # the usual case: every face
        ring_offsets, ring_points = face_offsets, face_points
    else:
        sizes = np.diff(face_offsets)[flat]
        ring_offsets = np.concatenate([[0], np.cumsum(sizes)])
        ring_points = face_points[_ranges(face_offsets[flat], sizes)]
    _, _, ring_face, ring_copy, same_way = _match_rings(ring_offsets, ring_points)
    if same_way:
        raise ValueError(f"{same_way} faces become one with another that runs the same way: "
                         "both their cells hold the merged face on the same side")
    first = flat[np.flatnonzero(ring_copy == 0)]          # each matched face's first copy
    representative = np.arange(nf)
    representative[flat] = first[ring_face]
    copy = (representative != np.arange(nf)).astype(np.uint8)
    kept = np.flatnonzero(copy == 0)
    number = np.full(nf, -1, np.int64)
    number[kept] = np.arange(kept.size)
    face_of = number[representative]
    kept_sizes = np.diff(face_offsets)[kept]
    offsets = np.concatenate([[0], np.cumsum(kept_sizes)])
    points = face_points[_ranges(face_offsets[kept], kept_sizes)]
    cell_faces = arrays.to_host(topology.cell_faces)
    sides = arrays.to_host(topology.cell_face_sides) ^ copy[cell_faces]
    merged = PolyhedralTopology(offsets, points, topology.cell_offsets, face_of[cell_faces],
                                sides, num_points=count, id_dtype=topology.id_dtype)
    counts, _ = _side_counts(merged)
    crowded = int(np.count_nonzero(counts.numpy() > 1))
    if crowded:
        raise ValueError(f"{crowded} merged faces have more than one cell on a side: more "
                         "than two cells meet at them")
    return merged, Take(tack.array(kept.astype(merged.id_dtype.numpy_dtype)))


def _equal_labels(positions, dtype, host=None):
    """Each point's smallest point of equal coordinates, as ``_merged_ids`` with no
    tolerance labels them, by sorting: the coordinates' bits (taken on the host, a
    view; -0 made 0 first, as it equals 0) sorted by a stable radix sort, and runs
    of points whose coordinates compare equal labelled with their first.
    ``positions`` are 3-vectors (a Tack array), ``host`` the same on the host if
    the caller has them. No atomics, and no locator: on AhmedML's 24M points
    0.4-0.8 s against the union-find's 2.3-3.7."""
    from tack.dataset.buckets import _radix_order

    n = arrays.size_of(positions)
    if host is None:
        host = arrays.to_host(positions)
    host = np.ascontiguousarray(host).reshape(n, 3)
    host = host + host.dtype.type(0)                  # -0 is 0's bits after this
    bits = host.view(np.uint32 if host.dtype.itemsize == 4 else np.uint64)
    keys = tack.array(bits, tack.vec3(tack.u32 if host.dtype.itemsize == 4 else tack.u64))
    del host, bits
    order = _radix_order(keys, n, dtype)
    del keys
    flags = tack.empty((n,), dtype)
    _run_starts(positions, order, flags)
    runs = tack.empty((n,), dtype)
    count = int(exclusive_scan(flags, runs, n))
    firsts = tack.empty((count,), dtype)
    _run_firsts(order, flags, runs, firsts)
    labels = tack.empty((n,), dtype)
    _run_labels(order, flags, runs, firsts, labels)
    return labels


def _merged_ids(data, tolerance):
    """``(new_id, kept, count)`` for ``data``'s points within ``tolerance`` of each other
    made one: each point's merged point, each merged point's input point (its set's
    smallest), and how many there are. With no tolerance, by sorting the
    coordinates (``_equal_labels``); otherwise union-find over the points through
    the locator's bins: each pair hooks the larger root under the smaller (an
    atomic minimum), until nothing changes, so a set's root is its smallest point
    whatever the order of the hooks."""
    n = data.num_points
    dtype = ids.at_least(tack.i32, n + 1)
    if tolerance == 0:
        labels = (_equal_labels(arrays.materialize(data.geometry.values), dtype) if n
                  else tack.empty((0,), dtype))
        return _numbered_roots(labels, n, dtype)
    if dtype not in get_backend().supported_atomic_dtypes:
        raise NotImplementedError(f"merging {n} points needs {dtype} atomics, which the "
                                  f"{get_backend().name} backend does not have")
    labels = tack.arange(n, dtype=dtype)
    if n:
        grid = cached_point_locator(data).grid()
        every = tack.empty((n,), tack.u8)
        every.fill(1)
        changed = tack.zeros((1,), tack.i32)
        while True:
            changed.fill(0)
            _hook_near(grid, 0, every, float(tolerance), labels, changed)
            _jump(labels)
            if not int(changed.numpy()[0]):
                break
    return _numbered_roots(labels, n, dtype)


def _numbered_roots(labels, n, dtype):
    """``_merged_ids``' result from each point's root (``labels``): the roots numbered
    in their order, each point its root's number."""
    flags = tack.empty((max(n, 1),), dtype)
    index = tack.empty((max(n, 1),), dtype)
    count = 0
    if n:
        _roots(labels, flags)
        count = int(exclusive_scan(flags, index, n))
    new_id = tack.empty((max(n, 1),), dtype)
    kept = tack.empty((max(count, 1),), dtype)
    if n:
        _numbered(labels, index, new_id, kept)
    return new_id, (kept if count else tack.empty((0,), dtype)), count


@on_inputs_device
def merge_points(data, tolerance=0.0, fields=None):
    """``data`` with each set of coincident points made one: points at most
    ``tolerance`` apart (inclusive; 0, the default, merges only points of equal
    coordinates) join, and so do chains of them, so a set may span more than
    ``tolerance``. A set becomes the point of its smallest id, which keeps that
    point's position, point fields, ghost marks and global id; the merged points
    keep the order of those ids, and the cells are the same cells on them, a
    cell's corners that merged kept as they are. Cell fields are unchanged.

    For meshes written without merging their points (each piece of a
    simulation's output storing a point once for every cell or block that
    uses it), which nothing can stitch by matching points across cells or
    pieces until each point is one. ``tolerance`` is in the dataset's spatial
    unit, or a quantity (``"1e-9 m"``); it should be far below the distance
    between points that are not copies, since every point compares itself with
    every point that close. Takes an unstructured, polyhedral, polygonal or
    polyline topology; a structured grid's points are its lattice. A polyhedral
    topology's faces are matched again on the merged points: the copies of a face
    that pieces each stored once, with one cell, become one face with both
    (OpenFOAM's processor boundaries, written without merging), its first copy
    kept with its values on faces; copies wound the same way, or a face with
    more than two cells, are refused. Polygonal and polyline topologies derive
    their facets from the merged loops and lines. Hidden cells are dropped
    first, as every filter that makes cells drops them."""
    tolerance = _length(data, tolerance, "merge_points's tolerance")
    if not tolerance >= 0:
        raise ValueError(f"tolerance must be at least zero, not {tolerance!r}")
    if isinstance(data.topology, StructuredTopology):
        raise TypeError("merge_points takes an unstructured topology: a structured "
                        "grid's points are its lattice")
    if not isinstance(data.topology, (UnstructuredTopology, PolyhedralTopology,
                                      PolylineTopology)):
        raise NotImplementedError(f"merge_points of a {type(data.topology).__name__} is "
                                  f"not supported yet")
    if not isinstance(data.geometry.space, H1) or data.geometry.space.order != 1:
        raise TypeError("merge_points needs an order-1 H1 geometry: one position per point")
    data = visible(data)
    topology = data.topology
    new_id, kept, count = _merged_ids(data, tolerance)
    faces = None
    if isinstance(topology, PolygonalTopology):
        merged = PolygonalTopology(topology.loop_offsets,
                                   _renumber(topology.loop_points, new_id, data.id_dtype),
                                   num_points=count, id_dtype=data.id_dtype)
    elif isinstance(topology, PolyhedralTopology):
        merged, faces = _merged_polyhedra(topology, new_id, count)
    elif isinstance(topology, PolylineTopology):
        merged = PolylineTopology(topology.line_offsets,
                                  _renumber(topology.line_points, new_id, data.id_dtype),
                                  num_points=count, id_dtype=data.id_dtype)
    elif topology.has_polyhedra:
        # The polyhedra's faces matched again as a polyhedral topology's are, the
        # shapes beside them renumbered. Face ids change (the shapes' faces come
        # after the polyhedra's), so values on faces are not carried.
        polyhedra, _ = _merged_polyhedra(SimpleNamespace(
            num_faces=topology.num_polyhedral_faces, face_offsets=topology.face_offsets,
            face_points=topology.face_points, cell_offsets=topology.cell_face_offsets,
            cell_faces=topology.cell_faces, cell_face_sides=topology.cell_face_sides,
            id_dtype=topology.id_dtype), new_id, count)
        merged = UnstructuredTopology(
            topology.types, topology.offsets,
            _renumber(topology.connectivity, new_id, data.id_dtype), num_points=count,
            polyhedra=(polyhedra.face_offsets, polyhedra.face_points, polyhedra.cell_offsets,
                       polyhedra.cell_faces, polyhedra.cell_face_sides))
    else:
        merged = UnstructuredTopology(topology.types, topology.offsets,
                                      _renumber(topology.connectivity, new_id, data.id_dtype),
                                      num_points=count)
    return carry(data, merged, points=Take(kept), cells=Same(), faces=faces, fields=fields)
