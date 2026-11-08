# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``vtk_to_dataset`` of an unstructured grid on a discrete GPU (CUDA, HIP, Level Zero).

There the conversion takes VTK's arrays to the device once and does its work on
them: the merge of equal points (sorted), the gathers it implies (positions,
point data, the connectivity and polyhedra's faces renumbered), the screen for
inside-out cells and their repair, the screen for collapsed hexahedra, and the
matching of polyhedra's faces. On the host it would run these as CPU kernels
over VTK's memory, build a merged VTK grid, and copy the result to the device
array by array: on jura (AhmedML, 21.5M cells) 3.8 s, where this takes about a
second, most of it the copy over PCIe.

It gives the dataset the host conversion gives, bit for bit, and is taken only
for what it does as that one does: a grid of linear shapes and polyhedra,
copied (not shared), merged exactly or not at all, without ``polyhedral``. Where
anything else turns up -- polylines or polygons, higher orders, implicit cell
arrays, collapsed hexahedra to convert, polyhedra to re-orient, a device without
f64 for the polyhedra's volumes -- it returns None before it has warned, and the
host conversion runs instead.
"""

import numpy as np

import tack

#: The device types this conversion runs on: memory of their own, behind a bus.
DEVICE_TYPES = ("cuda", "hip", "level_zero")


@tack.kernel
def _scatter_rows(values, entries, new):
    # values[entries[i]] = new[i]: the repaired corners of inside-out cells.
    for i in range(entries.shape[0]):
        values[entries[i]] = new[i]


@tack.kernel
def _place_counts(cells, count, per_cell):
    # per_cell[cells[i]] = count[i], every other cell's entry left (zero).
    for i in range(cells.shape[0]):
        per_cell[cells[i]] = count[i]


def device_for(grid, options):
    """The device ``grid`` converts on here, or None for the host conversion: a
    ``vtkUnstructuredGrid``, copied, with no ``polyhedral`` and no merge tolerance,
    while a discrete GPU is the default device."""
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    if not isinstance(grid, vtkUnstructuredGrid) or not options["copy"]:
        return None
    if options["polyhedral"] is not None:
        return None
    merge = options["merge_points"]
    if merge not in (None, False, True) and float(merge) != 0.0:
        return None
    device = tack.get_default_device()
    return device if device.type in DEVICE_TYPES else None


def _upload(host, dtype, device):
    """A host array on ``device`` as ``dtype`` (a scalar type): converted first in a
    parallel copy on the host where its type differs (VTK's 64-bit ids into 32, half
    the bytes over the bus), then copied as it is."""
    from tack.lang.array import _host_copy

    host = np.ascontiguousarray(host)
    if host.dtype != dtype.numpy_dtype:
        narrow = np.empty(host.shape, dtype.numpy_dtype)
        _host_copy(narrow, host)
        host = narrow
    return tack.array(host, dtype, device=device)


def convert(grid, options, shift):
    """``grid`` as ``_vtk_to_dataset`` makes it, the work on the default device, or
    None where the host conversion must run (nothing warned, nothing kept)."""
    device = device_for(grid, options)
    if device is None:
        return None
    with device:
        return _convert(grid, options, np.asarray(shift, np.float64), device)


def _convert(grid, options, shift, device):
    from vtkmodules.vtkCommonCore import vtkDataArray

    from tack.algorithms.sort import gather
    from tack.arrays import materialize, size_of
    from tack.dataset import H1, Constant, DataSet, Field, UnstructuredTopology, ids
    from tack.dataset.algorithms.merge_points import _equal_labels, _numbered_roots
    from tack.dataset.ghosts import Ghosts, hidden_by_points
    from tack.dataset.global_ids import GlobalIds
    from tack.dataset.polyhedra import _match_rings_on, _ranges
    from tack.dataset.shapes import _BY_ID
    from tack.interop.vtk import (
        _GHOST_ARRAY,
        _ORIENTATION,
        _array_to_field,
        _arrays_of,
        _attributes_to_fields,
        _collapsed_edges,
        _from_vtk,
        _ghost_marks,
        _global_ids,
        _lagrange,
        _negative_cells,
        _negative_volumes,
        _present_types,
        _suspect_frames,
        _units_label,
        _vtk_points,
        _warn_inside_out,
    )
    from tack.lang.types import from_numpy_dtype
    from tack.runtime.dispatch import get_backend

    dtype, orient, id_dtype = options["dtype"], options["orient"], options["id_dtype"]
    # ── what this conversion takes ───────────────────────────────────
    types = _from_vtk(grid.cell_types, tack.u8)
    cells = grid.cells
    offsets, connectivity = _from_vtk(cells.offsets_array), _from_vtk(cells.connectivity_array)
    x = _vtk_points(grid)
    n, ncells = grid.number_of_points, grid.number_of_cells
    if not (isinstance(types, np.ndarray) and isinstance(offsets, np.ndarray)
            and isinstance(connectivity, np.ndarray)) or not n or not ncells:
        return None
    if x.dtype not in (np.float32, np.float64) or n >= 2**31 or connectivity.size >= 2**31:
        return None
    types = np.ascontiguousarray(types, np.uint8)
    present = _present_types(types)
    if (present & ({4, 7} | _lagrange.HIGHER_ORDER | set(_lagrange.REFUSED))
            or present - set(_BY_ID) - {42}):
        return None
    if 42 in present and not get_backend().supports_f64:
        return None                                   # the polyhedra's volumes are f64
    exact = from_numpy_dtype(x.dtype)
    offsets = offsets.astype(np.int64, copy=False)
    # ── to the device ────────────────────────────────────────────────
    types_d = tack.array(types, tack.u8, device=device)
    offsets_d = _upload(offsets, tack.i32, device)
    connectivity_d = _upload(connectivity, tack.i32, device)
    positions = tack.array(x, tack.vec3(exact), device=device)
    # ── equal points made one ────────────────────────────────────────
    kept = None
    if options["merge_points"] not in (None, False):
        labels_dtype = ids.at_least(tack.i32, n + 1)
        labels = _equal_labels(positions, labels_dtype, host=x)
        new_id, kept, count = _numbered_roots(labels, n, labels_dtype)
        del labels
        if count == n:
            kept = None
        else:
            positions = gather(positions, kept)
            connectivity_d = gather(new_id, connectivity_d)
            n = count
    # ── hexahedra with collapsed edges: the host conversion's ────────
    if 12 in present:
        flags = tack.empty((ncells,), tack.u8)
        _collapsed_edges(types_d, offsets_d, connectivity_d, flags)
        if flags.numpy().any():
            return None
    # ── polyhedra: their faces matched among themselves ──────────────
    polyhedra = inside_out_polyhedra = None
    if 42 in present:
        listed_cells = np.flatnonzero(types == 42)
        locations, faces = grid.polyhedron_face_locations, grid.polyhedron_faces
        loc_offsets = np.asarray(locations.offsets_array).astype(np.int64, copy=False)
        loc_faces = np.asarray(locations.connectivity_array).astype(np.int64, copy=False)
        face_offsets = np.asarray(faces.offsets_array).astype(np.int64, copy=False)
        face_conn = np.asarray(faces.connectivity_array)
        count = loc_offsets[listed_cells + 1] - loc_offsets[listed_cells]
        listed = loc_faces[_ranges(loc_offsets[listed_cells], count)]
        sizes = face_offsets[listed + 1] - face_offsets[listed]
        ring_points = _upload(face_conn[_ranges(face_offsets[listed], sizes)], tack.i32,
                              device)
        del listed
        if kept is not None:
            ring_points = gather(new_id, ring_points)
        ring_offsets = tack.array(np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
                                  tack.i64, device=device)
        del sizes
        out_offsets, out_points, cell_faces, sides, same_way = _match_rings_on(
            device, ring_offsets, ring_points)
        if same_way:
            return None                               # refused, or re-oriented: the host's
        negative = tack.empty((count.size,), tack.u8)
        cell_rings = tack.array(np.concatenate([[0], np.cumsum(count)]).astype(np.int64),
                                tack.i64, device=device)
        _negative_volumes(positions, cell_rings, ring_offsets, ring_points, negative)
        inside_out_polyhedra = int(np.count_nonzero(negative.numpy()))
        if inside_out_polyhedra and orient:
            return None                               # turned outward on the host
        del ring_offsets, ring_points, cell_rings, negative
        per_cell = tack.zeros((ncells,), tack.i64)
        _place_counts(tack.array(listed_cells, tack.i64, device=device),
                      tack.array(count.astype(np.int64), tack.i64, device=device), per_cell)
        cell_face_offsets = tack.empty((ncells + 1,), tack.i64)
        from tack.algorithms.scan import exclusive_scan

        exclusive_scan(per_cell, cell_face_offsets, ncells, close=True)
        del per_cell
        # In the type the topology would choose for host arrays: a Tack array of i64
        # handed to it makes its ids i64 (``ids.choose``), where these are counts.
        extent = max(n, connectivity.size, 12 * ncells, 2 * size_of(out_points),
                     2 * size_of(out_offsets), size_of(cell_faces))
        wanted = ids.choose(id_dtype, extent)
        polyhedra = (*(ids.as_ids(v, wanted, check=False)
                       for v in (out_offsets, out_points, cell_face_offsets, cell_faces)),
                     sides)
    # ── inside-out shapes: screened here, decided in f64 on the host ──
    inverted = {}
    kinds = [k for k in _ORIENTATION if k in present]
    if kinds:
        flags = tack.empty((ncells,), tack.u8)
        _suspect_frames(types_d, offsets_d, connectivity_d, positions, flags)
        flagged = flags.numpy().astype(bool)
        del flags
        for kind in kinds:
            of_kind = np.flatnonzero(types == kind)
            corners = int(offsets[of_kind[0] + 1] - offsets[of_kind[0]])
            suspect = of_kind[flagged[of_kind]]
            if not suspect.size:
                continue
            entries = offsets[suspect][:, None] + np.arange(corners)
            rows_d = gather(connectivity_d, tack.array(entries.reshape(-1), tack.i64,
                                                       device=device))
            corner = gather(positions, rows_d).numpy().reshape(suspect.size, corners, 3)
            negative = _negative_cells(kind, corners, int(suspect[0]), corner)
            if negative.any():
                rows = rows_d.numpy().reshape(suspect.size, corners)
                inverted[kind] = (suspect[negative], corners, entries[negative],
                                  rows[negative])
    if inverted and not orient:
        _warn_inside_out({k: (c, m) for k, (c, m, _, _) in inverted.items()}, stacklevel=5)
    elif inverted:
        for kind, (_, corners, entries, rows) in inverted.items():
            turned = rows[:, list(_ORIENTATION[kind][1])]
            _scatter_rows(connectivity_d, tack.array(entries.reshape(-1), tack.i64,
                                                     device=device),
                          tack.array(turned.reshape(-1), connectivity_d.dtype, device=device))
    if inside_out_polyhedra:
        import warnings

        warnings.warn(f"{inside_out_polyhedra} polyhedra are inside out: their faces agree "
                      "with their neighbours' but point inward, so their volumes are "
                      "negative; vtk_to_dataset(..., orient=True) turns them outward",
                      stacklevel=5)
    topology = UnstructuredTopology(types_d, offsets_d, connectivity_d, num_points=n,
                                    id_dtype=id_dtype, polyhedra=polyhedra)
    # ── the geometry, and the data on points and cells ───────────────
    geometry_dtype = dtype or exact
    if not np.any(shift) and geometry_dtype == exact:
        geometry = positions
    else:                                             # an origin, or another type: as the host does it
        host = x if kept is None else x[kept.numpy()]
        host = host - shift if np.any(shift) else host
        geometry = tack.array(np.ascontiguousarray(host).astype(geometry_dtype.numpy_dtype,
                                                                copy=False),
                              tack.vec3(geometry_dtype), device=device)
    del positions

    def on_points(values):
        return values if kept is None else gather(materialize(values), kept)

    point_data, cell_data = grid.GetPointData(), grid.GetCellData()
    point_ids = point_data.GetGlobalIds()
    skipped = {_GHOST_ARRAY, point_ids.GetName() if point_ids is not None else None}
    space = H1(topology)
    fields = {name: Field(space, on_points(_array_to_field(array, dtype)), _units_label(array))
              for name, array in _arrays_of(point_data)
              if name and name not in skipped and isinstance(array, vtkDataArray)}
    for name, field in _attributes_to_fields(cell_data, Constant(topology), dtype).items():
        fields[name if name not in fields else (name, field.space)] = field
    marks = _ghost_marks(point_data)
    cell_marks, point_marks = _ghost_marks(cell_data), None if marks is None else on_points(marks)
    ghosts = (None if cell_marks is None and point_marks is None
              else Ghosts(cells=cell_marks, points=point_marks))
    ghosts = hidden_by_points(topology, ghosts)
    numbers = _global_ids(point_data)
    cell_ids, point_ids = _global_ids(cell_data), None if numbers is None else on_points(numbers)
    global_ids = (None if cell_ids is None and point_ids is None
                  else GlobalIds(cells=cell_ids, points=point_ids))
    unit = _units_label(grid.points.data) if grid.points is not None else None
    return DataSet(topology, geometry, fields=fields, dtype=geometry_dtype, ghosts=ghosts,
                   global_ids=global_ids, origin=options["origin"], unit=unit)
