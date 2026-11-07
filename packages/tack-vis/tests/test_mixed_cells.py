# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for mixed cells (#307): polyhedra beside shaped cells in one
``UnstructuredTopology``.

The oracle is the shaped mesh itself. Some of a mesh's shaped cells are made
polyhedra of the same points and faces (``_mixed``), each face stored once as
the shaped mesh's ``faces()`` has it. The mixed mesh must then derive the same
faces, sides, rows and boundary as the shaped one: a face between a shaped
cell and a polyhedron is one face with two sides, whichever of the two its
stored winding points out of.
"""

import numpy as np
import pytest
from test_dataset_api import _two_hexes_and_a_pyramid

import tack
import tack.dataset as td
from tack.dataset import shapes

pytestmark = pytest.mark.quick

POLYHEDRON = int(shapes.POLYHEDRON)


def _hexes(n=(4, 3, 3)):
    """An unstructured mesh of hexahedra over an ``n``-point grid, x fastest."""
    nx, ny, nz = n
    pts = np.array([[i, j, k] for k in range(nz) for j in range(ny) for i in range(nx)],
                   np.float64)
    pid = lambda i, j, k: i + nx * (j + ny * k)
    conn = [pid(i + a, j + b, k + c) for k in range(nz - 1) for j in range(ny - 1)
            for i in range(nx - 1) for a, b, c in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                                  (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
    cells = len(conn) // 8
    topology = td.UnstructuredTopology(np.full(cells, 12, np.uint8),
                                       np.arange(0, 8 * cells + 1, 8), np.array(conn),
                                       num_points=len(pts))
    return td.DataSet(topology, tack.array(pts, tack.vec3(tack.f64)))


def _faces_by_cell(topology):
    """``{(cell, local face): (face, slot)}`` from a shaped topology's faces."""
    found = {}
    for f, (c0, l0, c1, l1) in enumerate(topology.faces().sides.numpy()):
        found[(int(c0), int(l0))] = (f, 0)
        if c1 >= 0:
            found[(int(c1), int(l1))] = (f, 1)
    return found


def _mixed(topology, chosen):
    """``topology`` (shaped) with the cells ``chosen`` made polyhedra of the same
    points: each face of a chosen cell stored once, as the shaped mesh's
    ``faces()`` rows it (wound out of its side 0), the chosen cell's faces in its
    local order, so a polyhedron's local face k is the shape's."""
    chosen = sorted({int(c) for c in chosen})
    types = topology.types.numpy().copy()
    offsets = topology.offsets.numpy()
    rows = topology.faces().rows.numpy()
    by_cell = _faces_by_cell(topology)
    stored, face_points, face_offsets = {}, [], [0]
    cell_faces, sides, cell_face_offsets = [], [], [0]
    for c in range(topology.num_cells):
        if c in chosen:
            k = 0
            while (c, k) in by_cell:
                f, slot = by_cell[(c, k)]
                if f not in stored:
                    stored[f] = len(stored)
                    row = [int(p) for p in rows[f] if p >= 0]
                    face_points.extend(row)
                    face_offsets.append(len(face_points))
                cell_faces.append(stored[f])
                sides.append(slot)
                k += 1
            types[c] = POLYHEDRON
        cell_face_offsets.append(len(cell_faces))
    polyhedra = (np.array(face_offsets), np.array(face_points, np.int64),
                 np.array(cell_face_offsets), np.array(cell_faces, np.int64),
                 np.array(sides, np.uint8))
    return td.UnstructuredTopology(types, offsets, topology.connectivity.numpy(),
                                   num_points=topology.num_points, polyhedra=polyhedra)


def _face_records(topology):
    """Each face as ``{point set: (side 0's walk round it, sides)}``: what a face
    is, whatever its id."""
    faces = topology.faces()
    rows = faces.rows.numpy()
    kinds = faces.kinds.numpy()
    out = {}
    for f, side in enumerate(faces.sides.numpy()):
        if kinds[f] == int(shapes.POLYGON):
            fo, fp = topology.face_offsets.numpy(), topology.face_points.numpy()
            row = tuple(int(p) for p in fp[fo[f]:fo[f + 1]])
        else:
            row = tuple(int(p) for p in rows[f] if p >= 0)
        out[frozenset(row)] = (_cycle(row), tuple(int(x) for x in side))
    return out


def _cells_by_face(topology):
    """``{point set: {(cell, local face), ...}}``: which cells each face is between."""
    return {points: {sides[:2], sides[2:]} - {(-1, -1)}
            for points, (_, sides) in _face_records(topology).items()}


def _cycle(row):
    """A face's walk from its lowest point, in its direction: where a walk starts
    is not part of a face's winding."""
    k = row.index(min(row))
    return row[k:] + row[:k]


@pytest.mark.parametrize("chosen", [[0], [0, 3, 5], [1, 2, 4, 6], list(range(12))],
                         ids=["one", "some", "others", "all"])
def test_a_mixed_mesh_has_the_faces_of_the_shaped_one(chosen):
    shaped = _hexes().topology
    mixed = _mixed(shaped, chosen)
    assert mixed.has_polyhedra
    assert _face_records(mixed) == _face_records(shaped)
    assert mixed.faces().num_faces == shaped.faces().num_faces
    np.testing.assert_array_equal(
        np.sort([sorted(int(p) for p in r if p >= 0)
                 for r in mixed.faces().rows.numpy()[mixed.faces().boundary().numpy()]],
                axis=0),
        np.sort([sorted(int(p) for p in r if p >= 0)
                 for r in shaped.faces().rows.numpy()[shaped.faces().boundary().numpy()]],
                axis=0))


def test_a_mixed_meshs_polyhedral_faces_keep_their_ids():
    shaped = _hexes().topology
    mixed = _mixed(shaped, [0, 3, 5])
    faces = mixed.faces()
    nfp = mixed.num_polyhedral_faces
    fo, fp = mixed.face_offsets.numpy(), mixed.face_points.numpy()
    rows = faces.rows.numpy()
    for f in range(nfp):
        assert tuple(rows[f][:fo[f + 1] - fo[f]]) == tuple(fp[fo[f]:fo[f + 1]])
    # Every polyhedron's (cell, local face) names the face it is a side of.
    sides = faces.sides.numpy()
    cfo, cf = mixed.cell_face_offsets.numpy(), mixed.cell_faces.numpy()
    cfs = mixed.cell_face_sides.numpy()
    for c in (0, 3, 5):
        for k, e in enumerate(range(cfo[c], cfo[c + 1])):
            s = cfs[e]
            assert tuple(sides[cf[e]][2 * s:2 * s + 2]) == (c, k)


def test_mixed_groups_are_the_shapes_and_the_polyhedra():
    mixed = _mixed(_hexes().topology, [0, 3, 5])
    groups = mixed.groups()
    assert [g.shape for g in groups] == [shapes.Hexahedron, shapes.Polyhedron]
    hexes, polyhedra = groups
    assert sorted(hexes.args[1].numpy()) == [1, 2, 4, 6, 7, 8, 9, 10, 11]
    assert list(polyhedra.args[-2].numpy()) == [0, 3, 5]
    assert polyhedra.count == 3
    assert mixed.dimension == 3


@tack.kernel
def _face_counts(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.num_faces(c) * 100 + cells.num_points(c)


def test_a_polyhedral_groups_view_reads_its_cells_by_their_ids():
    mixed = _mixed(_hexes().topology, [0, 3, 5])
    out = tack.zeros((mixed.num_cells,), tack.i32)
    _face_counts(mixed.groups()[1].view(), out)
    expected = np.zeros(mixed.num_cells, np.int32)
    expected[[0, 3, 5]] = 608
    np.testing.assert_array_equal(out.numpy(), expected)


def test_mixed_shapes_and_polyhedra_have_the_shaped_faces():
    data = _two_hexes_and_a_pyramid()
    shaped = data.topology
    for chosen in ([0], [2], [1, 2], [0, 1, 2]):
        assert _face_records(_mixed(shaped, chosen)) == _face_records(shaped)


def test_polyhedra_need_their_faces():
    shaped = _hexes().topology
    types = shaped.types.numpy().copy()
    types[0] = POLYHEDRON
    topology = td.UnstructuredTopology(types, shaped.offsets.numpy(),
                                       shaped.connectivity.numpy())
    with pytest.raises(ValueError, match="polyhedra .type 42. need their faces"):
        topology.groups()


def test_validate_checks_the_polyhedra():
    shaped = _hexes().topology
    good = _mixed(shaped, [0, 3])
    assert good.validate() is good
    arrays = [getattr(good, name).numpy() for name in good._POLYHEDRAL]
    # Faces on a hexahedron: cell 1 takes polyhedron 3's six faces, leaving it none.
    cfo = arrays[2].copy()
    assert list(cfo[:5]) == [0, 6, 6, 6, 12]
    cfo[2] = cfo[3] = 12
    bad = td.UnstructuredTopology(good.types.numpy(), good.offsets.numpy(),
                                  good.connectivity.numpy(), polyhedra=(
                                      arrays[0], arrays[1], cfo, arrays[3], arrays[4]))
    with pytest.raises(ValueError, match="1 cells that are not polyhedra have faces; "
                                         "1 polyhedra have fewer than four faces"):
        bad.validate()
    # A polyhedron whose connectivity lacks one of its face points.
    conn = good.connectivity.numpy().copy()
    conn[0] = conn[1]
    bad = td.UnstructuredTopology(good.types.numpy(), good.offsets.numpy(), conn,
                                  num_points=good.num_points, polyhedra=tuple(arrays))
    with pytest.raises(ValueError, match="missing from their connectivity"):
        bad.validate()


def test_a_polyhedral_face_wound_like_its_shaped_neighbour_is_refused():
    shaped = _hexes().topology
    mixed = _mixed(shaped, [0])
    arrays = [getattr(mixed, name).numpy().copy() for name in mixed._POLYHEDRAL]
    fo, fp = arrays[0], arrays[1]
    # Cell 0's face shared with hexahedron 1, reversed: both cells now wind it out
    # of themselves.
    sides = shaped.faces().sides.numpy()
    by_cell = _faces_by_cell(shaped)
    k = next(k for k in range(6) if 1 in sides[by_cell[(0, k)][0]][[0, 2]])
    f = int(mixed.cell_faces.numpy()[k])
    fp[fo[f]:fo[f + 1]] = fp[fo[f]:fo[f + 1]][::-1]
    bad = td.UnstructuredTopology(mixed.types.numpy(), mixed.offsets.numpy(),
                                  mixed.connectivity.numpy(), num_points=mixed.num_points,
                                  polyhedra=tuple(arrays))
    with pytest.raises(ValueError, match="wound the same way|neither|either direction"):
        bad.faces()


def _pentagonal_prisms():
    """Two pentagonal prisms stacked along z, sharing a pentagon, as polyhedra,
    and a hexahedron against the lower prism's first side."""
    angles = 2 * np.pi * np.arange(5) / 5
    ring = np.stack([np.cos(angles), np.sin(angles)], 1)
    pts = [[x, y, z] for z in (0.0, 1.0, 2.0) for x, y in ring]   # 0-4, 5-9, 10-14
    # The hexahedron beyond side (0, 1): two points out from 0 and 1 at z 0 and 1.
    out = (ring[0] + ring[1]) * 1.5
    pts += [[*(ring[0] + out - (ring[0] + ring[1]) / 2), 0.0],
            [*(ring[1] + out - (ring[0] + ring[1]) / 2), 0.0]]           # 15, 16
    pts += [[p[0], p[1], 1.0] for p in pts[15:17]]                       # 17, 18
    # Faces stored once, wound out of their side 0.
    lower_bottom = [4, 3, 2, 1, 0]            # out of the lower prism (down)
    middle = [5, 6, 7, 8, 9]                  # out of the lower prism (up)
    upper_top = [10, 11, 12, 13, 14]          # out of the upper prism (up)
    lower_sides = [[i, (i + 1) % 5, 5 + (i + 1) % 5, 5 + i] for i in range(5)]
    upper_sides = [[5 + i, 5 + (i + 1) % 5, 10 + (i + 1) % 5, 10 + i] for i in range(5)]
    # The lower prism's side (0, 1) faces the hexahedron, wound out of the prism.
    faces = [lower_bottom, middle, upper_top, *lower_sides, *upper_sides]
    face_offsets = np.cumsum([0] + [len(f) for f in faces])
    lower = [0, 1, *list(range(3, 8))]
    upper = [1, 2, *list(range(8, 13))]
    cell_faces = lower + upper
    sides = [0] * len(lower) + [1, 0] + [0] * 5
    # The hexahedron: points 1, 0 at the prism, 15, 16 out, in VTK's order: its
    # bottom (1, 0, 15, 16)... wound so its face against the prism is (0, 1, 6, 5).
    hexahedron = [0, 15, 16, 1, 5, 17, 18, 6]
    types = [POLYHEDRON, POLYHEDRON, 12]
    connectivity = list(range(10)) + list(range(5, 15)) + hexahedron
    offsets = [0, 10, 20, 28]
    cell_face_offsets = [0, len(lower), len(lower) + len(upper), len(lower) + len(upper)]
    topology = td.UnstructuredTopology(
        np.array(types, np.uint8), np.array(offsets), np.array(connectivity),
        num_points=len(pts),
        polyhedra=(face_offsets, np.concatenate(faces), np.array(cell_face_offsets),
                   np.array(cell_faces), np.array(sides, np.uint8)))
    return td.DataSet(topology, tack.array(np.array(pts, np.float64), tack.vec3(tack.f64)))


def test_pentagons_are_polygon_faces_and_a_hexahedron_shares_a_prisms_side():
    data = _pentagonal_prisms()
    topology = data.topology
    assert topology.validate() is topology
    faces = topology.faces()
    kinds = faces.kinds.numpy()
    assert list(kinds[:3]) == [int(shapes.POLYGON)] * 3
    sides = faces.sides.numpy()
    # The middle pentagon: lower prism side 0, upper prism side 1.
    assert tuple(sides[1]) == (0, 1, 1, 0)
    # The prism's side (0, 1, 6, 5) is face 3: the prism's side 0, the hexahedron's 1.
    assert sides[3][0] == 0 and sides[3][2] == 2
    # Six faces of the hexahedron, one shared: 15 polyhedral faces, 5 more.
    assert faces.num_faces == 13 + 5
    groups = faces.groups()
    assert [g.shape for g in groups] == [shapes.Line, shapes.Triangle, shapes.Quad,
                                         shapes.Polygon]
    assert [g.count for g in groups] == [0, 0, 15, 3]
    assert len(faces.boundary().numpy()) == faces.num_faces - 2


@pytest.mark.parametrize("chosen", [[0, 3, 5], list(range(12))], ids=["some", "all"])
def test_a_mixed_mesh_has_the_edges_of_the_shaped_one(chosen):
    shaped = _hexes().topology
    mixed = _mixed(shaped, chosen)
    edges, expected = mixed.edges(), shaped.edges()
    # Numbered by (low, high) on both paths: the same edges, in the same order.
    np.testing.assert_array_equal(edges.rows.numpy(), expected.rows.numpy())
    rows = edges.rows.numpy()
    # Each polyhedral face's edge j runs from its point j to the next.
    fo, fp = mixed.face_offsets.numpy(), mixed.face_points.numpy()
    face_edge, sign = edges.face_edge.numpy(), edges.face_edge_sign.numpy()
    for f in range(mixed.num_polyhedral_faces):
        n = fo[f + 1] - fo[f]
        for j in range(n):
            a, b = fp[fo[f] + j], fp[fo[f] + (j + 1) % n]
            assert tuple(rows[face_edge[fo[f] + j]]) == (min(a, b), max(a, b))
            assert sign[fo[f] + j] == (1 if a < b else -1)
    # Each hexahedron's (cell, local edge) names its edge, as the shaped mesh's does.
    if len(chosen) < 12:
        (hexes,) = [g for g in mixed.groups() if g.shape is shapes.Hexahedron]
        start = edges.group_starts[hexes]
        (all_hexes,) = shaped.groups()
        for i, c in enumerate(hexes.args[1].numpy()):
            np.testing.assert_array_equal(
                edges.side_edge.numpy()[start + 12 * i:start + 12 * i + 12],
                expected.side_edge.numpy()[12 * c:12 * c + 12])


def test_a_mixed_meshs_faces_and_edges_move_with_it():
    mixed = _mixed(_hexes().topology, [0, 3, 5])
    moved = mixed._moved(tack.device("cpu"))
    assert moved.has_polyhedra
    assert _face_records(moved) == _face_records(mixed)


def _as_mixed(data, chosen):
    """``data`` (shaped) with cells ``chosen`` made polyhedra, fields kept."""
    out = td.DataSet(_mixed(data.topology, chosen), data.geometry.values)
    for name, field in data.fields.items():
        kind = "points" if field.space.__class__.__name__ == "H1" else "cells"
        out.fields[name] = td.Field(td.H1(out) if kind == "points" else td.Constant(out),
                                    field.values)
    return out


@pytest.mark.parametrize("cells", [[0, 1, 2, 3], [1, 2, 4, 6, 7], [3, 5], [8, 9]],
                         ids=["with-polyhedra", "both", "only-polyhedra", "only-hexahedra"])
def test_extracting_cells_of_a_mixed_mesh_keeps_its_polyhedra(cells):
    shaped = _hexes()
    shaped.fields["x"] = td.Field(td.H1(shaped), tack.array(shaped.positions()[:, 0]))
    chosen = [0, 3, 5]
    mixed = _as_mixed(shaped, chosen)
    out, expected = td.extract_cells(mixed, cells), td.extract_cells(shaped, cells)
    t, e = out.topology, expected.topology
    kept_polyhedra = [i for i, c in enumerate(cells) if c in chosen]
    assert t.has_polyhedra == bool(kept_polyhedra)
    types = e.types.numpy().copy() if hasattr(e.types, "numpy") else \
        np.full(e.num_cells, 12, np.uint8)
    types[kept_polyhedra] = POLYHEDRON
    np.testing.assert_array_equal(t.types.numpy(), types)
    np.testing.assert_array_equal(t.connectivity.numpy(), e.connectivity.numpy())
    assert t.num_points == e.num_points
    np.testing.assert_array_equal(out.fields["x"].values.numpy(),
                                  expected.fields["x"].values.numpy())
    if kept_polyhedra:
        assert t.validate() is t
    assert _face_records(t) == _face_records(e)


def test_a_kept_face_whose_side_zero_is_gone_is_turned():
    shaped = _hexes()
    # Cells 0 and 1 share a face; 0 is a hexahedron and 1 a polyhedron, so the
    # stored face winds out of the hexahedron, into the polyhedron (side 1).
    mixed = _as_mixed(shaped, [1])
    assert 1 in mixed.topology.cell_face_sides.numpy()
    out = td.extract_cells(mixed, [1])
    t = out.topology
    assert list(t.cell_face_sides.numpy()) == [0] * 6
    assert t.validate() is t
    assert _face_records(t) == _face_records(td.extract_cells(shaped, [1]).topology)
    # Kept with its hexahedron, it is not turned.
    both = td.extract_cells(mixed, [0, 1]).topology
    assert 1 in both.cell_face_sides.numpy()
    assert _face_records(both) == _face_records(td.extract_cells(shaped, [0, 1]).topology)


def test_threshold_of_a_mixed_mesh_is_the_shaped_meshs():
    shaped = _hexes()
    shaped.fields["x"] = td.Field(td.H1(shaped), tack.array(shaped.positions()[:, 0]))
    mixed = _as_mixed(shaped, [0, 3, 5])
    out, expected = td.threshold(mixed, "x", 0.0, 1.5), td.threshold(shaped, "x", 0.0, 1.5)
    np.testing.assert_array_equal(out.topology.connectivity.numpy(),
                                  expected.topology.connectivity.numpy())
    assert _face_records(out.topology) == _face_records(expected.topology)


# ── VTK ─────────────────────────────────────────────────────────────

def _vtk():
    return pytest.importorskip("tack.interop.vtk")


@pytest.mark.parametrize("chosen", [[0, 3, 5], list(range(12))], ids=["some", "all"])
def test_a_mixed_mesh_goes_to_vtk_and_back(chosen):
    vtk = _vtk()
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    data = _as_mixed(_hexes(), chosen)
    grid = vtk.dataset_to_vtk(data)
    types = np.asarray(grid.cell_types)
    assert list(np.flatnonzero(types == POLYHEDRON)) == chosen
    for c in chosen:
        cell = grid.GetCell(c)
        assert cell.cell_type == POLYHEDRON and cell.number_of_faces == 6
        assert cell.ComputeVolume() == pytest.approx(1.0)
    surface = vtkGeometryFilter()(grid)
    assert surface.number_of_cells == data.topology.faces().boundary().shape[0]
    back = vtk.vtk_to_dataset(grid)
    t = back.topology
    assert isinstance(t, td.UnstructuredTopology) and t.has_polyhedra
    np.testing.assert_array_equal(t.types.numpy(), types)
    assert t.validate() is t
    # The same faces between the same cells; which is side 0 is the stored
    # winding's, and VTK's copy of a face a polyhedron shares with a hexahedron is
    # the polyhedron's, outward for it.
    assert _cells_by_face(t) == _cells_by_face(data.topology)
    # The old reading, every cell a polyhedron, has the same faces.
    every = vtk.vtk_to_dataset(grid, polyhedral=True).topology
    assert isinstance(every, td.PolyhedralTopology)
    assert {frozenset(r) for r in _face_records(t)} == {
        frozenset(int(p) for p in every.face_points.numpy()[a:b])
        for a, b in zip(every.face_offsets.numpy()[:-1], every.face_offsets.numpy()[1:])}


def test_vtk_polyhedra_wound_the_same_way_beside_shapes_are_refused():
    vtk = _vtk()
    data = _as_mixed(_hexes(), [0, 1])
    grid = vtk.dataset_to_vtk(data)
    faces = grid.polyhedron_faces
    connectivity = np.asarray(faces.connectivity_array)
    offsets = np.asarray(faces.offsets_array)
    # Cell 1's copy of the face it shares with cell 0, turned: both wind it outward.
    sides = data.topology.faces().sides.numpy()
    shared = next(f for f, s in enumerate(sides) if {s[0], s[2]} == {0, 1})
    k = int(sides[shared][3] if sides[shared][2] == 1 else sides[shared][1])
    copy = np.asarray(grid.polyhedron_face_locations.offsets_array)[1] + k
    a, b = offsets[copy], offsets[copy + 1]
    connectivity[a:b] = connectivity[a:b][::-1].copy()
    with pytest.raises(ValueError, match="wound the same way"):
        vtk.vtk_to_dataset(grid)


def test_extract_cells_of_a_mixed_vtk_mesh_goes_back_to_vtk():
    vtk = _vtk()
    data = _as_mixed(_hexes(), [0, 3, 5])
    out = td.extract_cells(data, [2, 3, 4, 5])
    grid = vtk.dataset_to_vtk(out)
    assert [grid.GetCell(c).cell_type for c in range(4)] == [12, POLYHEDRON, 12, POLYHEDRON]
    assert all(grid.GetCell(c).ComputeVolume() == pytest.approx(1.0) for c in (1, 3))


# ── Which algorithms take mixed cells ───────────────────────────────

#: test_dataset_devices.CALLS that take a mixed mesh (#307 phase 1) and must give
#: the shaped mesh's answer; every other one refuses it by name.
READY = {
    "contour_1d", "threshold", "extract_geometry", "extract_cells", "extract_extent", "mask",
    "mask_points", "threshold_points", "extract_points", "boundary_faces", "extract_edges",
    "tubes", "glyphs", "warp", "transform", "cylindrical", "spherical", "cell_ids",
    "point_ids", "face_centers", "edge_lengths", "magnitude", "log_values", "dot", "cross",
    "composite", "implicit_values", "point_elevation", "cell_geometry", "face_geometry",
    "net_flux", "perot", "statistics", "entropy", "voxel_downsample",
    "voxel_downsample_nearest", "outliers_radius", "outliers_statistical", "point_normals",
    "point_normals_radius", "connected_components_radius", "deposit", "deposit_cic",
    "sph_interpolate", "sph_density", "connected_components_periodic",
    "point_normals_periodic", "halo_catalog", "halo_catalog_periodic",
}


def _summary(value):
    """What an algorithm's result is, independent of how faces are numbered."""
    if isinstance(value, td.DataSet):
        t = value.topology
        measure = (float(td.cell_geometry(value).measure.values.numpy().sum())
                   if t.num_cells and t.dimension > 0 else 0.0)
        return ("dataset", t.num_cells, value.num_points, round(measure, 5),
                tuple(np.round(np.sort(value.positions().reshape(-1)), 5)))
    if isinstance(value, td.Field):
        v = np.asarray(tack.arrays.to_host(value.values), float)
        v = v.reshape(v.shape[0], -1)
        return ("field", v.shape, tuple(np.round(np.sort(v.reshape(-1)), 5)))
    if isinstance(value, td.Set):
        ids = value.ids.numpy()
        if value.on == "faces":         # face ids are numbered differently: their points
            records = list(_face_records(value.topology))
            return ("faces", frozenset(k for k in records
                                       if _face_id(value.topology, k) in set(ids.tolist())))
        return ("set", value.on, tuple(np.sort(ids)))
    if isinstance(value, (tuple, list)):
        return tuple(_summary(v) for v in value)
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return type(value).__name__


@pytest.mark.parametrize("name", sorted(__import__("test_dataset_devices").CALLS))
def test_an_algorithm_takes_mixed_cells_or_refuses_them(name):
    import warnings

    import test_dataset_devices as devices

    cpu = tack.device("cpu")
    shaped = devices._hexes(cpu)
    mixed = td.DataSet(_mixed(shaped.topology, [0, 3, 5]), shaped.geometry.values)
    devices._with_fields(mixed, cpu)
    call = devices.CALLS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name not in READY:
            with pytest.raises(NotImplementedError, match="polyhedra|Polyhedron"):
                call(mixed)
            return
        assert _summary(call(mixed)) == _summary(call(shaped))


def test_face_algorithms_on_polygon_faces_are_the_all_polyhedra_readings():
    vtk = _vtk()
    mixed = _pentagonal_prisms()
    every = vtk.vtk_to_dataset(vtk.dataset_to_vtk(mixed), polyhedral=True, dtype=tack.f64)
    assert isinstance(every.topology, td.PolyhedralTopology)
    np.testing.assert_allclose(td.cell_geometry(mixed).measure.values.numpy(),
                               td.cell_geometry(every).measure.values.numpy(), rtol=1e-12)

    def by_points(data, values):
        faces = data.topology.faces()
        out = {}
        for f in range(faces.num_faces):
            if isinstance(data.topology, td.PolyhedralTopology):
                fo, fp = data.topology.face_offsets.numpy(), data.topology.face_points.numpy()
                points = frozenset(fp[fo[f]:fo[f + 1]].tolist())
            else:
                points = next(k for k in _face_records(data.topology)
                              if _face_id(data.topology, k) == f)
            out[points] = np.atleast_1d(values[f])
        return out

    for measure in (lambda d: td.face_geometry(d).area, td.face_centers):
        mine = by_points(mixed, tack.arrays.to_host(measure(mixed).values))
        theirs = by_points(every, tack.arrays.to_host(measure(every).values))
        assert mine.keys() == theirs.keys()
        for k in mine:
            np.testing.assert_allclose(mine[k], theirs[k], rtol=1e-12, atol=1e-12)
    # A face flux of 1 out of each face's side 0: the net flux of a cell is its
    # count of side-0 faces less its side-1 faces, whatever the numbering.
    for data in (mixed, every):
        ones = td.Field(td.Values(data, "faces"), tack.ones((data.topology.faces().num_faces,)))
        sides = data.topology.faces().sides.numpy()
        expected = np.zeros(data.num_cells)
        np.add.at(expected, sides[:, 0], 1)
        np.add.at(expected, sides[sides[:, 2] >= 0, 2], -1)
        np.testing.assert_allclose(td.net_flux(data, ones).values.numpy(), expected)


def _face_id(topology, points):
    """The id of the face of ``points`` (a point set) in ``topology.faces()``."""
    faces = topology.faces()
    rows, kinds = faces.rows.numpy(), faces.kinds.numpy()
    if topology.has_polyhedra:
        fo, fp = topology.face_offsets.numpy(), topology.face_points.numpy()
    for f in range(faces.num_faces):
        row = (fp[fo[f]:fo[f + 1]].tolist() if kinds[f] == int(shapes.POLYGON)
               else [int(p) for p in rows[f] if p >= 0])
        if frozenset(row) == points:
            return f
    return None
