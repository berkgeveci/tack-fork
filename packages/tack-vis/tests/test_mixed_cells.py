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
    # Phase 2: per-cell work, each group its own way.
    "gradients", "gradients_at_points", "cell_centers", "values_at_centers", "to_cells",
    "to_points", "project", "integrate", "flow_quantities", "cell_quality", "traces",
    "jump", "upwind_flux", "divergence",
    # Phase 3: filters that build a topology.
    "as_polyhedra", "merge_points", "merge_points_polyhedra", "clip", "clip_polyhedra",
    "clip_boolean", "tetrahedralize", "extract_surface", "external_faces", "surface_normals",
    "feature_edges", "contour_2d", "contour", "slice",
}

#: Calls of `READY` whose answer is not the shaped mesh's but that of the same call
#: on ``as_polyhedra`` of it: clip makes every cut cell a polyhedron, where the
#: shapes' tables cut it into shapes, and keeps whole cells as they are.
AS_POLYHEDRA = {"clip", "clip_boolean"}

#: Calls of `READY` with tests of their own: tetrahedralize adds points.
OWN_TESTS = {"tetrahedralize"}

#: Calls of `READY` whose cells are not the shaped mesh's: a polyhedron's cut is one
#: polygon where a shape's is triangles. Its points and measure are the same.
JOINED = {"contour", "slice"}

#: Calls of `READY` whose answer, a field on the cells, is the shaped mesh's on the
#: shaped cells and the all-polyhedra mesh's on the polyhedra: measures that differ
#: between a hexahedron and the same cell as a polyhedron.
PER_CELL = {"cell_quality"}


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
        # NaN (an empty bin's mean) as None, which equals itself.
        return ("field", v.shape, tuple(None if np.isnan(a) else a
                                        for a in np.round(np.sort(v.reshape(-1)), 5)))
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
        if name in OWN_TESTS:
            call(mixed)
            return
        if name in JOINED:
            got, expected = _summary(call(mixed)), _summary(call(shaped))
            assert got[:1] + got[2:] == expected[:1] + expected[2:]
            return
        if name in AS_POLYHEDRA:
            every = devices._with_fields(td.as_polyhedra(shaped), cpu)
            assert _summary(call(mixed)) == _summary(call(every))
            return
        if name in PER_CELL:
            every = devices._with_fields(td.as_polyhedra(shaped), cpu)
            chosen = np.zeros(shaped.num_cells, bool)
            chosen[[0, 3, 5]] = True
            expected = np.where(chosen, call(every).values.numpy(), call(shaped).values.numpy())
            np.testing.assert_allclose(call(mixed).values.numpy(), expected, rtol=1e-6)
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


# ── Phase 2: per-cell work, each group its own way ──────────────────

CHOSEN = [0, 3, 5, 7, 10]


def _three_ways(seed=0):
    """A distorted mesh of hexahedra with a field ``f`` that is not linear, as
    ``(shaped, mixed, every)``: the shaped mesh, the same with the cells
    ``CHOSEN`` made polyhedra, and every cell a polyhedron (``as_polyhedra``).
    Per cell, the mixed mesh's answer is the shaped mesh's on its shapes and the
    all-polyhedra mesh's on its polyhedra."""
    shaped = _hexes()
    x = shaped.positions() + np.random.default_rng(seed).uniform(-0.2, 0.2,
                                                                 (shaped.num_points, 3))
    shaped = td.DataSet(shaped.topology, tack.array(x, tack.vec3(tack.f64)))
    mixed = td.DataSet(_mixed(shaped.topology, CHOSEN), shaped.geometry.values)
    every = td.as_polyhedra(shaped)
    f = np.sin(x[:, 0]) * x[:, 1] + x[:, 2] ** 2
    c = np.cos(np.arange(shaped.num_cells, dtype=np.float64))
    for d in (shaped, mixed, every):
        d.fields["f"] = td.Field(td.H1(d), tack.array(f))
        d.fields["c"] = td.Field(td.Constant(d), tack.array(c))
    return shaped, mixed, every


def _per_cell(shaped, every, values):
    """What a per-cell answer on the mixed mesh must be, from ``values(data)``."""
    chosen = np.zeros(shaped.num_cells, bool)
    chosen[CHOSEN] = True
    a, b = values(shaped), values(every)
    return np.where(chosen.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)


@pytest.mark.parametrize("what", [
    "gradients", "gradients of cell data", "cell_centers", "values_at_centers",
    "values_at_centers of cell data", "integral means", "node means", "cell_quality"])
def test_per_cell_answers_are_each_groups_own(what):
    shaped, mixed, every = _three_ways()
    values = {
        "gradients": lambda d: td.gradients(d, "f").values.numpy(),
        "gradients of cell data": lambda d: td.gradients(d, "c").values.numpy(),
        "cell_centers": lambda d: td.cell_centers(d).values.numpy(),
        "values_at_centers": lambda d: td.values_at_centers(d, "f").values.numpy(),
        "values_at_centers of cell data": lambda d: td.values_at_centers(d, "c").values.numpy(),
        "integral means": lambda d: td.project(d, "f", td.Constant(d)).values.numpy(),
        "node means": lambda d: td.project(d, "f", td.Constant(d),
                                           average="nodes").values.numpy(),
        "cell_quality": lambda d: np.stack([v.values.numpy()
                                            for v in td.cell_quality(d).values()], 1),
    }[what]
    np.testing.assert_allclose(values(mixed), _per_cell(shaped, every, values), rtol=1e-12,
                               atol=1e-12)


def test_integrals_sum_each_groups_own():
    shaped, mixed, every = _three_ways()
    # Each cell's integral: the shapes' by quadrature, the polyhedra's on their
    # sub-tetrahedra; a cell field's its value times the cell's measure.
    measure = lambda d: td.cell_geometry(d).measure.values.numpy()
    means = lambda d: td.project(d, "f", td.Constant(d)).values.numpy()
    expected = (_per_cell(shaped, every, means) * _per_cell(shaped, every, measure)).sum()
    got = td.integrate(mixed, ["f", "c"])
    assert got.measure == pytest.approx(_per_cell(shaped, every, measure).sum(), rel=1e-12)
    assert got.values["f"] == pytest.approx(expected, rel=1e-12)
    c = mixed.fields["c"].values.numpy()
    assert got.values["c"] == pytest.approx((c * measure(mixed)).sum(), rel=1e-12)


def test_point_links_are_the_shaped_meshs_by_cell():
    shaped, mixed, _ = _three_ways()
    by_cell = {}
    for data in (shaped, mixed):
        t = data.topology
        start_of, total = td.topology.corner_layout(t)
        owner = np.empty(total, np.int64)
        offsets, entries = (a.numpy() for a in t.point_links())
        # Each entry's cell, from the layout: a shape group's cells NUM_POINTS
        # apart, the polyhedra by their corners' offsets.
        for group in t.groups():
            if not group.count:
                continue
            cells = _group_cells(t, group)
            if group.shape.REFERENCE:
                k = group.shape.NUM_POINTS
                owner[start_of[group]:start_of[group] + k * group.count] = np.repeat(cells, k)
            else:
                corners = td.topology.polyhedral_corners(t)[0].numpy()
                owner[start_of[group] + corners[0]:start_of[group] + corners[-1]] = \
                    np.repeat(cells, np.diff(corners))
        by_cell[data is mixed] = [sorted(owner[entries[offsets[p]:offsets[p + 1]]])
                                  for p in range(data.num_points)]
    assert by_cell[True] == by_cell[False]


def _group_cells(topology, group):
    """A group's cells' global ids, by their position in it."""
    found = tack.empty((group.count,), tack.i64)
    _cell_ids_of(group.view(), found)
    return found.numpy()


@tack.kernel
def _cell_ids_of(cells, out):
    for c in cells:
        out[cells.index(c)] = cells.entity_id(c)


def test_point_values_from_cells_are_the_shaped_meshs():
    shaped, mixed, every = _three_ways()
    for data in (mixed, every):
        np.testing.assert_allclose(td.project(data, "c", td.H1(data)).values.numpy(),
                                   td.project(shaped, "c", td.H1(shaped)).values.numpy(),
                                   rtol=1e-12)


def test_point_gradients_average_each_groups_own():
    shaped, mixed, every = _three_ways()
    # A linear field's gradient is exact on every cell, so at every point.
    x = mixed.positions()
    mixed.fields["l"] = td.Field(td.H1(mixed), tack.array(x @ [2.0, -1.0, 0.5]))
    np.testing.assert_allclose(td.gradients(mixed, "l", at="points").values.numpy(),
                               np.tile([2.0, -1.0, 0.5], (mixed.num_points, 1)), atol=1e-12)
    # A point among shapes alone has the shaped mesh's, among polyhedra alone the
    # all-polyhedra mesh's.
    kinds = {}
    for c, row in enumerate(shaped.topology.connectivity.numpy().reshape(-1, 8)):
        for p in row:
            kinds.setdefault(int(p), set()).add(c in CHOSEN)
    got, a, b = (td.gradients(d, "f", at="points").values.numpy()
                 for d in (mixed, shaped, every))
    shapes_only = [p for p, k in kinds.items() if k == {False}]
    polyhedra_only = [p for p, k in kinds.items() if k == {True}]
    assert shapes_only and polyhedra_only
    np.testing.assert_allclose(got[shapes_only], a[shapes_only], rtol=1e-12)
    np.testing.assert_allclose(got[polyhedra_only], b[polyhedra_only], rtol=1e-12)


def test_traces_are_the_shaped_meshs_face_by_face():
    shaped, mixed, _ = _three_ways()

    def by_points(data, field):
        values = td.project(data, field, td.SideTraces(data)).values.numpy().reshape(-1, 2, 4)
        rows, sides = data.topology.faces().rows.numpy(), data.topology.faces().sides.numpy()
        out = {}
        for f, row in enumerate(rows):
            for s in (0, 1):
                if sides[f, 2 * s] >= 0:
                    n = int((row >= 0).sum())
                    out[frozenset(row[:n].tolist()), int(sides[f, 2 * s])] = \
                        dict(zip(row[:n].tolist(), values[f, s, :n]))
        return out

    for field in ("f", "c"):
        mine, theirs = by_points(mixed, field), by_points(shaped, field)
        assert mine.keys() == theirs.keys()
        for k in mine:
            assert mine[k].keys() == theirs[k].keys()
            np.testing.assert_allclose(list(mine[k].values()),
                                       [theirs[k][p] for p in mine[k]], rtol=1e-12)
    for name in ("f", "c"):
        np.testing.assert_allclose(np.sort(td.jump(mixed, name).values.numpy()),
                                   np.sort(td.jump(shaped, name).values.numpy()), atol=1e-12)
    flux = lambda d: td.divergence(d, td.upwind_flux(d, "f", (1.0, 0.5, -0.25))).values.numpy()
    np.testing.assert_allclose(flux(mixed), flux(shaped), rtol=1e-12)


def test_traces_refuse_polygon_faces():
    data = _pentagonal_prisms()
    x = data.positions()
    data.fields["u"] = td.Field(td.H1(data), tack.array(x[:, 0].copy()))
    with pytest.raises(NotImplementedError, match="more than four points"):
        td.project(data, "u", td.SideTraces(data))


def test_point_data_crosses_polygon_faces_without_traces():
    # Point data beside polyhedra is continuous: no jump, and the upwind value at
    # a face's center its points' mean, on pentagons as on quads.
    data = _pentagonal_prisms()
    x = data.positions()
    u = np.sin(x[:, 0]) + x[:, 1] * x[:, 2]
    data.fields["u"] = td.Field(td.H1(data), tack.array(u))
    faces = data.topology.faces()
    assert not td.jump(data, "u").values.numpy().any()
    v = np.array([1.0, 0.5, -0.25])
    normals, areas = (f.values.numpy() for f in td.face_geometry(data))
    means = np.array([u[sorted(_face_points(data.topology, f))].mean()
                      for f in range(faces.num_faces)])
    np.testing.assert_allclose(td.upwind_flux(data, "u", tuple(v)).values.numpy(),
                               normals @ v * areas * means, rtol=1e-12, atol=1e-14)


def _face_points(topology, f):
    """Face ``f``'s point ids, from its row or, for a polygon, its stored walk."""
    faces = topology.faces()
    if faces.kinds.numpy()[f] == int(shapes.POLYGON):
        fo, fp = topology.face_offsets.numpy(), topology.face_points.numpy()
        return fp[fo[f]:fo[f + 1]].tolist()
    return [int(p) for p in faces.rows.numpy()[f] if p >= 0]


def test_a_mixed_mesh_refuses_spaces_that_need_reference_elements():
    _, mixed, _ = _three_ways()
    for space in (td.H1(mixed, order=2), td.L2(mixed)):
        with pytest.raises(NotImplementedError, match="needs a reference element"):
            td.project(mixed, "f", space)


def test_hidden_points_hide_the_polyhedra_around_them():
    from tack.dataset.ghosts import hidden_by_points

    _, mixed, _ = _three_ways()
    marks = np.zeros(mixed.num_points, np.uint8)
    marks[0] = td.Ghosts.HIDDENPOINT          # a corner of cell 0, a polyhedron, alone
    ghosts = hidden_by_points(mixed.topology, td.Ghosts(points=tack.array(marks)))
    hidden = (ghosts.cells.numpy() & td.Ghosts.HIDDENCELL) != 0
    assert hidden.tolist() == [c == 0 for c in range(mixed.num_cells)]


def _exploded(data):
    """``data`` (mixed) with every cell on its own copies of its points, and every
    polyhedron's faces its own one-sided copies, wound out of it: what pieces
    written without merging look like, cell by cell."""
    t = data.topology
    offsets, conn = t.offsets.numpy(), t.connectivity.numpy()
    fo, fp = t.face_offsets.numpy(), t.face_points.numpy()
    cfo, cf, cs = (a.numpy() for a in (t.cell_face_offsets, t.cell_faces, t.cell_face_sides))
    face_points, face_offsets, cell_faces, cell_face_offsets = [], [0], [], [0]
    for c in range(t.num_cells):
        own = {int(p): offsets[c] + j for j, p in enumerate(conn[offsets[c]:offsets[c + 1]])}
        for k in range(cfo[c], cfo[c + 1]):
            walk = fp[fo[cf[k]]:fo[cf[k] + 1]]
            walk = walk if cs[k] == 0 else walk[::-1]
            cell_faces.append(len(face_offsets) - 1)
            face_points.extend(own[int(p)] for p in walk)
            face_offsets.append(len(face_points))
        cell_face_offsets.append(len(cell_faces))
    topology = td.UnstructuredTopology(
        t.types.numpy(), offsets, np.arange(len(conn)), num_points=len(conn),
        polyhedra=(np.array(face_offsets), np.array(face_points), np.array(cell_face_offsets),
                   np.array(cell_faces), np.zeros(len(cell_faces), np.uint8)))
    return td.DataSet(topology, tack.array(data.positions()[conn], tack.vec3(tack.f64)))


# ── Phase 3: filters that build a topology ──────────────────────────

def test_as_polyhedra_of_a_mixed_mesh_is_the_all_polyhedra_mesh():
    shaped, mixed, every = _three_ways()
    out = td.as_polyhedra(mixed)
    assert isinstance(out.topology, td.PolyhedralTopology)
    for values in (lambda d: td.cell_geometry(d).measure.values.numpy(),
                   lambda d: td.gradients(d, "f").values.numpy()):
        np.testing.assert_allclose(values(out), values(every), rtol=1e-12)
    prisms = _pentagonal_prisms()
    np.testing.assert_allclose(td.cell_geometry(td.as_polyhedra(prisms)).measure.values.numpy(),
                               td.cell_geometry(prisms).measure.values.numpy(), rtol=1e-12)


def test_merging_an_exploded_mixed_mesh_gives_it_back():
    _, mixed, _ = _three_ways()
    exploded = _exploded(mixed)
    assert exploded.topology.faces().num_faces > mixed.topology.faces().num_faces
    merged = td.merge_points(exploded)
    merged.topology.validate()
    assert merged.num_points == mixed.num_points
    assert merged.topology.faces().num_faces == mixed.topology.faces().num_faces
    assert merged.topology.faces().boundary().shape[0] == \
        mixed.topology.faces().boundary().shape[0]
    # The same faces, the merged points numbered otherwise: by their positions.
    def faces_at(data):
        x = data.positions()
        return {frozenset(tuple(x[p]) for p in points) for points in _face_records(data.topology)}

    assert faces_at(merged) == faces_at(mixed)
    np.testing.assert_allclose(td.cell_geometry(merged).measure.values.numpy(),
                               td.cell_geometry(mixed).measure.values.numpy(), rtol=1e-12)


def test_check_winding_lists_the_polyhedra_wound_wrong():
    _, mixed, _ = _three_ways()
    assert td.check_winding(mixed).ids.numpy().size == 0
    t = mixed.topology
    sides = t.cell_face_sides.numpy().copy()
    sides[t.cell_face_offsets.numpy()[CHOSEN[1]]] ^= 1
    wrong = td.UnstructuredTopology(t.types, t.offsets, t.connectivity, num_points=t.num_points,
                                    polyhedra=(t.face_offsets, t.face_points,
                                               t.cell_face_offsets, t.cell_faces,
                                               tack.array(sides)))
    assert td.check_winding(wrong).ids.numpy().tolist() == [CHOSEN[1]]


@pytest.mark.parametrize("inside", [False, True])
@pytest.mark.parametrize("level", [0.9, 1.7, 3.2])
def test_clipping_a_mixed_mesh_keeps_whole_shapes_and_cuts_into_polyhedra(level, inside):
    shaped, mixed, every = _three_ways()
    out, expected = (td.clip(d, "f", level, inside=inside) for d in (mixed, every))
    out.topology.validate()
    # Cut cells are the all-polyhedra clip's; whole shapes stay shapes, whose
    # faces the polyhedra beside them share.
    assert (out.num_cells, out.num_points) == (expected.num_cells, expected.num_points)
    assert out.topology.faces().boundary().shape[0] == \
        expected.topology.faces().boundary().shape[0]
    assert td.cell_geometry(out).measure.values.numpy().sum() == pytest.approx(
        td.cell_geometry(expected).measure.values.numpy().sum(), rel=1e-12)
    np.testing.assert_allclose(np.sort(out.positions(), 0), np.sort(expected.positions(), 0),
                               atol=1e-12)
    f = mixed.fields["f"].values.numpy()
    corners = mixed.topology.connectivity.numpy().reshape(-1, 8)
    kept = (f >= level) if not inside else (f <= level)
    whole = [c for c in range(mixed.num_cells) if c not in CHOSEN and kept[corners[c]].all()]
    types = out.topology.types.numpy()
    assert (types == 12).sum() == len(whole)
    assert set(types.tolist()) <= {12, POLYHEDRON}


@pytest.mark.parametrize("make", ["straight", "distorted", "prisms"])
def test_tetrahedralized_polyhedra_meet_their_neighbours_face_to_face(make):
    if make == "straight":
        shaped = _hexes()
        data = td.DataSet(_mixed(shaped.topology, CHOSEN), shaped.geometry.values)
        data.fields["f"] = td.Field(td.H1(data), tack.array(data.positions()[:, 0] ** 2))
    elif make == "distorted":
        shaped, data, _ = _three_ways()
    else:
        shaped, data = None, _pentagonal_prisms()
        data.fields["f"] = td.Field(td.H1(data), tack.array(data.positions()[:, 1] ** 2))
    out = td.tetrahedralize(data)
    out.topology.validate()
    assert set(out.topology.types.numpy().tolist()) == {int(shapes.TETRA)}
    volumes = td.cell_geometry(out).measure.values.numpy()
    assert volumes.min() > 0
    # Face to face: the boundary is the input's, each quad split in two; the
    # shaped mesh tetrahedralized splits its boundary alike.
    boundary = out.topology.faces().boundary().shape[0]
    if shaped is not None:
        assert boundary == td.tetrahedralize(shaped).topology.faces().boundary().shape[0]
    if make != "distorted":         # planar faces: the volume is the input's
        assert volumes.sum() == pytest.approx(
            td.cell_geometry(data).measure.values.numpy().sum(), rel=1e-12)
    # The input points as they were, then each added point at its points' mean,
    # with their values' mean.
    n = data.num_points
    f = data.fields["f"].values.numpy()
    np.testing.assert_allclose(out.positions()[:n], data.positions())
    np.testing.assert_array_equal(out.fields["f"].values.numpy()[:n], f)
    t = data.topology
    conn, offsets = t.connectivity.numpy(), t.offsets.numpy()
    polyhedra = [c for c in range(t.num_cells) if t.types.numpy()[c] == POLYHEDRON]
    for k, c in enumerate(polyhedra):
        points = conn[offsets[c]:offsets[c + 1]]
        np.testing.assert_allclose(out.positions()[n + k], data.positions()[points].mean(0))
        assert out.fields["f"].values.numpy()[n + k] == pytest.approx(f[points].mean())


# ── 2D: polygons beside triangles and quads ─────────────────────────

POLYGON = int(shapes.POLYGON)


def _as_polygons_beside(data, chosen):
    """``data`` (a surface of triangles and quads) with the cells ``chosen`` made
    polygons of the same loops: a triangle's and a quad's points go round them."""
    t = data.topology
    types = t.types.numpy().copy()
    types[list(chosen)] = POLYGON
    topology = td.UnstructuredTopology(types, t.offsets.numpy(), t.connectivity.numpy(),
                                       num_points=t.num_points)
    out = td.DataSet(topology, data.geometry.values)
    for key, field in data.fields.items():
        if isinstance(field.space, td.H1):
            out.fields[key] = td.Field(td.H1(out), field.values)
    return out


def _quads(n=(5, 4), z=None):
    """A surface of quads over an ``n``-point grid in the plane z = 0 (or at heights
    ``z(x, y)``), x fastest."""
    nx, ny = n
    pts = np.array([[i, j, 0.0] for j in range(ny) for i in range(nx)], np.float64)
    if z is not None:
        pts[:, 2] = z(pts[:, 0], pts[:, 1])
    pid = lambda i, j: i + nx * j
    conn = [pid(i + a, j + b) for j in range(ny - 1) for i in range(nx - 1)
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))]
    cells = len(conn) // 4
    topology = td.UnstructuredTopology(np.full(cells, 9, np.uint8),
                                       np.arange(0, 4 * cells + 1, 4), np.array(conn),
                                       num_points=len(pts))
    return td.DataSet(topology, tack.array(pts, tack.vec3(tack.f64)))


def _surfaces(z=None):
    """A flat surface of 12 quads with the device sweep's fields, and the same with
    some quads made polygons. Flat: an edge's normal is then the opposite of
    itself seen from its other cell, so fluxes do not depend on which of the two
    is its side 0, which the shaped and the mixed derivations choose apart."""
    import test_dataset_devices as devices

    cpu = tack.device("cpu")
    shaped = devices._with_fields(_quads(z=z), cpu)
    mixed = devices._with_fields(_as_polygons_beside(shaped, [0, 3, 5, 10]), cpu)
    return shaped, mixed


def test_a_surface_of_quads_and_polygons_has_the_quads_faces():
    shaped, mixed = _surfaces()
    t = mixed.topology
    assert t.mixed and t.has_polygons and not t.has_polyhedra
    assert sorted(g.shape.__name__ for g in t.groups()) == ["Polygon", "Quad"]
    assert t.faces().num_faces == shaped.topology.faces().num_faces
    assert _cells_by_face(t) == _cells_by_face(shaped.topology)
    assert t.faces().boundary().shape[0] == shaped.topology.faces().boundary().shape[0]
    t.validate()


def test_polygons_wound_against_their_neighbours_are_refused():
    shaped, _ = _surfaces()
    t = shaped.topology
    types, conn = t.types.numpy().copy(), t.connectivity.numpy().copy()
    offsets = t.offsets.numpy()
    types[[0, 1]] = POLYGON
    # Turn every cell but polygon 0 round: polygon 0 now walks its edges the way
    # its neighbours do.
    for c in range(1, t.num_cells):
        conn[offsets[c]:offsets[c + 1]] = conn[offsets[c]:offsets[c + 1]][::-1]
    turned = td.UnstructuredTopology(types, offsets, conn, num_points=t.num_points)
    with pytest.raises(ValueError, match="wound|either direction|same way"):
        turned.faces()


@pytest.mark.parametrize("name", sorted(__import__("test_dataset_devices").CALLS))
def test_an_algorithm_takes_polygons_beside_shapes_or_refuses_them(name):
    import warnings

    import test_dataset_devices as devices

    shaped, mixed = _surfaces()
    call = CALLS_2D.get(name, devices.CALLS[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            expected = call(shaped)
        except Exception as error:             # what a surface of shapes refuses
            with pytest.raises((type(error), NotImplementedError)):
                call(mixed)
            return
        if name not in READY_2D:
            with pytest.raises(NotImplementedError, match="polyhedra|Polyhedron|polygon"):
                call(mixed)
            return
        got = call(mixed)
        if name in ORIENTED_2D:
            # Values on faces out of their side 0, which the two derivations may
            # choose apart: each face's by its points, out of the same cell.
            assert _out_of(shaped, expected) == pytest.approx(_out_of(mixed, got))
            return
        if name in POSITIONS_2D:
            np.testing.assert_allclose(np.sort(got.positions(), 0),
                                       np.sort(expected.positions(), 0), atol=1e-12)
            return
        assert _summary(got) == _summary(expected)


def _out_of(data, field):
    """A field on faces as ``{(face's points, cell on its side 0): value}``, the value
    turned to be out of the face's lower cell: what does not depend on which side
    is which."""
    faces = data.topology.faces()
    sides = faces.sides.numpy()
    values = np.asarray(tack.arrays.to_host(field.values), float)
    out = {}
    for points, (_, s) in _face_records(data.topology).items():
        f = _face_id(data.topology, points)
        cells = [c for c in (s[0], s[2]) if c >= 0]
        sign = 1.0 if sides[f, 0] == min(cells) else -1.0
        out[points] = sign * values[f]
    return out


#: The calls of `test_dataset_devices.CALLS` that take a surface of polygons beside
#: triangles and quads, each held to the shaped surface's answer.
READY_2D = set(READY) - {"clip", "clip_boolean", "clip_polyhedra", "as_polyhedra",
                         "merge_points_polyhedra"} | {"triangulate", "as_polygons"}

#: Calls made otherwise on surfaces: perot of a flux out of each face's side 0
#: (the sweep's ones are not turned with the sides, which the shaped and mixed
#: derivations may choose apart).
CALLS_2D = {"perot": lambda d: td.perot(d, td.upwind_flux(d, "s", (1.0, 0.5, 0.0)))}

#: ... whose values on faces are out of side 0 (`_out_of`); perot's of a field
#: given on the faces, which is not turned with them, only where both choose alike.
ORIENTED_2D = {"upwind_flux"}

#: ... compared by their positions: a map that curves the cells measures a polygon
#: (fanned) and a quad (bilinear) apart.
POSITIONS_2D = {"spherical", "cylindrical"}


def _merged_quads():
    """`_quads` with its first two quads one polygon of six points (two of them on
    its straight bottom and top), the other quads as they were."""
    shaped = _quads()
    t = shaped.topology
    conn = t.connectivity.numpy().reshape(-1, 4)
    a, b = conn[0], conn[1]                          # (0,0) (1,0) (1,1) (0,1), and right
    loop = [a[0], a[1], b[1], b[2], b[3], a[3]]
    cells = [loop] + [list(q) for q in conn[2:]]
    types = [POLYGON] + [9] * (len(cells) - 1)
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in cells])])
    topology = td.UnstructuredTopology(np.array(types, np.uint8), offsets,
                                       np.concatenate(cells), num_points=t.num_points)
    out = td.DataSet(topology, shaped.geometry.values)
    x = out.positions()
    out.fields["u"] = td.Field(td.H1(out), tack.array(np.sin(x[:, 0]) + x[:, 1] ** 2))
    return shaped, out


def test_a_six_point_polygon_shares_its_edges_with_the_quads_beside_it():
    shaped, merged = _merged_quads()
    faces = merged.topology.faces()
    # One edge fewer, the one the two quads shared; the boundary is the same.
    assert faces.num_faces == shaped.topology.faces().num_faces - 1
    assert faces.boundary().shape[0] == shaped.topology.faces().boundary().shape[0]
    # The polygon's six edges: its bottom two and its left on the boundary, its
    # right and its top two each shared with a quad.
    mine = [cells for cells in _cells_by_face(merged.topology).values()
            if 0 in {c for c, _ in cells}]
    assert sorted(len(cells) for cells in mine) == [1, 1, 1, 2, 2, 2]
    merged.topology.validate()
    np.testing.assert_allclose(td.cell_geometry(merged).measure.values.numpy(),
                               [2.0] + [1.0] * (merged.num_cells - 1))
    x = merged.positions()
    merged.fields["l"] = td.Field(td.H1(merged), tack.array(x @ [2.0, -1.0, 0.0]))
    np.testing.assert_allclose(td.gradients(merged, "l").values.numpy(),
                               np.tile([2.0, -1.0, 0.0], (merged.num_cells, 1)), atol=1e-12)
    # The integral of 2x - y over the rectangle [0, 4] x [0, 3].
    assert td.integrate(merged, "l").values["l"] == pytest.approx(2 * 8 * 3 - 4.5 * 4)


def test_triangulating_polygons_beside_quads():
    shaped, merged = _merged_quads()
    out = td.triangulate(merged)
    out.topology.validate()
    assert set(out.topology.types.numpy().tolist()) == {int(shapes.TRIANGLE)}
    # A quad's two triangles, the six-point polygon's four.
    assert out.num_cells == 2 * (merged.num_cells - 1) + 4
    assert td.cell_geometry(out).measure.values.numpy().sum() == pytest.approx(12.0)
    assert td.cell_geometry(out).measure.values.numpy().min() > 0
    np.testing.assert_array_equal(out.fields["u"].values.numpy(),
                                  merged.fields["u"].values.numpy())
    # The quads split as a shape-based surface's are.
    quads = td.triangulate(td.extract_cells(shaped, list(range(2, shaped.num_cells))))
    kept = td.triangulate(td.extract_cells(merged, list(range(1, merged.num_cells))))
    np.testing.assert_array_equal(kept.topology.connectivity.numpy(),
                                  quads.topology.connectivity.numpy())


def test_as_polygons_of_polygons_beside_quads_keeps_every_loop():
    _, merged = _merged_quads()
    out = td.as_polygons(merged)
    assert isinstance(out.topology, td.PolygonalTopology)
    offsets, loops = out.topology.loops()
    np.testing.assert_array_equal(offsets.numpy(), merged.topology.offsets.numpy())
    np.testing.assert_array_equal(loops.numpy(), merged.topology.connectivity.numpy())


@pytest.mark.parametrize("level", [1.7, 3.2, (1.2, 2.5)])
def test_a_mixed_contour_is_each_cells_own_and_conforms(level):
    shaped, mixed, every = _three_ways()
    for d in (shaped, mixed, every):
        d.fields["id"] = td.Field(td.Constant(d), tack.array(np.arange(d.num_cells, dtype=float)))
    out = {k: td.contour(d, "f", level) for k, d in (("s", shaped), ("m", mixed), ("e", every))}

    def area_by_cell(o):
        return np.bincount(o.fields["id"].values.numpy().astype(int),
                           td.cell_geometry(o).measure.values.numpy(), shaped.num_cells)

    got = out["m"]
    got.topology.validate()
    assert got.topology.mixed
    assert set(got.topology.types.numpy().tolist()) == {int(shapes.TRIANGLE), POLYGON}
    # The shapes' triangles and the polyhedra's polygons, each cell's as the shaped
    # and the all-polyhedra contours have it.
    np.testing.assert_allclose(area_by_cell(got),
                               _per_cell(shaped, every, lambda d: area_by_cell(out[
                                   "s" if d is shaped else "e"])), rtol=1e-12, atol=1e-14)
    # One point per crossing, shared across the faces between shapes and polyhedra:
    # no open edges but the mesh boundary's.
    assert got.num_points == out["s"].num_points
    assert got.topology.faces().boundary().shape[0] == \
        out["s"].topology.faces().boundary().shape[0]
