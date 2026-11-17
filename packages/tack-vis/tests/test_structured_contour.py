# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``contour`` of structured grids by flying edges (issue #316).

Images, rectilinear and curvilinear grids of solid cells are contoured by
flying edges (``algorithms._flying_edges``): crossings counted along the grid's
rows, numbered by a scan and written once, with no per-corner arrays and no
sort. Its output must be the general path's up to the order of points and
triangles -- the same triangles of the same cells, the same crossings to the
bit, the same values carried -- so each case here contours a grid both ways
(the general path by switching flying edges off) and compares the triangles
in the general path's order: by cell, then value, then in the case table's.

``slice`` of an image or a rectilinear grid by a ``Plane`` goes by the plane
cutter (``_flying_edges.cut``): flying edges over each row's stretch near the
plane, the distance computed where read. It must give the general
``slice_plane``'s output in the same sense, for planes through grid points,
along and across the rows, of either sign, several at once, and missing the
grid.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes
from tack.dataset.algorithms import _flying_edges

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def _general(monkeypatch, *args, **kwargs):
    with monkeypatch.context() as m:
        m.setattr(_flying_edges, "serves", lambda data, field: False)
        return td.contour(*args, **kwargs)


def _triangles(out, names):
    """Each triangle's corners' positions and the named fields at its corners
    (point fields) or on it (cell fields), in the general path's order."""
    width = 2 if out.topology.groups()[0].shape is shapes.Line else 3
    rows = out.topology.connectivity.numpy().reshape(-1, width).astype(np.int64)
    cells = out.fields["cell"].values.numpy()
    levels = (out.fields["contour_value"].values.numpy() if "contour_value" in out.fields
              else np.zeros(len(rows)))
    order = np.lexsort((levels, cells))      # stable: the case table's order within
    corners = out.geometry.values.numpy()[rows[order]]
    fields = {}
    for name in names:
        values = out.fields[name].values.numpy()
        fields[name] = values[rows[order]] if out.fields[name].space.on == "points" \
            else values[order]
    return corners, fields


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def _check(monkeypatch, data, value, names=("s",)):
    data.fields["cell"] = td.Field(td.Constant(data), tack.array(
        np.arange(data.num_cells, dtype=np.int64), tack.i64))
    names = [*names, "cell"]
    fast = td.contour(data, "s", value)
    slow = _general(monkeypatch, data, "s", value)
    assert fast.num_cells == slow.num_cells and fast.num_points == slow.num_points
    assert fast.num_cells > 0
    a, fa = _triangles(fast, names)
    b, fb = _triangles(slow, names)
    assert _same(a, b)
    for name in names:
        assert _same(fa[name], fb[name]), name
    if np.ndim(value):
        assert _same(np.sort(fast.fields["contour_value"].values.numpy()),
                     np.sort(slow.fields["contour_value"].values.numpy()))
    return fast


def _dtypes():
    from tack.runtime.dispatch import get_backend

    return [tack.f32] + ([tack.f64] if get_backend().supports_f64 else [])


def _with(data, values, dtype, name="s"):
    data.fields[name] = td.Field(td.H1(data), tack.array(np.asarray(values).astype(
        dtype.numpy_dtype)))
    return data


@pytest.mark.parametrize("several", [False, True])
def test_an_image_contours_as_the_general_path_does(backend, monkeypatch, several):
    for dtype in _dtypes():
        data = td.wavelet((-8, 8, -7, 7, -6, 6), dtype=dtype)
        data.fields["s"] = data.fields["RTData"]
        x = data.geometry.values.numpy()
        data.fields["v"] = td.Field(td.H1(data), tack.array(x.astype(dtype.numpy_dtype),
                                                           tack.vec(3, dtype)))
        _check(monkeypatch, data, [100.0, 150.0, 200.0, 250.0] if several else 150.0,
               names=("s", "v"))


def test_crossings_that_land_on_points_are_those_points(backend, monkeypatch):
    # Values rounded to tenths meet the levels exactly at many points: every
    # crossing there is the point, shared by the edges around it, and a
    # triangle two of whose crossings land on one point is dropped.
    for dtype in _dtypes():
        data = td.uniform_grid((9, 7, 6), spacing=(0.5, 0.3, 0.8), dtype=dtype)
        x = data.geometry.values.numpy().astype(float)
        out = _check(monkeypatch, _with(data, np.round(
            np.sin(x[:, 0]) + np.cos(2 * x[:, 1]) + 0.3 * x[:, 2], 1), dtype), [0.5, 1.0, 1.2])
        assert out.num_points < 3 * out.num_cells


def test_cells_with_a_nan_have_no_surface(backend, monkeypatch):
    rng = np.random.default_rng(4)
    data = td.uniform_grid((9, 7, 6))
    x = data.geometry.values.numpy().astype(float)
    values = np.sin(x[:, 0]) + np.cos(x[:, 1]) + 0.2 * x[:, 2]
    values[rng.integers(0, len(values), 20)] = np.nan
    out = _check(monkeypatch, _with(data, values, tack.f32), [0.5, 1.0])
    # Every point is a triangle's: a crossing whose cells all hold a NaN is not kept.
    used = np.unique(out.topology.connectivity.numpy())
    assert len(used) == out.num_points


def test_rectilinear_and_curvilinear_grids_contour_alike(backend, monkeypatch):
    rng = np.random.default_rng(5)
    for dtype in _dtypes():
        axes = [np.cumsum(rng.uniform(0.1, 1.0, n)) for n in (10, 8, 7)]
        grid = td.rectilinear_grid(*axes, dtype=dtype)
        x = grid.geometry.values.numpy().astype(float)
        _check(monkeypatch, _with(grid, x[:, 0] ** 2 + x[:, 1] - x[:, 2], dtype), [5.0, 10.0])
        image = td.uniform_grid((8, 7, 6), dtype=dtype)
        x = image.geometry.values.numpy().astype(float)
        bent = x + 0.2 * np.sin(x[:, [1, 2, 0]])
        curvilinear = td.DataSet(image.topology, tack.array(bent.astype(dtype.numpy_dtype),
                                                            tack.vec(3, dtype)))
        _check(monkeypatch, _with(curvilinear, np.linalg.norm(bent - 3.0, axis=1), dtype),
               [2.0, 3.0])


def test_ghost_marks_are_carried(backend, monkeypatch):
    pieces = td.ghost_cells(td.split(td.wavelet(8), (2, 1, 1)))
    piece = pieces[0]
    piece.fields["s"] = piece.fields["RTData"]
    out = _check(monkeypatch, piece, 150.0)
    general = _general(monkeypatch, piece, "s", 150.0)
    marks = out.ghosts.cells.numpy()
    order = np.lexsort((np.zeros(out.num_cells), out.fields["cell"].values.numpy()))
    reference = general.ghosts.cells.numpy()
    reference_order = np.lexsort((np.zeros(general.num_cells),
                                  general.fields["cell"].values.numpy()))
    assert _same(marks[order], reference[reference_order]) and marks.any()


def test_what_flying_edges_does_not_take_goes_the_general_way(backend):
    image = td.uniform_grid((6, 5, 4))
    x = image.geometry.values.numpy()
    image = _with(image, x[:, 0] + x[:, 1], tack.f32)
    assert _flying_edges.serves(image, image.fields["s"])
    cell_values = td.uniform_grid((6, 5, 4))
    cell_values.fields["s"] = td.Field(td.Constant(cell_values), tack.array(
        np.arange(cell_values.num_cells, dtype=np.float32)))
    tets = td.tetrahedralize(image)
    for data in (cell_values, tets):
        assert not _flying_edges.serves(data, data.fields["s"])
    # Unmerged points go the general way too: each triangle its own.
    out = td.contour(image, "s", 4.5, merge_points=False)
    assert out.num_points == 3 * out.num_cells


# ── The plane cutter: slice of images and rectilinear grids by a Plane ──

def _general_slice(monkeypatch, *args, **kwargs):
    with monkeypatch.context() as m:
        m.setattr(_flying_edges, "cuts", lambda data, function: False)
        m.setattr(_flying_edges, "serves", lambda data, field: False)
        return td.slice_plane(*args, **kwargs)


def _check_cut(monkeypatch, data, origin, normal, value=0.0):
    assert _flying_edges.cuts(data, td.Plane(origin, normal))
    data.fields["cell"] = td.Field(td.Constant(data), tack.array(
        np.arange(data.num_cells, dtype=np.int64), tack.i64))
    x = data.geometry.values.numpy()
    data.fields["p"] = td.Field(td.H1(data), tack.array(np.sin(x).astype(data.dtype.numpy_dtype),
                                                       tack.vec(3, data.dtype)))
    fast = td.slice_plane(data, origin, normal, value=value)
    slow = _general_slice(monkeypatch, data, origin, normal, value=value)
    assert fast.num_cells == slow.num_cells and fast.num_points == slow.num_points
    if fast.num_cells:
        a, fa = _triangles(fast, ["p", "cell"])
        b, fb = _triangles(slow, ["p", "cell"])
        assert _same(a, b) and _same(fa["p"], fb["p"]) and _same(fa["cell"], fb["cell"])
    return fast


@pytest.mark.parametrize("plane", [
    ((4.3, 3.1, 2.2), (0.2, 0.3, 1.0)),             # tilted
    ((3.0, 0.0, 0.0), (1.0, 0.0, 0.0)),             # through grid points, across the rows
    ((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),             # through grid points, along the rows
    ((2.0, 2.0, 2.0), (-1.0, 0.5, -0.25)),          # a negative normal
])
def test_a_plane_cuts_an_image_as_the_general_path_does(backend, monkeypatch, plane):
    for dtype in _dtypes():
        image = td.uniform_grid((20, 17, 15), spacing=(0.5, 0.4, 0.3), dtype=dtype)
        assert _check_cut(monkeypatch, image, *plane).num_cells > 0


def test_planes_cut_rotated_far_and_rectilinear_grids_alike(backend, monkeypatch):
    rng = np.random.default_rng(3)
    rotation = np.array([[np.cos(0.4), -np.sin(0.4), 0.0], [np.sin(0.4), np.cos(0.4), 0.0],
                         [0.0, 0.0, 1.0]])
    for dtype in _dtypes():
        rotated = td.uniform_grid((12, 11, 10), spacing=(0.5, 0.4, 0.3), direction=rotation,
                                  origin=(1.0, 2.0, 3.0), dtype=dtype)
        _check_cut(monkeypatch, rotated, (3.0, 4.0, 4.0), (0.3, -0.2, 1.0))
        far = td.uniform_grid((12, 11, 10), origin=(1e4, -2e3, 5e2), spacing=(0.25, 0.5, 1.0),
                              dtype=dtype)
        _check_cut(monkeypatch, far, (1e4 + 1.3, -2e3 + 2.1, 5e2 + 4.0), (0.1, 0.7, 0.3))
        axes = [np.cumsum(rng.uniform(0.1, 1.0, n)) for n in (14, 12, 11)]
        grid = td.rectilinear_grid(*axes, dtype=dtype)
        _check_cut(monkeypatch, grid, [a.mean() for a in axes], (0.4, -0.3, 0.8))
        _check_cut(monkeypatch, grid, (axes[0][5], 0.0, 0.0), (1.0, 0.0, 0.0))
        _check_cut(monkeypatch, grid, (0.0, axes[1][4], 0.0), (0.0, 1.0, 0.0))


def test_parallel_planes_in_one_pass_and_a_plane_missing_the_grid(backend, monkeypatch):
    image = td.uniform_grid((20, 17, 15), spacing=(0.5, 0.4, 0.3))
    several = _check_cut(monkeypatch, image, (4.3, 3.1, 2.2), (0.2, 0.3, 1.0),
                         value=[-1.0, 0.0, 0.7])
    assert len(np.unique(several.fields["contour_value"].values.numpy())) == 3
    assert _check_cut(monkeypatch, image, (100.0, 0.0, 0.0), (1.0, 0.2, 0.0)).num_cells == 0


def test_what_the_cutter_does_not_take_goes_the_general_way(backend):
    plane = td.Plane((1.0, 1.0, 1.0), (0.3, 0.2, 1.0))
    image = td.uniform_grid((5, 4, 3))
    curvilinear = td.DataSet(image.topology, tack.array(image.geometry.values.numpy(),
                                                       tack.vec(3, tack.f32)))
    for data in (image, curvilinear, _hidden(td.uniform_grid((5, 4, 3)), [0])):
        assert _flying_edges.cuts(data, plane)
    for data, function in ((td.tetrahedralize(image), plane),
                           (image, td.Sphere((1.0, 1.0, 1.0), 1.0)),
                           (td.uniform_grid((5, 4)), plane)):
        assert not _flying_edges.cuts(data, function)


# ── Hidden and refined cells, left out in place ─────────────────────

def _hidden(data, cells=None, seed=6):
    """``data`` with some cells hidden, refined, or both (left out), and some
    duplicates (kept, their marks carried)."""
    n = data.num_cells
    marks = np.zeros(n, np.uint8)
    if cells is None:
        rng = np.random.default_rng(seed)
        marks[rng.random(n) < 0.15] = td.Ghosts.HIDDENCELL
        marks[rng.random(n) < 0.15] |= td.Ghosts.REFINEDCELL
        marks[rng.random(n) < 0.1] |= td.Ghosts.DUPLICATECELL
    else:
        marks[cells] = td.Ghosts.HIDDENCELL
    data.ghosts = td.Ghosts(cells=tack.array(marks, tack.u8))
    return data


def _cell_marks(out):
    """The output's cell marks in the general path's order of triangles."""
    order = np.lexsort((np.zeros(out.num_cells), out.fields["cell"].values.numpy()))
    return out.ghosts.cells.numpy()[order]


@pytest.mark.parametrize("kind", ["image", "rectilinear", "curvilinear"])
def test_hidden_and_refined_cells_are_left_out_in_place(backend, monkeypatch, kind):
    # The general path extracts the visible cells (an unstructured mesh) and
    # contours that; flying edges skips them in place, to the same output.
    rng = np.random.default_rng(7)
    for dtype in _dtypes():
        grid = td.uniform_grid((14, 12, 10), spacing=(0.5, 0.4, 0.3), dtype=dtype)
        if kind == "rectilinear":
            grid = td.rectilinear_grid(*[np.cumsum(rng.uniform(0.1, 0.6, n))
                                         for n in (14, 12, 10)], dtype=dtype)
        elif kind == "curvilinear":
            x = grid.geometry.values.numpy().astype(float)
            grid = td.DataSet(grid.topology, tack.array(
                (x + 0.05 * np.sin(3 * x[:, [1, 2, 0]])).astype(dtype.numpy_dtype),
                tack.vec(3, dtype)))
        x = grid.geometry.values.numpy().astype(float)
        centre = x.mean(axis=0)
        radius = float(np.linalg.norm(x - centre, axis=1).mean())
        data = _hidden(_with(grid, np.linalg.norm(x - centre, axis=1), dtype))
        fast = _check(monkeypatch, data, [0.6 * radius, radius])
        slow = _general(monkeypatch, data, "s", [0.6 * radius, radius])
        assert _same(_cell_marks(fast), _cell_marks(slow))
        assert (_cell_marks(fast) & td.Ghosts.DUPLICATECELL).any()
        cut = _check_cut(monkeypatch, data, centre, (0.3, 0.2, 1.0))
        assert _same(_cell_marks(cut), _cell_marks(_general_slice(monkeypatch, data, centre,
                                                                  (0.3, 0.2, 1.0))))


def test_a_plane_cuts_a_curvilinear_grid_as_the_general_path_does(backend, monkeypatch):
    for dtype in _dtypes():
        image = td.uniform_grid((14, 12, 10), spacing=(0.5, 0.4, 0.3), dtype=dtype)
        x = image.geometry.values.numpy().astype(float)
        bent = td.DataSet(image.topology, tack.array(
            (x + 0.05 * np.sin(3 * x[:, [1, 2, 0]])).astype(dtype.numpy_dtype),
            tack.vec(3, dtype)))
        assert _check_cut(monkeypatch, bent, x.mean(axis=0), (0.3, 0.2, 1.0),
                          value=[-0.2, 0.3]).num_cells > 0


# ── A sheet's lines ─────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["image", "rectilinear", "curvilinear"])
def test_a_sheet_contours_to_the_general_paths_lines(backend, monkeypatch, kind):
    # Lines keep their orientation (the values above on their right), share their
    # points, and drop where both ends land on one point, as the general path's.
    rng = np.random.default_rng(9)
    for dtype in _dtypes():
        grid = td.uniform_grid((30, 25), spacing=(0.3, 0.2), dtype=dtype)
        if kind == "rectilinear":
            grid = td.rectilinear_grid(np.cumsum(rng.uniform(0.1, 0.6, 22)),
                                       np.cumsum(rng.uniform(0.1, 0.6, 19)), dtype=dtype)
        elif kind == "curvilinear":
            x = grid.geometry.values.numpy().astype(float)
            bent = x + 0.1 * np.sin(2 * x[:, [1, 0, 2]])
            bent[:, 2] = 0.05 * np.sin(x[:, 0])
            grid = td.DataSet(grid.topology, tack.array(bent.astype(dtype.numpy_dtype),
                                                       tack.vec(3, dtype)))
        assert _flying_edges.serves(grid, td.Field(td.H1(grid), tack.zeros((grid.num_points,))))
        x = grid.geometry.values.numpy().astype(float)
        data = _with(grid, np.sin(3 * x[:, 0]) * np.cos(4 * x[:, 1]), dtype)
        out = _check(monkeypatch, data, [-0.3, 0.0, 0.4])
        assert out.topology.groups()[0].shape is shapes.Line


def test_a_sheets_values_on_points_nans_and_hidden_cells(backend, monkeypatch):
    rng = np.random.default_rng(10)
    for dtype in _dtypes():
        grid = td.uniform_grid((20, 18), spacing=(0.5, 0.5), dtype=dtype)
        x = grid.geometry.values.numpy().astype(float)
        values = np.round(np.sin(x[:, 0]) + np.cos(x[:, 1]), 1)
        _check(monkeypatch, _with(grid, values, dtype), [0.0, 0.5, 1.0])
        missing = values.copy()
        missing[rng.integers(0, len(missing), 15)] = np.nan
        out = _check(monkeypatch, _with(td.uniform_grid((20, 18), spacing=(0.5, 0.5),
                                                        dtype=dtype), missing, dtype), [0.0, 0.5])
        assert len(np.unique(out.topology.connectivity.numpy())) == out.num_points
        hidden = _hidden(_with(td.uniform_grid((20, 18), spacing=(0.5, 0.5), dtype=dtype),
                               values, dtype))
        fast = _check(monkeypatch, hidden, [0.0, 0.5])
        assert _same(_cell_marks(fast), _cell_marks(_general(monkeypatch, hidden, "s",
                                                             [0.0, 0.5])))
