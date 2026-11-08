# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``vtk_to_dataset`` on a discrete GPU (``tack.interop.vtk._device_read``).

With CUDA, HIP or Level Zero the default device, an unstructured grid's arrays go
to the device once and the conversion's work runs there. The dataset must be the
host conversion's, array for array and bit for bit; what that path does not do
falls back to the host conversion, warnings and all. Run on every GPU here, and
on a simulated one -- a CPU backend of its own memory domain, the path's device
types widened to take it -- so it runs anywhere.
"""

import warnings

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.arrays import to_host
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytest.importorskip("vtkmodules.vtkCommonDataModel")

pytestmark = pytest.mark.quick


class _DeviceCPU(CPUBackend):
    """A CPU backend posing as a discrete GPU's: memory of its own."""

    display_name = "simulated GPU"

    @property
    def memory_domain(self):
        return ("cpu", 7)


def _gpus():
    from conftest import available_backends

    return [a for a in available_backends if a in ("cuda", "hip", "level_zero")]


@pytest.fixture(params=["simulated", *_gpus()])
def on_device(request, monkeypatch):
    """Converts on the device named, through the device path: ``on_device(convert)``
    runs ``convert()`` with that device the default, and returns its dataset and
    whether the device path made it -- found by a spy, a frame in the way of a
    warning's ``stacklevel``; ``spy=False`` leaves it out (and says None)."""
    from tack.interop.vtk import _device_read

    if request.param == "simulated":
        device = dispatch.as_device(_DeviceCPU())
        types = (*_device_read.DEVICE_TYPES, "cpu")
    else:
        device, types = tack.device(request.param), _device_read.DEVICE_TYPES
    real = _device_read._convert
    taken = []

    def spied(*args):
        out = real(*args)
        taken.append(out is not None)
        return out

    def run(convert, spy=True):
        taken.clear()
        with monkeypatch.context() as m:
            m.setattr(_device_read, "DEVICE_TYPES", types)
            if spy:
                m.setattr(_device_read, "_convert", spied)
            tack.set_default_device(device)
            try:
                data = convert()
            finally:
                tack.set_default_device("cpu")
        return data, (any(taken) if spy else None)

    run.device = device
    tack.set_default_device("cpu")
    yield run
    tack.set_default_device("cpu")


def _arrays(data):
    t = data.topology
    out = {n: getattr(t, n) for n in ("types", "offsets", "connectivity", "face_offsets",
                                      "face_points", "cell_face_offsets", "cell_faces",
                                      "cell_face_sides") if hasattr(t, n)}
    out["geometry"] = data.geometry.values
    for key, field in data.fields.items():
        out[f"field {key}"] = field.values
    if data.ghosts is not None:
        out["ghost cells"], out["ghost points"] = data.ghosts.cells, data.ghosts.points
    if data.global_ids is not None:
        out["cell ids"], out["point ids"] = data.global_ids.cells, data.global_ids.points
    return {k: v for k, v in out.items() if v is not None}


def _assert_same(expected, got, device=None):
    """Every array of ``expected`` (the host conversion) in ``got``: type, shape and
    bits; and ``got``'s explicit arrays on ``device``."""
    want, have = _arrays(expected), _arrays(got)
    assert sorted(want) == sorted(have)
    for key in want:
        a, b = np.asarray(to_host(want[key])), np.asarray(to_host(have[key]))
        assert (a.dtype, a.shape) == (b.dtype, b.shape), key
        np.testing.assert_array_equal(a, b, err_msg=key)
        if device is not None and isinstance(have[key], tack.Array):
            assert have[key].device == device, key
    assert got.origin == expected.origin
    assert got.unit == expected.unit


def _mixed_grid(far=False):
    """Hexahedra, wedges and polyhedra side by side on merged points, with point
    and cell fields of several types, ghost marks (a hidden point) and global ids."""
    import vtkmodules.vtkCommonDataModel as dm
    from vtkmodules.vtkCommonCore import vtkIdTypeArray, vtkPoints, vtkUnsignedCharArray
    from vtkmodules.vtkFiltersCore import vtkAppendFilter
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    append = vtkAppendFilter(merge_points=True)
    for shift, kind in ((0, dm.VTK_HEXAHEDRON), (4, dm.VTK_POLYHEDRON), (-4, dm.VTK_WEDGE),
                        (8, dm.VTK_TETRA), (12, dm.VTK_PYRAMID)):
        block = vtkCellTypeSource(cell_type=kind, blocks_dimensions=(3, 3, 3))()
        x = np.asarray(block.points.data).copy()
        x[:, 0] += shift + (1e6 if far else 0)
        block.points = vtkPoints(data=x)
        append.AddInputData(block)
    append.Update()
    grid = append.GetOutput()
    n, c = grid.number_of_points, grid.number_of_cells
    x = np.asarray(grid.points.data)
    grid.point_data["T"] = x[:, 0].astype(np.float32)
    grid.point_data["v"] = x.astype(np.float64)
    grid.point_data["k"] = np.arange(n, dtype=np.int32)
    grid.cell_data["c"] = np.arange(c, dtype=np.float32)
    grid.point_data["p"] = np.ones(n, np.float32)
    grid.cell_data["p"] = np.full(c, 2.0, np.float32)
    marks = vtkUnsignedCharArray()
    marks.SetName("vtkGhostType")
    marks.SetNumberOfTuples(n)
    marks.Fill(0)
    marks.SetValue(5, 2)                                    # HIDDENPOINT
    grid.GetPointData().AddArray(marks)
    for attributes, size in ((grid.GetPointData(), n), (grid.GetCellData(), c)):
        numbers = vtkIdTypeArray()
        numbers.SetName("ids")
        numbers.SetNumberOfTuples(size)
        for i in range(size):
            numbers.SetValue(i, 1000 + i)
        attributes.SetGlobalIds(numbers)
    return grid


def _repeated(grid):
    """``grid`` with its even and odd cells each on their own copies of their points,
    as a mesh written per processor and put together unmerged."""
    from vtkmodules.vtkCommonCore import vtkIdList
    from vtkmodules.vtkFiltersCore import vtkAppendFilter, vtkExtractCells

    append = vtkAppendFilter(merge_points=False)
    for parity in (0, 1):
        ids = vtkIdList()
        for c in range(parity, grid.number_of_cells, 2):
            ids.InsertNextId(c)
        append.AddInputData(vtkExtractCells(cell_list=ids)(grid))
    append.Update()
    return append.GetOutput()


def _flipped(grid, kinds=(12, 13)):
    """``grid`` with every other cell of ``kinds`` turned inside out."""
    from tack.interop.vtk import _ORIENTATION

    types = np.asarray(grid.cell_types)
    offsets = np.asarray(grid.cells.offsets_array)
    corners = np.asarray(grid.cells.connectivity_array)
    for c in np.flatnonzero(np.isin(types, kinds))[::2]:
        row = corners[offsets[c]:offsets[c + 1]]
        row[:] = row[list(_ORIENTATION[int(types[c])][1])].copy()
    grid.Modified()
    return grid


@pytest.mark.parametrize("merge", [None, True])
@pytest.mark.parametrize("dtype", ["default", None])
def test_the_device_reads_what_the_host_reads(on_device, merge, dtype):
    from tack.interop.vtk import vtk_to_dataset

    grid = _repeated(_mixed_grid()) if merge else _mixed_grid()
    options = {"merge_points": merge} if dtype == "default" else {"merge_points": merge,
                                                                  "dtype": None}
    expected = vtk_to_dataset(grid, **options)
    got, taken = on_device(lambda: vtk_to_dataset(grid, **options))
    assert taken
    if merge:
        assert got.num_points < grid.number_of_points
    _assert_same(expected, got, on_device.device)


def test_far_points_keep_their_origin_on_the_device(on_device):
    from tack.interop.vtk import vtk_to_dataset

    grid = _repeated(_mixed_grid(far=True))
    expected = vtk_to_dataset(grid, merge_points=True)
    assert expected.origin is not None
    got, taken = on_device(lambda: vtk_to_dataset(grid, merge_points=True))
    assert taken
    _assert_same(expected, got, on_device.device)


def test_inside_out_cells_warn_or_turn_on_the_device(on_device):
    from tack.interop.vtk import vtk_to_dataset

    grid = _flipped(_mixed_grid())
    with pytest.warns(UserWarning, match="cells are inside out") as host:
        expected = vtk_to_dataset(grid)
    with pytest.warns(UserWarning, match="cells are inside out") as device:
        got, taken = on_device(lambda: vtk_to_dataset(grid))
    assert taken
    assert [str(w.message) for w in device] == [str(w.message) for w in host]
    with pytest.warns(UserWarning, match="cells are inside out") as device:
        on_device(lambda: vtk_to_dataset(grid), spy=False)
    assert {w.filename for w in device} == {__file__}             # the caller's line
    _assert_same(expected, got, on_device.device)
    expected = vtk_to_dataset(grid, orient=True, merge_points=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got, taken = on_device(lambda: vtk_to_dataset(grid, orient=True, merge_points=True))
    assert taken
    _assert_same(expected, got, on_device.device)
    assert (td.cell_geometry(got).measure.values.numpy() > 0).all()


def _collapsed_hexahedra():
    """Hexahedra, one written as a wedge with corners repeated (Tecplot's brick)."""
    import vtkmodules.vtkCommonDataModel as dm
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    grid = vtkCellTypeSource(cell_type=dm.VTK_HEXAHEDRON, blocks_dimensions=(2, 2, 2))()
    offsets = np.asarray(grid.cells.offsets_array)
    corners = np.asarray(grid.cells.connectivity_array)
    row = corners[offsets[0]:offsets[1]]
    row[3], row[7] = row[2], row[6]
    grid.Modified()
    return grid


def _inside_out_polyhedra():
    import vtkmodules.vtkCommonDataModel as dm
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    grid = vtkCellTypeSource(cell_type=dm.VTK_POLYHEDRON, blocks_dimensions=(3, 2, 2))()
    faces = grid.polyhedron_faces
    offsets = np.asarray(faces.offsets_array)
    points = np.asarray(faces.connectivity_array)
    for f in range(len(offsets) - 1):
        points[offsets[f]:offsets[f + 1]] = points[offsets[f]:offsets[f + 1]][::-1].copy()
    grid.Modified()
    return grid


def _polylines():
    import vtkmodules.vtkCommonDataModel as dm
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid

    grid = vtkUnstructuredGrid()
    grid.points = vtkPoints(data=np.random.default_rng(0).random((6, 3)))
    grid.cells = (dm.VTK_POLY_LINE, vtkCellArray(offsets=np.array([0, 3, 6]),
                                                 connectivity=np.arange(6)))
    return grid


@pytest.mark.parametrize("case,options,warns", [
    ("collapsed hexahedra", {}, None),
    ("inside-out polyhedra", {"orient": True}, None),
    ("inside-out polyhedra", {}, "polyhedra are inside out"),
    ("polylines", {}, None),
])
def test_what_the_device_does_not_do_the_host_does(on_device, case, options, warns):
    """Collapsed hexahedra to convert, polyhedra to turn outward, polylines: the host
    conversion, its result on the device as before. Inside-out polyhedra left as
    they are stay on the device path, with the host's warning."""
    from tack.interop.vtk import vtk_to_dataset

    grid = {"collapsed hexahedra": _collapsed_hexahedra,
            "inside-out polyhedra": _inside_out_polyhedra,
            "polylines": _polylines}[case]()
    with warnings.catch_warnings(record=True) as host:
        warnings.simplefilter("always")
        expected = vtk_to_dataset(grid, **options)
    with warnings.catch_warnings(record=True) as device:
        warnings.simplefilter("always")
        got, taken = on_device(lambda: vtk_to_dataset(grid, **options))
    assert [str(w.message) for w in device] == [str(w.message) for w in host]
    if warns:
        assert taken and any(warns in str(w.message) for w in device)
    else:
        assert not taken
    _assert_same(expected, got)


def test_the_cpu_and_metal_keep_the_host_conversion(monkeypatch):
    """Only a discrete GPU's default device takes the device path."""
    from tack.interop.vtk import _device_read

    options = {"copy": True, "polyhedral": None, "merge_points": True}
    grid = _mixed_grid()
    tack.set_default_device("cpu")
    assert _device_read.device_for(grid, options) is None
    assert _device_read.device_for(grid, {**options, "copy": False}) is None
    assert _device_read.device_for(grid, {**options, "merge_points": 1e-6}) is None
    assert "metal" not in _device_read.DEVICE_TYPES
