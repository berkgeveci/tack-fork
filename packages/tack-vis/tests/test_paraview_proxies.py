# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``tack.interop.paraview``: the proxies' XML, and the proxies driven as ParaView
drives them -- every property's command called with its default, then the
values set -- against the wrappers given the same values. Nothing here needs
ParaView; loading them in ParaView is checked by hand (pvpython, against
ParaView's own filters)."""

from xml.dom.minidom import parseString

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.all")
from tack.interop import paraview as pv
from tack.interop.paraview import _generate
from tack.interop.vtk import algorithms as ta

pytestmark = pytest.mark.quick

#: Commands every proxy has from VTK's algorithm, not from the generator.
_VTK_COMMANDS = {"SetInputConnection", "SetInputArrayToProcess"}


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _wavelet():
    """The wavelet, with a vector, a 9-component and two particle fields beside RTData."""
    source = vtk.vtkRTAnalyticSource(whole_extent=(-8, 8, -8, 8, -8, 8))
    source.Update()
    image = source.GetOutput()
    points = np.array([image.GetPoint(i) for i in range(image.number_of_points)])
    n = len(points)
    image.point_data["v"] = vtk.vtkFloatArray(
        np.c_[points[:, 1], -points[:, 0], 0.1 * points[:, 2]].astype(np.float32))
    image.point_data["g9"] = vtk.vtkFloatArray(
        np.tile(np.eye(3).ravel(), (n, 1)).astype(np.float32))
    image.point_data["h"] = vtk.vtkFloatArray(np.full(n, 1.5, np.float32))
    image.point_data["mass"] = vtk.vtkFloatArray(np.ones(n, np.float32))
    return image


def _surface(image):
    return vtk.vtkDataSetSurfaceFilter()(image)


def _lines(image):
    return vtk.vtkFeatureEdges()(_surface(image))


def _points(image):
    """The image's points as vertices, with its point fields."""
    n = image.number_of_points
    cloud = vtk.vtkPolyData()
    cloud.points = vtk.vtkPoints(
        data=np.array([image.GetPoint(i) for i in range(n)], np.float64))
    cloud.verts = vtk.vtkCellArray(offsets=np.arange(n + 1, dtype=np.int64),
                                   connectivity=np.arange(n, dtype=np.int64))
    for name in image.point_data.keys():
        cloud.GetPointData().AddArray(image.GetPointData().GetArray(name))
    return cloud


def _proxy_element(cls):
    return parseString(cls._pvsm_proxy_xml).getElementsByTagName("SourceProxy")[0]


def _properties(cls):
    """The proxy's property elements, in order (not the ones in groups)."""
    return [node for node in _proxy_element(cls).childNodes
            if node.nodeType == node.ELEMENT_NODE and node.tagName.endswith("Property")]


def _as_paraview(cls, data):
    """A proxy of ``cls`` on ``data`` with every property at its default, set as
    ParaView sets it: a field to the first array of its width (an optional one to
    none), a repeated property to none, an empty string as None. A proxy property
    is left for the test."""
    algorithm = cls()
    algorithm.SetInputDataObject(0, data)
    domains = {d.getAttribute("name"): d
               for d in _proxy_element(cls).getElementsByTagName("InputArrayDomain")}
    for prop in _properties(cls):
        command = prop.getAttribute("command")
        if prop.tagName in ("InputProperty", "ProxyProperty") or prop.getAttribute(
                "repeat_command"):
            continue
        if command == "SetInputArrayToProcess":
            index = int(prop.getAttribute("default_values").split(";")[0])
            width = domains[f"input_array_{index}"].getAttribute("number_of_components")
            none = prop.getElementsByTagName("ArrayListDomain")[0].getAttribute("none_string")
            algorithm.SetInputArrayToProcess(index, 0, 0, 0, none or {"3": "v", "9": "g9"}.get(
                width, "RTData"))
            continue
        values = prop.getAttribute("default_values")
        if prop.tagName == "StringVectorProperty":
            getattr(algorithm, command)(values or None)
            continue
        convert = {"IntVectorProperty": int, "DoubleVectorProperty": float}[prop.tagName]
        getattr(algorithm, command)(*[convert(v) for v in values.split()])
    return algorithm


def _run(algorithm):
    assert algorithm.Update(), algorithm.error
    assert algorithm.error is None, algorithm.error
    return algorithm.GetOutputDataObject(0)


def _with_input(algorithm, data):
    algorithm.SetInputDataObject(0, data)
    return algorithm


def _plane(origin=(0, 0, 0), normal=(1, 0, 0)):
    return vtk.vtkPlane(origin=origin, normal=normal)


def test_every_panel_makes_a_proxy():
    assert set(pv.__all__) == {f"Tack{name}" for name in pv._panels.PANELS}
    for name in pv.__all__:
        cls = getattr(pv, name)
        assert issubclass(cls, ta.TackAlgorithm)
        assert cls.__module__ == "tack.interop.paraview"


def test_every_wrapper_has_a_proxy_but_those_named():
    """A new wrapper needs a panel, or a place on this list with the reason."""
    wrappers = {name for name in ta.__all__
                if isinstance(getattr(ta, name), type)
                and issubclass(getattr(ta, name), ta.TackAlgorithm)
                and getattr(ta, name) is not ta.TackAlgorithm}
    without = {"Interpolate",   # its function is a @tack.func
               "Composite",     # its fields are in order, which a list cannot say
               "SlicePlane"}    # Slice with a Plane
    assert wrappers - set(pv._panels.PANELS) == without


@pytest.mark.parametrize("name", pv.__all__)
def test_the_xml_names_what_the_class_has(name):
    """The XML parses; the class it names is this one; every command and every
    property a group or a domain names exists; names are unique."""
    cls = getattr(pv, name)
    proxy = _proxy_element(cls)
    assert proxy.getAttribute("class") == f"tack.interop.paraview.{name}"
    assert proxy.getAttribute("si_class") == "vtkSIPythonSourceProxy"
    properties = _properties(cls)
    names = [p.getAttribute("name") for p in properties]
    assert len(names) == len(set(names))
    for prop in properties:
        for attribute in ("command", "set_number_command"):
            command = prop.getAttribute(attribute)
            if command and command not in _VTK_COMMANDS:
                assert callable(getattr(cls, command, None)), (name, command)
    for node in proxy.getElementsByTagName("Property"):
        # A group's member, or a domain's required property.
        assert node.getAttribute("name") in names, (name, node.getAttribute("name"))
    for node in proxy.getElementsByTagName("PropertyWidgetDecorator"):
        if node.getAttribute("property"):
            assert node.getAttribute("property") in names, (name, node.toxml())
    hints = proxy.getElementsByTagName("ShowInMenu")
    assert [h.getAttribute("category") for h in hints] == ["Tack"]


#: The input each proxy runs on at its defaults, where not the wavelet.
_INPUTS = {"SurfaceNormals": _surface, "FeatureEdges": _surface, "Triangulate": _surface,
           "AsPolygons": _surface, "EdgeLengths": _surface, "FaceCenters": _surface,
           "FaceGeometry": _surface, "MergePoints": _surface, "Tubes": _lines,
           "GhostPoints": _points, "ConnectedComponents": _points, "Outliers": _points,
           "PointNormals": _points, "VoxelDownsample": _points, "SphDensity": _points,
           "Deposit": _points, "Glyphs": _points}

#: What a proxy needs set beyond its defaults: an implicit function, a transform, a
#: field of the right kind, a box.
_SETUP = {
    "Clip": lambda a: a.SetClipFunction(_plane()),
    "Slice": lambda a: a.SetCutFunction(_plane()),
    "ExtractGeometry": lambda a: a.SetImplicitFunction(_plane()),
    "ExtractPoints": lambda a: a.SetImplicitFunction(_plane()),
    "ImplicitValues": lambda a: a.SetImplicitFunction(vtk.vtkSphere(radius=4.0)),
    "Transform": lambda a: a.SetTransform(vtk.vtkTransform()),
    "ExtractExtent": lambda a: a.SetVOI(0, 4, 0, 4, 0, 4),
    "Voxelize": lambda a: a.SetSamplingDimensions(9, 9, 9),
    "Deposit": lambda a: a.SetSamplingDimensions(9, 9, 9),
    "Magnitude": lambda a: a.SetInputArrayToProcess(0, 0, 0, 0, "v"),
    "Dot": lambda a: (a.SetInputArrayToProcess(0, 0, 0, 0, "v"),
                      a.SetInputArrayToProcess(1, 0, 0, 0, "v")),
}

#: Proxies this test does not run: a second input, time steps, an AMR hierarchy, an
#: input of a kind the wavelet is not (a unit, polyhedra, a cell set, halos).
_NOT_RUN = {"Probe", "SphInterpolate", "Streamlines", "Advect", "Advection", "Pathlines",
            "Streaklines", "RunningStatistics", "TimeDerivative", "AverageDown",
            "HaloCatalog", "Convert", "CheckWinding", "ExtractCells", "Contour"}


@pytest.mark.parametrize("name", [n for n in pv.__all__ if n[4:] not in _NOT_RUN])
def test_runs_at_paraviews_defaults(name):
    """Every proxy, its properties at their defaults as ParaView sets them, runs on
    an input of its kind (Contour has no values by default: its own test)."""
    short = name[4:]
    algorithm = _as_paraview(getattr(pv, name), _INPUTS.get(short, lambda i: i)(_wavelet()))
    _SETUP.get(short, lambda a: None)(algorithm)
    out = _run(algorithm)
    assert out is not None


def test_a_device_named_on_the_panel_is_the_filters_device():
    algorithm = pv.TackShrink()
    assert algorithm.device is None
    algorithm.SetDeviceIndex(_generate.DEVICES.index("cpu"))
    assert algorithm.device == "cpu"
    algorithm.SetDeviceIndex(0)
    assert algorithm.device is None


def test_contour_values_as_paraview_sets_them():
    image = _wavelet()
    algorithm = _as_paraview(pv.TackContour, image)
    assert _run(algorithm).GetNumberOfCells() == 0      # no values, no surface
    algorithm.SetNumberOfContours(2)
    algorithm.SetValue(0, 120.0)
    algorithm.SetValue(1, 200.0)
    out = _run(algorithm)
    expected = _run(_with_input(ta.Contour(field="RTData", value=(120.0, 200.0)), image))
    assert out.GetNumberOfCells() == expected.GetNumberOfCells() > 0
    # ParaView's defaults: normals on, and an image's points as floats.
    assert out.GetPointData().GetNormals().GetName() == "normals"
    assert out.GetPoints().GetDataType() == vtk.VTK_FLOAT
    algorithm.SetOutputPointsPrecision(1)
    assert _run(algorithm).GetPoints().GetDataType() == vtk.VTK_DOUBLE


def test_contour_normals_and_gradients():
    """The wrapper's normals are unit, and on an image the gradient's direction
    negated; the gradient comes along when asked for, the field stays unless not."""
    image = _wavelet()
    out = _run(_with_input(ta.Contour(field="RTData", value=150.0, compute_normals=True,
                                      compute_gradients=True), image))
    normals = np.asarray(out.GetPointData().GetArray("normals"))
    gradient = np.asarray(out.GetPointData().GetArray("gradient"))
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    unit = gradient / np.linalg.norm(gradient, axis=1)[:, None]
    assert np.allclose(normals, -unit, atol=1e-5)
    assert out.GetPointData().GetNormals().GetName() == "normals"
    assert "RTData" in out.point_data.keys()
    tets = vtk.vtkDataSetTriangleFilter()(image)
    out = _run(_with_input(ta.Contour(field="RTData", value=150.0, compute_normals=True,
                                      compute_scalars=False), tets))
    normals = np.asarray(out.GetPointData().GetArray("normals"))
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    assert "RTData" not in out.point_data.keys()


def test_clip_by_a_function_ignores_the_value_paraview_fills_in():
    image = _wavelet()
    algorithm = _as_paraview(pv.TackClip, image)
    algorithm.SetClipFunction(_plane())
    algorithm.SetValue(157.0)       # what the range domain fills in, once it can
    out = _run(algorithm)
    expected = _run(_with_input(ta.Clip(by=td.Plane((0, 0, 0), (1, 0, 0)), inside=True),
                                image))
    assert out.GetNumberOfCells() == expected.GetNumberOfCells() > 0
    assert out.GetBounds()[1] == 0.0
    algorithm.SetClipFunction(None)     # Scalar, a null proxy
    algorithm.SetValue(150.0)
    by_field = _run(algorithm)
    expected = _run(_with_input(ta.Clip(by="RTData", value=150.0, inside=True), image))
    assert by_field.GetNumberOfCells() == expected.GetNumberOfCells() > 0


def test_a_function_changed_in_place_runs_the_filter_again():
    """ParaView sets the plane's origin on the plane: the filter watches it."""
    image = _wavelet()
    algorithm = _as_paraview(pv.TackSlice, image)
    plane = _plane()
    algorithm.SetCutFunction(plane)
    assert _run(algorithm).GetBounds()[0:2] == (0.0, 0.0)
    plane.SetOrigin(2.5, 0, 0)
    assert _run(algorithm).GetBounds()[0:2] == (2.5, 2.5)
    # A function replaced is no longer watched.
    algorithm.SetCutFunction(_plane((1, 0, 0)))
    before = algorithm.GetMTime()
    plane.SetOrigin(3.5, 0, 0)
    assert algorithm.GetMTime() == before


def test_slice_offsets_as_paraview_sets_them():
    image = _wavelet()
    algorithm = _as_paraview(pv.TackSlice, image)
    algorithm.SetCutFunction(_plane())
    algorithm.SetNumberOfContours(3)
    for i, offset in enumerate((-2.0, 0.0, 2.0)):
        algorithm.SetValue(i, offset)
    out = _run(algorithm)
    expected = _run(_with_input(ta.Slice(function=td.Plane((0, 0, 0), (1, 0, 0)),
                                         value=(-2.0, 0.0, 2.0)), image))
    assert out.GetNumberOfCells() == expected.GetNumberOfCells() > 0


class _ParaViewBox(vtk.vtkBox):
    """A ``vtkBox`` the conversion takes for ParaView's ``vtkPVBox``."""

    def GetClassName(self):
        return "vtkPVBox"


class _ParaViewCylinder(vtk.vtkCylinder):
    def GetClassName(self):
        return "vtkPVCylinder"


def _turned_cylinder(center, radius, scale=1.0):
    """A cylinder along x through ``center``, as ``vtkPVCylinder`` makes one: VTK's
    along y, turned about its center by a transform from world to its space."""
    cylinder = _ParaViewCylinder()
    cylinder.SetCenter(*center)
    cylinder.SetRadius(radius)
    turn = vtk.vtkTransform()
    turn.Translate(*center)
    turn.RotateWXYZ(-90, 0, 0, 1)
    turn.Scale(scale, 1, 1)
    turn.Translate(*[-c for c in center])
    turn.Inverse()
    cylinder.SetTransform(turn)
    return cylinder


def _placed_box(rotation=None):
    """A unit box placed as ``vtkPVBox`` places it: scaled, (turned,) moved, the
    transform from world to the box's space."""
    box = _ParaViewBox()
    box.SetBounds(0, 1, 0, 1, 0, 1)
    place = vtk.vtkTransform()
    place.Translate(-2, -1, 0)
    if rotation:
        place.RotateWXYZ(*rotation)
    place.Scale(4, 3, 2)
    place.Inverse()
    box.SetTransform(place)
    return box


def test_paraviews_box_and_cylinder_as_tacks():
    box = _generate.implicit_function(_placed_box())
    assert isinstance(box, td.Box)
    lower, upper = (-2, -1, 0), (2, 2, 2)
    image = _wavelet()
    by_paraview = _run(_with_input(ta.ExtractGeometry(function=box), image))
    by_tack = _run(_with_input(ta.ExtractGeometry(function=td.Box(lower, upper)), image))
    assert by_paraview.GetNumberOfCells() == by_tack.GetNumberOfCells() > 0
    # Turned by a right angle it is still along the axes.
    turned = _generate.implicit_function(_placed_box((90, 0, 0, 1)))
    assert isinstance(turned, td.Box)
    with pytest.raises(ValueError, match="turned Box"):
        _generate.implicit_function(_placed_box((30, 0, 0, 1)))
    cylinder = _generate.implicit_function(_turned_cylinder((1, 2, 0), 3.0))
    by_paraview = _run(_with_input(ta.ExtractGeometry(function=cylinder), image))
    by_tack = _run(_with_input(ta.ExtractGeometry(
        function=td.Cylinder((1, 2, 0), (1, 0, 0), 3.0)), image))
    assert by_paraview.GetNumberOfCells() == by_tack.GetNumberOfCells() > 0
    with pytest.raises(ValueError, match="scaled"):
        _generate.implicit_function(_turned_cylinder((1, 2, 0), 3.0, scale=2.0))
    # VTK's own pass through as tack.interop.vtk converts them.
    assert isinstance(_generate.implicit_function(_plane()), td.Plane)


def test_an_optional_field_left_at_none_is_none():
    cloud = _points(_wavelet())
    algorithm = _as_paraview(pv.TackGlyphs, cloud)
    unscaled = _run(_with_input(ta.Glyphs(vectors="v"), cloud))
    assert _run(algorithm).GetNumberOfPoints() == unscaled.GetNumberOfPoints()
    np.testing.assert_allclose(_run(algorithm).GetBounds(), unscaled.GetBounds())
    algorithm.SetInputArrayToProcess(1, 0, 0, 0, "h")
    scaled = _run(_with_input(ta.Glyphs(vectors="v", scale_by="h"), cloud))
    np.testing.assert_allclose(_run(algorithm).GetBounds(), scaled.GetBounds())


def test_zero_leaves_an_optional_number_to_the_function():
    algorithm = pv.TackConnectedComponents()
    algorithm.SetRadius(0.0)
    assert algorithm.radius is None
    algorithm.SetRadius(1.5)
    assert algorithm.radius == 1.5
    algorithm.SetMinSize(0)
    assert algorithm.min_size is None


def test_an_empty_string_is_none():
    algorithm = pv.TackProject()
    algorithm.SetResultArrayName(None)      # ParaView's empty string
    assert algorithm.name is None
    algorithm.SetResultArrayName("p")
    assert algorithm.name == "p"


def test_deposit_onto_the_inputs_bounds():
    """The sampling grid over the input's bounds, as Resample To Image makes it."""
    cloud = _points(_wavelet())
    algorithm = _as_paraview(pv.TackDeposit, cloud)
    algorithm.SetSamplingDimensions(9, 9, 9)
    out = _run(algorithm)
    grid = vtk.vtkImageData(origin=(-8, -8, -8), spacing=(2, 2, 2), dimensions=(9, 9, 9))
    expected = _run(_with_input(ta.Deposit(grid=grid), cloud))
    assert out.GetDimensions() == (9, 9, 9)
    np.testing.assert_array_equal(np.asarray(out.cell_data["deposit"]),
                                  np.asarray(expected.cell_data["deposit"]))
    algorithm.SetUseInputBounds(0)
    algorithm.SetSamplingBounds(0, 8, 0, 8, 0, 8)
    assert _run(algorithm).GetBounds() == (0, 8, 0, 8, 0, 8)


@pytest.mark.parametrize("method, lower, upper", [
    (0, 100.0, 200.0), (1, float("-inf"), 100.0), (2, 200.0, float("inf"))])
def test_threshold_methods(method, lower, upper):
    image = _wavelet()
    algorithm = _as_paraview(pv.TackThreshold, image)
    algorithm.SetLowerThreshold(100.0)
    algorithm.SetUpperThreshold(200.0)
    algorithm.SetThresholdMethod(method)
    expected = _run(_with_input(ta.Threshold(field="RTData", lower=lower, upper=upper),
                                image))
    assert _run(algorithm).GetNumberOfCells() == expected.GetNumberOfCells() > 0


def test_a_required_parameter_needs_a_property():
    with pytest.raises(TypeError, match="Streamlines.seeds is required"):
        _generate.proxy("TackBroken", ta.Streamlines, _generate.Panel())


def test_a_default_without_a_kind_needs_a_panel():
    class Odd(ta.Shrink):
        factor = ta.Parameter({"a": 1}, "a default no property takes")

    with pytest.raises(TypeError, match="Odd.factor"):
        _generate.proxy("TackOdd", Odd, _generate.Panel())
