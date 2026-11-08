# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``tack.interop.paraview``: the proxies' XML, and the proxies driven as ParaView
drives them -- every property's command called with its default, then the
values set -- against the wrappers given the same values. Nothing here needs
ParaView; loading them in ParaView is checked by hand (pvpython, against
ParaView's own filters)."""

from xml.dom.minidom import parseString

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
    source = vtk.vtkRTAnalyticSource(whole_extent=(-8, 8, -8, 8, -8, 8))
    source.Update()
    return source.GetOutput()


def _proxy_element(cls):
    return parseString(cls._pvsm_proxy_xml).getElementsByTagName("SourceProxy")[0]


def _properties(cls):
    """The proxy's property elements, in order (not the ones in groups)."""
    return [node for node in _proxy_element(cls).childNodes
            if node.nodeType == node.ELEMENT_NODE and node.tagName.endswith("Property")]


def _as_paraview(cls, data):
    """A proxy of ``cls`` on ``data`` with every property at its default, set as
    ParaView sets it (a field to the first array, a repeated property to none)."""
    algorithm = cls()
    algorithm.SetInputDataObject(0, data)
    for prop in _properties(cls):
        command = prop.getAttribute("command")
        if prop.tagName == "InputProperty" or prop.getAttribute("repeat_command"):
            continue
        if command == "SetInputArrayToProcess":
            index = int(prop.getAttribute("default_values"))
            algorithm.SetInputArrayToProcess(index, 0, 0, 0, "RTData")
            continue
        values = prop.getAttribute("default_values").split()
        convert = {"IntVectorProperty": int, "DoubleVectorProperty": float,
                   "StringVectorProperty": str}[prop.tagName]
        getattr(algorithm, command)(*[convert(v) for v in values])
    return algorithm


def _run(algorithm):
    assert algorithm.Update(), algorithm.error
    return algorithm.GetOutputDataObject(0)


def test_every_panel_makes_a_proxy():
    assert set(pv.__all__) == {f"Tack{name}" for name in pv._panels.PANELS}
    for name in pv.__all__:
        cls = getattr(pv, name)
        assert issubclass(cls, ta.TackAlgorithm)
        assert cls.__module__ == "tack.interop.paraview"


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
    hints = proxy.getElementsByTagName("ShowInMenu")
    assert [h.getAttribute("category") for h in hints] == ["Tack"]


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


def _with_input(algorithm, data):
    algorithm.SetInputDataObject(0, data)
    return algorithm


def test_contour_normals_and_gradients():
    """The wrapper's normals are unit, and on an image the gradient's direction
    negated; the gradient comes along when asked for, the field stays unless not."""
    import numpy as np

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
    tets = vtk.vtkDataSetTriangleFilter(input_data=image)
    tets.Update()
    out = _run(_with_input(ta.Contour(field="RTData", value=150.0, compute_normals=True,
                                      compute_scalars=False), tets.GetOutput()))
    normals = np.asarray(out.GetPointData().GetArray("normals"))
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    assert "RTData" not in out.point_data.keys()


def test_clip_by_plane_ignores_the_value_paraview_fills_in():
    image = _wavelet()
    algorithm = _as_paraview(pv.TackClip, image)
    algorithm.SetValue(157.0)       # what the range domain fills in, once it can
    out = _run(algorithm)
    expected = _run(_with_input(ta.Clip(by=td.Plane((0, 0, 0), (1, 0, 0)), inside=True),
                                image))
    assert out.GetNumberOfCells() == expected.GetNumberOfCells() > 0
    assert out.GetBounds()[1] == 0.0
    algorithm.SetClipType(1)
    algorithm.SetValue(150.0)
    by_field = _run(algorithm)
    expected = _run(_with_input(ta.Clip(by="RTData", value=150.0, inside=True), image))
    assert by_field.GetNumberOfCells() == expected.GetNumberOfCells() > 0


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


@pytest.mark.parametrize("name", ["TackSlicePlane", "TackGradients", "TackElevation",
                                  "TackCellQuality", "TackExtractSurface", "TackShrink"])
def test_runs_at_its_defaults(name):
    """A proxy of one input at ParaView's defaults gives what its wrapper gives."""
    out = _run(_as_paraview(getattr(pv, name), _wavelet()))
    assert out.GetNumberOfCells() > 0


def test_a_required_parameter_needs_a_property():
    with pytest.raises(TypeError, match="Streamlines.seeds is required"):
        _generate.proxy("TackBroken", ta.Streamlines, _generate.Panel())


def test_a_default_without_a_kind_needs_a_panel():
    class Odd(ta.Shrink):
        factor = ta.Parameter({"a": 1}, "a default no property takes")

    with pytest.raises(TypeError, match="Odd.factor"):
        _generate.proxy("TackOdd", Odd, _generate.Panel())
