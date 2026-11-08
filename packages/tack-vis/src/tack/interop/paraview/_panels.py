# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The panels of the proxies: what each takes from ParaView's filter of the kind
(names, labels, widgets, defaults) where the wrapper's parameters cannot say,
and the mixins behind the properties a panel adds. A state or a script moves
between ParaView's filter and Tack's by changing the proxy's name, for the
properties both have."""

from ._generate import Panel, Property

_INPUT = '<RequiredProperties><Property name="Input" function="Input"/></RequiredProperties>'


def _range_of(field, mode=None):
    """An ``ArrayRangeDomain`` over the array the property ``field`` selects."""
    attrs = f' default_mode="{mode}"' if mode else ""
    return (f'<ArrayRangeDomain name="range"{attrs}><RequiredProperties>'
            f'<Property name="Input" function="Input"/>'
            f'<Property name="{field}" function="ArraySelection"/>'
            "</RequiredProperties></ArrayRangeDomain>")


def _bounds(mode):
    return (f'<BoundsDomain name="range" mode="normal" default_mode="{mode}">'
            f"{_INPUT}</BoundsDomain>")


def _shown_when(prop, value):
    return (f'<Hints><PropertyWidgetDecorator type="GenericDecorator" mode="visibility" '
            f'property="{prop}" value="{value}"/></Hints>')


# ── Mixins ──────────────────────────────────────────────────────────


class _ContourValues:
    """ParaView's ContourValues: ``SetNumberOfContours``, then ``SetValue`` for each.
    No values, no surface (as ParaView's Contour gives)."""

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self._values = []

    def SetValue(self, index, value):
        self._values[int(index)] = float(value)
        self._set_values()

    def SetNumberOfContours(self, count):
        self._values = (self._values + [0.0] * int(count))[:int(count)]
        self._set_values()

    def _set_values(self):
        # One value as a number: a sequence would add "contour_value" for nothing.
        self.value = self._values[0] if len(self._values) == 1 else tuple(self._values)

    def RequestData(self, request, inInfo, outInfo):
        if not self._values:
            self.GetOutputData(outInfo, 0).Initialize()
            return 1
        return super().RequestData(request, inInfo, outInfo)


class _PointsPrecision:
    """VTK's OutputPointsPrecision: the output's coordinates as floats, doubles, or
    as the input's (floats for an image, which has no points, as VTK gives)."""

    SINGLE, DOUBLE, SAME = 0, 1, 2

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self._precision = self.SAME

    def SetOutputPointsPrecision(self, precision):
        self._precision = int(precision)
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        if not super().RequestData(request, inInfo, outInfo):
            return 0
        from vtkmodules.vtkCommonDataModel import vtkImageData

        precision = self._precision
        if precision == self.SAME and any(
                isinstance(leaf, vtkImageData)
                for leaf in _leaves(self.GetInputData(inInfo, 0, 0))):
            precision = self.SINGLE
        if precision != self.SAME:
            _set_points_type(self.GetOutputData(outInfo, 0), precision == self.DOUBLE)
        return 1


def _leaves(data):
    """The datasets in ``data``: itself, or a composite's leaves."""
    from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet

    if data is None:
        return []
    if not isinstance(data, vtkCompositeDataSet):
        return [data]
    iterator = data.NewIterator()
    iterator.SkipEmptyNodesOn()
    iterator.InitTraversal()
    leaves = []
    while not iterator.IsDoneWithTraversal():
        leaves.append(iterator.GetCurrentDataObject())
        iterator.GoToNextItem()
    return leaves


def _set_points_type(data, double):
    """Every point set in ``data`` with its coordinates as doubles or floats."""
    from vtkmodules.vtkCommonCore import VTK_DOUBLE, VTK_FLOAT, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPointSet

    kind = VTK_DOUBLE if double else VTK_FLOAT
    for leaf in _leaves(data):
        points = leaf.GetPoints() if isinstance(leaf, vtkPointSet) else None
        if points is not None and points.GetDataType() != kind:
            converted = vtkPoints()
            converted.SetDataType(kind)
            converted.SetNumberOfPoints(points.GetNumberOfPoints())
            converted.GetData().DeepCopy(points.GetData())
            leaf.SetPoints(converted)


class _PlaneOrField:
    """``by`` as ParaView's Clip Type: a plane (``Origin``, ``Normal``) or the field
    the Scalars menu selects."""

    PLANE, SCALAR = 0, 1

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self._type, self._origin, self._normal = self.PLANE, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        self._set_by()

    def SetClipType(self, index):
        self._type = int(index)
        self._set_by()

    def SetOrigin(self, *origin):
        self._origin = tuple(float(v) for v in origin)
        self._set_by()

    def SetNormal(self, *normal):
        self._normal = tuple(float(v) for v in normal)
        self._set_by()

    def _set_by(self):
        import tack.dataset as td

        self.by = td.Plane(self._origin, self._normal) if self._type == self.PLANE else None

    def arguments(self):
        arguments = super().arguments()
        if self._type == self.PLANE:
            # The Value is the field's (its range domain fills it in): at the plane.
            arguments["value"] = 0.0
        return arguments


class _ThresholdMethod:
    """ParaView's ThresholdMethod: between the thresholds, below the lower or above
    the upper."""

    BETWEEN, BELOW, ABOVE = 0, 1, 2

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self._method = self.BETWEEN

    def SetThresholdMethod(self, method):
        self._method = int(method)
        self.Modified()

    def arguments(self):
        arguments = super().arguments()
        if self._method == self.BELOW:
            arguments["lower"], arguments["upper"] = float("-inf"), arguments["lower"]
        elif self._method == self.ABOVE:
            arguments["lower"], arguments["upper"] = arguments["upper"], float("inf")
        return arguments


# ── Panels ──────────────────────────────────────────────────────────

_PRECISION = """
<IntVectorProperty name="OutputPointsPrecision" command="SetOutputPointsPrecision"
    number_of_elements="1" default_values="2" animateable="0" panel_visibility="advanced">
  <EnumerationDomain name="enum">
    <Entry text="Single" value="0"/><Entry text="Double" value="1"/>
    <Entry text="Same as input" value="2"/>
  </EnumerationDomain>
  <Documentation>The type of the output's coordinates (floats for an image
  input).</Documentation>
</IntVectorProperty>"""

_PLANE = """
<DoubleVectorProperty name="Origin" command="SetOrigin" number_of_elements="3"
    default_values="0 0 0">%s
  <Documentation>A point of the plane.</Documentation>
</DoubleVectorProperty>
<DoubleVectorProperty name="Normal" command="SetNormal" number_of_elements="3"
    default_values="1 0 0">
  <DoubleRangeDomain name="range"/>
  <Documentation>The plane's normal.</Documentation>
</DoubleVectorProperty>""" % _bounds("mid")


def _plane_group(shown=""):
    return ('<PropertyGroup label="Plane Parameters" panel_widget="InteractivePlane">'
            '<Property function="Origin" name="Origin"/>'
            '<Property function="Normal" name="Normal"/>'
            f'<Property function="Input" name="Input"/>{shown}</PropertyGroup>')


#: The panels by wrapper; a wrapper without one has no proxy yet.
PANELS = {
    "Contour": Panel(
        properties={
            "field": Property(name="SelectInputScalars", label="Contour By",
                              attribute="point", components=1),
            "compute_normals": Property(
                name="ComputeNormals", default=True,
                domain='<Hints><PropertyWidgetDecorator type="InputDataTypeDecorator" '
                       'name="vtkPolyData" exclude="1" mode="enabled_state"/></Hints>'),
            "compute_gradients": Property(name="ComputeGradients"),
            "compute_scalars": Property(name="ComputeScalars", advanced=True),
            "OutputPointsPrecision": _PRECISION,
            "value": """
<DoubleVectorProperty name="ContourValues" label="Isosurfaces" command="SetValue"
    number_of_elements="0" number_of_elements_per_command="1" repeat_command="1"
    set_number_command="SetNumberOfContours" use_index="1" animateable="1">%s
  <Documentation>The values to contour at. Tack makes the surfaces of every value in one
  pass over the cells, with each triangle's value in a cell array "contour_value" when
  there are several.</Documentation>
</DoubleVectorProperty>""" % _range_of("SelectInputScalars"),
            "merge_points": Property(name="MergePoints", advanced=True,
                                     group="Point Locator"),
            "component": None,
            "dimension": None,
        },
        order=("field", "compute_normals", "compute_gradients", "compute_scalars",
               "OutputPointsPrecision", "value", "merge_points"),
        hints='<Visibility replace_input="2"/>',
        groups='<PropertyGroup label="Isosurfaces"><Property name="ContourValues"/>'
               "</PropertyGroup>",
        mixins=(_ContourValues, _PointsPrecision)),

    "Clip": Panel(
        properties={
            "ClipType": """
<IntVectorProperty name="ClipType" label="Clip Type" command="SetClipType"
    number_of_elements="1" default_values="0" animateable="0">
  <EnumerationDomain name="enum"><Entry text="Plane" value="0"/>
  <Entry text="Scalar" value="1"/></EnumerationDomain>
  <Documentation>Clip by a plane, or by a field at a value.</Documentation>
</IntVectorProperty>""" + _PLANE,
            "by": Property(name="SelectInputScalars", label="Scalars", attribute="point",
                           components=1, domain=_shown_when("ClipType", 1),
                           doc="The point field to clip by."),
            "value": Property(name="Value", domain=_range_of("SelectInputScalars")
                              + _shown_when("ClipType", 1)),
            "inside": Property(name="Invert", default=True,
                               doc="Keep the part below the value (behind the plane) "
                                   "instead of above."),
        },
        order=("ClipType", "by", "value", "inside"),
        hints='<Visibility replace_input="2"/>',
        groups=_plane_group(_shown_when("ClipType", 0)),
        mixins=(_PlaneOrField,)),

    "SlicePlane": Panel(
        label="Tack Slice",
        properties={
            "origin": Property(name="Origin", domain=_bounds("mid")),
            "normal": Property(name="Normal", domain='<DoubleRangeDomain name="range"/>'),
            "value": Property(name="Offset", advanced=True,
                              doc="The slice's distance from the plane, along its "
                                  "normal."),
            "merge_points": Property(name="MergePoints", advanced=True),
        },
        hints='<Visibility replace_input="2"/>',
        groups=_plane_group()),

    "Threshold": Panel(
        properties={
            "field": Property(name="SelectInputScalars", label="Scalars"),
            "component": Property(
                name="SelectedComponent", kind="int", default=0,
                domain='<NumberOfComponentsDomain name="comps" enable_magnitude="0">'
                       '<RequiredProperties><Property function="Input" name="Input"/>'
                       '<Property function="ArraySelection" name="SelectInputScalars"/>'
                       "</RequiredProperties></NumberOfComponentsDomain>",
                doc="A vector field's component to threshold."),
            "lower": Property(name="LowerThreshold", default=0.0,
                              domain=_range_of("SelectInputScalars", "min")),
            "upper": Property(name="UpperThreshold", default=0.0,
                              domain=_range_of("SelectInputScalars", "max")),
            "ThresholdMethod": """
<IntVectorProperty name="ThresholdMethod" label="Threshold Method"
    command="SetThresholdMethod" number_of_elements="1" default_values="0">
  <EnumerationDomain name="enum"><Entry text="Between" value="0"/>
  <Entry text="Below Lower Threshold" value="1"/>
  <Entry text="Above Upper Threshold" value="2"/></EnumerationDomain>
  <Documentation>Keep the cells between the thresholds, below the lower one or above the
  upper one.</Documentation>
</IntVectorProperty>""",
            "all_points": Property(name="AllScalars",
                                   doc="A cell is kept when the field at all its points "
                                       "is in range; off, at any of them."),
        },
        order=("field", "component", "lower", "upper", "ThresholdMethod", "all_points"),
        hints='<Visibility replace_input="2"/>',
        mixins=(_ThresholdMethod,)),

    "Gradients": Panel(
        properties={
            "field": Property(name="ScalarArray", label="Scalar Array"),
            "at": Property(choices=(("Cells", "cells"), ("Points", "points")),
                           doc="Where the gradient is computed: at each cell's center "
                               "or at the points (each cell's gradient there, "
                               "averaged)."),
            "name": Property(name="ResultArrayName", label="Result Array Name"),
        }),

    "Elevation": Panel(
        properties={
            "low_point": Property(name="LowPoint", domain=_bounds("min")),
            "high_point": Property(name="HighPoint", domain=_bounds("max")),
            "range": Property(name="ScalarRange", domain='<DoubleRangeDomain name="range"/>'),
            "name": Property(name="ResultArrayName", label="Result Array Name"),
        },
        order=("range", "low_point", "high_point"),
        groups='<PropertyGroup panel_widget="InteractiveLine">'
               '<Property function="Point1WorldPosition" name="LowPoint"/>'
               '<Property function="Point2WorldPosition" name="HighPoint"/>'
               '<Property function="Input" name="Input"/></PropertyGroup>'),

    "CellQuality": Panel(
        properties={
            "measures": Property(
                name="Measure", choices=(
                    ("All", None), ("Scaled Jacobian", ("scaled_jacobian",)),
                    ("Jacobian Ratio", ("jacobian_ratio",)), ("Condition", ("condition",)),
                    ("Edge Ratio", ("edge_ratio",)), ("Min Edge", ("min_edge",)),
                    ("Max Edge", ("max_edge",))),
                doc="The measure computed, or all of them, each a cell array."),
        }),

    "ExtractSurface": Panel(),

    "Probe": Panel(ports=("Source",)),

    "Streamlines": Panel(
        ports=("Seeds",),
        properties={
            "field": Property(name="Vectors", components=3),
            "seeds": None,                  # the Seeds port's points
            "integrator": Property(choices=(("Runge-Kutta 4", "rk4"),
                                            ("Runge-Kutta 4-5", "rk45"),
                                            ("Euler", "euler"))),
            "step_unit": Property(choices=(("Time", "time"), ("Length", "length"))),
            "neighbors": Property(advanced=True),
        }),

    "Shrink": Panel(
        properties={
            "factor": Property(name="ShrinkFactor",
                               domain='<DoubleRangeDomain name="range" min="0" max="1"/>'),
        }),

    "FeatureEdges": Panel(
        properties={
            "boundary": Property(name="BoundaryEdges"),
            "feature": Property(name="FeatureEdges"),
            "non_manifold": Property(name="NonManifoldEdges"),
            "manifold": Property(name="ManifoldEdges"),
            "angle": Property(name="FeatureAngle",
                              domain='<DoubleRangeDomain name="range" min="0" max="180"/>'),
        },
        order=("boundary", "feature", "non_manifold", "manifold", "angle")),
}
