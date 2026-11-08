# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The panels of the proxies: what each takes from ParaView's filter of the kind
(names, labels, widgets, defaults) where the wrapper's parameters cannot say,
and the mixins behind the properties a panel adds. A state or a script moves
between ParaView's filter and Tack's by changing the proxy's name, for the
properties both have."""

import inspect

from ._generate import FUNCTIONS, Panel, Property, _watch

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

    _DEFAULT = ()

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self._values = list(self._DEFAULT)

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


class _SliceOffsets(_ContourValues):
    """ParaView's Slice Offset Values: one, zero, until ParaView sets them."""

    _DEFAULT = (0.0,)


class _ClipFunction:
    """``by`` as ParaView's Clip Type: an implicit function's proxy (Plane, Box, ...),
    or ``None`` for Scalar, the field the Scalars menu selects."""

    def SetClipFunction(self, function):
        _watch(self, "by", function)

    def arguments(self):
        arguments = super().arguments()
        if self.by is not None:
            # The Value is the field's (its range domain fills it in): at the function.
            arguments["value"] = 0.0
        return arguments


class _SamplingGrid:
    """ParaView's Resample To Image: the image a filter samples onto (``grid``), over
    the input's bounds (every rank's) or the bounds given, with so many points along
    each axis."""

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self._use_input_bounds, self._bounds, self._dimensions = True, (0, 1, 0, 1, 0, 1), \
            (100, 100, 100)

    def SetUseInputBounds(self, use):
        self._use_input_bounds = bool(use)
        self.Modified()

    def SetSamplingBounds(self, *bounds):
        self._bounds = tuple(float(b) for b in bounds)
        self.Modified()

    def SetSamplingDimensions(self, *dimensions):
        self._dimensions = tuple(max(int(d), 1) for d in dimensions)
        self.Modified()

    def _prepare(self, inInfo):
        super()._prepare(inInfo)
        from vtkmodules.vtkCommonDataModel import vtkImageData

        bounds = self._bounds
        if self._use_input_bounds:
            bounds = _bounds_over(self.GetInputData(inInfo, 0, 0), self._run_world)
        spacing = [(bounds[2 * a + 1] - bounds[2 * a]) / (n - 1) if n > 1 else 1.0
                   for a, n in enumerate(self._dimensions)]
        grid = vtkImageData(origin=bounds[0::2], spacing=spacing,
                            dimensions=self._dimensions)
        # Made for this run from its input: set without Modified(), which would run
        # the filter again.
        self.__dict__[inspect.getattr_static(type(self), "grid").slot] = grid


def _bounds_over(data, world):
    """The bounds of ``data``'s leaves on every rank of ``world``."""
    boxes = [leaf.GetBounds() for leaf in _leaves(data) if leaf.GetNumberOfPoints()]
    mine = tuple((min, max)[i % 2](b[i] for b in boxes) for i in range(6)) if boxes else None
    boxes = [b for b in world._allgather(mine) if b is not None]
    if not boxes:
        return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    return tuple((min, max)[i % 2](b[i] for b in boxes) for i in range(6))


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

def _functions(*names):
    """ParaView's implicit functions ``names`` as a property's proxies."""
    return tuple(("implicit_functions", name) for name in names)


def _result(doc=None):
    """A new array's name, as ParaView names the property."""
    return Property(name="ResultArrayName", label="Result Array Name", doc=doc)


def _side():
    return Property(name="ExtractInside", label="Extraction Side",
                    choices=(("outside", False), ("inside", True)),
                    doc="Keep what is inside the function (where it is negative), or "
                        "outside.")


def _weights(*extra):
    return Property(choices=(("None", None), *extra),
                    doc="How each value is weighed: equally, or by each cell's measure "
                        "(its volume, area or length).")


def _scaled_extent(factor):
    """A length, by default ``factor`` times the input's largest extent."""
    return (f'<BoundsDomain name="range" mode="scaled_extent" scale_factor="{factor}">'
            f"{_INPUT}</BoundsDomain>")


def _unset(kind, doc, **extra):
    """An optional number: zero (``unset``) leaves it to the function."""
    return Property(kind=kind, unset=0 if kind == "int" else 0.0, doc=doc, **extra)


_SAMPLING = """
<IntVectorProperty name="UseInputBounds" command="SetUseInputBounds"
    number_of_elements="1" default_values="1">
  <BooleanDomain name="bool"/>
  <Documentation>Sample over the input's bounds (every rank's), or the bounds
  given.</Documentation>
</IntVectorProperty>
<IntVectorProperty name="SamplingDimensions" command="SetSamplingDimensions"
    number_of_elements="3" default_values="100 100 100">
  <IntRangeDomain name="range" min="1 1 1"/>
  <Documentation>The image's points along each axis.</Documentation>
</IntVectorProperty>
<DoubleVectorProperty name="SamplingBounds" command="SetSamplingBounds"
    number_of_elements="6" default_values="0 1 0 1 0 1">
  <BoundsDomain name="bounds">%s</BoundsDomain>
  <Hints><PropertyWidgetDecorator type="GenericDecorator" mode="visibility"
      property="UseInputBounds" value="0"/></Hints>
  <Documentation>The image's bounds, when not the input's.</Documentation>
</DoubleVectorProperty>""" % _INPUT

_INTEGRATORS = (("Runge-Kutta 4", "rk4"), ("Euler", "euler"))

def _kernels():
    return Property(choices=(("Cubic", "cubic"), ("Quartic", "quartic"),
                             ("Quintic", "quintic"), ("Wendland", "wendland")))


def _particles(**extra):
    """A temporal particle filter's panel: the velocity, the seeds on their port."""
    return Panel(
        ports=("Seeds",),
        properties={
            "field": Property(name="Vectors", components=3),
            "integrator": Property(choices=_INTEGRATORS),
            "step_size": _unset("double", "The longest substep, in the steps' time unit; "
                                          "zero, one substep from step to step."),
            "copy": Property(advanced=True),
            **extra,
        })


#: The panels by wrapper; a wrapper without one has no proxy yet (``Interpolate``,
#: whose function is a ``@tack.func``, and ``Composite``, whose fields are in order).
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
            "ClipFunction": """
<ProxyProperty name="ClipFunction" label="Clip Type" command="SetClipFunction">
  <ProxyListDomain name="proxy_list">%s<Proxy group="implicit_functions" name="Scalar"/>
  </ProxyListDomain>
  <Documentation>Clip by an implicit function, or by a field at a value
  (Scalar).</Documentation>
</ProxyProperty>""" % "".join(f'<Proxy group="{g}" name="{n}"/>' for g, n in
                              _functions(*FUNCTIONS)),
            "by": Property(name="SelectInputScalars", label="Scalars", attribute="point",
                           components=1, domain=_shown_when("ClipFunction", "Scalar"),
                           doc="The point field to clip by."),
            "value": Property(name="Value", domain=_range_of("SelectInputScalars")
                              + _shown_when("ClipFunction", "Scalar")),
            "inside": Property(name="Invert", default=True,
                               doc="Keep the part below the value (inside the function) "
                                   "instead of above."),
        },
        order=("ClipFunction", "by", "value", "inside"),
        hints='<Visibility replace_input="2"/>',
        mixins=(_ClipFunction,)),

    "Slice": Panel(
        properties={
            "function": Property(name="CutFunction", label="Slice Type",
                                 proxies=_functions(*FUNCTIONS),
                                 doc="The implicit function whose zero set is the "
                                     "slice."),
            "value": """
<DoubleVectorProperty name="ContourValues" label="Slice Offset Values" command="SetValue"
    number_of_elements="1" default_values="0" number_of_elements_per_command="1"
    repeat_command="1" set_number_command="SetNumberOfContours" use_index="1"
    animateable="1">
  <BoundsDomain name="bounds">%s</BoundsDomain>
  <Documentation>The function's values to slice at, every one in one pass, each output
  cell's in a cell array "contour_value" when there are several.</Documentation>
</DoubleVectorProperty>""" % _INPUT,
            "merge_points": Property(name="MergePoints", advanced=True),
        },
        order=("function", "value", "merge_points"),
        hints='<Visibility replace_input="2"/>',
        mixins=(_SliceOffsets,)),

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

    "ThresholdPoints": Panel(
        properties={
            "field": Property(name="SelectInputScalars", label="Scalars"),
            "lower": Property(name="LowerThreshold", kind="double", default=0.0,
                              domain=_range_of("SelectInputScalars", "min")),
            "upper": Property(name="UpperThreshold", kind="double", default=0.0,
                              domain=_range_of("SelectInputScalars", "max")),
        }),

    "ExtractGeometry": Panel(
        label="Tack Extract Cells By Region",
        properties={
            "function": Property(name="ImplicitFunction", label="Intersect With",
                                 proxies=_functions("Plane", "Box", "Sphere"),
                                 doc="The implicit function the cells are taken by."),
            "inside": _side(),
            "boundary": Property(name="ExtractIntersected", label="Extract intersected",
                                 doc="Keep the cells the function cuts as well."),
        },
        hints='<Visibility replace_input="2"/>'),

    "ExtractPoints": Panel(
        properties={
            "function": Property(name="ImplicitFunction", label="Intersect With",
                                 proxies=_functions("Plane", "Box", "Sphere"),
                                 doc="The implicit function the points are taken by."),
            "inside": _side(),
        },
        hints='<Visibility replace_input="2"/>'),

    "ImplicitValues": Panel(
        properties={
            "function": Property(name="ImplicitFunction", label="Implicit Function",
                                 proxies=_functions(*FUNCTIONS),
                                 doc="The implicit function sampled at the points."),
            "name": _result(),
        }),

    "ExtractCells": Panel(
        properties={
            "cells": Property(name="Cells", kind="string", default="",
                              doc="The cells: an integer cell field's name (the cells "
                                  "where it is nonzero) or a cell set's."),
        }),

    "ExtractExtent": Panel(
        label="Tack Extract Subset",
        properties={
            "extent": Property(name="VOI", kind="int", count=6, default=(0,) * 6,
                               domain=f'<ExtentDomain name="extent">{_INPUT}</ExtentDomain>',
                               doc="The box of points kept: the low and high index along "
                                   "each axis."),
            "rate": Property(name="SampleRate", doc="Every so many points along each "
                                                    "axis."),
        }),

    "Mask": Panel(
        properties={"stride": Property(name="Stride", kind="int", default=2,
                                       doc="Keep every so many cells.")}),

    "MaskPoints": Panel(
        properties={"stride": Property(name="OnRatio", kind="int", default=2,
                                       doc="Keep every so many points.")}),

    "Gradients": Panel(
        properties={
            "field": Property(name="ScalarArray", label="Scalar Array"),
            "at": Property(choices=(("Cells", "cells"), ("Points", "points")),
                           doc="Where the gradient is computed: at each cell's center "
                               "or at the points (each cell's gradient there, "
                               "averaged)."),
            "name": _result(),
        }),

    "FlowQuantities": Panel(
        properties={"gradient": Property(name="GradientArray", label="Gradient",
                                         components=9)}),

    "Elevation": Panel(
        properties={
            "low_point": Property(name="LowPoint", domain=_bounds("min")),
            "high_point": Property(name="HighPoint", domain=_bounds("max")),
            "range": Property(name="ScalarRange", domain='<DoubleRangeDomain name="range"/>'),
            "name": _result(),
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

    "Transform": Panel(
        properties={
            "matrix": Property(name="Transform", proxies=(("extended_sources",
                                                           "Transform3"),),
                               doc="The translation, rotation and scale."),
        }),

    "Warp": Panel(
        properties={
            "direction": Property(name="Vectors", components=3),
            "scale": Property(name="ScaleFactor"),
            "scale_by": Property(name="ScaleBy", label="Scale By", optional=True,
                                 doc="A field the displacement is also scaled by."),
        }),

    "Glyphs": Panel(
        properties={
            "vectors": Property(name="OrientationArray", label="Orientation Array",
                                components=3),
            "scale": Property(name="ScaleFactor", domain=_scaled_extent(0.1)),
            "scale_by": Property(name="ScaleArray", label="Scale Array", optional=True,
                                 doc="A field each glyph is also scaled by."),
            "resolution": Property(name="Resolution"),
        }),

    "Tubes": Panel(
        properties={
            "radius": Property(name="Radius", kind="double", default=1.0,
                               domain=_scaled_extent(0.01)),
            "sides": Property(name="NumberOfSides"),
            "radius_by": Property(name="RadiusBy", label="Radius By", optional=True,
                                  doc="A field the radius is also scaled by."),
        }),

    "ExtractSurface": Panel(),

    "Probe": Panel(ports=("Source",)),

    "Deposit": Panel(
        properties={
            "grid": None,                   # the sampling grid's
            "field": Property(name="Field", optional=True,
                              doc="The field deposited; none, the points counted."),
            "reduce": Property(choices=(("Sum, Or Count", None), ("Sum", "sum"),
                                        ("Mean", "mean"),
                                        ("Count", "count"), ("Min", "min"),
                                        ("Max", "max")),
                               doc="How the values reaching a cell combine (by default "
                                   "summed, or the points counted without a field); "
                                   "Min and Max with Nearest only."),
            "method": Property(choices=(("Nearest", "nearest"), ("Cloud In Cell", "cic")),
                               doc="Each point to the cell holding it, or spread over "
                                   "that cell's corners."),
            "name": _result(),
            "_sampling": _SAMPLING,
        },
        order=("_sampling",),
        mixins=(_SamplingGrid,)),

    "Voxelize": Panel(
        properties={"grid": None, "_sampling": _SAMPLING},
        order=("_sampling",),
        mixins=(_SamplingGrid,)),

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

    "Advect": Panel(
        ports=("Seeds",),
        properties={
            "field": Property(name="Vectors", components=3),
            "seeds": None,
            "step_size": Property(name="StepSize", kind="double", default=0.01),
            "steps": Property(name="NumberOfSteps", kind="int", default=100),
            "integrator": Property(choices=(("Runge-Kutta 4", "rk4"),
                                            ("Runge-Kutta 4-5", "rk45"),
                                            ("Euler", "euler"))),
            "step_unit": Property(choices=(("Time", "time"), ("Length", "length"))),
            "tolerance": _unset("double", "Runge-Kutta 4-5's error per step; zero, the "
                                          "function's.", advanced=True),
            "min_step": _unset("double", "Runge-Kutta 4-5's smallest step; zero, the "
                                         "function's.", advanced=True),
            "max_step": _unset("double", "Runge-Kutta 4-5's largest step; zero, the "
                                         "function's.", advanced=True),
            "max_length": _unset("double", "Stop a particle that has gone so far; zero, "
                                           "never."),
            "terminal_speed": _unset("double", "Stop a particle at or below this speed; "
                                               "zero, never.", advanced=True),
            "neighbors": Property(advanced=True),
        }),

    "Advection": _particles(),
    "Pathlines": _particles(),
    "Streaklines": _particles(),
    "RunningStatistics": Panel(),
    "TimeDerivative": Panel(properties={"copy": Property(advanced=True)}),

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

    "SurfaceNormals": Panel(
        properties={
            "consistent": Property(name="Consistency"),
            "weights": Property(choices=(("None", None), ("Area", "area")),
                                doc="Each cell's normal weighed equally in its points' "
                                    "sum, or by its area."),
        }),

    "Triangulate": Panel(),
    "Tetrahedralize": Panel(),
    "AsPolygons": Panel(),
    "AsPolyhedra": Panel(),
    "BoundaryFaces": Panel(),
    "ExtractEdges": Panel(),
    "Subdivide": Panel(properties={
        "level": _unset("int", "The sub-cells along each parametric axis; zero, the "
                               "function's.")}),
    "MergePoints": Panel(properties={
        "tolerance": Property(name="Tolerance", doc="How far apart two points made one "
                                                    "may be.")}),
    "RemoveGhosts": Panel(),

    # Fields made from fields.
    "Magnitude": Panel(properties={"name": _result()}),
    "LogValues": Panel(properties={
        "name": _result(),
        "min_value": _unset("double", "Values below it are taken as it; zero, the "
                                      "smallest positive normal float.")}),
    "Dot": Panel(properties={"a": Property(name="VectorA", label="A"),
                             "b": Property(name="VectorB", label="B"),
                             "name": _result()}),
    "Cross": Panel(properties={"a": Property(name="VectorA", label="A", components=3),
                               "b": Property(name="VectorB", label="B", components=3),
                               "name": _result()}),
    "Project": Panel(properties={
        "to": Property(choices=(("Points", "points"), ("Cells", "cells")),
                       doc="Where the field goes."),
        "average": Property(choices=(("Integral", "integral"), ("Nodes", "nodes"))),
        "name": Property(name="ResultArrayName", label="Result Array Name", kind="string",
                         default="", doc="The new array's name; empty, the field's."),
    }),
    "ValuesAtCenters": Panel(properties={
        "name": Property(name="ResultArrayName", label="Result Array Name", kind="string",
                         default="", doc="The new array's name; empty, the field's."),
    }),
    "Jump": Panel(properties={"name": _result()}),
    "UpwindFlux": Panel(properties={
        "velocity": Property(name="Velocity", kind="double", count=3,
                             default=(1.0, 0.0, 0.0), doc="The constant velocity."),
        "name": _result()}),
    "AverageDown": Panel(),

    # Geometry and ids.
    "CellCenters": Panel(properties={"name": _result()}),
    "CellGeometry": Panel(),
    "CellIds": Panel(properties={"name": _result()}),
    "PointIds": Panel(properties={"name": _result()}),
    "FaceCenters": Panel(properties={"name": _result()}),
    "FaceGeometry": Panel(),
    "EdgeLengths": Panel(properties={"name": _result()}),
    "CheckWinding": Panel(properties={"name": _result()}),
    "Cylindrical": Panel(),
    "Spherical": Panel(),
    "Convert": Panel(properties={
        "unit": Property(name="Unit", kind="string", default="m",
                         doc="The unit to convert to, such as m, km or ft."),
    }),

    # Reductions over the whole input.
    "Statistics": Panel(properties={"weights": _weights(("Volume", "volume"))}),
    "Entropy": Panel(properties={"weights": _weights(("Volume", "volume"))}),
    "Integrate": Panel(label="Tack Integrate Variables"),

    # Pieces.
    "Split": Panel(properties={
        "pieces": Property(name="NumberOfPieces", kind="int", default=4,
                           doc="The pieces the grid is cut into.")}),
    "MergePieces": Panel(properties={
        "root": Property(advanced=True),
        "tolerance": Property(name="Tolerance")}),
    "GenerateGlobalIds": Panel(properties={"tolerance": Property(name="Tolerance")}),
    "GhostCells": Panel(properties={"tolerance": Property(name="Tolerance",
                                                          advanced=True)}),

    # Point clouds and particles.
    "GhostPoints": Panel(properties={
        "radius": _unset("double", "Copy the points within this distance of a piece; "
                                   "zero, none."),
        "k": _unset("int", "Copy every point among an owned point's k nearest; zero, "
                           "none.", default=10)}),
    "ConnectedComponents": Panel(properties={
        "radius": _unset("double", "Points within this distance are connected; zero, "
                                   "cells through their points."),
        "min_size": _unset("int", "Regions of fewer points are labelled -1; zero, "
                                  "none.")}),
    "Outliers": Panel(properties={
        "radius": _unset("double", "The radius test's radius; zero, the statistical "
                                   "test."),
        "min_neighbors": _unset("int", "The radius test: fewer other points within the "
                                       "radius is an outlier."),
        "k": _unset("int", "The statistical test's neighbours; zero, the radius test.",
                    default=10)}),
    "PointNormals": Panel(properties={
        "k": _unset("int", "The nearest points a normal is fitted to; zero, those within "
                           "the radius.", default=10),
        "radius": _unset("double", "The points within it a normal is fitted to.")}),
    "VoxelDownsample": Panel(properties={
        "size": Property(name="VoxelSize", kind="double", default=1.0,
                         domain=_scaled_extent(0.01)),
        "keep": Property(choices=(("Centroid", "centroid"), ("First", "first"),
                                  ("Nearest", "nearest")),
                         doc="The point a voxel keeps: its points' mean, its first, or "
                             "the one nearest the mean.")}),
    "SphDensity": Panel(label="Tack SPH Density", properties={"kernel": _kernels()}),
    "SphInterpolate": Panel(label="Tack SPH Interpolate", ports=("Source",),
                            properties={"kernel": _kernels()}),
    "HaloCatalog": Panel(properties={
        "velocity": Property(kind="string", default="",
                             doc="A 3-vector point field: each halo's mean velocity and "
                                 "dispersion; empty, none."),
        "center": Property(choices=(("Most Bound Particle", "mbp"),
                                    ("Center Of Mass", "com")))}),
}
