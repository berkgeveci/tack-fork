# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""ParaView proxies made from ``tack.interop.vtk.algorithms``' wrappers.

A proxy is a subclass of a wrapper with a setter for each property and its
ServerManager XML as ``_pvsm_proxy_xml``, which ParaView's Python plugin loader
collects from the plugin module's namespace (``get_plugin_xmls``). Its class is
named in the XML by its module, ``tack.interop.paraview``, which every process
running the filter imports. Nothing here imports ParaView.

A property is inferred from the wrapper's parameter: its default's type gives its
kind (a bool a check box, an int or a float a number, a tuple of numbers that
many), a field parameter (``FIELDS``) an array menu. A ``Panel`` gives what the
parameters cannot: ParaView's names, labels and widgets, choices, parameters
hidden, properties of its own (with the methods they call, in a mixin).

A property may also hold a ParaView proxy (``proxies``): an implicit function
(Plane, Box, Sphere, Cylinder, with their widgets) or a transform. The filter
watches the object it is given, since ParaView changes it in place, and
converts ParaView's own functions (``vtkPVBox``, ``vtkPVCylinder``) to Tack's.
"""

import importlib.util
import inspect
import re
import weakref
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

from tack.interop.vtk.algorithms import Parameter

#: The module ParaView imports a proxy's class from.
MODULE = "tack.interop.paraview"

#: Parameters no panel shows unless it says so: the fields a filter carries
#: (every one by default) and what only higher-order cells use.
HIDDEN = frozenset({"fields", "subdivision", "tolerance", "exact"})

_UNSET = object()

#: The entry of an optional field's menu that chooses no field.
NO_FIELD = "None"

#: ParaView's implicit functions a filter offers, by default.
FUNCTIONS = ("Plane", "Box", "Sphere", "Cylinder")


@dataclass
class Property:
    """A property of a panel, setting a wrapper's parameter. Anything left ``None``
    is inferred from the parameter."""

    #: ParaView's name for it, by default the parameter's in CamelCase.
    name: str = None
    label: str = None
    #: ``"bool"``, ``"int"``, ``"double"``, ``"string"``, ``"enum"`` or ``"field"``.
    kind: str = None
    #: The number of elements (a point is 3).
    count: int = None
    default: object = _UNSET
    #: An enumeration's entries: ``(text, value the parameter takes)``.
    choices: tuple = ()
    advanced: bool = False
    doc: str = None
    #: XML put inside the property: domains, hints.
    domain: str = ""
    #: The label of the property group it goes in.
    group: str = None
    #: A field's: ``"point"`` or ``"cell"`` where it must be one, and its number of
    #: components where it must have so many; ``optional`` adds an entry for none.
    attribute: str = None
    components: int = None
    optional: bool = False
    #: A number that means the parameter's ``None`` (what the function works out).
    unset: object = None
    #: A proxy property's proxies, ``(group, name)``: the object ParaView makes of
    #: the one chosen is the parameter's value.
    proxies: tuple = ()


@dataclass
class Panel:
    """What a proxy's panel takes from ParaView where the wrapper cannot say."""

    label: str = None
    #: ``{parameter: Property}``, or ``None`` to hide one; ``{name: str}`` for a
    #: property of the panel's own, its XML whole (its commands the mixin's).
    properties: dict = field(default_factory=dict)
    #: The order of the properties, by those keys; the rest follow in the wrapper's.
    order: tuple = ()
    #: The input ports' names, after ``"Input"``.
    ports: tuple = ()
    #: XML in the proxy's ``<Hints>``.
    hints: str = ""
    #: XML after the properties (``PropertyGroup``s with a widget).
    groups: str = ""
    #: Classes put before the wrapper, for what the panel's own properties call.
    mixins: tuple = ()


def available_devices():
    """The device types whose packages (or, for Level Zero, runtime) are present.
    Nothing is initialized: a device is made only when a filter runs on it."""
    import ctypes

    found = ["cpu"]
    for device, module in (("cuda", "cuda.bindings"), ("hip", "hip"), ("metal", "Metal")):
        try:
            if importlib.util.find_spec(module) is not None:
                found.append(device)
        except (ImportError, ValueError):
            pass
    for library in ("libze_loader.so.1", "ze_loader.dll"):
        try:
            ctypes.CDLL(library)
            found.append("level_zero")
            break
        except OSError:
            pass
    return found


#: The devices a proxy offers, the first meaning "where the input's memory is".
DEVICES = ("(input's)", *available_devices())


def _camel(name):
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))


def _words(name):
    """A label from a parameter's or a ParaView name: ``step_size``, ``StepSize`` →
    ``Step Size``."""
    if "_" in name or name.islower():
        return " ".join(part.capitalize() for part in name.split("_"))
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def _text(doc):
    """A docstring as plain text for ParaView: reST's literals unquoted, lines joined."""
    return " ".join((doc or "").replace("``", "").split())


def _values(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (tuple, list)):
        return " ".join(_values(v) for v in value)
    return str(value)


def _element(tag, attrs, inner=""):
    text = "".join(f" {k}={quoteattr(str(v))}" for k, v in attrs.items() if v is not None)
    return f"<{tag}{text}>{inner}</{tag}>"


def _parameters(wrapper):
    """The wrapper's parameters, the function's in its order, then the class's own;
    not ``device`` and ``world``, which every proxy handles alike."""
    names = list(getattr(wrapper, "_ARGUMENTS", ()))
    for klass in reversed(wrapper.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, Parameter) and name not in names:
                names.append(name)
    return [n for n in names if n not in ("device", "world")]


def _infer(wrapper, parameter, given):
    """``given`` (a ``Property``, or ``None`` for one to infer) completed from the
    parameter; ``None`` where it shows nothing."""
    prop = given or Property()
    default = inspect.getattr_static(wrapper, parameter).default
    # A number that stands for None gives the kind of a parameter None by default.
    sample = prop.unset if default is None and prop.unset is not None else default
    if prop.kind is None:
        if parameter in wrapper.FIELDS:
            prop.kind = "field"
        elif prop.proxies:
            prop.kind = "proxy"
        elif prop.choices:
            prop.kind = "enum"
        elif given is None and (parameter in HIDDEN or default is None):
            return None
        elif isinstance(sample, bool):
            prop.kind = "bool"
        elif isinstance(sample, int):
            prop.kind = "int"
        elif isinstance(sample, float):
            prop.kind = "double"
        elif isinstance(sample, str):
            prop.kind = "string"
        elif (isinstance(sample, tuple) and sample
              and all(isinstance(v, (int, float)) for v in sample)):
            prop.kind = "double"
        else:
            raise TypeError(f"{wrapper.__name__}.{parameter}: no property for a default "
                            f"of {default!r}; give the panel one")
    if prop.count is None:
        prop.count = len(default) if isinstance(default, tuple) else 1
    if prop.default is _UNSET:
        if default is None and prop.unset is not None:
            prop.default = prop.unset
        elif prop.kind == "enum":
            values = [value for _, value in prop.choices]
            prop.default = values.index(default) if default in values else 0
        else:
            prop.default = default
    if prop.name is None:
        prop.name = _camel(parameter)
    if prop.label is None:
        prop.label = _words(prop.name)
    if prop.doc is None:
        prop.doc = inspect.getattr_static(wrapper, parameter).__doc__
    return prop


def _watch(algorithm, parameter, value):
    """``value`` (a VTK object, or ``None``) as ``parameter``'s, the algorithm modified
    whenever it is: ParaView sets an implicit function's or a transform's properties
    on the object itself."""
    watched = algorithm.__dict__.setdefault("_watched", {})
    old = watched.pop(parameter, None)
    if old is not None:
        old[0].RemoveObserver(old[1])
    if value is not None:
        owner = weakref.ref(algorithm)

        def modified(caller, event):
            algorithm = owner()
            if algorithm is not None:
                algorithm.Modified()

        watched[parameter] = (value, value.AddObserver("ModifiedEvent", modified))
    setattr(algorithm, parameter, value)


def _setter(parameter, prop):
    """The method ParaView calls with the property's elements."""
    if prop.kind == "proxy":
        def set_value(self, value):
            _watch(self, parameter, value)
    elif prop.kind == "bool":
        def set_value(self, value):
            setattr(self, parameter, bool(value))
    elif prop.kind == "enum":
        choices = [value for _, value in prop.choices]

        def set_value(self, index):
            setattr(self, parameter, choices[int(index)])
    elif prop.kind == "string":
        # ParaView sends an empty string as None: the parameter's None.
        def set_value(self, value):
            setattr(self, parameter, None if value is None or value == "" else str(value))
    else:
        convert = int if prop.kind == "int" else float
        unset = prop.unset
        if prop.count == 1:
            def set_value(self, value):
                value = convert(value)
                setattr(self, parameter, None if unset is not None and value == unset
                        else value)
        else:
            def set_value(self, *values):
                setattr(self, parameter, tuple(convert(v) for v in values))
    set_value.__name__ = f"Set{prop.name}"
    return set_value


def _property_xml(prop, index=None):
    """A property's XML; ``index`` is a field's ``SetInputArrayToProcess`` index."""
    doc = _element("Documentation", {}, escape(_text(prop.doc)))
    common = {"name": prop.name, "label": prop.label,
              "panel_visibility": "advanced" if prop.advanced else None}
    if prop.kind == "proxy":
        proxies = "".join(_element("Proxy", {"group": group, "name": name})
                          for group, name in prop.proxies)
        return _element("ProxyProperty", {**common, "command": f"Set{prop.name}"},
                        _element("ProxyListDomain", {"name": "proxy_list"}, proxies)
                        + prop.domain + doc)
    if prop.kind == "field":
        domain = _element("ArrayListDomain", {
            "name": "array_list",
            "attribute_type": "Vectors" if prop.components == 3 else "Scalars",
            "input_domain_name": f"input_array_{index}",
            "none_string": NO_FIELD if prop.optional else None},
            '<RequiredProperties><Property name="Input" function="Input"/>'
            "</RequiredProperties>")
        # An optional field's default names none: ParaView would choose the active
        # array of its kind.
        defaults = {"default_values": f"{index};0;0;0;{NO_FIELD}",
                    "default_values_delimiter": ";"} if prop.optional else {
                        "default_values": index}
        return _element("StringVectorProperty", {
            **common, "command": "SetInputArrayToProcess", "number_of_elements": 5,
            "element_types": "0 0 0 0 2", **defaults, "animateable": 0},
            domain + prop.domain + doc)
    kind = {"bool": "IntVectorProperty", "int": "IntVectorProperty",
            "enum": "IntVectorProperty", "double": "DoubleVectorProperty",
            "string": "StringVectorProperty"}[prop.kind]
    domain = prop.domain
    if prop.kind == "bool":
        domain = '<BooleanDomain name="bool"/>' + domain
    elif prop.kind == "enum":
        entries = "".join(_element("Entry", {"text": text, "value": i})
                          for i, (text, _) in enumerate(prop.choices))
        domain = _element("EnumerationDomain", {"name": "enum"}, entries) + domain
    return _element(kind, {
        **common, "command": f"Set{prop.name}", "number_of_elements": prop.count,
        "default_values": _values(prop.default)}, domain + doc)


def _device_xml():
    entries = "".join(_element("Entry", {"text": name, "value": i})
                      for i, name in enumerate(DEVICES))
    return _element("IntVectorProperty", {
        "name": "Device", "command": "SetDeviceIndex", "number_of_elements": 1,
        "default_values": 0, "animateable": 0},
        _element("EnumerationDomain", {"name": "enum"}, entries)
        + "<Documentation>The device Tack runs the filter on: where the input's "
          "memory is, or one named. The input moves there once.</Documentation>")


def _set_device_index(self, index):
    index = int(index)
    self.device = None if index == 0 else DEVICES[index]


def _none_for_no_field(cls):
    """``cls``'s ``_input_array``, ``None`` where an optional field's menu says none."""
    def _input_array(self, index):
        value = super(cls, self)._input_array(index)
        name = value[0] if isinstance(value, tuple) else value
        return None if name == NO_FIELD else value

    return _input_array


def implicit_function(value):
    """An implicit function as Tack's: ParaView's ``vtkPVBox`` (a ``vtkBox`` placed by a
    transform, which may turn it only by right angles, Tack's ``Box`` being along the
    axes) and ``vtkPVCylinder`` (a ``vtkCylinder`` turned by a transform), and the
    rest as ``tack.interop.vtk.algorithms`` converts them. Python has only VTK's
    classes of the two, so they are read as those and their transforms."""
    import tack.dataset as td
    from tack.interop.vtk.algorithms import _implicit

    name = value.GetClassName() if hasattr(value, "GetClassName") else None
    if name not in ("vtkPVBox", "vtkPVCylinder"):
        return _implicit(value)
    # The transform takes world positions into the function's space.
    transform = value.GetTransform()
    inverse = None if transform is None else transform.GetLinearInverse()
    if name == "vtkPVCylinder":
        center, axis = value.GetCenter(), value.GetAxis()
        if inverse is not None:
            rows = _rows(transform)
            if not all(abs(sum(a * b for a, b in zip(rows[i], rows[j])) - (i == j)) < 1e-9
                       for i in range(3) for j in range(3)):
                raise ValueError("a Cylinder scaled by its transform has no Tack "
                                 "implicit function")
            center, axis = inverse.TransformPoint(center), inverse.TransformVector(axis)
        return td.Cylinder(center, axis, value.GetRadius())
    bounds = value.GetBounds()
    lower, upper = bounds[0::2], bounds[1::2]
    if inverse is not None:
        for row in _rows(transform):
            magnitudes = sorted(abs(v) for v in row)
            if magnitudes[1] > 1e-9 * magnitudes[2]:
                raise ValueError("a turned Box has no Tack implicit function: Tack's Box "
                                 "is along the axes")
        a, b = inverse.TransformPoint(lower), inverse.TransformPoint(upper)
        lower, upper = tuple(map(min, a, b)), tuple(map(max, a, b))
    return td.Box(lower, upper)


def _rows(transform):
    """A linear transform's 3x3 part, by rows."""
    matrix = transform.GetMatrix()
    return [[matrix.GetElement(i, j) for j in range(3)] for i in range(3)]


def _inputs_xml(wrapper, panel, fields):
    """The input ports' properties, port 0's with the field parameters' domains."""
    required = wrapper.INPUTS if wrapper.REQUIRED_INPUTS is None else wrapper.REQUIRED_INPUTS
    names = ("Input", *panel.ports)
    xml = []
    for port in range(wrapper.INPUTS):
        inner = ('<ProxyGroupDomain name="groups"><Group name="sources"/>'
                 '<Group name="filters"/></ProxyGroupDomain>'
                 '<DataTypeDomain name="input_type" composite_data_supported="1">'
                 '<DataType value="vtkDataSet"/></DataTypeDomain>')
        if port == 0:
            for index, prop in enumerate(fields):
                inner += _element("InputArrayDomain", {
                    "name": f"input_array_{index}", "attribute_type": prop.attribute,
                    "number_of_components": prop.components,
                    "optional": 1 if prop.optional else None})
        if port >= required:
            inner += "<Hints><Optional/></Hints>"
        xml.append(_element("InputProperty", {
            "name": names[port] if port < len(names) else f"Input{port}",
            "command": "SetInputConnection", "port_index": port}, inner))
    return xml


def proxy(name, wrapper, panel):
    """The proxy class ``name``: ``wrapper`` with ``panel``'s properties."""
    entries, fields_by_index = {}, {}
    for parameter in _parameters(wrapper):
        given = panel.properties.get(parameter, _UNSET)
        if given is None:
            continue
        if isinstance(given, str):
            entries[parameter] = given
            continue
        prop = _infer(wrapper, parameter, None if given is _UNSET else given)
        if prop is None:
            if parameter in getattr(wrapper, "_REQUIRED", ()):
                raise TypeError(f"{wrapper.__name__}.{parameter} is required, and "
                                "the panel shows no property for it")
            continue
        entries[parameter] = prop
    for key, given in panel.properties.items():
        if isinstance(given, str) and key not in entries:
            entries[key] = given
    order = [k for k in panel.order if k in entries]
    order += [k for k in entries if k not in order]

    namespace = {"__module__": MODULE, "__qualname__": name,
                 "__doc__": wrapper.__doc__, "SetDeviceIndex": _set_device_index,
                 "_implicit_function": staticmethod(implicit_function)}
    properties, groups = [], {}
    for key in order:
        entry = entries[key]
        if isinstance(entry, str):
            properties.append(entry)
            continue
        if entry.kind == "field":
            index = wrapper.FIELDS.index(key)
            properties.append(_property_xml(entry, index))
            fields_by_index[index] = entry
        else:
            properties.append(_property_xml(entry))
            namespace[f"Set{entry.name}"] = _setter(key, entry)
        if entry.group:
            groups.setdefault(entry.group, []).append(entry.name)
    # The fields' domains by SetInputArrayToProcess's index, which names them.
    fields = [fields_by_index.get(i, Property(kind="field"))
              for i in range(len(wrapper.FIELDS))]

    properties.append(_device_xml())
    groups.setdefault("Tack", []).append("Device")
    group_xml = "".join(
        _element("PropertyGroup", {"label": label},
                 "".join(f'<Property name="{n}"/>' for n in names))
        for label, names in groups.items())
    label = panel.label or f"Tack {_words(wrapper.__name__)}"
    hints = _element("Hints", {}, '<ShowInMenu category="Tack"/>' + panel.hints)
    short = _text(wrapper.__doc__).split(". ")[0].rstrip(".") + "."
    doc = _element("Documentation", {"short_help": short}, escape(_text(wrapper.__doc__)))
    body = (doc + "".join(_inputs_xml(wrapper, panel, fields)) + "".join(properties)
            + group_xml + panel.groups + hints)
    xml = _element("ServerManagerConfiguration", {}, _element(
        "ProxyGroup", {"name": "filters"}, _element(
            "SourceProxy", {"name": name, "class": f"{MODULE}.{name}", "label": label,
                            "si_class": "vtkSIPythonSourceProxy"}, body)))
    cls = type(name, (*panel.mixins, wrapper), namespace)
    if any(prop.optional for prop in fields):
        cls._input_array = _none_for_no_field(cls)
    cls._pvsm_proxy_xml = xml
    return cls
