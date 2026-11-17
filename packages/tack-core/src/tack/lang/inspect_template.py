# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Describe a device-class object: what it brings to a kernel, and where its
methods come from.

Usage:
    print(tack.inspect_template(obj))                 # the whole object
    print(tack.inspect_template(obj, "corner_value")) # one method

The plan is ``docs/design/plans/inspect-template.md``. Inspection uses the
compiler's own classification (``template_rewrite``) and leaves no trace:
it reads instance attributes from the instance's dictionary and class values
from the classes' dictionaries, so no property or metaclass code computes
what it shows; it follows lowering's rules where the source could mislead (a
held device function shadows a method of its name; ``len(obj)`` is
``__len__`` whatever ``len`` is bound to); it keeps no class's constant names that lowering would not have kept;
it enters no ``template_scope``; it looks names up without the disk cache's
recorder; and it reads nothing from a device, launches nothing, compiles
nothing and makes no backend. The text it returns is for people: its layout
is not a stable interface.
"""

import ast
import os

from tack.lang.array import Array
from tack.lang.call_bindings import _DYNAMIC, _UNBOUND, CallBindings
from tack.lang.func import Func
from tack.lang.template_rewrite import (
    UNREAD,
    _classify_own,
    _classify_tree,
    _is_template_static,
    _meta_lookup,
    _number_tuple,
    _own_func_attrs,
    _static,
    iteration_space,
    scalar_path,
    template_children,
    template_func_owners,
)

#: Entries of a tuple shown before the rest are counted.
_TUPLE_SHOWN = 8
#: Levels of held objects described before the rest are counted.
_DEPTH = 8
#: Characters of source shown for one call.
_CALL_TEXT = 60

# The syntax each protocol method serves (``_Indexing``).
_PROTOCOL = {"__getitem__": "obj[i]", "__setitem__": "obj[i] = v", "__len__": "len(obj)"}


def inspect_template(obj, name=None) -> str:
    """Describe a device-class object, or one of its methods, as text.

    ``obj`` is an object of a ``@tack.device_class`` class that exists
    already: one of yours, an implicit array of `tack.arrays`, or a dataset
    view. Without ``name`` the text gives its class, what is compiled in and
    what is passed at run time, its arrays (never their values), the objects
    and device functions it holds, and its ``@tack.func`` methods with the
    class that defines each and its source location. With ``name``, a method
    of the object's class (or a device function it holds), it gives that
    method's defining class, source location and signature, and the calls in
    its source that reach another method or device function, with what each
    resolves to. Those calls are read from the source text, not from what a
    kernel inlines.

    Inspection runs nothing: no property or metaclass code for the values it
    shows, no device read, launch or compilation, no disk-cache lookup, and no
    backend is made.

    Raises:
        TypeError: ``obj`` is a class, or not an object of a device class; or
            ``name`` is a member that is not a device function.
        AttributeError: ``obj`` has no member ``name``.
    """
    if isinstance(obj, type):
        raise TypeError(f"inspect_template takes an object, not the class "
                        f"{obj.__qualname__}: what is compiled depends on its attributes")
    if not _is_template_static(obj):
        raise TypeError(f"inspect_template takes an object of a @tack.device_class "
                        f"class, not a {type(obj).__qualname__}")
    if name is None:
        return _Report(obj).text()
    return _method_report(obj, name)


# -- reading without running anything -------------------------------------------

def _read(obj, name):
    """An instance attribute's value from the instance's dictionary, or `UNREAD`
    where the class defines a data descriptor (a property) of that name, whose
    value lowering would get by running it."""
    if _class_descriptor(type(obj), name):
        return UNREAD
    return vars(obj)[name]


def _class_descriptor(cls, name):
    for klass in cls.__mro__:
        if name in vars(klass):
            value = vars(klass)[name]
            kind = type(value)
            return _defines(kind, "__set__") or _defines(kind, "__delete__")
    return False


def _defines(kind, name):
    """Whether a class or one of its bases defines ``name``, read from their
    dictionaries (``hasattr`` would run a metaclass's lookup)."""
    return any(name in vars(k) for k in kind.__mro__)


def _owner(cls, name):
    """The class along ``cls``'s method resolution order whose ``name`` is used."""
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass
    return None


def _number(value):
    # The built-in forms, so a subclass's own __repr__ (tack.constant's) never runs.
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return int.__repr__(value)
    return float.__repr__(value)


def _numbers(values):
    shown = ", ".join(_number(v) for v in values[:_TUPLE_SHOWN])
    if len(values) > _TUPLE_SHOWN:
        shown += f", ... ({len(values)} entries)"
    elif len(values) == 1:
        shown += ","
    return f"({shown})"


def _class_name(cls):
    return f"{cls.__qualname__} (in {cls.__module__})"


def _location(func):
    """``path:line`` of a device function's ``def``: the path from its top-level
    package (``tack/dataset/views.py``), or the file's name for a script."""
    code = func.func.__code__
    line = code.co_firstlineno + func._funcdef.lineno - 1
    path = code.co_filename
    parts = func.func.__module__.split(".")
    pieces = path.replace(os.sep, "/").split("/")
    if parts != ["__main__"] and pieces[-1] in (parts[-1] + ".py", "__init__.py"):
        keep = len(parts) + (pieces[-1] == "__init__.py")
        path = "/".join(pieces[-keep:])
    else:
        path = pieces[-1]
    return f"{path}:{line}"


def _signature(func):
    return f"{func.name}({ast.unparse(func._funcdef.args)})"


def _device_label(array):
    backend = array._device       # as recorded: never the default device
    if backend is None:
        return "device not recorded"
    if getattr(backend, "multi_device", False):
        return f"{backend.name}:{backend.ordinal}"
    return backend.name


def _describe_array(array):
    parts = [f"shape {tuple(array.shape)}", str(array.dtype)]
    if array.layout != "aos":
        parts.append(array.layout)
    parts.append(_device_label(array))
    if not array._writable:
        parts.append("read-only")
    return ", ".join(parts)


def _group(values):
    """Values by attribute, a tuple's entries (``shape__0``) back into one tuple;
    an `UNREAD` value stays `UNREAD`."""
    grouped = {}
    for key, value in values.items():
        path = scalar_path(key)
        if value is UNREAD:
            grouped[key] = value
        elif len(path) == 2:
            grouped.setdefault(path[0], []).append((path[1], value))
        else:
            grouped[key] = value
    return {name: (UNREAD if value is UNREAD
                   else _numbers([v for _, v in sorted(value)]) if isinstance(value, list)
                   else _number(value))
            for name, value in grouped.items()}


# -- the whole object --------------------------------------------------------

class _Report:
    def __init__(self, obj):
        self.obj = obj
        self.lines = []
        self.seen = {}          # id -> path of an object already described

    def text(self):
        obj = self.obj
        cls = type(obj)
        self.lines.append(_class_name(cls))
        bases = ", ".join(b.__qualname__ for b in cls.__bases__)
        self.lines.append(f"  bases: {bases}")
        self.lines.append("  method resolution order: "
                          + ", ".join(k.__qualname__ for k in cls.__mro__))
        composition = vars(cls).get("_tack_composition")
        if composition is not None:
            kind, shape, mixins = composition
            self.lines.append(f"  view: {kind.__qualname__} of {shape.__qualname__}, parts "
                              + (", ".join(m.__qualname__ for m in mixins) or "none"))
        try:
            _classify_tree(obj, _read, inspecting=True)
        except TypeError as error:
            self.lines.append(f"  lowering would refuse it: {error}")
        self._object(obj, (), "")
        return "\n".join(self.lines)

    def _object(self, obj, path, indent):
        self.seen[id(obj)] = path
        cls = type(obj)
        try:
            scalars, arrays, runtime = _classify_own(obj, _read, inspecting=True)
        except TypeError as error:
            self.lines.append(f"{indent}  lowering would refuse it: {error}")
            scalars, arrays, runtime = {}, {}, {}
        meta = _meta_lookup(cls)
        if meta is not None:
            self.lines.append(f"{indent}  its metaclass {meta.__qualname__} defines "
                              "__getattribute__: class values are shown as the classes "
                              "hold them; lowering reads them through it")
        self._section(indent, "compiled in", self._constants(obj, scalars))
        self._section(indent, "passed at run time", [
            f"{name} = {text}" for name, text in _group(runtime).items()])
        self._section(indent, "arrays", [
            f"{name}: {_describe_array(array)}" for name, array in arrays.items()])
        self._section(indent, "held device functions", [
            f"{name}: @tack.func {func.func.__module__}.{func.func.__qualname__}, "
            f"{_location(func)}" for name, func in _own_func_attrs(obj).items()])
        self._section(indent, "not passed to kernels", self._others(obj, scalars, arrays,
                                                                    runtime))
        self._section(indent, "methods", self._methods(obj))
        self._iteration(cls, indent)
        self._children(obj, path, indent)

    def _section(self, indent, title, rows):
        """A titled list; a row given as ``(left, right)`` is aligned in two columns."""
        if not rows:
            return
        self.lines.append(f"{indent}{title}:")
        width = max((len(row[0]) for row in rows if isinstance(row, tuple)), default=0)
        for row in rows:
            text = f"{row[0].ljust(width)}  {row[1]}" if isinstance(row, tuple) else row
            self.lines.append(f"{indent}  {text}")

    def _constants(self, obj, scalars):
        rows = []
        own = vars(obj)
        for name, text in _group(scalars).items():
            if text is UNREAD:
                owner = _owner(type(obj), name)
                where = owner.__qualname__ if owner else "its metaclass"
                rows.append((f"{name}: not read", f"Python would run code for it ({where})"))
                continue
            if name in own:
                where = "the object's own value, compiled in"
                owner = _owner(type(obj), name)
                if owner is not None:
                    where += f" (over {owner.__qualname__}'s)"
            else:
                owner = _owner(type(obj), name)
                where = owner.__qualname__ if owner is not None else "?"
            rows.append((f"{name} = {text}", where))
        return rows

    def _others(self, obj, scalars, arrays, runtime):
        taken = {scalar_path(k)[0] for k in (*scalars, *runtime)} | set(arrays)
        rows = []
        for name in vars(obj):
            if name.startswith("_") or name in taken:
                continue
            if _class_descriptor(type(obj), name):
                rows.append(f"{name}: not read, a property of the class")
                continue
            value = vars(obj)[name]
            if _is_template_static(value) or isinstance(value, Func):
                continue
            rows.append(f"{name}: {type(value).__qualname__}")
        return rows

    def _methods(self, obj):
        rows = []
        held = _own_func_attrs(obj)
        for name, (func, owner, static) in sorted(template_func_owners(type(obj)).items()):
            notes = [owner.__qualname__, _location(func)]
            if name in held:
                notes.append("not used: the object holds a device function of this name")
            if name in _PROTOCOL:
                notes.append(f"serves {_PROTOCOL[name]}")
            if static:
                notes.append("static")
            if func._host is not None:
                notes.append("has a host implementation")
            rows.append((_signature(func), ", ".join(notes)))
        return rows

    def _iteration(self, cls, indent):
        declared = _static(cls, "__tack_iterate__")
        if declared is None:
            return
        if declared is UNREAD:
            self.lines.append(f"{indent}iterates: not read, Python would run code for "
                              "__tack_iterate__")
            return
        try:
            names = iteration_space(cls, _static)
        except TypeError as error:
            self.lines.append(f"{indent}iterates: refused, {error}")
            return
        if len(names) == 1:
            loop = f"range(obj.{names[0]})"
        else:
            loop = f"ndindex over {', '.join(names)}, the first fastest"
        self.lines.append(f"{indent}iterates: for c in obj runs {loop}")

    def _children(self, obj, path, indent):
        children = template_children(obj, _is_template_static)
        if not children:
            return
        self.lines.append(f"{indent}held objects:")
        for name, child in children.items():
            where = (*path, name)
            label = ".".join(where)
            if id(child) in self.seen:
                earlier = self.seen[id(child)]
                if earlier in [where[:k] for k in range(len(where))]:
                    self.lines.append(f"{indent}  {label}: holds itself, through "
                                      f"{label}: a template is a tree")
                else:
                    self.lines.append(f"{indent}  {label}: the same object as "
                                      f"{'.'.join(earlier) or 'the object itself'}")
                continue
            self.lines.append(f"{indent}  {label}: {_class_name(type(child))}")
            if len(where) >= _DEPTH:
                self.lines.append(f"{indent}    ... ({_count(child)} more objects held "
                                  "below, not shown)")
                continue
            self._object(child, where, indent + "    ")


def _count(obj, _seen=None):
    seen = set() if _seen is None else _seen
    total = 0
    for child in template_children(obj, _is_template_static).values():
        if id(child) not in seen:
            seen.add(id(child))
            total += 1 + _count(child, seen)
    return total


# -- one method --------------------------------------------------------------

def _member_kind(obj, name):
    """What ``name`` is on ``obj``, when it is not a device function, else None."""
    cls = type(obj)
    owner = _owner(cls, name)
    if name in vars(obj) and owner is None:
        value = vars(obj)[name]
        if _is_template_static(value):
            return "a held object"
        if isinstance(value, Array):
            return "an array"
        if isinstance(value, (int, float)) or _number_tuple(value):
            return "a value passed at run time"
        return f"an attribute holding a {type(value).__qualname__}"
    if owner is None:
        return None
    value = vars(owner)[name]
    if isinstance(value, (int, float)) or _number_tuple(value):
        return "a compiled-in constant"
    if callable(value) or isinstance(value, (staticmethod, classmethod)):
        return "a Python method, not a @tack.func"
    return f"a class attribute holding a {type(value).__qualname__}"


def _method_report(obj, name):
    cls = type(obj)
    methods = template_func_owners(cls)
    held = _own_func_attrs(obj)
    # A held device function shadows a method of its name, as lowering has it.
    if name in held:
        func = held[name]
        static = True
        title = (f"{name}: a device function the object holds, "
                 f"{func.func.__module__}.{func.func.__qualname__}")
        if name in methods:
            title += f" (over {methods[name][1].__qualname__}'s method, not used)"
    elif name in methods:
        func, owner, static = methods[name]
        title = f"{name}: defined by {owner.__qualname__}" + (" (static)" if static else "")
    else:
        kind = _member_kind(obj, name)
        if kind is None:
            raise AttributeError(f"a {cls.__qualname__} has no member {name!r}")
        raise TypeError(f"{cls.__qualname__}.{name} is {kind}: inspect_template "
                        "takes the name of a device function")
    lines = [title, f"source: {_location(func)}", f"signature: {_signature(func)}"]
    calls = _Calls(obj, func, static).find()
    if calls:
        width = max(len(text) for text, _ in calls)
        lines.append("calls in its source:")
        lines.extend(f"  {text.ljust(width)}  -> {target}" for text, target in calls)
    else:
        lines.append("calls in its source: none to other methods or device functions")
    return "\n".join(lines)


def _text(node):
    text = ast.unparse(node)
    return text if len(text) <= _CALL_TEXT else text[:_CALL_TEXT - 3] + "..."


def _method_target(owner_obj, name):
    """What ``owner_obj.name(...)`` reaches: a method, a held device function, or why not."""
    cls = type(owner_obj)
    held = _own_func_attrs(owner_obj)
    if name in held:                # it shadows a method of its name, as lowering has it
        func = held[name]
        return (f"held device function {func.func.__module__}.{func.func.__qualname__}  "
                f"{_location(func)}")
    methods = template_func_owners(cls)
    if name in methods:
        func, owner, _ = methods[name]
        return f"{owner.__qualname__}.{name}  {_location(func)}"
    return f"not defined: {cls.__qualname__} has no @tack.func {name}"


class _Calls:
    """The calls in one device function's source that reach another method or
    device function, in source order, with what each resolves to. Read from
    the text: nothing is run, lowered or followed further."""

    def __init__(self, obj, func, static):
        self.obj = obj
        self.func = func
        args = func._funcdef.args.args
        self.self_name = args[0].arg if (not static and func._is_method and args) else None
        self.bindings = CallBindings(func._funcdef, python_func=func.func)
        self.found = []

    def find(self):
        body = self.func._funcdef
        updates = {id(n.target): n for n in ast.walk(body)
                   if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript)}
        for node in ast.walk(body):
            if isinstance(node, ast.Call):
                self._call(node)
            elif isinstance(node, ast.Subscript):
                self._subscript(node, updates.get(id(node)))
        self.found.sort(key=lambda entry: entry[0])
        return [(text, target) for _, text, target in self.found]

    def _add(self, node, text, target):
        self.found.append(((node.lineno, node.col_offset, len(self.found)), text, target))

    def _object_at(self, expr):
        """The held object ``expr`` names (``self``, ``self.a.b``), else None."""
        if self.self_name is None:
            return None
        names = []
        while isinstance(expr, ast.Attribute):
            names.append(expr.attr)
            expr = expr.value
        if not (isinstance(expr, ast.Name) and expr.id == self.self_name):
            return None
        current = self.obj
        for name in reversed(names):
            value = vars(current).get(name)
            if value is None or not _is_template_static(value):
                return None
            current = value
        return current

    def _call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute):
            receiver = self._object_at(func.value)
            if receiver is not None:
                self._add(node, _text(node), _method_target(receiver, func.attr))
                return
        # len(obj) is obj.__len__() whatever ``len`` is bound to, as _Indexing
        # rewrites it.
        if isinstance(func, ast.Name) and func.id == "len" and len(node.args) == 1:
            receiver = self._object_at(node.args[0])
            if receiver is not None:
                self._add(node, _text(node), _method_target(receiver, "__len__"))
                return
        if isinstance(func, ast.Name) and func.id in self.bindings.locals:
            self._add(node, _text(node), "unresolved: a local name, not followed")
            return
        if not any(self._object_at(arg) is not None for arg in node.args):
            return
        value = self.bindings.lookup(func)
        if isinstance(value, Func):
            target = (f"@tack.func {value.func.__module__}.{value.func.__qualname__}  "
                      f"{_location(value)}")
        elif value is _UNBOUND:
            target = "unresolved: the name is not bound"
        elif value is _DYNAMIC:
            target = "unresolved"
        else:
            target = f"not a @tack.func: a {type(value).__qualname__}"
        self._add(node, _text(node), target)

    def _subscript(self, node, update):
        """``obj[i]``, ``obj[i] = v``, or (``update``, the statement) ``obj[i] += v``,
        which ``_Indexing`` rewrites to a get and a set."""
        receiver = self._object_at(node.value)
        if receiver is None:
            return
        text = _text(update if update is not None else node)
        if update is not None:
            names = ("__getitem__", "__setitem__")
        elif isinstance(node.ctx, ast.Store):
            names = ("__setitem__",)
        else:
            names = ("__getitem__",)
        for name in names:
            self._add(node, text, _method_target(receiver, name))
