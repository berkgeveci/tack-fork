# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``tack.inspect_template``: what a device-class object brings to a kernel, and
where its methods come from, described without running anything
(``docs/design/plans/inspect-template.md``)."""

import ast
import gc
import types
import weakref

import numpy as np
import pytest

import tack
import tack.arrays as ta
from tack.lang import inspect_template as it
from tack.lang import template_rewrite
from tack.lang.call_bindings import CallBindings
from tack.lang.func import Func
from tack.runtime import disk_cache

pytestmark = pytest.mark.quick


@tack.func
def _doubled(obj, i):
    return obj.get(i) * 2.0


@tack.func
def _cubic(r, h):
    return r * r * r / h


@tack.func
def _scaled(obj, i):
    return obj.get(i) * 3.0


# A module holding a device function, for module-qualified calls.
_helpers = types.ModuleType("_helpers")
_helpers.scaled = _scaled


@tack.device_class
class _Leaf:
    K = 3

    def __init__(self, n):
        self.data = tack.zeros((n,), tack.vec3(tack.f32))

    @tack.func
    def get(self, i):
        return self.data[i][0] * self.K


@tack.device_class
class _Base:
    K = 3
    SHAPE = (1, 2)

    @tack.func
    def get(self, i):
        return self.data[i] * self.K

    @tack.func
    def twice(self, i):
        return _doubled(self, i) + self.get(i)

    @tack.func
    def thrice(self, i):
        return _helpers.scaled(self, i)


@tack.device_class
class _Grid(_Base):
    __tack_iterate__ = "n"

    def __init__(self, n):
        self.data = tack.zeros((n,))
        self.scale = 2.0
        self.n = n
        self.K = 5                      # over _Base's: still compiled in
        self.shape = (n, 1)
        self.label = "a label"
        self.kernel = _cubic
        self.inner = _Leaf(n)

    @tack.func
    def get(self, i):
        return self.data[i] + self.scale

    @tack.func
    def __getitem__(self, i):
        return self.get(i)

    @tack.func
    def __setitem__(self, i, v):
        self.data[i] = v

    @tack.func
    def __len__(self):
        return self.n

    @staticmethod
    @tack.func
    def unit():
        return 1.0

    @tack.func
    def calls(self, i):
        self[i] += 1.0
        self[i] = self[i] * 2.0
        if self.K == 3:
            self.inner.get(i)
        f = self.get
        return (len(self) + len(self.inner) + self.kernel(1.0, 2.0) + f(i)
                + _absent(self) + self.inner[i])


def _grid(n=4):
    return _Grid(n)


# -- the full report ----------------------------------------------------------

def test_the_report_tells_constants_from_run_time_values():
    text = tack.inspect_template(_grid())
    assert text.splitlines()[0] == f"_Grid (in {__name__})"
    assert "  bases: _Base" in text
    assert "method resolution order: _Grid, _Base, object" in text
    compiled = text.split("compiled in:")[1].split("passed at run time:")[0]
    assert "K = 5" in compiled and "the object's own value, compiled in (over _Base's)" in compiled
    assert "SHAPE = (1, 2)" in compiled and "_Base" in compiled
    runtime = text.split("passed at run time:")[1].split("arrays:")[0]
    for line in ("scale = 2.0", "n = 4", "shape = (4, 1)"):
        assert line in runtime
    assert "data: shape (4,), tack.f32, cpu" in text
    assert "kernel: @tack.func" in text and "_cubic" in text
    assert "label: str" in text.split("not passed to kernels:")[1]


def test_the_report_names_each_methods_defining_class_and_source():
    text = tack.inspect_template(_grid())
    methods = {line.split("(")[0].strip(): line for line in
               text.split("methods:")[1].split("iterates:")[0].splitlines() if line.strip()}
    assert "_Base, test_inspect_template.py:" in methods["twice"]          # inherited
    assert methods["get"].split()[-2] == "_Grid,"                          # overridden
    assert "static" in methods["unit"]
    assert "serves obj[i]" in methods["__getitem__"]
    assert "serves obj[i] = v" in methods["__setitem__"]
    assert "serves len(obj)" in methods["__len__"]
    line = int(methods["get"].rsplit(":", 1)[1])
    assert _Grid.get.func.__code__.co_firstlineno < line              # the def, not its decorator
    assert "iterates: for c in obj runs range(obj.n)" in text


def test_held_objects_are_described_under_their_path():
    text = tack.inspect_template(_grid())
    held = text.split("held objects:")[1]
    assert f"inner: _Leaf (in {__name__})" in held
    assert "data: shape (4,), tack.vec(3, tack.f32), cpu" in held


def test_arrays_say_their_layout_and_element():
    @tack.device_class
    class Holder:
        def __init__(self):
            self.soa = tack.zeros((5,), tack.vec3(tack.f32), layout="soa")
            self.matrices = tack.zeros((2, 3), tack.mat(2, 2, tack.f32))

    text = tack.inspect_template(Holder())
    assert "soa: shape (5,), tack.vec(3, tack.f32), soa, cpu" in text
    assert "matrices: shape (2, 3)" in text and "mat" in text


def test_implicit_arrays_are_described_as_what_they_hold():
    text = tack.inspect_template(ta.Permutation(tack.arange(4), ta.CountingArray(10)))
    assert text.startswith("Permutation (in tack.arrays)")
    assert "base: CountingArray (in tack.arrays)" in text
    assert "start = 0" in text and "length = 10" in text
    assert "has a host implementation" in text
    assert "values: shape (4,), tack.i32, cpu" in text


def test_an_object_held_twice_is_described_once():
    @tack.device_class
    class Pair:
        def __init__(self, leaf):
            self.a = leaf
            self.b = leaf

    text = tack.inspect_template(Pair(_Leaf(2)))
    assert "b: the same object as a" in text
    assert text.count("K = 3") == 1


def test_a_cycle_and_a_collision_are_reported_not_raised():
    @tack.device_class
    class Node:
        pass

    a, b = Node(), Node()
    a.child, b.child = b, a
    text = tack.inspect_template(a)
    assert "lowering would refuse it" in text and "a template is a tree" in text
    assert "child.child: holds itself" in text

    @tack.device_class
    class Clash:
        def __init__(self):
            self.base = _Leaf(1)
            self.base__K = 2

    text = tack.inspect_template(Clash())
    assert "flatten to the same name" in text


def test_output_is_bounded():
    @tack.device_class
    class Long:
        TABLE = tuple(range(20))

    text = tack.inspect_template(Long())
    assert "TABLE = (0, 1, 2, 3, 4, 5, 6, 7, ... (20 entries))" in text

    @tack.device_class
    class Link:
        def __init__(self, child=None):
            if child is not None:
                self.next = child

    chain = None
    for _ in range(12):
        chain = Link(chain)
    text = tack.inspect_template(chain)
    assert "(3 more objects held below, not shown)" in text


def test_a_property_over_an_attribute_is_not_run():
    @tack.device_class
    class Guarded:
        @property
        def scale(self):
            raise AssertionError("inspection ran a property")

    obj = Guarded()
    obj.__dict__["scale"] = 2.0
    text = tack.inspect_template(obj)
    assert "scale: not read, a property of the class" in text


# -- the compiler's own classification, read without traces ------------------------

def _objects():
    return [_grid(), _Leaf(3), ta.CountingArray(10), ta.Permutation(tack.arange(4), ta.CountingArray(10)),
            ta.Concatenate(tack.arange(3), tack.arange(2)), ta.ConstantArray(2.0, 5)]


@pytest.mark.parametrize("index", range(6))
def test_classification_is_the_compilers(index):
    obj = _objects()[index]
    assert (template_rewrite._classify_tree(obj, it._read, inspecting=True)
            == template_rewrite.classify_template_attrs(obj))


def test_inspection_keeps_no_constant_names():
    @tack.device_class
    class Fresh:
        K = 2

        @tack.func
        def get(self, i):
            return self.K * i

    obj = Fresh()
    tack.inspect_template(obj)
    assert Fresh not in template_rewrite._CONSTANT_NAMES

    @tack.kernel
    def use(obj, out):
        for i in range(out.shape[0]):
            out[i] = obj.get(i)

    out = tack.zeros((3,), tack.i32)
    use(obj, out)
    assert template_rewrite._CONSTANT_NAMES[Fresh] == ("K",)
    # Once lowering has kept the names, inspection shows what lowering will use:
    # a constant added to the class later is not one.
    Fresh.LATER = 7
    assert "LATER" not in tack.inspect_template(obj)


def test_inspection_runs_nothing(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("inspection touched a device, a kernel or the disk cache")

    from tack.lang import array, kernel
    from tack.runtime import dispatch, kernel_utils
    obj = _grid()        # made first: preparing the object may use the device
    for owner, name in [(dispatch, "get_backend"), (dispatch, "as_device"),
                        (array.Array, "numpy"), (array.Array, "_host_values"),
                        (array.Array, "__getitem__"), (array.DeviceBuffer, "read_range"),
                        (array.NumpyBuffer, "read_range"), (array.NumpyBuffer, "to_numpy"),
                        (kernel.Kernel, "get_ir"), (kernel.Kernel, "__call__"),
                        (kernel_utils, "resolve_variant"), (disk_cache, "recorder")]:
        monkeypatch.setattr(owner, name, refuse)
    tack.inspect_template(obj)
    for name in ("twice", "thrice", "calls", "get"):
        tack.inspect_template(obj, name)
    assert getattr(template_rewrite._scope, "memo", None) is None


def test_inspection_keeps_no_reference_to_the_object():
    obj = _grid()
    ref = weakref.ref(obj)
    tack.inspect_template(obj)
    tack.inspect_template(obj, "calls")
    del obj
    gc.collect()
    assert ref() is None


def test_lookup_reports_to_no_recorder():
    obj = _grid()
    with disk_cache._record([_Base.thrice.func]) as rec:
        text = tack.inspect_template(obj, "thrice")
        tack.inspect_template(obj, "twice")
    assert rec.queries == [] and rec.owners == [_Base.thrice.func] and rec.cacheable
    assert "_helpers.scaled(self, i)  -> @tack.func" in text


def _device_funcs():
    import tack.math
    found = []
    for module in (tack.math, ta):
        for value in vars(module).values():
            candidates = [value] if isinstance(value, Func) else []
            if isinstance(value, type):
                candidates = [f for f, _, _ in template_rewrite.template_func_owners(value).values()]
            found.extend(candidates)
    found.extend(f for f, _, _ in template_rewrite.template_func_owners(_Grid).values())
    return found


def test_lookup_answers_as_resolve_does():
    checked = 0
    for func in _device_funcs():
        bindings = CallBindings(func._funcdef, python_func=func.func)
        for node in ast.walk(func._funcdef):
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
                assert bindings.lookup(node.func) is bindings.resolve(node.func)
                checked += 1
    assert checked > 50


# -- one method ----------------------------------------------------------------

def _calls(text):
    rows = text.split("calls in its source:\n")[1].splitlines()
    return [tuple(part.strip() for part in row.split("->", 1)) for row in rows]


def test_one_method_names_its_class_source_and_signature():
    text = tack.inspect_template(_grid(), "twice")
    lines = text.splitlines()
    assert lines[0] == "twice: defined by _Base"
    assert lines[1].startswith("source: test_inspect_template.py:")
    assert lines[2] == "signature: twice(self, i)"
    calls = dict(_calls(text))
    assert calls["_doubled(self, i)"].startswith(f"@tack.func {__name__}._doubled")
    assert calls["self.get(i)"].startswith("_Grid.get")      # the object's, not _Base's


def test_every_kind_of_call_is_listed_in_source_order():
    calls = _calls(tack.inspect_template(_grid(), "calls"))
    texts = [text for text, _ in calls]
    targets = dict(calls)
    # self[i] += 1.0: a get and a set, as _Indexing rewrites it
    assert calls[0][0] == "self[i] += 1.0" and calls[0][1].startswith("_Grid.__getitem__")
    assert calls[1][0] == "self[i] += 1.0" and calls[1][1].startswith("_Grid.__setitem__")
    # self[i] = self[i] * 2.0: the store's target first, then the load
    assert calls[2][0] == "self[i]" and calls[2][1].startswith("_Grid.__setitem__")
    assert calls[3][0] == "self[i]" and calls[3][1].startswith("_Grid.__getitem__")
    assert targets["self.inner.get(i)"].startswith("_Leaf.get")   # listed though constants decide
    assert targets["f(i)"] == "unresolved: a local name, not followed"
    assert targets["len(self)"].startswith("_Grid.__len__")
    assert targets["len(self.inner)"] == "not defined: _Leaf has no @tack.func __len__"
    assert targets["self.kernel(1.0, 2.0)"].startswith("held device function")
    assert targets["_absent(self)"] == "unresolved: the name is not bound"
    assert targets["self.inner[i]"] == "not defined: _Leaf has no @tack.func __getitem__"
    assert texts.index("self.inner.get(i)") < texts.index("len(self)") < texts.index("f(i)")


def test_an_update_through_a_held_object_reaches_both_protocol_methods():
    @tack.device_class
    class Outer:
        def __init__(self):
            self.base = _grid(2)

        @tack.func
        def bump(self, i):
            self.base[i] += 1.0

    calls = _calls(tack.inspect_template(Outer(), "bump"))
    assert [text for text, _ in calls] == ["self.base[i] += 1.0"] * 2
    assert calls[0][1].startswith("_Grid.__getitem__")
    assert calls[1][1].startswith("_Grid.__setitem__")


def test_a_held_device_function_can_be_asked_about():
    text = tack.inspect_template(_grid(), "kernel")
    assert text.startswith(f"kernel: a device function the object holds, {__name__}._cubic")
    assert "calls in its source: none" in text


def test_selectors_are_checked():
    obj = _grid()
    with pytest.raises(TypeError, match="not the class _Grid"):
        tack.inspect_template(_Grid)
    with pytest.raises(TypeError, match="@tack.device_class"):
        tack.inspect_template(np.zeros(3))
    with pytest.raises(AttributeError, match="no member 'nope'"):
        tack.inspect_template(obj, "nope")
    with pytest.raises(TypeError, match="K is a compiled-in constant"):
        tack.inspect_template(obj, "K")
    with pytest.raises(TypeError, match="data is an array"):
        tack.inspect_template(obj, "data")
    with pytest.raises(TypeError, match="inner is a held object"):
        tack.inspect_template(obj, "inner")
    with pytest.raises(TypeError, match="scale is a value passed at run time"):
        tack.inspect_template(obj, "scale")
    with pytest.raises(TypeError, match="__init__ is a Python method"):
        tack.inspect_template(obj, "__init__")


def test_it_is_public():
    assert "inspect_template" in tack.__all__
    assert tack.inspect_template is it.inspect_template


# -- as lowering has it ---------------------------------------------------------

@tack.func
def _two():
    return 2.0


@tack.kernel
def _each_call(obj, out):
    for i in range(out.shape[0]):
        out[i] = obj.call()


def test_a_held_device_function_shadows_a_method_of_its_name():
    """Lowering calls the device function an object holds over a method of the same
    name (``_template_nodes``); the report, the method query and the calls say so."""
    @tack.device_class
    class Shadow:
        def __init__(self):
            self.f = _two

        @tack.func
        def f(self):
            return 1.0

        @tack.func
        def call(self):
            return self.f()

    obj = Shadow()
    out = tack.zeros((1,), tack.f32)
    _each_call(obj, out)
    assert out.numpy()[0] == 2.0
    call = tack.inspect_template(obj, "call")
    assert f"self.f()  -> held device function {__name__}._two" in call
    assert f"{Shadow.__qualname__}.f" not in call
    first = tack.inspect_template(obj, "f").splitlines()[0]
    assert first == (f"f: a device function the object holds, {__name__}._two "
                     f"(over {Shadow.__qualname__}'s method, not used)")
    assert "not used: the object holds a device function of this name" in \
        tack.inspect_template(obj)


def test_len_of_the_object_is_its_len_whatever_len_is_bound_to():
    """``_Indexing`` rewrites ``len(self)`` to ``self.__len__()`` from the syntax alone,
    so a local ``len`` does not change the call."""
    @tack.device_class
    class Sized:
        @tack.func
        def __len__(self):
            return 3

        @tack.func
        def call(self):
            len = 99
            return len(self) + 0 * len

    obj = Sized()
    out = tack.zeros((1,), tack.f32)
    _each_call(obj, out)
    assert out.numpy()[0] == 3.0
    assert f"len(self)  -> {Sized.__qualname__}.__len__" in tack.inspect_template(obj, "call")


def test_inspection_runs_no_metaclass_code():
    """Class values are read from the classes' dictionaries: a metaclass's
    ``__getattribute__`` never runs, and a value only a metaclass's code gives
    (a descriptor of its) is shown as not read."""
    reads = []

    class Counting(type):
        def __getattribute__(cls, name):
            if name in ("K", "__tack_iterate__", "_device_class"):
                reads.append(name)
            return super().__getattribute__(name)

    @tack.device_class
    class Metered(metaclass=Counting):
        K = 5
        __tack_iterate__ = "n"

        def __init__(self):
            self.n = 4
            self.inner = _Leaf(1)

        @tack.func
        def get(self, i):
            return self.K * i + self.inner.get(i)

    obj = Metered()
    reads.clear()
    text = tack.inspect_template(obj)
    method = tack.inspect_template(obj, "get")
    assert reads == []
    assert "<locals>.Counting defines __getattribute__" in text
    assert "K = 5" in text.split("compiled in:")[1]
    assert "iterates: for c in obj runs range(obj.n)" in text
    assert "inner: _Leaf" in text
    assert "self.inner.get(i)  -> _Leaf.get" in method
    assert template_rewrite.classify_template_attrs(obj)[0]["K"] == 5    # as lowering reads it

    class Computed(type):
        K = property(lambda cls: 7)       # a data descriptor of the metaclass: Python's K

    @tack.device_class
    class Overridden(metaclass=Computed):
        K = 5

        @tack.func
        def get(self, i):
            return self.K * i

    obj = Overridden()
    assert template_rewrite.classify_template_attrs(obj)[0]["K"] == 7
    text = tack.inspect_template(obj)
    assert "K: not read" in text.split("compiled in:")[1]
    assert "K = 5" not in text
