# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""C++ functions called from kernels (a prototype, CPU only).

A `Library` names C++ source: headers, or text. A function of it,
`library.function(name, result=, args=)`, is called in a kernel like a math
builtin, on scalars (a vector argument passes its components)::

    lib = tack.cpp.Library(headers=["poly.h"], include_dirs=[here])
    smooth = lib.function("demo::smooth", result=tack.cpp.T, args=(tack.cpp.T, tack.cpp.T))

    @tack.kernel
    def apply(x, out, w):
        for i in range(x.shape[0]):
            out[i] = smooth(x[i], w)

Nothing is compiled until a kernel that calls the function is: the variant
knows its argument types, so Tack writes one ``extern "C"`` function per
(function, types) it needs, which is where a template is instantiated, has
clang compile that to LLVM bitcode, and links the bitcode into the kernel's
module before LLVM optimizes it. The library's code is inlined into the
loop, vectorized with it, and stored with the variant in the disk cache: a
process that finds the variant there needs no clang.

The C++ compiler is clang, found as the tests' host checks find theirs:
``TACK_CLANGXX``, else the ``clang++`` beside ``TACK_CLANG``, else ``clang++`` on
the path. It is needed only where such a kernel is compiled. Its LLVM must not be newer than
llvmlite's, which reads bitcode of its own version and older.

``T`` is one floating type per call. An array given for a ``ptr(T)`` decides
it; else the scalars given for a ``T``: f64 if one is, f32 if not, as the
math builtins choose. Float literals do not decide, as they decide nothing
elsewhere in Tack: where only they (or nothing) are given for ``T``, it is
the kernel's float type, f64 if an array argument of the kernel is f64, else
f32. ``function(..., T=tack.f64)``, or ``f.of(tack.f64)`` for a function
already declared, fixes it instead. Any other entry is a Tack scalar type, to
which the argument converts.

``ptr(type)`` takes an array: an array argument of the kernel (also one a
device class holds) or a ``tack.local_array``, passed as the address of its
first scalar (a vector array's first component). Its element type must be
the declared one; ``ptr(T)`` fixes ``T`` for the call. The pointer is
``const`` unless ``ptr(type, writes=True)``, which counts as a store to that
array wherever Tack asks what a kernel writes. Offsets and lengths are
scalars of their own. A function with ``result=None`` returns nothing and is
called as a statement. Debug mode checks none of what the C++ reads or writes.

``obj(method=(result, args), ...)`` takes a ``@tack.device_class`` object,
for a template parameter the C++ calls methods on::

    integrate = lib.function("demo::integrate", result=T,
                             args=(tack.cpp.obj(value=(T, (T,))), T, T))
    ...
        out[i] = integrate(field, a[i], b[i])     # field.value(x) is a @tack.func

The C++ receives a struct Tack writes for the call, with one method per name
declared, typed as declared, which calls the object's method; both are
inlined, so the method's arithmetic ends up inside the C++ loop. The struct
lasts for the call: the C++ must not keep it. A method takes scalars,
``ptr(type)`` (a pointer the C++ passes, which the method indexes as an
array, unchecked even in debug mode, and stores through only if
``writes=True``) and ``vec(type, n)`` (``std::array<type, n>`` in C++, a
vector in the method); it returns a scalar, a ``vec`` or nothing (None).

The protocol methods take their C++ spellings: ``__getitem__`` is
``operator[]``, ``__len__`` is ``size()`` and ``__call__`` is ``operator()``,
so a template written for any indexable container, or any callable, takes a
device class that defines them, ``tack.arrays``' implicit arrays among them.
With both of the first two the struct has ``begin()`` and ``end()``,
random-access iterators, for range-for loops and ``<algorithm>``.
``indexable(type)`` is the ``obj`` of a container of scalars, read-only:
``c[i]`` is a value. ``indexable(type, writable=True)`` declares
``__setitem__`` too: then on a container that is not ``const``, ``c[i]`` is a
reference to the element (a proxy object, as ``std::vector<bool>``'s:
``auto x = c[i]`` holds the reference, not the value) that reads, assigns,
updates (``+=`` and the like) and swaps, so ``std::sort`` sorts it, and every
assignment is a call of ``__setitem__``.

The function is called from every worker thread, any number of times, so it
must be safe to: no state of its own.
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from tack.lang.types import ScalarType, f32, f64

__all__ = ["ExternFunc", "Library", "Object", "Pointer", "T", "Vector", "indexable", "obj",
           "ptr", "vec"]


class _TypeVariable:
    def __repr__(self):
        return "tack.cpp.T"

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_type_variable, ())


T = _TypeVariable()


def _type_variable():
    return T


class Pointer:
    """An array argument: the address of its first scalar, of ``dtype`` (a
    scalar type or ``T``), which the function stores through if ``writes``."""

    def __init__(self, dtype, writes=False):
        self.dtype = dtype
        self.writes = bool(writes)

    def __repr__(self):
        return f"tack.cpp.ptr({self.dtype!r}, writes={self.writes})"

    def __eq__(self, other):
        return (isinstance(other, Pointer) and other.dtype is self.dtype
                and other.writes == self.writes)

    def __hash__(self):
        return hash((id(self.dtype), self.writes))

    def __deepcopy__(self, memo):
        return self

    @property
    def name(self):
        return ("w" if self.writes else "p") + self.dtype.name


def ptr(dtype, writes=False):
    """The type of an array argument; see the module's notes."""
    return Pointer(dtype, writes)


class Vector:
    """A vector of ``n`` scalars of ``dtype`` (a scalar type or ``T``) that a
    method C++ calls takes or returns: ``std::array<dtype, n>`` in C++."""

    def __init__(self, dtype, n):
        if not (isinstance(n, int) and 1 <= n <= 16):
            raise ValueError(f"tack.cpp.vec takes 1 to 16 components, not {n!r}")
        self.dtype = dtype
        self.n = n

    def __repr__(self):
        return f"tack.cpp.vec({self.dtype!r}, {self.n})"

    def __deepcopy__(self, memo):
        return self

    @property
    def name(self):
        return f"v{self.n}{self.dtype.name}"


def vec(dtype, n):
    """The type of a vector a method C++ calls takes or returns."""
    return Vector(dtype, n)


class Object:
    """A device-class argument: ``methods`` maps each method the C++ calls to
    its ``(result type, argument types)``, scalar types or ``T``."""

    def __init__(self, methods):
        self.methods = {name: (result, tuple(args))
                        for name, (result, args) in methods.items()}

    def __repr__(self):
        return f"tack.cpp.obj({self.methods!r})"

    def __deepcopy__(self, memo):
        return self


def obj(**methods):
    """The type of a device-class argument; see the module's notes."""
    return Object(methods)


def indexable(dtype, index=None, writable=False):
    """The type of a device-class argument the C++ indexes and measures:
    ``c[i]`` gives a ``dtype`` (a scalar type or ``T``) for an ``index`` (i64
    unless given), and ``c.size()`` its length, an i64. If ``writable``,
    ``c[i] = v`` calls the object's ``__setitem__``."""
    from tack.lang.types import i64
    methods = {"__getitem__": (dtype, (index or i64,)), "__len__": (i64, ())}
    if writable:
        methods["__setitem__"] = (None, (index or i64, dtype))
    return Object(methods)


#: How the C++ spells a protocol method. ``__setitem__`` is reached through
#: the element reference ``operator[]`` gives (`_container`).
_CPP_METHODS = {"__getitem__": "operator[]", "__len__": "size", "__call__": "operator()",
                "__setitem__": "tack_setitem"}

#: What every library's text is preceded by: the iterator of a container
#: whose elements are reached by index (`_container`).
_HELPERS = """
#include <cstddef>
#include <iterator>
namespace tack_cpp {
template <class C, class V, class R> struct iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = V;
  using difference_type = std::ptrdiff_t;
  using reference = R;
  using pointer = void;
  C* c;
  difference_type i;
  R operator*() const { return (*c)[i]; }
  R operator[](difference_type n) const { return (*c)[i + n]; }
  iterator& operator++() { ++i; return *this; }
  iterator operator++(int) { iterator t = *this; ++i; return t; }
  iterator& operator--() { --i; return *this; }
  iterator operator--(int) { iterator t = *this; --i; return t; }
  iterator& operator+=(difference_type n) { i += n; return *this; }
  iterator& operator-=(difference_type n) { i -= n; return *this; }
  friend iterator operator+(iterator a, difference_type n) { a.i += n; return a; }
  friend iterator operator+(difference_type n, iterator a) { a.i += n; return a; }
  friend iterator operator-(iterator a, difference_type n) { a.i -= n; return a; }
  friend difference_type operator-(const iterator& a, const iterator& b) { return a.i - b.i; }
  friend bool operator==(const iterator& a, const iterator& b) { return a.i == b.i; }
  friend bool operator!=(const iterator& a, const iterator& b) { return a.i != b.i; }
  friend bool operator<(const iterator& a, const iterator& b) { return a.i < b.i; }
  friend bool operator>(const iterator& a, const iterator& b) { return a.i > b.i; }
  friend bool operator<=(const iterator& a, const iterator& b) { return a.i <= b.i; }
  friend bool operator>=(const iterator& a, const iterator& b) { return a.i >= b.i; }
};
}
"""


def _container(struct, methods) -> list:
    """The members a struct with ``methods`` (name -> concrete ``(result,
    arguments)``) adds to its own for the container protocol: iterators,
    and with ``__setitem__`` element references."""
    if "__getitem__" not in methods:
        return []
    value = _c_type(methods["__getitem__"][0])
    index = _c_type(methods["__getitem__"][1][0])
    sized = "__len__" in methods
    ends = ("{ return {this, 0}; }", "{ return {this, std::ptrdiff_t(size())}; }")
    reading = f"tack_cpp::iterator<const {struct}, {value}, {value}>"
    lines = [f"  using value_type = {value};"]
    if sized:
        lines += [f"  {reading} begin() const {ends[0]}", f"  {reading} end() const {ends[1]}"]
    if "__setitem__" not in methods:
        return lines
    updates = [f"    reference& operator{op}=({value} v) "
               f"{{ return *this = {value}({value}(*this) {op} v); }}" for op in "+-*/"]
    lines += [
        "  struct reference {",
        f"    const {struct}* s; {index} i;",
        f"    operator {value}() const {{ return (*s)[i]; }}",
        f"    reference& operator=({value} v) {{ s->tack_setitem(i, v); return *this; }}",
        f"    reference& operator=(const reference& o) {{ return *this = {value}(o); }}",
        *updates,
        (f"    friend void swap(reference a, reference b) "
         f"{{ {value} t = a; a = {value}(b); b = t; }}"),
        "  };",
        f"  reference operator[]({index} i) {{ return {{this, i}}; }}"]
    if sized:
        writing = f"tack_cpp::iterator<{struct}, {value}, reference>"
        lines += [f"  {writing} begin() {ends[0]}", f"  {writing} end() {ends[1]}"]
    return lines


def _c_type(t):
    if isinstance(t, Vector):
        return f"std::array<{_C_TYPES[t.dtype.name]}, {t.n}>"
    if isinstance(t, Pointer):
        return ("" if t.writes else "const ") + _C_TYPES[t.dtype.name] + "*"
    return _C_TYPES[t.name]


_C_TYPES = {"i8": "int8_t", "u8": "uint8_t", "i16": "int16_t", "u16": "uint16_t",
            "i32": "int32_t", "u32": "uint32_t", "i64": "int64_t", "u64": "uint64_t",
            "f32": "float", "f64": "double"}

#: What every shim is compiled with besides the library's own flags. Contraction
#: as Tack's own arithmetic has it: within one expression. No stack checks or
#: protectors: Apple clang's stack probe ("probe-stack"="__chkstk_darwin") is a
#: function attribute llvmlite's LLVM aborts the process on.
_FLAGS = ("-std=c++17", "-O2", "-fno-exceptions", "-fno-rtti", "-ffp-contract=on",
          "-fno-math-errno", "-fno-stack-check", "-fno-stack-protector")


class Library:
    """C++ source a kernel may call into: ``headers`` (included by these names),
    found through ``include_dirs``, and/or ``source`` text. ``flags`` go to
    clang.

    Its identity, for the disk cache, is its text and flags, the contents of
    every file clang reads for it (system headers too) and clang's version
    (`fingerprint`). Read once a process: a header edited while a process
    runs is noticed by the next one."""

    def __init__(self, headers=(), *, source="", include_dirs=(), flags=()):
        self.headers = tuple(str(h) for h in headers)
        self.source = source
        self.include_dirs = tuple(str(Path(d).resolve()) for d in include_dirs)
        self.flags = tuple(flags)
        self._fingerprint = None
        self._base = None

    def function(self, name, *, result, args, T=None):
        """The C++ function ``name`` (qualified; ``{T}`` in it is replaced by the
        call's floating type, as in ``"demo::norm<{T}, 3>"``), returning
        ``result`` for arguments of types ``args``. ``T``, f32 or f64, fixes
        that type for every call instead."""
        return ExternFunc(self, name, result, tuple(args), T)

    def __deepcopy__(self, memo):
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fingerprint"] = None     # another process reads the files again
        return state

    def base(self) -> str:
        """The library as written: its text, headers' names, include directories
        and flags."""
        if self._base is None:
            self._base = hashlib.sha256(repr((
                self.headers, self.source, self.include_dirs, self.flags, _FLAGS,
            )).encode()).hexdigest()
        return self._base

    def fingerprint(self) -> str:
        """The library by content, for the disk cache: `base`, clang's version
        and the contents of every file clang read for it. Without a disk cache
        nothing is stored, and `base` is enough."""
        if self._fingerprint is None:
            from tack.runtime.disk_cache import _cache
            if _cache() is None:
                self._fingerprint = self.base()
            else:
                self._fingerprint = _dependencies(self).fingerprint
        return self._fingerprint


class ExternFunc:
    """A C++ function a kernel calls; made by `Library.function`."""

    def __init__(self, library, name, result, args, fixed=None):
        if fixed is not None and fixed not in (f32, f64):
            raise TypeError(f"{name}: T is f32 or f64, not {fixed!r}")
        self.fixed = fixed

        def scalar(t):
            return t is T or (isinstance(t, ScalarType) and t.name in _C_TYPES)
        if result is not None and not scalar(result):
            raise TypeError(f"{name}: the result {result!r} is not a scalar type, "
                            "tack.cpp.T or None")
        for t in args:
            if isinstance(t, Object):
                for method, (r, params) in t.methods.items():
                    def held(p):
                        return isinstance(p, (Vector, Pointer)) and scalar(p.dtype)
                    if not ((r is None or scalar(r) or (held(r) and isinstance(r, Vector)))
                            and all(scalar(p) or held(p) for p in params)):
                        raise TypeError(
                            f"{name}: the method '{method}' takes scalars, tack.cpp.ptr "
                            "and tack.cpp.vec, and returns a scalar, a tack.cpp.vec or None")
            elif not (scalar(t) or (isinstance(t, Pointer) and scalar(t.dtype))):
                raise TypeError(f"{name}: {t!r} is not a scalar type, tack.cpp.T, "
                                "a tack.cpp.ptr of one or a tack.cpp.obj")
        self.library = library
        self.name = name
        self.result = result
        self.args = args

    def __call__(self, *args):
        raise TypeError(f"the C++ function {self.name} is called in a kernel or a "
                        "@tack.func, not from Python")

    def of(self, dtype):
        """This function with ``T`` fixed to ``dtype`` (f32 or f64)."""
        return ExternFunc(self.library, self.name, self.result, self.args, dtype)

    def __deepcopy__(self, memo):
        return self

    def fingerprint(self) -> tuple:
        return ("cpp", self.name, repr(self.result), repr(self.args), repr(self.fixed),
                self.library.fingerprint())

    @property
    def writes(self) -> tuple:
        """The positions of the arguments the function stores through."""
        return tuple(k for k, t in enumerate(self.args)
                     if isinstance(t, Pointer) and t.writes)

    def bind(self, arg_types, weak=(), default=f32):
        """``(result type, argument types)`` of a call given arguments of
        ``arg_types``: a scalar's type, or for an array a `Pointer` of its
        element type. ``weak`` holds the positions of float literal arguments,
        which do not decide ``T``; ``default`` is ``T`` where nothing does (the
        kernel's float type). A function without a result gives i32 (always 0)."""
        passed = self.passed(self.args)
        if len(arg_types) != len(passed):
            raise TypeError(f"{self.name}() takes {len(passed)} arguments "
                            f"({len(arg_types)} given; a vector counts one per component)")
        pointed = set()
        for k, (given, declared) in enumerate(zip(arg_types, passed)):
            if isinstance(declared, Pointer) != isinstance(given, Pointer):
                raise TypeError(
                    f"{self.name}(): argument {k + 1} is "
                    + ("an array, where a scalar is declared" if isinstance(given, Pointer)
                       else "a scalar, where an array is declared (tack.cpp.ptr)"))
            if not isinstance(declared, Pointer):
                continue
            if declared.dtype is T:
                if given.dtype not in (f32, f64):
                    raise TypeError(f"{self.name}(): argument {k + 1} is an array of "
                                    f"{given.dtype.name}, where a floating type is declared")
                pointed.add(given.dtype)
            elif given.dtype is not declared.dtype:
                raise TypeError(f"{self.name}(): argument {k + 1} is an array of "
                                f"{given.dtype.name}, where {declared.dtype.name} is declared")
        if len(pointed) > 1:
            raise TypeError(f"{self.name}(): the arrays given for tack.cpp.T are of "
                            "f32 and f64; one call has one T")
        deciding = [given for k, (given, declared) in enumerate(zip(arg_types, passed))
                    if declared is T and k not in weak]
        if self.fixed is not None:
            floating = self.fixed
            if pointed and pointed != {floating}:
                raise TypeError(f"{self.name}(): T is fixed to {floating.name}, and an "
                                f"array of {pointed.pop().name} is given for it")
        elif pointed:
            floating = pointed.pop()    # a pointer cannot convert, so it decides
        elif deciding:
            floating = f64 if any(given is f64 for given in deciding) else f32
        else:
            floating = default

        def concrete(t):
            if isinstance(t, Pointer):
                return Pointer(concrete(t.dtype), t.writes)
            if isinstance(t, Vector):
                return Vector(concrete(t.dtype), t.n)
            if isinstance(t, Object):
                return Object({name: (concrete(r), [concrete(p) for p in params])
                               for name, (r, params) in t.methods.items()})
            return floating if t is T else t
        from tack.lang.types import i32
        return (i32 if self.result is None else concrete(self.result),
                tuple(concrete(t) for t in self.args))

    @staticmethod
    def passed(args) -> tuple:
        """The entries of ``args`` a call passes values for: all but objects."""
        return tuple(t for t in args if not isinstance(t, Object))

    def symbol(self, result, args, tag="") -> str:
        """The C symbol of the instantiation: by library content, so two
        libraries (or two versions of one) never share it. ``tag`` tells apart
        the instantiations of calls that pass objects, one per call."""
        text = "".join(c if c.isalnum() else "_" for c in self.name)
        digest = hashlib.sha256(repr((self.library.fingerprint(), self.name)).encode())
        types = "_".join("obj" if isinstance(t, Object) else t.name for t in (result, *args))
        return f"tack_cpp_{digest.hexdigest()[:12]}_{text}_{types}{tag}"

    def shim(self, result, args, objects=None, tag="") -> str:
        """The ``extern "C"`` function clang compiles: the instantiation.

        ``objects`` gives, per declared position of an object, its methods as
        ``(name, callback symbol, C types the callback takes after the
        method's own arguments)``: the values Tack captured for it, which the
        instantiation takes after the call's arguments and the object's
        struct holds."""
        objects = objects or {}
        floating = result if self.result is T else next(
            (getattr(a, "dtype", a) for a, declared in zip(args, self.args)
             if getattr(declared, "dtype", declared) is T), f32)
        target = self.name.replace("{T}", _C_TYPES[floating.name])
        symbol = self.symbol(result, args, tag)
        lines, params, captures, names, setup = [], [], [], [], []
        for k, t in enumerate(args):
            if not isinstance(t, Object):
                params.append(f"{_c_type(t)} a{k}")
                names.append(f"a{k}")
                continue
            struct, fields, given, methods = f"{symbol}_o{k}", [], [], []
            for m, (method, callback, captured) in enumerate(objects[k]):
                r, takes = t.methods[method]
                held = [f"m{m}c{j}" for j in range(len(captured))]
                # The callback takes a vector as its components, returns one
                # through a pointer to them, and then the values captured.
                own, passes, arguments = [], [], []
                for j, p in enumerate(takes):
                    if isinstance(p, Vector):
                        own.append(f"const {_c_type(p)}& x{j}")
                        passes += [_C_TYPES[p.dtype.name]] * p.n
                        arguments += [f"x{j}[{c}]" for c in range(p.n)]
                    else:
                        own.append(f"{_c_type(p)} x{j}")
                        passes.append(_c_type(p))
                        arguments.append(f"x{j}")
                if isinstance(r, Vector):
                    passes.append(_C_TYPES[r.dtype.name] + "*")
                    returns, gives = "void", _c_type(r)
                    body = (f"{gives} r; {callback}({', '.join([*arguments, 'r.data()', *held])});"
                            " return r;")
                elif r is None:
                    returns = gives = "void"
                    body = f"{callback}({', '.join([*arguments, *held])});"
                else:
                    returns = gives = _c_type(r)
                    body = f"return {callback}({', '.join([*arguments, *held])});"
                lines.append(f'extern "C" {returns} {callback}('
                             + ", ".join([*passes, *captured]) + ");")
                fields += [f"{c} {h};" for c, h in zip(captured, held)]
                given += [f"o{k}{h}" for h in held]
                captures += [f"{c} o{k}{h}" for c, h in zip(captured, held)]
                methods.append(
                    f"  __attribute__((always_inline)) {gives} "
                    f"{_CPP_METHODS.get(method, method)}({', '.join(own)}) const {{ {body} }}")
            methods += _container(struct, t.methods)
            lines.append("namespace { struct " + struct + " {\n  " + " ".join(fields) + "\n"
                         + "\n".join(methods) + "\n}; }")
            setup.append(f"{struct} o{k}{{{', '.join(given)}}};")
            names.append(f"o{k}")
        call = f"{target}({', '.join(names)})"
        body = " ".join([*setup, f"{call}; return 0;" if self.result is None
                         else f"return {call};"])
        lines.append(f'extern "C" __attribute__((always_inline)) {_C_TYPES[result.name]} '
                     f"{symbol}({', '.join(params + captures)}) {{ {body} }}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compiling: clang to bitcode, linked into a kernel's module
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_bitcode = {}       # shim source and flags -> bitcode
_clang = None


def _find_clang():
    """Where clang++ is, or None: ``TACK_CLANGXX``; else the ``clang++`` beside
    ``TACK_CLANG`` (the C driver, ``clang-14`` giving ``clang++-14``); else
    ``clang++`` on the path. A named compiler that is not there is not looked
    for elsewhere."""
    named = os.environ.get("TACK_CLANGXX")
    if not named and os.environ.get("TACK_CLANG"):
        c = Path(os.environ["TACK_CLANG"])
        named = str(c.with_name(c.name.replace("clang", "clang++", 1)))
    return shutil.which(named or "clang++")


def clang() -> str:
    """The compiler, or an error saying how to name one."""
    global _clang
    if _clang is None:
        found = _find_clang()
        if not found:
            raise RuntimeError(
                "a kernel calls a C++ function (tack.cpp), which needs clang to compile "
                "it: put clang++ on the path or name one in TACK_CLANGXX")
        _clang = found
    return _clang


def _prefix(library) -> str:
    """What every instantiation of ``library`` starts with: its headers and
    its own text. Parsed once per process (`_precompiled`)."""
    lines = ["#include <array>", "#include <cstdint>", _HELPERS]
    lines += [f'#include "{h}"' for h in library.headers]
    lines.append(library.source)
    return "\n".join(lines) + "\n"


def _source(shims) -> str:
    return "\n".join(sorted(set(shims))) + "\n"


def _command(library) -> list:
    return [clang(), *_FLAGS, *library.flags, *(f"-I{d}" for d in library.include_dirs)]


_folder = None
_headers = {}       # library and command -> its precompiled header's path, or None
_build_lock = threading.Lock()


def _scratch() -> Path:
    """This process's directory for precompiled headers, removed at exit."""
    global _folder
    if _folder is None:
        import atexit
        _folder = Path(tempfile.mkdtemp(prefix="tack-cpp-"))
        atexit.register(shutil.rmtree, _folder, ignore_errors=True)
    return _folder


def _precompiled(library, command=None):
    """``(path of library's precompiled header or None, the files clang read
    for it or None)``, built on first use. The path is None if clang would not
    build one (the instantiation then includes the text, and reports what is
    wrong with it); the files are listed then too, unless clang cannot read
    them. A library such as Eigen takes a second to parse; without this every
    kernel that calls it would pay that again, where with it an instantiation
    costs a third as much."""
    command = command or _command(library)
    key = hashlib.sha256(repr((command, library.base())).encode()).hexdigest()
    with _build_lock:
        if key not in _headers:
            header = _scratch() / f"{key[:16]}.h"
            header.write_text(_prefix(library))
            out, listed = header.with_suffix(".pch"), header.with_suffix(".d")
            done = subprocess.run([*command, "-x", "c++-header", str(header), "-o", str(out),
                                   "-MD", "-MF", str(listed)], capture_output=True, text=True)
            if done.returncode != 0:
                out = None
                done = subprocess.run([*command, "-x", "c++-header", str(header), "-M",
                                       "-MF", str(listed)], capture_output=True, text=True)
            files = _listed(listed, header) if done.returncode == 0 else None
            _headers[key] = (out, files)
        return _headers[key]


def _listed(path, header) -> list:
    """The files a make rule lists as ``header``'s prerequisites, less itself."""
    text = path.read_text().replace("\\\n", " ")
    names = text.split(":", 1)[1].split() if ":" in text else []
    files = {str(Path(n).resolve()) for n in (n.replace("\\ ", " ") for n in names)}
    files.discard(str(Path(header).resolve()))
    return sorted(files)


# ---------------------------------------------------------------------------
# What a library depends on, for the disk cache
# ---------------------------------------------------------------------------

class _Dependencies:
    """The files a library was compiled from, with what each held, and the
    clang that compiled it: an entry of the disk cache (``.cppdeps``), by the
    library's `Library.base`. ``fingerprint`` is the library's identity."""

    def __init__(self, files, clang_version):
        self.files = files                  # [(path, size, mtime_ns, sha256)]
        self.clang_version = clang_version

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(repr((self.clang_version, [
            (path, digest) for path, _, _, digest in self.files])).encode()).hexdigest()


def _digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


_versions = {}


def _clang_version():
    """The version clang reports, or None where there is none."""
    found = _find_clang()
    if found is None:
        return None
    if found not in _versions:
        done = subprocess.run([found, "--version"], capture_output=True, text=True)
        _versions[found] = done.stdout if done.returncode == 0 else None
    return _versions[found]


def _dependencies(library) -> _Dependencies:
    """``library``'s dependencies, as the disk cache records them, checked.

    A record is used while every file it names holds what it did, and clang
    (if there is one) is the version it names: a file whose size and
    modification time are as recorded is taken to hold what it did, any other
    is read again. Otherwise -- no record, a file changed or gone, another
    clang -- the files are listed afresh, by building the library's
    precompiled header, which compiling will want anyway; a changed file may
    include others. Without clang a record is all there is: if it does not
    hold, the fingerprint names that, and compiling asks for clang."""
    from tack.runtime.disk_cache import _cache
    cache = _cache()
    key = library.base()
    entry = cache.get(key, "cppdeps")
    record = _Dependencies([tuple(f) for f in entry["files"]], entry["clang"]) if entry else None
    version = _clang_version()
    if record is not None and (version is None or version == record.clang_version):
        files, changed = [], False
        for path, size, mtime, digest in record.files:
            try:
                st = os.stat(path)
            except OSError:
                changed = True
                break
            if (st.st_size, st.st_mtime_ns) != (size, mtime):
                if _digest_file(path) != digest:
                    changed = True
                    break
                size, mtime = st.st_size, st.st_mtime_ns
            files.append((path, size, mtime, digest))
        if not changed:
            fresh = _Dependencies(files, record.clang_version)
            if files != record.files:
                _store(cache, key, fresh)           # spare the next process the reads
            return fresh
    if version is None:
        return _Dependencies([("<no clang to list the library's files>", 0, 0, key)], None)
    _, listed = _precompiled(library)
    if listed is None:
        return _Dependencies([("<clang could not read the library>", 0, 0, key)], version)
    files = []
    for path in listed:
        st = os.stat(path)
        files.append((path, st.st_size, st.st_mtime_ns, _digest_file(path)))
    fresh = _Dependencies(files, version)
    _store(cache, key, fresh)
    return fresh


def _store(cache, key, record):
    """The record as an entry of plain data (the cache unpickles only its own)."""
    if cache.writes:
        cache.put(key, "cppdeps", {"key": key, "files": record.files,
                                   "clang": record.clang_version})


def _compile(library, source) -> bytes:
    """The bitcode of ``source`` (instantiations) after ``library``'s prefix."""
    command = _command(library)
    key = hashlib.sha256(repr((command, source, library.fingerprint())).encode()).hexdigest()
    with _lock:
        found = _bitcode.get(key)
    if found is not None:
        return found
    header, _ = _precompiled(library, command)
    with tempfile.TemporaryDirectory(prefix="tack-cpp-") as folder:
        src, out = Path(folder) / "shim.cpp", Path(folder) / "shim.bc"
        done = None
        if header is not None:
            src.write_text(source)
            done = subprocess.run([*command, "-include-pch", str(header), "-c", "-emit-llvm",
                                   str(src), "-o", str(out)], capture_output=True, text=True)
        if done is None or done.returncode != 0:
            # Without the precompiled header: the whole text, whose errors
            # clang then reports against what the user wrote.
            whole = _prefix(library) + source
            src.write_text(whole)
            done = subprocess.run([*command, "-c", "-emit-llvm", str(src), "-o", str(out)],
                                  capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"clang could not compile the C++ a kernel calls:\n"
                                   f"{done.stderr}\n--- the source ---\n{whole}")
        found = out.read_bytes()
    with _lock:
        _bitcode[key] = found
    return found


def uses(ir_func) -> bool:
    """Whether a kernel calls a C++ function."""
    from tack.lang import ir
    from tack.lang.ir_traversal import walk_ir
    return any(isinstance(node, ir.IRCall) and node.extern is not None
               for node in walk_ir(ir_func))


def link(module, calls):
    """Link the instantiations the code generator recorded, ``calls``
    (``(library, shim source, symbols to make internal)`` each), into
    ``module``, an llvmlite module not yet optimized: the optimizer inlines
    them, and the kernel's functions they call back, into the kernel."""
    import llvmlite.binding as llvm
    by_library = {}
    for library, shim, symbols in calls:
        entry = by_library.setdefault(library.fingerprint(), (library, [], []))
        entry[1].append(shim)
        entry[2].extend(symbols)
    for library, shims, symbols in by_library.values():
        data = _compile(library, _source(shims))
        try:
            other = llvm.parse_bitcode(data)
        except RuntimeError as error:
            version = ".".join(str(v) for v in llvm.llvm_version_info)
            raise RuntimeError(
                f"llvmlite (LLVM {version}) cannot read the bitcode {clang()} wrote: "
                f"{error}. Name a clang of LLVM {llvm.llvm_version_info[0]} or older "
                "in TACK_CLANGXX") from error
        other.triple = module.triple
        other.data_layout = module.data_layout
        for symbol in set(symbols):
            # Both sides were written here; a difference would run as a wild call.
            try:
                ours, theirs = module.get_function(symbol), other.get_function(symbol)
            except NameError:
                continue
            if str(ours.global_value_type) != str(theirs.global_value_type):
                raise RuntimeError(
                    f"tack.cpp: {symbol} is {ours.global_value_type} in the kernel and "
                    f"{theirs.global_value_type} in the C++ (a bug in Tack)")
        # Everything the C++ defines becomes internal to this kernel's module:
        # once inlined nothing is left of it, and what is not inlined (a large
        # template function, a static table) must not meet another kernel's
        # copy of the same name in the JIT's one symbol table.
        # (A private one, such as a switch's table, is local already, and the
        # link may rename it.)
        local = (llvm.Linkage.private, llvm.Linkage.internal)
        defined = [f.name for f in other.functions
                   if not f.is_declaration and f.linkage not in local]
        defined += [g.name for g in other.global_variables
                    if not g.is_declaration and g.linkage not in local]
        module.link_in(other)
        for symbol in {*symbols, *defined}:
            try:
                module.get_function(symbol).linkage = "internal"
            except NameError:
                module.get_global_variable(symbol).linkage = "internal"
