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

clang is found as ``TACK_CLANG``, else ``clang++`` on the path, and is needed
only where such a kernel is compiled. Its LLVM must not be newer than
llvmlite's, which reads bitcode of its own version and older.

``T`` is one floating type per call: f64 if an argument given for a ``T`` is
f64, else f32, as the math builtins choose theirs. Any other entry is a Tack
scalar type, to which the argument converts.

``ptr(type)`` takes an array: an array argument of the kernel (also one a
device class holds) or a ``tack.local_array``, passed as the address of its
first scalar (a vector array's first component). Its element type must be
the declared one; ``ptr(T)`` fixes ``T`` for the call. The pointer is
``const`` unless ``ptr(type, writes=True)``, which counts as a store to that
array wherever Tack asks what a kernel writes. Offsets and lengths are
scalars of their own. A function with ``result=None`` returns nothing and is
called as a statement. Debug mode checks none of what the C++ reads or writes.

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

__all__ = ["ExternFunc", "Library", "Pointer", "T", "ptr"]


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


def _c_type(t):
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
    clang. ``depends`` names further files whose contents the code depends on
    (headers the named ones include), so that the disk cache notices a change
    to them."""

    def __init__(self, headers=(), *, source="", include_dirs=(), flags=(), depends=()):
        self.headers = tuple(str(h) for h in headers)
        self.source = source
        self.include_dirs = tuple(str(Path(d).resolve()) for d in include_dirs)
        self.flags = tuple(flags)
        self.depends = tuple(str(d) for d in depends)
        self._fingerprint = None

    def function(self, name, *, result, args):
        """The C++ function ``name`` (qualified; ``{T}`` in it is replaced by the
        call's floating type, as in ``"demo::norm<{T}, 3>"``), returning
        ``result`` for arguments of types ``args``."""
        return ExternFunc(self, name, result, tuple(args))

    def __deepcopy__(self, memo):
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fingerprint"] = None     # another process reads the files again
        return state

    def _find(self, header):
        for folder in self.include_dirs:
            path = Path(folder) / header
            if path.is_file():
                return path
        path = Path(header)
        return path if path.is_file() else None

    def fingerprint(self) -> str:
        """The library by content: its text, flags, and the bytes of its named
        headers and ``depends``. A header clang finds on its own paths counts
        by name alone."""
        if self._fingerprint is None:
            h = hashlib.sha256()
            h.update(repr((self.headers, self.source, self.flags, _FLAGS)).encode())
            for name in self.headers + self.depends:
                path = self._find(name)
                h.update(b"\0" + name.encode() + b"\0")
                h.update(path.read_bytes() if path is not None else b"<system>")
            self._fingerprint = h.hexdigest()
        return self._fingerprint


class ExternFunc:
    """A C++ function a kernel calls; made by `Library.function`."""

    def __init__(self, library, name, result, args):
        def scalar(t):
            return t is T or (isinstance(t, ScalarType) and t.name in _C_TYPES)
        if result is not None and not scalar(result):
            raise TypeError(f"{name}: the result {result!r} is not a scalar type, "
                            "tack.cpp.T or None")
        for t in args:
            if not (scalar(t) or (isinstance(t, Pointer) and scalar(t.dtype))):
                raise TypeError(f"{name}: {t!r} is not a scalar type, tack.cpp.T or "
                                "a tack.cpp.ptr of one")
        self.library = library
        self.name = name
        self.result = result
        self.args = args

    def __call__(self, *args):
        raise TypeError(f"the C++ function {self.name} is called in a kernel or a "
                        "@tack.func, not from Python")

    def __deepcopy__(self, memo):
        return self

    def fingerprint(self) -> tuple:
        return ("cpp", self.name, repr(self.result), repr(self.args),
                self.library.fingerprint())

    @property
    def writes(self) -> tuple:
        """The positions of the arguments the function stores through."""
        return tuple(k for k, t in enumerate(self.args)
                     if isinstance(t, Pointer) and t.writes)

    def bind(self, arg_types):
        """``(result type, argument types)`` of a call given arguments of
        ``arg_types``: a scalar's type, or for an array a `Pointer` of its
        element type. A function without a result gives i32 (always 0)."""
        if len(arg_types) != len(self.args):
            raise TypeError(f"{self.name}() takes {len(self.args)} arguments "
                            f"({len(arg_types)} given; a vector counts one per component)")
        pointed = set()
        for k, (given, declared) in enumerate(zip(arg_types, self.args)):
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
        if pointed:
            floating = pointed.pop()    # a pointer cannot convert, so it decides
        else:
            floating = f64 if any(given is f64 for given, declared
                                  in zip(arg_types, self.args) if declared is T) else f32

        def concrete(t):
            if isinstance(t, Pointer):
                return Pointer(concrete(t.dtype), t.writes)
            return floating if t is T else t
        from tack.lang.types import i32
        return (i32 if self.result is None else concrete(self.result),
                tuple(concrete(t) for t in self.args))

    def symbol(self, result, args) -> str:
        """The C symbol of the instantiation: by library content, so two
        libraries (or two versions of one) never share it."""
        text = "".join(c if c.isalnum() else "_" for c in self.name)
        tag = hashlib.sha256(repr((self.library.fingerprint(), self.name)).encode())
        types = "_".join(t.name for t in (result, *args))
        return f"tack_cpp_{tag.hexdigest()[:12]}_{text}_{types}"

    def shim(self, result, args) -> str:
        """The ``extern "C"`` function clang compiles: the instantiation."""
        floating = result if self.result is T else next(
            (getattr(a, "dtype", a) for a, declared in zip(args, self.args)
             if getattr(declared, "dtype", declared) is T), f32)
        target = self.name.replace("{T}", _C_TYPES[floating.name])
        params = ", ".join(f"{_c_type(t)} a{k}" for k, t in enumerate(args))
        names = ", ".join(f"a{k}" for k in range(len(args)))
        body = (f"{target}({names}); return 0;" if self.result is None
                else f"return {target}({names});")
        return (f"__attribute__((always_inline)) {_C_TYPES[result.name]} "
                f"{self.symbol(result, args)}({params}) {{ {body} }}")


# ---------------------------------------------------------------------------
# Compiling: clang to bitcode, linked into a kernel's module
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_bitcode = {}       # shim source and flags -> bitcode
_clang = None


def clang() -> str:
    """The compiler, or an error saying how to name one."""
    global _clang
    if _clang is None:
        found = shutil.which(os.environ.get("TACK_CLANG") or "clang++")
        if not found:
            raise RuntimeError(
                "a kernel calls a C++ function (tack.cpp), which needs clang to compile "
                "it: put clang++ on the path or name one in TACK_CLANG")
        _clang = found
    return _clang


def _source(library, uses) -> str:
    lines = ["#include <cstdint>"]
    lines += [f'#include "{h}"' for h in library.headers]
    lines += [library.source, 'extern "C" {']
    lines += sorted(func.shim(result, args) for func, result, args in uses)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _compile(library, source) -> bytes:
    command = [clang(), *_FLAGS, *library.flags]
    command += [f"-I{d}" for d in library.include_dirs]
    key = hashlib.sha256(repr((command, source, library.fingerprint())).encode()).hexdigest()
    with _lock:
        found = _bitcode.get(key)
    if found is not None:
        return found
    with tempfile.TemporaryDirectory(prefix="tack-cpp-") as folder:
        src, out = Path(folder) / "shim.cpp", Path(folder) / "shim.bc"
        src.write_text(source)
        done = subprocess.run([*command, "-c", "-emit-llvm", str(src), "-o", str(out)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"clang could not compile the C++ a kernel calls:\n"
                               f"{done.stderr}\n--- the source ---\n{source}")
        found = out.read_bytes()
    with _lock:
        _bitcode[key] = found
    return found


def uses(ir_func) -> list:
    """The ``(function, result, argument types)`` a typed kernel calls."""
    from tack.lang import ir
    from tack.lang.ir_traversal import walk_ir
    found = {}
    for node in walk_ir(ir_func):
        if isinstance(node, ir.IRCall) and node.extern is not None:
            result, args = node.extern_types
            found[node.extern.symbol(result, args)] = (node.extern, result, args)
    return [found[k] for k in sorted(found)]


def link(module, calls):
    """Link what ``calls`` (`uses`) name into ``module``, an llvmlite module
    not yet optimized, as internal functions the optimizer inlines."""
    import llvmlite.binding as llvm
    by_library = {}
    for func, result, args in calls:
        by_library.setdefault(func.library.fingerprint(), (func.library, []))[1].append(
            (func, result, args))
    for library, group in by_library.values():
        data = _compile(library, _source(library, group))
        try:
            other = llvm.parse_bitcode(data)
        except RuntimeError as error:
            version = ".".join(str(v) for v in llvm.llvm_version_info)
            raise RuntimeError(
                f"llvmlite (LLVM {version}) cannot read the bitcode {clang()} wrote: "
                f"{error}. Name a clang of LLVM {llvm.llvm_version_info[0]} or older "
                "in TACK_CLANG") from error
        other.triple = module.triple
        other.data_layout = module.data_layout
        module.link_in(other)
        for func, result, args in group:
            # Internal: once inlined, nothing is left of it.
            module.get_function(func.symbol(result, args)).linkage = "internal"
