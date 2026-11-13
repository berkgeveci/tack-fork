# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tack IR to Slang, for shaders: one iteration of a Tack kernel as a Slang function.

A Tack kernel ``for k in range(n): out[k] = <expression of ids[k], pcs[k]>`` becomes

    float3 / float <name>(int id, float3 pc)

whose body is the kernel's own, with ``ids``, ``pcs`` and ``out`` function-local
arrays of one element (the parallel loop's index is 0), so nothing in the body
changes. The kernel's other arrays (a view's connectivity, values, tables, ...) are
storage buffers at bindings this module chooses; its scalars are words of one more
buffer. A field view's ``value(c, pc)`` or a domain view's ``position(c, pc)`` and
Jacobian are exported this way and called from a hand-written renderer shader.

The target is WGSL through Slang, so there is no f64 and no 64-bit or narrow
integer: f64 is computed in f32, i64/u64 and the 8/16-bit types in 32 bits, and
the host uploads those arrays narrowed to match (``Export.buffers``). That is a
renderer's precision, not Tack's semantics; the integer wrapping of narrow types is
not kept. The generator is the Metal one (no f64 either) with Slang's spellings.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from tack import arrays
from tack.codegen.float_division import float_division_helpers
from tack.codegen.integer_division import integer_division_helpers
from tack.codegen.integer_ops import IntegerCodeGen
from tack.codegen.msl_gen import MSLCodeGen
from tack.codegen.tables import Tables
from tack.lang import ir
from tack.lang.array import Array
from tack.lang.ir_traversal import walk_ir
from tack.lang.types import f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

#: Slang types for Tack's: 32 bits at most, as WGSL has.
SLANG_TYPES = {i8: "int", i16: "int", i32: "int", i64: "int",
               u8: "uint", u16: "uint", u32: "uint", u64: "uint", f32: "float", f64: "float"}
_NUMPY = {"int": np.int32, "uint": np.uint32, "float": np.float32}
_READ = {"float": "asfloat", "int": "asint", "uint": "asuint"}


@dataclass
class Export:
    """One generated function: its Slang source, and the buffers it reads, each
    ``(binding, numpy array)`` in the type the shader declares."""

    name: str
    source: str
    buffers: list = field(default_factory=list)
    helpers: list = field(default_factory=list)      # blocks, shared by name between exports


class SlangCodeGen(MSLCodeGen):
    """The Metal generator with Slang's spellings, emitting one iteration of a
    kernel as a function (see the module's docstring)."""

    _LABEL = "Slang"
    _INDEX_TYPE = "int"
    _HAS_F64 = False
    _integer_type_map = SLANG_TYPES

    def __init__(self, ir_func, name, slots, results, binding, group=0, header_base=0):
        original = [p.name for p in ir_func.params]
        super().__init__(ir_func)
        # The GPU renaming keeps the parameters' order: the slots by position.
        renamed = {o: p.name for o, p in zip(original, self.ir_func.params, strict=True)}
        slots = {renamed[n]: (t, length, n) for n, (t, length) in slots.items()}
        self._tables = Tables("metal")
        self._integers = IntegerCodeGen(SLANG_TYPES, bitcast=True)
        self._name = name
        self._slots = slots              # parameter -> (Slang type, length, its own name)
        out = list(slots)[-1]            # the result: (Slang type, expression of {out})
        self._results = (results[0], results[1].format(out=out))
        self._binding = binding          # the first binding number this function takes
        self._group = group
        # Where the function's header starts in its buffer: 0 alone, after the others'
        # headers in a buffer several functions share (export_shared).
        self._header_base = header_base
        self._globals = {}               # parameter -> its buffer's name
        self._aliases = {}               # a local holding an array -> the array it names

    def generate(self, effective_args):
        # The Metal generator reads its module's type table and integer type, not the
        # instance's: Slang's for as long as this runs.
        from tack.codegen import msl_gen

        saved = msl_gen._MSL_TYPE_MAP, msl_gen._INT
        msl_gen._MSL_TYPE_MAP, msl_gen._INT = SLANG_TYPES, "int"
        try:
            return self._generate(effective_args)
        finally:
            msl_gen._MSL_TYPE_MAP, msl_gen._INT = saved

    def _generate(self, effective_args):
        func = self.ir_func
        # Every array and scalar of the function in one buffer of 32-bit words, read by
        # offset with a bit cast: one binding whatever the view holds (WebGPU's default
        # is 8 storage buffers a stage; a curved geometry's view alone holds 18 arrays).
        # The buffer opens with a header of fixed size: each array's offset, then the
        # scalars; the arrays follow. The function reads the offsets at run time, so its
        # source does not depend on the arrays' lengths: a mesh of other sizes (in situ,
        # each step's) keeps the same source, and the renderer's compiled shader.
        lines, scalars, array_params = [], [], []
        pool = f"{self._name}__pool"
        # The packing's layout: per parameter, None (a slot) or (is an array, Slang type).
        self.layout = []
        for param in func.params:
            self._param_types[param.name] = param.type_annotation
            if param.name in self._slots:
                self._array_params.add(param.name)
                self.layout.append(None)
                continue
            t = SLANG_TYPES[param.type_annotation]
            self.layout.append((bool(param._is_array), t))
            if param._is_array:
                self._array_params.add(param.name)
                array_params.append((param, t))
            else:
                scalars.append(param)
        offset_names = []
        for k, (param, t) in enumerate(array_params):
            offset_names.append((f"{self._name}__at{k}", k))
            self._globals[param.name] = (pool, offset_names[-1][0], t)
        scalars = [(param, len(array_params) + k) for k, param in enumerate(scalars)]
        packed = pack(self.layout, effective_args)
        if self._binding is None:
            # The buffer is the function's first parameter: the caller's own (a field of
            # a renderer's hook type, which the renderer binds).
            buffers = [(None, packed)]
            self.next_binding = None
        else:
            lines.append(f"[[vk::binding({self._binding}, {self._group})]] "
                         f"StructuredBuffer<uint> {pool};")
            buffers = [(self._binding, packed)]
            self.next_binding = self._binding + 1

        self._lines = lines
        ret_type, _ = self._results
        args = ", ".join(([f"StructuredBuffer<uint> {pool}"] if self._binding is None else [])
                         + [f"{t} {own}" if length == 1 else f"float3 {own}"
                            for n, (t, length, own) in self._slots.items()
                            if n != _result_slot(self._slots)])
        self._emit(f"{ret_type} {self._name}({args})")
        self._emit("{")
        self._indent += 1
        for n, (t, length, own) in self._slots.items():
            self._emit(f"{t} {n}[{length}];")
            if n == _result_slot(self._slots):
                continue
            if length == 1:
                self._emit(f"{n}[0] = {own};")
            else:
                for c in range(length):
                    self._emit(f"{n}[{c}] = {own}[{c}];")
        for name, slot in offset_names:
            self._emit(f"int {name} = asint({pool}[{self._header_base + slot}]);")
        for param, offset in scalars:
            t = SLANG_TYPES[param.type_annotation]
            self._emit(f"{t} {param.name} = {_READ[t]}({pool}[{self._header_base + offset}]);")
            self._local_vars[param.name] = t
            self._declared_vars.add(param.name)
        self._declare_locals_at_kernel_scope(func.body)
        self._emit_body(func.body)
        self._emit(f"return {self._results[1]};")
        self._indent -= 1
        self._emit("}")
        helpers = (
            self._tables.declarations("static const", SLANG_TYPES)
            + float_division_helpers(self._float_division_helpers, SLANG_TYPES, "")
            + integer_division_helpers(self._integer_division_helpers, SLANG_TYPES, "")
            + self._integers.definitions("")
        )
        source = "\n".join(lines) + "\n"
        blocks = [b.strip() for b in "\n".join(helpers).split("\n\n") if b.strip()]
        return Export(self._name, _slang_spelling(source, self._name), buffers,
                      [_slang_spelling(b, self._name) for b in blocks])

    # The parallel loop is one iteration, at index 0; a sequential one counts in int.
    def _emit_parallel_for(self, node):
        for var in (node.dims or [node.var]):
            self._emit(f"int {var} = 0;")
            self._local_vars[var] = "int"
            self._declared_vars.add(var)
        self._emit_body(node.body)

    def _emit_sequential_for(self, node):
        start, end = self._expr(node.start), self._expr(node.end)
        step = self._expr(node.step) if node.step else None
        var = node.var
        incr = f"{var} += {step}" if step else f"{var}++"
        self._emit(f"for (int {var} = {start}; {var} < {end}; {incr}) {{")
        outer = (var in self._declared_vars, self._local_vars.get(var))
        self._local_vars[var] = "int"
        self._declared_vars.add(var)
        self._indent += 1
        self._emit_body(node.body)
        self._indent -= 1
        self._emit("}")
        self._leave_loop_scope(var, outer)

    def _emit_stmt(self, node):
        if isinstance(node, ir.IRReturn) or (isinstance(node, ir.IRContinue) and node.outermost):
            self._emit(f"return {self._results[1]};")
        elif isinstance(node, (ir.IRAtomicOp, ir.IRBarrier, ir.IRSharedAlloc, ir.IRPrint)):
            raise NotImplementedError(f"{type(node).__name__} in a shader function")
        else:
            super()._emit_stmt(node)

    def _array_root(self, name):
        """The parameter (a pooled array or a slot) a name holding an array names, or None."""
        name = self._aliases.get(name, name)
        return name if name in self._globals or name in self._slots else None

    def _expr_array_load(self, node):
        root = self._array_root(node.array.name) if isinstance(node.array, ir.IRName) else None
        if root in self._globals:
            pool, offset, t = self._globals[root]
            return f"{_READ[t]}({pool}[{offset} + int({self._expr(node.index)})])"
        return super()._expr_array_load(node)

    def _emit_array_store(self, node):
        root = self._array_root(node.array.name) if isinstance(node.array, ir.IRName) else None
        if root in self._globals:
            raise NotImplementedError("a store to a dataset array in a shader function")
        super()._emit_array_store(node)

    def _alias(self, node):
        """Record ``node`` (an assignment) if it binds a local to an array: True if so.
        One local naming two arrays (chosen at run time) is refused."""
        if not (isinstance(node.value, ir.IRName)
                and self._array_root(node.value.name) is not None):
            return False
        root = self._aliases.get(node.value.name, node.value.name)
        if self._aliases.get(node.target, root) != root:
            raise NotImplementedError(f"'{node.target}' names two arrays: a buffer chosen at "
                                      "run time has no WGSL spelling")
        self._aliases[node.target] = root
        self._declared_vars.add(node.target)
        return True

    def _emit_assign(self, node):
        # Inlined device functions pass arrays down as locals holding them; WGSL has no
        # pointers to buffers, and each names one buffer, so the name is resolved here.
        if not self._alias(node):
            super()._emit_assign(node)

    def _emit_if(self, node):
        # The branches' array locals first, so that none is hoisted as a variable.
        for stmt in walk_ir(node):
            if isinstance(stmt, ir.IRAssign):
                self._alias(stmt)
        super()._emit_if(node)

    def _expr(self, node):
        if isinstance(node, ir.IRName) and self._array_root(node.name) in self._slots:
            return self._array_root(node.name)
        return super()._expr(node)


def _result_slot(slots):
    return list(slots)[-1]


def pack(layout, effective_args):
    """The words of one function's buffer: a header (each array's offset, then the
    scalars) and the arrays, in the generator's ``layout`` (per parameter: None for a
    slot, else (is an array, Slang type))."""
    data, scalars = [], []
    for kind, arg in zip(layout, effective_args, strict=True):
        if kind is None:
            continue
        is_array, t = kind
        if is_array:
            host = np.asarray(arrays.to_host(arrays.materialize(arg))).reshape(-1)
            data.append(host.astype(_NUMPY[t]).view(np.uint32))
        else:
            scalars.append(np.array([arg], _NUMPY[t]).view(np.uint32))
    header = len(data) + len(scalars)
    if not header:
        return np.zeros(1, np.uint32)
    offsets = header + np.cumsum([0] + [len(d) for d in data[:-1]])
    return np.concatenate([offsets.astype(np.uint32)[:len(data)], *scalars, *data])


def _slang_spelling(source, prefix):
    """Metal's spellings the shared helpers use, in Slang's; tables named per function."""
    source = re.sub(r"as_type<(\w+)>", r"bit_cast<\1>", source)
    source = source.replace("__attribute__((noinline))", "")
    source = re.sub(r"\bulong\b", "uint", source)
    source = re.sub(r"\blong\b", "int", source)
    source = re.sub(r"(\d+)U?LL?\b", r"\1", source)
    return source.replace("__tack_table_", f"__tack_table_{prefix}_")


#: export's code by the call's structure: (Export, layout, next binding).
_EXPORTS = {}


def _structure(kernel, args, effective_args, *rest):
    """What the generated code depends on: the templates' keys (classes, compile-time
    constants, arrays' ranks and types, nesting) and each expanded array's type and
    inner dimensions (compiled in); not the lengths, which the buffer's header holds."""
    from tack.lang.kernel import _template_key
    from tack.runtime.kernel_utils import _detect_template_args

    templates = _detect_template_args(kernel, args)
    return (id(kernel), *rest,
            tuple((i, _template_key(obj)) for i, (_, obj) in sorted(templates.items())),
            tuple(("a", str(a.dtype), tuple(a.shape[1:]), getattr(a, "_vector_n", None),
                   getattr(a, "_matrix_shape", None)) if isinstance(a, Array)
                  else ("s", type(a).__name__) for a in effective_args))


def export(kernel, args, name, slots, results, binding, group=0, header_base=0):
    """``kernel`` called with ``args`` (Tack arrays, views), its one iteration as a Slang
    function ``name``: ``slots`` names the kernel's per-iteration parameters in order,
    each ``(Slang type, length)``, the last the result; ``results`` is ``(return type,
    expression of the slots)``.

    The code is generated once per structure (``_structure``); a later call of the same
    structure (a new mesh, another step) only packs its arrays into the buffer.
    ``header_base`` is where the function's header starts in its buffer
    (``export_shared``). The result keeps the packing's ``layout`` and the call's
    ``effective_args``."""
    from dataclasses import replace

    from tack.lang.inspect_kernel import _prepare_ir
    from tack.runtime.kernel_utils import _detect_template_args, _expand_template_args

    effective_args = _expand_template_args(args, _detect_template_args(kernel, args))
    key = _structure(kernel, args, effective_args, name, repr(slots), repr(results), binding,
                     group, header_base)
    hit = _EXPORTS.get(key)
    if hit is not None:
        out, layout, next_binding = hit
        words = pack(layout, effective_args)
        out = replace(out, buffers=[(out.buffers[0][0], words)])
        out.next_binding = next_binding
        out.layout, out.effective_args = layout, effective_args
        return out
    ir_func, effective_args = _prepare_ir(kernel, args)
    for node in walk_ir(ir_func.body):
        if isinstance(node, (ir.IRAtomicOp, ir.IRBarrier, ir.IRSharedAlloc)):
            raise NotImplementedError(f"{name}: {type(node).__name__} has no place in a shader")
    gen = SlangCodeGen(ir_func, name, slots, results, binding, group, header_base)
    out = gen.generate(effective_args)
    out.next_binding = gen.next_binding
    out.layout, out.effective_args = gen.layout, effective_args
    # Kept without its words, which the next call packs afresh.
    _EXPORTS[key] = (replace(out, buffers=[(b, None) for b, _ in out.buffers]), gen.layout,
                     gen.next_binding)
    return out


def _header_size(layout):
    return sum(1 for kind in layout if kind is not None)


def pack_shared(parts):
    """One buffer for several functions (``parts``: per function its ``(layout,
    effective_args)``, in order): their headers one after another, then every array
    once -- an array two functions read (a topology's connectivity, read by the
    geometry's view and every field's) is stored once, by identity. Each header holds
    its arrays' offsets in the whole buffer, then its scalars."""
    sizes = [_header_size(layout) for layout, _ in parts]
    total = sum(sizes)
    data, where = [], {}
    headers = []
    for layout, effective_args in parts:
        offsets, scalars = [], []
        for kind, arg in zip(layout, effective_args, strict=True):
            if kind is None:
                continue
            is_array, t = kind
            if is_array:
                key = (id(arg), t)
                if key not in where:
                    host = np.asarray(arrays.to_host(arrays.materialize(arg))).reshape(-1)
                    where[key] = total + sum(len(d) for d in data)
                    data.append(host.astype(_NUMPY[t]).view(np.uint32))
                offsets.append(where[key])
            else:
                scalars.append(np.array([arg], _NUMPY[t]).view(np.uint32))
        headers.append(np.concatenate([np.array(offsets, np.uint32), *scalars])
                       if offsets or scalars else np.zeros(0, np.uint32))
    words = np.concatenate([*headers, *data]) if total or data else np.zeros(1, np.uint32)
    return words if len(words) else np.zeros(1, np.uint32)


def export_shared(specs):
    """Several functions reading one buffer, passed to each as its first parameter:
    ``specs`` are ``export``'s arguments without ``binding`` (each ``(kernel, args,
    name, slots, results)``). The functions and the buffer's words (``pack_shared``)."""
    first = [export(*spec, binding=None) for spec in specs]
    bases = np.cumsum([0] + [_header_size(e.layout) for e in first[:-1]])
    outs = [export(*spec, binding=None, header_base=int(b)) for spec, b in zip(specs, bases)]
    return outs, pack_shared([(e.layout, e.effective_args) for e in outs])


def is_array(arg):
    return isinstance(arg, Array)


def joined(*exports):
    """The exports' sources after their helpers, each helper once."""
    helpers = list(dict.fromkeys(b for e in exports for b in e.helpers))
    return "\n\n".join(helpers) + "\n\n" + "\n".join(e.source for e in exports)
