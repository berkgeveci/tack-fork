# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conditions that constants decide while a kernel is lowered.

A device class's constants (``DIM = 3``, a mask of cell kinds) and literals
make many conditions known before any argument is: in AhmedML's analysis
kernels, a quarter of the IR sat in branches that could never run (half of
the streamlines kernel's), which every later pass and the code generator
walked, and only LLVM removed. Lowering takes the arm such a condition
picks and never lowers the other (`KernelTransformer.visit_If`,
`visit_IfExp`). The condition itself stays, in an ``if`` with empty arms
before the arm: it is evaluated as the kernel runs, as it was, so an
operand's bounds check still runs in debug mode (``data[i + 1] > 0 and
False``), and typed, so a literal past 64 bits or a mix of ``u64`` and
``i64`` is still refused.

Only integer and Boolean values are decided, with the operations whose
meaning does not depend on a width: comparisons, ``not``, ``&``, ``|``,
``^``, ``and``, ``or``. Floating-point conditions are left to run, since a
literal's f32 value may compare differently from Python's double. An
``and`` with any operand false is false and an ``or`` with any operand true
is true, whatever the others are, since the condition is still evaluated
in full where the kernel would have (an inlined call's effects come before
it).
"""

from tack.lang import ir
from tack.lang.types import ScalarType

_COMPARE = {
    '==': lambda a, b: a == b, '!=': lambda a, b: a != b,
    '<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
    '>': lambda a, b: a > b, '>=': lambda a, b: a >= b,
}
_BITWISE = {'&': lambda a, b: a & b, '|': lambda a, b: a | b, '^': lambda a, b: a ^ b}


def _fits(value: int, dtype) -> bool:
    kind = dtype.numpy_dtype.kind if isinstance(dtype, ScalarType) else None
    if kind not in ('i', 'u'):
        return False
    bits = dtype.bits
    if kind == 'u':
        return 0 <= value < (1 << bits)
    return -(1 << (bits - 1)) <= value < (1 << (bits - 1))


def constant_integer(node):
    """The integer ``node`` is, decided by constants alone, or None."""
    if isinstance(node, ir.IRConstant):
        return int(node.value) if type(node.value) in (int, bool) else None
    if isinstance(node, ir.IRCast):
        value = constant_integer(node.value)
        return value if value is not None and _fits(value, node.dtype) else None
    if isinstance(node, ir.IRUnaryOp) and node.op == 'not':
        value = constant_integer(node.operand)
        return None if value is None else int(not value)
    if isinstance(node, ir.IRCompare) and node.op in _COMPARE:
        left, right = constant_integer(node.left), constant_integer(node.right)
        return None if left is None or right is None else int(_COMPARE[node.op](left, right))
    if isinstance(node, ir.IRBinOp) and node.op in _BITWISE:
        left, right = constant_integer(node.left), constant_integer(node.right)
        return None if left is None or right is None else _BITWISE[node.op](left, right)
    if isinstance(node, ir.IRBoolOp) and node.op in ('and', 'or'):
        values = [constant_integer(v) for v in node.values]
        if node.op == 'and':
            if any(v is not None and not v for v in values):
                return 0
            return 1 if None not in values else None
        if any(v is not None and v for v in values):
            return 1
        return 0 if None not in values else None
    return None


def constant_truth(condition):
    """Whether a lowered condition is true, when constants alone decide it; else
    None (it is decided as the kernel runs)."""
    if isinstance(condition, list):
        return None
    value = constant_integer(condition)
    return None if value is None else bool(value)
