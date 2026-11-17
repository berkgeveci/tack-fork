# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Branches that constants decide are resolved while lowering
(lang/lowering/constant_conditions.py): only the arm taken is lowered."""

import numpy as np
import pytest

import tack

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

NOT_A_CONSTANT = 0.5                # a module float: a kernel reading it fails to lower


def _depth(line):
    return len(line) - len(line.lstrip())


def _branches(kernel, *args):
    """The branches left in the kernel's IR for these arguments: those with an
    arm. A decided condition stays as an ``If`` with empty arms, so that it is
    still evaluated and typed. Debug mode's bounds checks (an ``If`` around a
    store, an ``IfExp`` around a load) are not the kernel's branches."""
    lines = tack.inspect(kernel, *args, mode="ir").splitlines()

    def branch(k, line):
        if line.count("IfExp(") > line.count("IfExp((Bounds["):
            return True
        text = line.lstrip()
        return (text.startswith("If (") and not text.startswith("If (Bounds[")
                and k + 1 < len(lines) and _depth(lines[k + 1]) > _depth(line))

    return [line for k, line in enumerate(lines) if branch(k, line)]


@tack.device_class
class Shape:
    SIDES = 3

    @tack.func
    def sides(self):
        if self.SIDES == 3:
            return 30
        if self.SIDES == 4:
            return 40
        return 0


@tack.device_class
class Square(Shape):
    SIDES = 4


@tack.kernel
def by_class(shape, out):
    for i in range(out.shape[0]):
        out[i] = shape.sides()


@pytest.fixture(autouse=True)
def _cpu():
    tack.set_default_device("cpu")


@pytest.mark.parametrize("shape, expected", [(Shape(), 30), (Square(), 40)])
def test_a_class_constant_picks_the_arm(shape, expected):
    out = tack.zeros((4,), tack.i32)
    by_class(shape, out)
    np.testing.assert_array_equal(out.numpy(), np.full(4, expected))
    assert _branches(by_class, shape, out) == []


@tack.kernel
def unlowerable_arm(out):
    for i in range(out.shape[0]):
        if 0 == 1:
            out[i] = NOT_A_CONSTANT      # would raise NameError if lowered
        else:
            out[i] = 7


def test_the_arm_not_taken_is_not_lowered():
    out = tack.zeros((3,))
    unlowerable_arm(out)
    np.testing.assert_array_equal(out.numpy(), np.full(3, 7))


@tack.kernel
def barrier_never_run(out):
    for i in range(out.shape[0]):
        if 1 > 2:
            tack.barrier()
        out[i] = i


def test_a_primitive_in_an_arm_not_taken_is_not_required():
    out = tack.zeros((4,), tack.i32)
    barrier_never_run(out)              # the CPU has no workgroups
    np.testing.assert_array_equal(out.numpy(), np.arange(4))


@tack.kernel
def false_and_runtime(data, out):
    for i in range(out.shape[0]):
        if (6 & 1) != 0 and data[i] > 0:
            out[i] = 1
        else:
            out[i] = 2


@tack.kernel
def true_or_runtime(data, out):
    for i in range(out.shape[0]):
        out[i] = 1 if data[i] > 0 or not False else 2


def test_and_or_are_decided_by_one_operand():
    data = tack.array(np.array([1.0, -1.0], np.float32))
    out = tack.zeros((2,), tack.i32)
    false_and_runtime(data, out)
    np.testing.assert_array_equal(out.numpy(), [2, 2])
    true_or_runtime(data, out)
    np.testing.assert_array_equal(out.numpy(), [1, 1])
    assert _branches(false_and_runtime, data, out) == []
    assert _branches(true_or_runtime, data, out) == []


@tack.kernel
def runtime_branch(data, out):
    for i in range(out.shape[0]):
        if 0.5 < 1.0 and data[i] > 0:
            out[i] = 1
        else:
            out[i] = 2


def test_float_and_runtime_conditions_are_left_to_run():
    data = tack.array(np.array([1.0, -1.0], np.float32))
    out = tack.zeros((2,), tack.i32)
    runtime_branch(data, out)
    np.testing.assert_array_equal(out.numpy(), [1, 2])
    assert len(_branches(runtime_branch, data, out)) == 1


@tack.kernel
def bound_in_the_arm_not_taken(out):
    for i in range(out.shape[0]):
        if 1 == 2:
            x = 1
        out[i] = x


def test_a_name_bound_only_in_the_arm_not_taken_is_unbound():
    """The error says why: the condition that decided it, and where."""
    with pytest.raises(NameError, match=r"name 'x' .* assigned only in the arm of "
                                        r"`if 1 == 2` \(line \d+\) that constants decide against"):
        bound_in_the_arm_not_taken(tack.zeros((2,), tack.i32))


@tack.kernel
def joined_in_a_statement(out):
    for i in range(out.shape[0]):
        if 1 == 1:
            x = tack.i32(2147483647)
        else:
            x = tack.i64(0)
        out[i] = x + 1


@tack.kernel
def joined_in_an_expression(out):
    for i in range(out.shape[0]):
        x = tack.i32(2147483647) if 1 == 1 else tack.i64(0)
        out[i] = x + 1


@tack.kernel
def joined_at_run_time(flag, out):
    for i in range(out.shape[0]):
        if flag == 1:
            x = tack.i32(2147483647)
        else:
            x = tack.i64(0)
        out[i] = x + 1


@pytest.mark.parametrize("kernel", [joined_in_a_statement, joined_in_an_expression])
def test_a_decided_branch_takes_the_types_of_the_arm_taken(kernel):
    """As in C++'s `if constexpr`, the arm not taken has no part in the types:
    `x` is the arm taken's i32, so `x + 1` wraps. A branch decided as the
    kernel runs joins its arms' types: `x` is i64 there."""
    out = tack.zeros((1,), tack.i64)
    kernel(out)
    assert out.numpy()[0] == -2147483648
    joined_at_run_time(1, out)
    assert out.numpy()[0] == 2147483648


@tack.kernel
def checked_before_false(data, out):
    for i in range(out.shape[0]):
        if data[i + 1] > 0 and False:
            out[i] = 1
        else:
            out[i] = 2


@tack.kernel
def checked_before_true(data, out):
    for i in range(out.shape[0]):
        out[i] = 1 if data[i + 1] > 0 or True else 2


@pytest.mark.parametrize("kernel", [checked_before_false, checked_before_true])
def test_operands_before_the_deciding_constant_are_still_evaluated(kernel):
    """`data[i + 1] > 0 and False` is false whatever the load gives, but the load
    still runs, and in debug mode its bounds check still raises."""
    data = tack.array(np.array([1.0], np.float32))
    out = tack.zeros((1,), tack.i32)
    tack.set_debug(True)
    try:
        with pytest.raises(IndexError, match=r"data\[i \+ 1\]"):
            kernel(data, out)
    finally:
        tack.set_debug(False)
    assert _branches(kernel, data, out) == []


@tack.kernel
def literal_past_64_bits(out):
    for i in range(out.shape[0]):
        if 18446744073709551616 > 0:
            out[i] = 1


@tack.kernel
def u64_against_i64(out):
    for i in range(out.shape[0]):
        out[i] = 1 if tack.u64(18446744073709551615) > tack.i64(-1) else 2


@pytest.mark.parametrize("kernel, message", [
    (literal_past_64_bits, "outside Tack's 64-bit range"),
    (u64_against_i64, "Cannot implicitly mix"),
])
def test_a_decided_condition_is_still_typed(kernel, message):
    with pytest.raises(TypeError, match=message):
        kernel(tack.zeros((1,), tack.i32))
