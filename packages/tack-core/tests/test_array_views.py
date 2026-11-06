# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Array views: ``a[i:j]`` is an array of rows ``i`` to ``j`` over ``a``'s storage.

A view's buffer is of a subclass of its root's buffer class (`lang/array.py`,
`view_buffer`), its pointer shifted (Metal: an offset into the MTLBuffer). It
frees nothing, keeps the root alive, and shares the root's write count and its
being handed outside Tack. These tests hold it to being an array like any
other, on every backend: kernels, host reads and writes, reductions, copies,
DLPack; and to writing only its own rows.
"""

import gc

import numpy as np
import pytest

import tack
from tack import arrays
from tack.lang.array import view_buffer
from tack.runtime.kernel_utils import arrays_disjoint

pytestmark = pytest.mark.quick


@tack.kernel
def _scale(x, s):
    for i in range(x.shape[0]):
        x[i] = x[i] * s


@tack.kernel
def _add(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@tack.kernel
def _rows_sum(m, out):
    for i in range(m.shape[0]):
        total = 0.0
        for j in range(m.shape[1]):
            total += m[i, j]
        out[i] = total


@tack.kernel
def _bump(v):
    for i in range(v.shape[0]):
        v[i] += tack.vec3(1.0, 2.0, 3.0)


@tack.kernel
def _count(counts, keys):
    for i in range(keys.shape[0]):
        tack.atomic_add(counts, keys[i], 1)


@tack.kernel
def _past_the_end(x, n):
    for i in range(1):
        x[n] = 1.0


def _floats(n, start=0.0):
    return tack.array(np.arange(n, dtype=np.float32) + start)


# -- what a view is --------------------------------------------------------

def test_a_view_reads_its_rows(backend):
    a = _floats(10)
    v = a[2:5]
    assert v.shape == (3,) and v.dtype == tack.f32 and v.device == a.device
    np.testing.assert_array_equal(v.numpy(), [2, 3, 4])
    assert v[0] == 2.0 and v[-1] == 4.0
    np.testing.assert_array_equal(np.asarray(v), [2, 3, 4])


def test_bounds_follow_python(backend):
    a = _floats(10)
    np.testing.assert_array_equal(a[-3:].numpy(), [7, 8, 9])
    np.testing.assert_array_equal(a[:2].numpy(), [0, 1])
    np.testing.assert_array_equal(a[8:100].numpy(), [8, 9])
    np.testing.assert_array_equal(a[:].numpy(), np.arange(10))
    np.testing.assert_array_equal(a[2:-2].numpy(), np.arange(2, 8))


def test_an_empty_range_is_a_new_empty_array(backend):
    a = _floats(10)
    for empty in (a[5:2], a[10:], a[3:3]):
        assert empty.shape == (0,) and empty.dtype == tack.f32
        assert empty.numpy().shape == (0,)
    m = tack.zeros((4, 3), tack.vec2(tack.f32))
    assert m[4:].shape == (0, 3) and m[4:].dtype == tack.vec2(tack.f32)


def test_rows_of_a_multidimensional_array(backend):
    host = np.arange(24, dtype=np.float32).reshape(6, 4)
    m = tack.array(host)
    v = m[1:4]
    assert v.shape == (3, 4)
    np.testing.assert_array_equal(v.numpy(), host[1:4])
    out = tack.zeros((3,), tack.f32)
    _rows_sum(v, out)
    np.testing.assert_array_equal(out.numpy(), host[1:4].sum(axis=1))
    assert v[2, 3] == host[3, 3]


def test_rows_of_vector_and_matrix_arrays(backend):
    host = np.arange(18, dtype=np.float32).reshape(6, 3)
    a = tack.array(host, tack.vec3(tack.f32))
    v = a[2:4]
    assert v.dtype == tack.vec3(tack.f32) and v.shape == (2,)
    np.testing.assert_array_equal(v.numpy(), host[2:4])
    _bump(v)
    expected = host.copy()
    expected[2:4] += [1, 2, 3]
    np.testing.assert_array_equal(a.numpy(), expected)
    mats = tack.array(np.arange(20, dtype=np.float32).reshape(5, 2, 2), tack.mat2(tack.f32))
    np.testing.assert_array_equal(mats[3:].numpy(), np.arange(20).reshape(5, 2, 2)[3:])
    assert mats[3:].dtype == tack.mat2(tack.f32)


def test_a_view_of_a_view_is_a_view_of_the_root(backend):
    a = _floats(10)
    w = a[1:9][2:4]
    np.testing.assert_array_equal(w.numpy(), [3, 4])
    assert w._buffer._root is a._buffer and w._buffer._start == 3
    _scale(w, 10.0)
    np.testing.assert_array_equal(a.numpy(), [0, 1, 2, 30, 40, 5, 6, 7, 8, 9])


def _same_view(a, b):
    """Two arrays alike in every attribute, their buffers too (but the buffer)."""
    assert type(a) is type(b) and type(a._buffer) is type(b._buffer)
    left, right = dict(vars(a)), dict(vars(b))
    left.pop("_buffer"), right.pop("_buffer")
    left.pop("_sig", None), right.pop("_sig", None)
    assert left == right
    keys = set(vars(a._buffer)) | set(vars(b._buffer))
    for k in keys:
        x, y = vars(a._buffer).get(k), vars(b._buffer).get(k)
        if isinstance(x, np.ndarray):
            assert x.__array_interface__ == y.__array_interface__, k
        elif hasattr(x, "ptr"):             # Warp's array over the view's memory
            assert (x.ptr, x.shape, x.dtype) == (y.ptr, y.shape, y.dtype), k
        elif k != "_view":                  # Metal's NumPy view of its memory
            assert x == y or x is y, k


@pytest.mark.parametrize("kind", ["scalars", "rows", "vectors", "matrices", "view"])
def test_many_views_at_once_are_the_views_one_at_a_time(backend, kind):
    base = np.arange(48, dtype=np.float32)
    whole = {"scalars": lambda: tack.array(base),
             "rows": lambda: tack.array(base.reshape(12, 4)),
             "vectors": lambda: tack.array(base.reshape(16, 3), tack.vec3(tack.f32)),
             "matrices": lambda: tack.array(base.reshape(12, 2, 2), tack.mat2(tack.f32)),
             "view": lambda: tack.array(base)[5:45]}[kind]()
    n = whole.shape[0]
    offsets = [0, 1, 1, 4, n - 2, n]
    views = whole._row_views(np.array(offsets))
    assert len(views) == len(offsets) - 1
    for view, a, b in zip(views, offsets[:-1], offsets[1:]):
        one = whole[a:b]
        np.testing.assert_array_equal(view.numpy(), one.numpy())
        assert view.shape == one.shape and view.dtype == one.dtype
        if b > a:
            _same_view(view, one)
    tack.copyto(views[2], np.full(views[2].numpy().shape, -1, np.float32))
    assert (whole.numpy()[1:4] == -1).all()


def test_many_views_refuse_offsets_out_of_order_or_range():
    x = _floats(6)
    for offsets in ([0, 3, 2], [-1, 2], [0, 7]):
        with pytest.raises(ValueError, match="nondecreasing"):
            x._row_views(offsets)
    s = tack.soa(_floats(4), _floats(4))
    np.testing.assert_array_equal(s._row_views([0, 1, 4])[1].numpy(), s[1:4].numpy())


def test_refused_forms():
    a = _floats(10)
    with pytest.raises(ValueError, match="no step"):
        a[::2]
    with pytest.raises(ValueError, match="no step"):
        a[8:2:-1]
    with pytest.raises(TypeError, match="no rows"):
        tack.zeros((), tack.f32)[0:1]
    with pytest.raises(TypeError, match="a view of rows"):
        tack.zeros((3, 3), tack.f32)[0:1, 1]
    with pytest.raises(TypeError, match="not written by element"):
        a[2:4] = 1.0
    with pytest.raises(BufferError, match="no allocation of its own"):
        a[2:4]._buffer.export_memory()


# -- writing through a view ---------------------------------------------------

def test_a_kernel_writes_only_the_views_rows(backend):
    a = _floats(10)
    _scale(a[3:6], -1.0)
    np.testing.assert_array_equal(a.numpy(), [0, 1, 2, -3, -4, -5, 6, 7, 8, 9])


def test_host_writes_reach_the_parent(backend):
    a = _floats(8)
    a[2:4].fill(-1.0)
    tack.copyto(a[5:7], [50.0, 60.0])
    tack.copyto(a[0:1], _floats(1, 100.0))
    np.testing.assert_array_equal(a.numpy(), [100, 1, -1, -1, 4, 50, 60, 7])
    vectors = tack.zeros((4,), tack.vec3(tack.f32))
    vectors[1:3].fill([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(vectors.numpy(), [[0, 0, 0], [1, 2, 3], [1, 2, 3], [0, 0, 0]])


def test_views_of_one_array_in_one_kernel(backend):
    a = _floats(9)
    # Two disjoint views read, a third written: rows 6-8 = rows 0-2 + rows 3-5.
    _add(a[0:3], a[3:6], a[6:9])
    np.testing.assert_array_equal(a.numpy(), [0, 1, 2, 3, 4, 5, 3, 5, 7])
    # In place through three views of the same rows: each iteration reads and
    # writes its own element only, so no two iterations conflict.
    b = _floats(5)
    _add(b[1:4], b[1:4], b[1:4])
    np.testing.assert_array_equal(b.numpy(), [0, 2, 4, 6, 4])


def test_disjoint_views_are_disjoint_for_the_cpu():
    """The CPU's noalias variant needs the views' byte ranges, not their root's."""
    a = tack.array(np.arange(9, dtype=np.float32), device="cpu")
    ir = _add.get_ir({}, template_args=None, texture_arrays={}).functions[0]
    assert arrays_disjoint(ir, [a[0:3], a[3:6], a[6:9]])
    assert not arrays_disjoint(ir, [a[0:3], a[3:6], a[5:8]])
    assert not arrays_disjoint(ir, [a[0:3], a[3:6], a])


def test_atomics_into_a_view(backend):
    counts = tack.zeros((6,), tack.i32)
    keys = tack.array(np.array([0, 1, 1, 2, 2, 2], np.int32))
    _count(counts[3:6], keys)
    np.testing.assert_array_equal(counts.numpy(), [0, 0, 0, 1, 2, 3])


# -- reading and copying --------------------------------------------------------

def test_reductions_copies_and_reshapes(backend):
    a = _floats(10)
    v = a[4:8]
    assert v.sum() == 4 + 5 + 6 + 7 and v.min() == 4.0 and v.max() == 7.0
    assert v.mean() == 5.5
    c = v.copy()
    _scale(c, 0.0)
    np.testing.assert_array_equal(a.numpy(), np.arange(10))      # a copy is its own
    np.testing.assert_array_equal(v.astype(tack.i32).numpy(), [4, 5, 6, 7])
    np.testing.assert_array_equal(v.reshape(2, 2).numpy(), [[4, 5], [6, 7]])
    np.testing.assert_array_equal(tack.concat([a[8:], a[:2]]).numpy(), [8, 9, 0, 1])


def test_dlpack_shares_the_views_memory(backend):
    a = _floats(10)
    v = a[3:6]
    back = tack.from_dlpack(v)
    np.testing.assert_array_equal(back.numpy(), [3, 4, 5])
    _scale(back, 2.0)
    np.testing.assert_array_equal(a.numpy(), [0, 1, 2, 6, 8, 10, 6, 7, 8, 9])


def test_a_metal_view_handed_out_comes_back_at_its_offset(backend):
    """Metal: a view's host memory, handed to another library (VTK's copy=False
    sharing does this) and wrapped again, is the view, not its MTLBuffer's start."""
    if backend != "metal":
        pytest.skip("Metal's host wrapping (_wrap_host)")
    a = _floats(10)
    view = a[3:6]                          # held, as the sharing library holds it
    host = view._buffer.host_view()
    back = tack.array_from_ptr(host, (3,), tack.f32, device="metal", writable=True)
    assert back._buffer._offset == 12
    _scale(back, 2.0)
    np.testing.assert_array_equal(a.numpy(), [0, 1, 2, 6, 8, 10, 6, 7, 8, 9])


def test_numpy_reads_a_cpu_view_through_dlpack():
    a = _floats(10)
    np.testing.assert_array_equal(np.from_dlpack(a[3:6]), [3, 4, 5])


def test_soa_rows():
    x, y = _floats(5), _floats(5, 10.0)
    s = tack.soa(x, y)
    v = s[1:3]
    assert v.layout == "soa" and v.shape == (2,)
    np.testing.assert_array_equal(v.numpy(), [[1, 11], [2, 12]])
    assert s[5:].shape == (0,)


# -- the allocation's state ---------------------------------------------------------

def test_writes_through_any_view_change_every_revision(backend):
    a = _floats(10)
    v, w = a[0:3], a[5:8]
    before = (arrays._revision(a), arrays._revision(v), arrays._revision(w))
    _scale(v, 2.0)
    after = (arrays._revision(a), arrays._revision(v), arrays._revision(w))
    assert all(x != y for x, y in zip(before, after))
    before = arrays._revision(v)
    tack.copyto(a, np.zeros(10, np.float32))
    assert arrays._revision(v) != before


def test_a_view_keeps_its_root_alive(backend):
    a = _floats(1000)
    v = a[990:]
    root = a._buffer
    del a
    gc.collect()
    junk = [_floats(1000, 5000.0) for _ in range(4)]      # would reuse a freed buffer
    np.testing.assert_array_equal(v.numpy(), np.arange(990, 1000))
    assert v._buffer._root is root
    del junk


def test_a_read_only_array_gives_read_only_views():
    host = np.arange(6, dtype=np.float32)
    a = tack.array_from_ptr(host.ctypes.data, (6,), tack.f32, device="cpu")
    v = a[1:3]
    with pytest.raises(RuntimeError, match="read-only"):
        v.fill(0.0)
    with pytest.raises(ValueError, match="read-only|not writable|writ"):
        _scale(v, 2.0)


def test_shared_is_the_roots(backend):
    a = _floats(4)
    v = a[1:3]
    v._buffer._shared = True
    assert getattr(a._buffer, "_shared", False) is True
    assert a[0:1]._buffer._shared is True


def test_view_buffer_counts_scalars():
    """`view_buffer` takes scalar elements: a vec3 array's row is three of them."""
    a = tack.array(np.arange(12, dtype=np.float32).reshape(4, 3), tack.vec3(tack.f32))
    buffer = view_buffer(a._buffer, 3, 6)
    assert buffer.nbytes == 24
    np.testing.assert_array_equal(buffer.to_numpy(), np.arange(3, 9))


# -- debug mode ---------------------------------------------------------------------

def test_debug_mode_checks_against_the_view(backend):
    """A view's bounds are its own rows, not its root's."""
    a = _floats(10)
    was = tack.runtime.dispatch.debug_enabled()
    tack.set_debug(True)
    try:
        _past_the_end(a[0:4], 3)                  # the last row: fine
        with pytest.raises(IndexError):
            _past_the_end(a[0:4], 4)              # inside the root, past the view
    finally:
        tack.set_debug(was)
