# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernels that call C++ (tack.cpp): templates instantiated by the call's types,
compiled by clang to bitcode and inlined into the kernel."""

import shutil
import subprocess
import sys

import numpy as np
import pytest

import tack
import tack.cpp
from tack.cpp import T, ptr

_HEADER = """
#pragma once
#include <cmath>
namespace demo {
template <class V> struct Cubic {
  V a, b, c, d;
  V value(V x) const { return ((a * x + b) * x + c) * x + d; }
  V slope(V x) const { return (V(3) * a * x + V(2) * b) * x + c; }
};
template <class V> inline V root(V a, V b, V c, V d, V x0, int steps) {
  Cubic<V> f{a, b, c, d};
  V x = x0;
  for (int k = 0; k < steps; ++k) x -= f.value(x) / f.slope(x);
  return x;
}
template <class V> inline V damped(V x, V w) { return x * std::exp(-w * x * x); }
template <class V, int N> inline V length(V x, V y, V z) {
  V v[3] = {x, y, z}, s = 0;
  for (int i = 0; i < N; ++i) s += v[i] * v[i];
  return std::sqrt(s);
}
inline int twice(int k) { return 2 * k; }
}
"""


def _readable_clang():
    """Whether there is a clang whose bitcode llvmlite reads."""
    import os

    import llvmlite.binding as llvm
    clang = shutil.which(os.environ.get("TACK_CLANG") or "clang++")
    if clang is None:
        return False
    done = subprocess.run([clang, "-x", "c++", "-c", "-emit-llvm", "-", "-o", "-"],
                          input=b"int f() { return 1; }", capture_output=True)
    if done.returncode != 0:
        return False
    try:
        llvm.parse_bitcode(done.stdout)
    except RuntimeError:
        return False
    return True


pytestmark = [
    pytest.mark.quick,
    pytest.mark.skipif(not _readable_clang(), reason="no clang whose bitcode llvmlite reads"),
]


@pytest.fixture(autouse=True)
def _cpu():
    with tack.device("cpu"):
        yield


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cpp")
    (folder / "demo.h").write_text(_HEADER)
    return tack.cpp.Library(headers=["demo.h"], include_dirs=[folder])


@pytest.mark.parametrize("dtype", [tack.f32, tack.f64])
def test_a_template_is_instantiated_by_the_arguments(lib, dtype):
    damped = lib.function("demo::damped", result=T, args=(T, T))

    @tack.kernel
    def apply(x, out, w):
        for i in range(x.shape[0]):
            out[i] = damped(x[i], w) + 0.5

    xs = np.linspace(-2, 2, 257).astype(dtype.numpy_dtype)
    out = tack.zeros(257, dtype)
    apply(tack.array(xs), out, 0.25)
    want = xs * np.exp(-xs.dtype.type(0.25) * xs * xs) + 0.5
    assert out.numpy().dtype == xs.dtype
    np.testing.assert_allclose(out.numpy(), want, rtol=4 * np.finfo(xs.dtype).eps)


def test_called_through_a_device_function_with_literals_and_an_integer(lib):
    root = lib.function("demo::root", result=T, args=(T, T, T, T, T, tack.i32))

    @tack.func
    def newton(d):
        return root(1.0, 0.0, -2.0, d, 2.0, 8)

    @tack.kernel
    def roots(d, out):
        for i in range(d.shape[0]):
            out[i] = newton(d[i])

    ds = np.linspace(-5, -1, 100)
    out = tack.zeros(100, tack.f64)
    roots(tack.array(ds), out)
    x = out.numpy()
    # f64 arithmetic throughout: the literals took the call's type.
    assert np.abs(x**3 - 2 * x + ds).max() < 1e-13


def test_a_vector_passes_its_components_and_a_name_may_fix_template_arguments(lib):
    length = lib.function("demo::length<{T}, 3>", result=T, args=(T, T, T))

    @tack.kernel
    def lengths(v, out):
        for i in range(v.shape[0]):
            out[i] = length(v[i])

    vs = np.random.default_rng(0).random((50, 3)).astype(np.float32)
    out = tack.zeros(50, tack.f32)
    lengths(tack.array(vs, dtype=tack.vec(3, tack.f32)), out)
    np.testing.assert_allclose(out.numpy(), np.linalg.norm(vs, axis=1), rtol=1e-6)


def test_integer_functions(lib):
    twice = lib.function("demo::twice", result=tack.i32, args=(tack.i32,))

    @tack.kernel
    def double(out):
        for i in range(out.shape[0]):
            out[i] = twice(i) + 1

    out = tack.zeros(10, tack.i32)
    double(out)
    assert out.numpy().tolist() == [2 * i + 1 for i in range(10)]


def test_the_library_is_inlined_and_vectorized(lib, no_debug):
    root = lib.function("demo::root", result=T, args=(T, T, T, T, T, tack.i32))

    @tack.kernel
    def roots(d, out):
        for i in range(d.shape[0]):
            out[i] = root(1.0, 0.0, -2.0, d[i], 2.0, 8)

    d, out = tack.zeros(64, tack.f64), tack.zeros(64, tack.f64)
    text = tack.inspect(roots, d, out, mode="optimized")
    assert "tack_cpp_" not in text
    assert " x double>" in text


def test_wrong_argument_count_and_python_calls_are_refused(lib):
    damped = lib.function("demo::damped", result=T, args=(T, T))

    @tack.kernel
    def bad(x, out):
        for i in range(x.shape[0]):
            out[i] = damped(x[i])

    x = tack.zeros(4, tack.f32)
    with pytest.raises(Exception, match="takes 2 arguments"):
        bad(x, x)
    with pytest.raises(TypeError, match="called in a kernel"):
        damped(1.0, 2.0)


def test_a_compile_error_shows_clangs_message():
    broken = tack.cpp.Library(source="template <class V> V f(V x) { return x.nope(); }")
    f = broken.function("f", result=T, args=(T,))

    @tack.kernel
    def apply(x):
        for i in range(x.shape[0]):
            x[i] = f(x[i])

    with pytest.raises(Exception, match="nope"):
        apply(tack.zeros(4, tack.f32))


_MESH = """
#include <cmath>
namespace demo {
// A class template over array data, with methods the functions call.
template <class V, int N> struct Points {
  const V* p;
  V edge(int a, int b) const {
    V s = 0;
    for (int c = 0; c < 3; ++c) { V d = p[3 * a + c] - p[3 * b + c]; s += d * d; }
    return std::sqrt(s);
  }
  V perimeter() const {
    V s = 0;
    for (int k = 0; k < N; ++k) s += edge(k, (k + 1) % N);
    return s;
  }
};
template <class V> V perimeter(const V* points, const long long* ids, long long cell) {
  V local[9];
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 3; ++c) local[3 * k + c] = points[3 * ids[3 * cell + k] + c];
  return Points<V, 3>{local}.perimeter();
}
template <class V> V perimeter_of(const V* local) { return Points<V, 3>{local}.perimeter(); }
template <class V>
void centroid(const V* points, const long long* ids, long long cell, V* out) {
  for (int c = 0; c < 3; ++c) {
    V s = 0;
    for (int k = 0; k < 3; ++k) s += points[3 * ids[3 * cell + k] + c];
    out[3 * cell + c] = s / 3;
  }
}
}
"""


@pytest.fixture(scope="module")
def mesh():
    rng = np.random.default_rng(0)
    points = rng.random((200, 3))
    triangles = rng.integers(0, 200, (500, 3))

    def edge(a, b):
        return np.linalg.norm(points[triangles[:, a]] - points[triangles[:, b]], axis=1)
    return (tack.cpp.Library(source=_MESH), points, triangles,
            edge(0, 1) + edge(1, 2) + edge(2, 0))


@pytest.mark.parametrize("dtype", [tack.f32, tack.f64])
def test_arrays_pass_as_pointers_and_fix_the_template_type(mesh, dtype):
    lib, pts, tri, want = mesh
    perimeter = lib.function("demo::perimeter", result=T,
                             args=(ptr(T), ptr(tack.i64), tack.i64))

    @tack.kernel
    def perimeters(points, ids, out):
        for i in range(out.shape[0]):
            out[i] = perimeter(points, ids, i)

    points = tack.array(pts.astype(dtype.numpy_dtype), dtype=tack.vec(3, dtype))
    out = tack.zeros(len(tri), dtype)
    perimeters(points, tack.array(tri.ravel().astype(np.int64)), out)
    np.testing.assert_allclose(out.numpy(), want, rtol=20 * np.finfo(dtype.numpy_dtype).eps)


def test_a_local_array_passes_as_a_pointer(mesh):
    lib, pts, tri, want = mesh
    perimeter_of = lib.function("demo::perimeter_of", result=T, args=(ptr(T),))

    @tack.kernel
    def perimeters(points, ids, out):
        for i in range(out.shape[0]):
            local = tack.local_array(tack.f64, 9)
            for k in range(3):
                p = points[ids[3 * i + k]]
                for c in range(3):
                    local[3 * k + c] = p[c]
            out[i] = perimeter_of(local)

    out = tack.zeros(len(tri), tack.f64)
    perimeters(tack.array(pts, dtype=tack.vec(3, tack.f64)),
               tack.array(tri.ravel().astype(np.int64)), out)
    np.testing.assert_allclose(out.numpy(), want, rtol=1e-14)


def test_a_function_without_a_result_stores_through_a_pointer(mesh):
    lib, pts, tri, _ = mesh
    centroid = lib.function("demo::centroid", result=None,
                            args=(ptr(T), ptr(tack.i64), tack.i64, ptr(T, writes=True)))

    @tack.kernel
    def centroids(points, ids, out):
        for i in range(ids.shape[0] // 3):
            centroid(points, ids, i, out)

    points = tack.array(pts, dtype=tack.vec(3, tack.f64))
    ids = tack.array(tri.ravel().astype(np.int64))
    out = tack.zeros(len(tri), tack.vec(3, tack.f64))
    centroids(points, ids, out)
    np.testing.assert_allclose(out.numpy(), pts[tri].mean(axis=1), rtol=1e-14)

    # Tack counts the pointer as a store: a read-only array is refused there,
    # and accepted where the function only reads.
    host = np.zeros(len(tri) * 3)
    readonly = tack.array_from_ptr(host.ctypes.data, (len(tri) * 3,), tack.f64)
    with pytest.raises(ValueError, match="parameter 'out'.*read-only"):
        centroids(points, ids, readonly)
    frozen = np.ascontiguousarray(pts).ravel()
    centroids(tack.array_from_ptr(frozen.ctypes.data, (frozen.size,), tack.f64), ids, out)


def test_array_arguments_are_checked(mesh):
    lib, pts, tri, _ = mesh
    perimeter = lib.function("demo::perimeter", result=T,
                             args=(ptr(T), ptr(tack.i64), tack.i64))

    @tack.kernel
    def perimeters(points, ids, out):
        for i in range(out.shape[0]):
            out[i] = perimeter(points, ids, i)

    @tack.kernel
    def scalar_for_array(points, out):
        for i in range(out.shape[0]):
            out[i] = perimeter(points, i, i)

    points = tack.array(pts, dtype=tack.vec(3, tack.f64))
    out = tack.zeros(len(tri), tack.f64)
    with pytest.raises(Exception, match="argument 2 is an array of i32, where i64"):
        perimeters(points, tack.array(tri.ravel().astype(np.int32)), out)
    with pytest.raises(Exception, match="argument 2 is a scalar, where an array"):
        scalar_for_array(points, out)


def test_a_storing_call_outside_the_parallel_loop_is_refused(mesh):
    lib = mesh[0]
    centroid = lib.function("demo::centroid", result=None,
                            args=(ptr(T), ptr(tack.i64), tack.i64, ptr(T, writes=True)))

    @tack.kernel
    def outside(points, ids, out):
        centroid(points, ids, 0, out)
        for i in range(out.shape[0]):
            out[i] = 0.0

    a = tack.zeros(9, tack.f64)
    with pytest.raises(Exception, match="C\\+\\+ call that may store"):
        outside(a, tack.zeros(3, tack.i64), a)


_SCRIPT = """
import sys, numpy as np, tack, tack.cpp
lib = tack.cpp.Library(headers=["demo.h"], include_dirs=[sys.argv[1]])
scale = lib.function("demo::scale", result=tack.cpp.T, args=(tack.cpp.T,))

@tack.kernel
def apply(x):
    for i in range(x.shape[0]):
        x[i] = scale(x[i])

x = tack.ones(4, tack.f32)
apply(x)
print(x.numpy()[0], tack.cache.stats().get("variant_hit", 0))
"""


def test_disk_cache_needs_no_clang_and_notices_a_changed_header(tmp_path):
    """A stored variant holds the library's code, so a second process compiles
    nothing; a header's contents are in the key."""
    import os
    header = tmp_path / "demo.h"
    header.write_text("namespace demo { template <class V> V scale(V x) { return 2 * x; } }")
    script = tmp_path / "run.py"
    script.write_text(_SCRIPT)
    env = dict(os.environ, TACK_CACHE="write", TACK_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("TACK_DEBUG", None)

    def run(**extra):
        done = subprocess.run([sys.executable, str(script), str(tmp_path)],
                              env=dict(env, **extra), capture_output=True, text=True)
        return done.returncode, done.stdout.split(), done.stderr

    code, out, err = run()
    assert code == 0, err
    assert out == ["2.0", "0"]
    code, out, err = run(TACK_CLANG="/no/such/clang++")
    assert code == 0, err
    assert out == ["2.0", "1"]
    header.write_text("namespace demo { template <class V> V scale(V x) { return 3 * x; } }")
    code, out, err = run(TACK_CLANG="/no/such/clang++")
    assert code != 0 and "needs clang" in err
    code, out, err = run()
    assert out == ["3.0", "0"]
