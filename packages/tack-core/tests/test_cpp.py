# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernels that call C++ (tack.cpp): templates instantiated by the call's types,
compiled by clang to bitcode and inlined into the kernel."""

import subprocess
import sys

import numpy as np
import pytest

import tack
import tack.arrays
import tack.cpp
from tack.cpp import T, indexable, obj, ptr, vec

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

    import llvmlite.binding as llvm
    clang = tack.cpp._find_clang()
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


def test_literals_leave_t_to_the_kernel_and_t_can_be_fixed(lib):
    """Float literals decide nothing, so a call given only literals for T takes
    the kernel's float type; `of` and ``T=`` fix it."""
    damped = lib.function("demo::damped", result=T, args=(T, T))

    @tack.kernel
    def constant(out):
        for i in range(out.shape[0]):
            out[i] = damped(0.1, 0.25)

    damped32 = damped.of(tack.f32)

    @tack.kernel
    def narrow(out):
        for i in range(out.shape[0]):
            out[i] = damped32(0.1, 0.25)

    exact = 0.1 * np.exp(-0.25 * 0.01)
    out = tack.zeros(4, tack.f64)
    constant(out)
    assert out.numpy()[0] == pytest.approx(exact, rel=1e-15)     # f64: the kernel's
    narrow(out)
    assert out.numpy()[0] == np.float32(np.float32(0.1) * np.exp(np.float32(-0.0025)))
    out32 = tack.zeros(4, tack.f32)
    constant(out32)
    assert out32.numpy()[0] == pytest.approx(exact, rel=1e-6)
    wide = lib.function("demo::damped", result=T, args=(T, T), T=tack.f64)

    @tack.kernel
    def widened(x, out):
        for i in range(out.shape[0]):
            out[i] = wide(x[i], 0.25)        # f32 argument, f64 computation

    wide_out = tack.zeros(1, tack.f64)
    widened(tack.full(1, 0.1, tack.f32), wide_out)
    x = float(np.float32(0.1))
    assert wide_out.numpy()[0] == pytest.approx(x * np.exp(-0.25 * x * x), rel=1e-15)
    with pytest.raises(TypeError, match="f32 or f64"):
        damped.of(tack.i32)


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


def test_private_definitions_of_the_library_link():
    """A switch clang turns into a private lookup table: already local, the link
    may rename it, and Tack must not look for it to make it internal."""
    lib = tack.cpp.Library(source="""
template <class V> V weight(int k) {
  switch (k) { case 0: return V(0.5); case 1: return V(1.5); case 2: return V(2.5);
               case 3: return V(4.0); case 4: return V(8.5); default: return V(0); }
}""")
    weight = lib.function("weight<{T}>", result=T, args=(tack.i32,))

    @tack.kernel
    def weights(out):
        for i in range(out.shape[0]):
            out[i] = weight(i)

    for dtype in (tack.f32, tack.f64):
        out = tack.zeros(6, dtype)
        weights(out)
        assert out.numpy().tolist() == [0.5, 1.5, 2.5, 4.0, 8.5, 0.0]


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


_CALLS_METHODS = """
namespace demo {
// Simpson's rule over any F with value(x), and Newton steps with value and slope.
template <class F, class V> V integrate(const F& f, V a, V b, int n) {
  V h = (b - a) / n, s = f.value(a) + f.value(b);
  for (int k = 1; k < n; ++k) s += f.value(a + k * h) * (k % 2 ? 4 : 2);
  return s * h / 3;
}
template <class F, class V> V newton(const F& f, V x, int steps) {
  for (int k = 0; k < steps; ++k) x -= f.value(x) / f.slope(x);
  return x;
}
template <class F> int visit(const F& f, int n) {
  int s = 0;
  for (int k = 0; k < n; ++k) s += f.mark(k);
  return s;
}
}
"""


@tack.device_class
class _Poly:
    """A polynomial less a shift: an array, a runtime scalar and two methods."""

    def __init__(self, coefficients, shift, dtype=tack.f64):
        self.c = tack.array(np.asarray(coefficients, dtype=dtype.numpy_dtype))
        self.shift = shift

    @tack.func
    def value(self, x):
        r = 0.0 * x
        for k in range(self.c.shape[0]):
            r = r * x + self.c[k]
        return r - self.shift

    @tack.func
    def slope(self, x):
        r = 0.0 * x
        n = self.c.shape[0]
        for k in range(n - 1):
            r = r * x + (n - 1 - k) * self.c[k]
        return r


@tack.device_class
class _Scaled:
    """A template holding a template, with a method that calls its."""

    def __init__(self, base, factor):
        self.base = base
        self.factor = factor

    @tack.func
    def value(self, x):
        return self.factor * self.base.value(x)


@pytest.fixture(scope="module")
def calls_methods():
    lib = tack.cpp.Library(source=_CALLS_METHODS)
    return (lib.function("demo::integrate", result=T,
                         args=(obj(value=(T, (T,))), T, T, tack.i32)),
            lib.function("demo::newton", result=T,
                         args=(obj(value=(T, (T,)), slope=(T, (T,))), T, tack.i32)),
            lib)


_COEFFICIENTS = [1.0, 0.0, -2.0, 1.0]


def _integral(a, b, shift):
    antiderivative = np.polyint(np.poly1d(_COEFFICIENTS))
    return antiderivative(b) - antiderivative(a) - shift * (b - a)


@pytest.mark.parametrize("dtype", [tack.f32, tack.f64])
def test_cpp_calls_a_device_classs_method(calls_methods, dtype):
    integrate = calls_methods[0]

    @tack.kernel
    def integrals(f, a, b, out):
        for i in range(out.shape[0]):
            out[i] = integrate(f, a[i], b[i], 64)

    np_type = dtype.numpy_dtype
    a = np.linspace(0, 1, 100).astype(np_type)
    b = (a + 1.5).astype(np_type)
    poly = _Poly(_COEFFICIENTS, 0.5, dtype)
    out = tack.zeros(100, dtype)
    integrals(poly, tack.array(a), tack.array(b), out)
    tolerance = 1e-13 if dtype is tack.f64 else 2e-5
    np.testing.assert_allclose(out.numpy(), _integral(a, b, 0.5), rtol=tolerance)
    # A runtime scalar of the object, read by the method C++ calls: no new variant.
    poly.shift = 0.25
    integrals(poly, tack.array(a), tack.array(b), out)
    np.testing.assert_allclose(out.numpy(), _integral(a, b, 0.25), rtol=tolerance)


def test_cpp_calls_two_methods_and_both_are_inlined(calls_methods, no_debug):
    newton = calls_methods[1]

    @tack.kernel
    def roots(f, x0, out):
        for i in range(out.shape[0]):
            out[i] = newton(f, x0[i], 20)

    poly = _Poly(_COEFFICIENTS, 0.25)
    x0 = tack.array(np.linspace(1.2, 3, 50))
    out = tack.zeros(50, tack.f64)
    roots(poly, x0, out)
    assert np.abs(np.polyval(_COEFFICIENTS, out.numpy()) - 0.25).max() < 1e-13
    text = tack.inspect(roots, poly, x0, out, mode="optimized")
    # No function of either side is left, called or defined.
    assert "@tack_cpp_" not in text and "@tack_cb_" not in text


def test_objects_held_by_objects_and_passed_through_device_functions(calls_methods):
    integrate = calls_methods[0]

    @tack.func
    def area(f, a, b):
        return integrate(f, a, b, 64)

    @tack.kernel
    def nested(f, a, b, out):
        for i in range(out.shape[0]):
            out[i] = area(f, a[i], b[i]) + integrate(f.base, a[i], b[i], 64)

    a = np.linspace(0, 1, 20)
    b = a + 1.5
    out = tack.zeros(20, tack.f64)
    nested(_Scaled(_Poly(_COEFFICIENTS, 0.5), 3.0), tack.array(a), tack.array(b), out)
    np.testing.assert_allclose(out.numpy(), 4.0 * _integral(a, b, 0.5), rtol=1e-13)


def test_a_method_cpp_calls_may_store_and_tack_counts_it(calls_methods):
    visit = calls_methods[2].function("demo::visit", result=tack.i32,
                                      args=(obj(mark=(tack.i32, (tack.i32,))), tack.i32))

    @tack.device_class
    class Rows:
        def __init__(self, seen, width):
            self.seen = seen
            self.width = width
            self.row = 0

        @tack.func
        def mark(self, k):
            self.seen[self.row * self.width + k] = k + 1
            return k

    @tack.kernel
    def fill(rows, out):
        for i in range(out.shape[0]):
            out[i] = visit(rows, 4)

    seen = tack.zeros(4, tack.i32)
    out = tack.zeros(1, tack.i32)
    fill(Rows(seen, 4), out)
    assert seen.numpy().tolist() == [1, 2, 3, 4] and out.numpy().tolist() == [6]
    host = np.zeros(4, np.int32)
    readonly = tack.array_from_ptr(host.ctypes.data, (4,), tack.i32)
    with pytest.raises(ValueError, match="read-only"):
        fill(Rows(readonly, 4), out)


def test_objects_are_checked(calls_methods):
    integrate = calls_methods[0]
    plain = calls_methods[2].function("demo::integrate", result=T, args=(T, T, T, tack.i32))

    @tack.device_class
    class NoValue:
        def __init__(self):
            self.k = 1.0

    @tack.device_class
    class VectorValue:
        def __init__(self):
            self.k = 1.0

        @tack.func
        def value(self, x):
            return tack.vec2(x, self.k)

    @tack.kernel
    def integrals(f, out):
        for i in range(out.shape[0]):
            out[i] = integrate(f, 0.0, 1.0, 8)

    @tack.kernel
    def undeclared(f, out):
        for i in range(out.shape[0]):
            out[i] = plain(f, 0.0, 1.0, 8)

    out = tack.zeros(2, tack.f64)
    with pytest.raises(Exception, match="has no @tack.func method 'value'"):
        integrals(NoValue(), out)
    with pytest.raises(Exception, match="returns a vector"):
        integrals(VectorValue(), out)
    with pytest.raises(Exception, match="does not make a tack.cpp.obj"):
        undeclared(_Poly(_COEFFICIENTS, 0.0), out)


_GENERIC = """
namespace demo {
// For any container with size() and operator[]: the mean of a window, clipped.
template <class C> auto window(const C& c, long long at, long long half) {
  decltype(c[0] + 0) s = 0;
  long long n = 0;
  for (long long k = at - half; k <= at + half; ++k)
    if (k >= 0 && k < (long long)c.size()) { s += c[k]; ++n; }
  return s / n;
}
template <class C, class V> V window_mean(const C& c, long long at, long long half, V) {
  return window(c, at, half);
}
// For any callable: a central difference.
template <class F, class V> V derivative(const F& f, V x, V h) {
  return (f(x + h) - f(x - h)) / (2 * h);
}
}
"""


def _window_means(values, half):
    return np.array([values[max(0, i - half):i + half + 1].mean()
                     for i in range(len(values))])


@pytest.fixture(scope="module")
def generic():
    return tack.cpp.Library(source=_GENERIC)


def test_a_container_template_takes_implicit_arrays(generic):
    """`__getitem__` is ``operator[]`` and `__len__` is ``size()``, so a template
    for any indexable container reads tack.arrays' implicit arrays, composed ones
    too, and each kind compiles to its own arithmetic."""
    window_mean = generic.function("demo::window_mean", result=T,
                                   args=(indexable(T), tack.i64, tack.i64, T))

    @tack.kernel
    def smooth(c, out, half):
        for i in range(out.shape[0]):
            out[i] = window_mean(c, i, half, out[i])

    n = 200
    values = np.random.default_rng(0).random(n)
    reverse = np.arange(n)[::-1].copy()
    cases = [
        (tack.arrays.CountingArray(n, 0.0, 1.0, dtype=tack.f64), np.arange(n, dtype=float)),
        (tack.arrays.ConstantArray(2.5, n), np.full(n, 2.5)),
        (tack.arrays.Permutation(tack.array(reverse), tack.array(values)), values[::-1]),
        (tack.arrays.Permutation(tack.array(reverse),
                                 tack.arrays.CountingArray(n, 1.0, 0.5, dtype=tack.f64)),
         (1.0 + 0.5 * np.arange(n))[::-1]),
    ]
    for array, host in cases:
        out = tack.zeros(n, tack.f64)
        smooth(array, out, 3)
        np.testing.assert_allclose(out.numpy(), _window_means(host, 3), rtol=1e-14)


def test_a_device_class_with_the_protocol_and_a_callable_one(generic):
    window_mean = generic.function("demo::window_mean", result=T,
                                   args=(indexable(T, index=tack.i32), tack.i64, tack.i64, T))
    derivative = generic.function("demo::derivative", result=T,
                                  args=(obj(__call__=(T, (T,))), T, T))

    @tack.device_class
    class Squares:
        def __init__(self, n):
            self.n = n

        @tack.func
        def __getitem__(self, k):
            return tack.f32(k * k)

        @tack.func
        def __len__(self):
            return self.n

    @tack.device_class
    class Cubic:
        def __init__(self, k):
            self.k = k

        @tack.func
        def __call__(self, x):
            return self.k * x * x * x

    @tack.kernel
    def smooth(c, out):
        for i in range(out.shape[0]):
            out[i] = window_mean(c, i, 2, out[i])

    @tack.kernel
    def slopes(f, x, out):
        for i in range(out.shape[0]):
            out[i] = derivative(f, x[i], 1e-3)

    out = tack.zeros(20, tack.f32)
    smooth(Squares(20), out)
    np.testing.assert_allclose(out.numpy(), _window_means(np.arange(20.0) ** 2, 2), rtol=1e-6)
    xs = np.linspace(-1, 1, 50)
    slope = tack.zeros(50, tack.f64)
    slopes(Cubic(2.0), tack.array(xs), slope)
    np.testing.assert_allclose(slope.numpy(), 6 * xs**2 + 2e-6, atol=1e-11)


_ALGORITHMS = """
#include <algorithm>
namespace demo {
template <class C, class V> V total(const C& c, V) {
  V s = 0;
  for (auto x : c) s += x;
  return s;
}
template <class C, class V> V largest(const C& c, V) {
  return *std::max_element(c.begin(), c.end());
}
template <class C> int sort_row(C& c, long long row, long long width) {
  std::sort(c.begin() + row * width, c.begin() + (row + 1) * width);
  return 0;
}
template <class C> int bump(C& c, long long i) {
  c[i] += 1;
  c[i] *= 2;
  return 0;
}
}
"""


@pytest.fixture(scope="module")
def algorithms():
    lib = tack.cpp.Library(source=_ALGORITHMS)
    return (lib.function("demo::total", result=T, args=(indexable(T), T)),
            lib.function("demo::largest", result=T, args=(indexable(T), T)),
            lib.function("demo::sort_row", result=tack.i32,
                         args=(indexable(T, writable=True), tack.i64, tack.i64)),
            lib.function("demo::bump", result=tack.i32,
                         args=(indexable(T, writable=True), tack.i64)))


def test_containers_iterate_for_range_for_and_algorithms(algorithms):
    total, largest = algorithms[:2]

    @tack.kernel
    def summary(c, out):
        for i in range(out.shape[0]):
            out[i] = total(c, out[i]) + largest(c, out[i])

    out = tack.zeros(1, tack.f64)
    summary(tack.arrays.CountingArray(10, 1.0, 1.0, dtype=tack.f64), out)
    assert out.numpy()[0] == 55.0 + 10.0
    values = np.random.default_rng(2).random(30)
    summary(tack.arrays.Permutation(tack.array(np.arange(30)[::-1].copy()),
                                    tack.array(values)), out)
    assert out.numpy()[0] == pytest.approx(values.sum() + values.max(), rel=1e-14)


def test_writable_containers_sort_and_update_in_place(algorithms):
    """Element references assign through ``__setitem__``: std::sort sorts a
    Tack array through a view, and compound assignment updates it."""
    sort_row, bump = algorithms[2:]

    @tack.kernel
    def sort_rows(c, width, done):
        for i in range(done.shape[0]):
            done[i] = sort_row(c, i, width)

    @tack.kernel
    def bumps(c, done):
        for i in range(done.shape[0]):
            done[i] = bump(c, i)

    rows, width = 40, 17
    data = np.random.default_rng(3).random(rows * width)
    a = tack.array(data)
    sort_rows(tack.arrays.View(a, 0, rows * width), width, tack.zeros(rows, tack.i32))
    np.testing.assert_array_equal(a.numpy().reshape(rows, width),
                                  np.sort(data.reshape(rows, width), axis=1))
    b = tack.array(np.arange(5.0))
    bumps(tack.arrays.Strided(b, 1), tack.zeros(5, tack.i32))
    np.testing.assert_array_equal(b.numpy(), (np.arange(5.0) + 1) * 2)
    # The view's array is written, as Tack sees it: a read-only one is refused.
    frozen = np.arange(5.0)
    readonly = tack.array_from_ptr(frozen.ctypes.data, (5,), tack.f64)
    with pytest.raises(ValueError, match="read-only"):
        bumps(tack.arrays.View(readonly, 0, 5), tack.zeros(5, tack.i32))


def test_a_missing_protocol_method_is_named(generic):
    window_mean = generic.function("demo::window_mean", result=T,
                                   args=(indexable(T), tack.i64, tack.i64, T))

    @tack.device_class
    class NoLength:
        def __init__(self):
            self.k = 1.0

        @tack.func
        def __getitem__(self, k):
            return self.k

    @tack.kernel
    def smooth(c, out):
        for i in range(out.shape[0]):
            out[i] = window_mean(c, i, 2, out[i])

    with pytest.raises(Exception, match="has no @tack.func method '__len__'"):
        smooth(NoLength(), tack.zeros(4, tack.f64))


_CALLBACKS = """
namespace demo {
// The work of a vector field along a segment, by the trapezoid rule: the
// points are the C++'s, the field Tack's, taking and returning vectors.
template <class F, class V> V work(const F& f, V x1, V y1, int n) {
  V s = 0;
  for (int k = 0; k < n; ++k) {
    V t0 = V(k) / n, t1 = V(k + 1) / n;
    std::array<V, 2> a{t0 * x1, t0 * y1}, b{t1 * x1, t1 * y1};
    std::array<V, 2> fa = f.field(a), fb = f.field(b);
    s += V(0.5) * ((fa[0] + fb[0]) * (b[0] - a[0]) + (fa[1] + fb[1]) * (b[1] - a[1]));
  }
  return s;
}
// Tack fills a buffer the C++ owns, then reads it back.
template <class F, class V> V moments(const F& f, V x) {
  V buffer[4];
  f.fill(buffer, x);
  return f.weigh(buffer);
}
}
"""


@pytest.fixture(scope="module")
def callbacks():
    return tack.cpp.Library(source=_CALLBACKS)


@pytest.mark.parametrize("dtype", [tack.f32, tack.f64])
def test_methods_cpp_calls_take_and_return_vectors(callbacks, dtype):
    work = callbacks.function(
        "demo::work", result=T,
        args=(obj(field=(vec(T, 2), (vec(T, 2),))), T, T, tack.i32))

    @tack.device_class
    class Gradient:
        """The gradient of k x y: its work from the origin is k x1 y1."""

        def __init__(self, k):
            self.k = k

        @tack.func
        def field(self, p):
            return tack.vec2(self.k * p[1], self.k * p[0])

    @tack.kernel
    def works(f, ends, out):
        for i in range(out.shape[0]):
            out[i] = work(f, ends[i], 1.5, 16)

    ends = np.linspace(0.5, 2, 10).astype(dtype.numpy_dtype)
    out = tack.zeros(10, dtype)
    works(Gradient(3.0), tack.array(ends), out)
    np.testing.assert_allclose(out.numpy(), 3.0 * ends * 1.5,
                               rtol=50 * np.finfo(dtype.numpy_dtype).eps)


def test_methods_cpp_calls_take_pointers_to_its_memory(callbacks):
    moments = callbacks.function(
        "demo::moments", result=T,
        args=(obj(fill=(None, (ptr(T, writes=True), T)), weigh=(T, (ptr(T),))), T))

    @tack.device_class
    class Powers:
        def __init__(self, weights):
            self.w = weights

        @tack.func
        def fill(self, buffer, x):
            v = 1.0 + 0.0 * x
            for k in range(4):
                buffer[k] = v
                v = v * x

        @tack.func
        def weigh(self, buffer):
            s = 0.0 * buffer[0]
            for k in range(4):
                s += self.w[k] * buffer[k]
            return s

    @tack.kernel
    def polynomial(f, x, out):
        for i in range(out.shape[0]):
            out[i] = moments(f, x[i])

    x = np.linspace(-1, 1, 10)
    w = np.array([1.0, -2.0, 0.5, 3.0])
    out = tack.zeros(10, tack.f64)
    polynomial(Powers(tack.array(w)), tack.array(x), out)
    np.testing.assert_allclose(out.numpy(), np.polyval(w[::-1], x), rtol=1e-14)
    # Stores into C++ memory write none of the kernel's arrays: a read-only one works.
    polynomial(Powers(tack.array_from_ptr(w.ctypes.data, (4,), tack.f64)),
               tack.array(x), out)
    np.testing.assert_allclose(out.numpy(), np.polyval(w[::-1], x), rtol=1e-14)


def test_callback_declarations_are_held_to(callbacks):
    moments = callbacks.function(
        "demo::moments", result=T,
        args=(obj(fill=(None, (ptr(T), T)), weigh=(T, (ptr(T),))), T))
    work = callbacks.function(
        "demo::work", result=T,
        args=(obj(field=(vec(T, 3), (vec(T, 2),))), T, T, tack.i32))

    @tack.device_class
    class Writer:
        def __init__(self):
            self.k = 1.0

        @tack.func
        def fill(self, buffer, x):
            buffer[0] = x

        @tack.func
        def weigh(self, buffer):
            return buffer[0]

        @tack.func
        def field(self, p):
            return p

    @tack.kernel
    def stores(f, out):
        for i in range(out.shape[0]):
            out[i] = moments(f, 1.0)

    @tack.kernel
    def short(f, out):
        for i in range(out.shape[0]):
            out[i] = work(f, 1.0, 1.0, 4)

    out = tack.zeros(2, tack.f64)
    with pytest.raises(Exception, match="stores through a pointer its declaration makes read-only"):
        stores(Writer(), out)
    with pytest.raises(Exception, match="declared to return a vector of 3, which returns 2"):
        short(Writer(), out)
    with pytest.raises(TypeError, match="returns a scalar, a tack.cpp.vec or None"):
        callbacks.function("demo::work", result=T,
                           args=(obj(field=(ptr(T), (T,))), T, T, tack.i32))


def _eigen():
    """Eigen's include directory, if it is installed where it usually is."""
    import pathlib
    for folder in ("/opt/homebrew/include/eigen3", "/usr/include/eigen3",
                   "/usr/local/include/eigen3"):
        if pathlib.Path(folder, "Eigen", "Dense").is_file():
            return folder
    return None


@pytest.mark.skipif(_eigen() is None, reason="Eigen is not installed")
def test_a_real_library_eigen_through_pointers_in_two_kernels():
    """Eigen's closed-form eigenvalues of symmetric 3x3 matrices, mapped over
    Tack's arrays. The solver is too large to inline, so each kernel keeps its
    own internal copy: two kernels in one process must not share or clash."""
    lib = tack.cpp.Library(include_dirs=[_eigen()], flags=["-DNDEBUG"], source="""
#include <Eigen/Dense>
template <class V> void eigenvalues(const V* a, long long i, V* out) {
  Eigen::Map<const Eigen::Matrix<V, 3, 3, Eigen::RowMajor>> m(a + 9 * i);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<V, 3, 3>> solver;
  solver.computeDirect(m, Eigen::EigenvaluesOnly);
  Eigen::Map<Eigen::Matrix<V, 3, 1>>(out + 3 * i) = solver.eigenvalues();
}""")
    eigenvalues = lib.function("eigenvalues", result=None,
                               args=(ptr(T), tack.i64, ptr(T, writes=True)))

    @tack.kernel
    def spectrum(a, out):
        for i in range(out.shape[0]):
            eigenvalues(a, i, out)

    @tack.kernel
    def doubled(a, out):
        for i in range(out.shape[0]):
            eigenvalues(a, i, out)
            out[i] = out[i] * 2.0

    g = np.random.default_rng(1).standard_normal((300, 3, 3))
    sym = (g + g.transpose(0, 2, 1)) / 2
    a = tack.array(sym, dtype=tack.mat(3, 3, tack.f64))
    out, twice = (tack.zeros(300, tack.vec(3, tack.f64)) for _ in range(2))
    spectrum(a, out)
    doubled(a, twice)
    spectrum(a, out)
    want = np.linalg.eigvalsh(sym)
    np.testing.assert_allclose(out.numpy(), want, atol=1e-12)
    np.testing.assert_allclose(twice.numpy(), 2 * want, atol=1e-12)


def test_a_library_is_parsed_once_for_all_its_instantiations(monkeypatch):
    """The library's text goes into one precompiled header a process; each
    kernel's instantiation is compiled against it."""
    real = subprocess.run
    runs = []

    def run(command, **kwargs):
        runs.append("header" if "c++-header" in command else
                    "with header" if "-include-pch" in command else "whole")
        return real(command, **kwargs)

    monkeypatch.setattr(tack.cpp.subprocess, "run", run)
    lib = tack.cpp.Library(source="template <class V> V cube(V x) { return x * x * x; }")
    cube = lib.function("cube", result=T, args=(T,))

    @tack.kernel
    def cubes(x):
        for i in range(x.shape[0]):
            x[i] = cube(x[i])

    @tack.kernel
    def sixth_powers(x):
        for i in range(x.shape[0]):
            x[i] = cube(cube(x[i]) * 1.0) / cube(x[i])

    for dtype in (tack.f32, tack.f64):
        x = tack.full(4, 2.0, dtype)
        cubes(x)
        sixth_powers(x)
        assert x.numpy().tolist() == [8.0 ** 6] * 4     # (2^3)^9 / (2^3)^3
    # One header; the f32 and the f64 instantiation once each, shared by both kernels.
    assert runs == ["header", "with header", "with header"]


def test_an_error_in_the_librarys_own_text_is_clangs_message():
    """No precompiled header can be built, so the whole text is compiled and
    clang's error names the user's line."""
    broken = tack.cpp.Library(source="template <class V> V f(V x) { return x + ; }")
    f = broken.function("f", result=T, args=(T,))

    @tack.kernel
    def apply(x):
        for i in range(x.shape[0]):
            x[i] = f(x[i])

    with pytest.raises(Exception, match="expected expression"):
        apply(tack.zeros(4, tack.f32))


_SCRIPT = """
import sys, numpy as np, tack, tack.cpp
lib = tack.cpp.Library(headers=["demo.h"], include_dirs=[sys.argv[1]], flags=["-I" + sys.argv[1] + "/inner"])
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
    nothing; the contents of every header clang read are in the key, one the
    library names and one that header includes alike, and a header touched
    without change keeps the entry."""
    import os
    (tmp_path / "inner").mkdir()
    factor = tmp_path / "inner" / "factor.h"
    factor.write_text("#define FACTOR 1\n")
    header = tmp_path / "demo.h"
    header.write_text('#include "factor.h"\n'
                      "namespace demo { template <class V> V scale(V x) { return 2 * x; } }")
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
    code, out, err = run(TACK_CLANGXX="/no/such/clang++")
    assert code == 0, err
    assert out == ["2.0", "1"]
    os.utime(header)            # touched, not changed
    code, out, err = run(TACK_CLANGXX="/no/such/clang++")
    assert code == 0, err
    assert out == ["2.0", "1"]
    header.write_text('#include "factor.h"\n'
                      "namespace demo { template <class V> V scale(V x) "
                      "{ return 3 * FACTOR * x; } }")
    code, out, err = run(TACK_CLANGXX="/no/such/clang++")
    assert code != 0 and "needs clang" in err
    code, out, err = run()
    assert out == ["3.0", "0"]
    # A header the library never names, which the named one includes.
    factor.write_text("#define FACTOR 2\n")
    code, out, err = run()
    assert code == 0, err
    assert out == ["6.0", "0"]
    code, out, err = run(TACK_CLANGXX="/no/such/clang++")
    assert out == ["6.0", "1"]
