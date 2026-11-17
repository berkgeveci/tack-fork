# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``L2`` with an order per cell (p-adaptive), orders 0 to 5 in one field: each cell
read as the space of its order is, in one basis, its values in blocks; and the
filters over such a field (#423)."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _quadratic(x):
    return 1.0 + x[0] * x[0] - 0.5 * x[1] * x[2] + 0.25 * x[2]


def _quadratic_at(x):
    return 1.0 + x[:, 0] ** 2 - 0.5 * x[:, 1] * x[:, 2] + 0.25 * x[:, 2]


def _quadratic_gradient(x):
    return np.stack([2 * x[:, 0], -0.5 * x[:, 2], -0.5 * x[:, 1] + 0.25], axis=1)


@tack.func
def _ball(x):
    return ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)
            + (x[2] - 0.5) * (x[2] - 0.5))


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _hexes():
    grid = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), dtype=tack.f64)
    hexes = td.extract_cells(grid, np.arange(grid.num_cells))
    return td.DataSet(hexes.topology, hexes.geometry)


def _wedges():
    hexes = _hexes()
    conn = []
    for h in hexes.topology.connectivity.numpy().reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = td.topology.uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    topology = td.UnstructuredTopology(types, offsets,
                                       tack.array(np.array(conn, np.int32).reshape(-1)),
                                       num_points=hexes.num_points)
    return td.DataSet(topology, hexes.positions())


def _mesh(cells, curved=False):
    data = {"hexahedra": _hexes, "tetrahedra": lambda: td.tetrahedralize(_hexes()),
            "wedges": _wedges}[cells]()
    data = td.DataSet(data.topology, data.geometry)
    if curved:
        data.geometry = td.interpolate(data, _bend, td.H1(data, order=2),
                                       dtype=tack.vec3(tack.f64))
    return data


def _orders(data, low=2, high=5, seed=0):
    return np.random.default_rng(seed).integers(low, high + 1, data.num_cells)


_BASES = [("hexahedra", "gauss_lobatto"), ("hexahedra", "gauss_legendre"),
          ("hexahedra", "equispaced"), ("hexahedra", "bernstein"), ("tetrahedra", "equispaced"),
          ("tetrahedra", "bernstein"), ("wedges", "equispaced")]


@pytest.mark.parametrize("cells, basis", _BASES)
def test_each_cell_holds_the_polynomials_of_its_order(cells, basis):
    """Orders 2 to 5 in one field: each cell's values are its order's nodes, in
    blocks, and a quadratic is itself everywhere, read through each order's view."""
    data = _mesh(cells)
    orders = _orders(data)
    space = td.L2(data, order=orders, basis=basis)
    assert space.varies and space.orders() == [2, 3, 4, 5] and space.max_order == 5
    nodes = {"hexahedra": lambda p: (p + 1) ** 3,
             "tetrahedra": lambda p: (p + 1) * (p + 2) * (p + 3) // 6,
             "wedges": lambda p: (p + 1) * (p + 2) // 2 * (p + 1)}[cells]
    offsets = space.offsets.numpy()
    np.testing.assert_array_equal(np.diff(offsets), [nodes(p) for p in orders])
    field = td.interpolate(data, _quadratic, space, dtype=tack.f64)
    for c in (0, 7, data.num_cells - 1):
        fixed = td.interpolate(data, _quadratic, space.at_order(orders[c]), dtype=tack.f64)
        first = fixed.space.offsets.numpy()[c]
        np.testing.assert_allclose(field.values.numpy()[offsets[c]:offsets[c + 1]],
                                   fixed.values.numpy()[first:first + nodes(orders[c])],
                                   rtol=0, atol=1e-14)
    points = np.random.default_rng(1).random((300, 3)) * 0.98 + 0.01
    data.fields["w"] = field
    probed = td.probe(data, points).fields["w"].values.numpy()
    # The locator finds a point in a wedge to about 1e-7 at any order.
    np.testing.assert_allclose(probed, _quadratic_at(points), rtol=0,
                               atol=1e-5 if cells == "wedges" else 1e-12)


def test_orders_0_to_5_side_by_side():
    """Order 0 is the cell's centre value, order 1 its corners (a closed basis), the
    others their nodes; Gauss-Legendre's order 1 is its 2^d interior nodes."""
    data = _mesh("hexahedra")
    orders = np.arange(data.num_cells) % 6
    space = td.L2(data, order=orders)
    assert space.orders() == [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(np.diff(space.offsets.numpy()),
                                  [1 if p == 0 else 8 if p == 1 else (p + 1) ** 3 for p in orders])
    legendre = td.L2(data, order=orders, basis="gauss_legendre")
    np.testing.assert_array_equal(np.diff(legendre.offsets.numpy()),
                                  [1 if p == 0 else (p + 1) ** 3 for p in orders])
    for s in (space, legendre):
        field = td.interpolate(data, _quadratic, s, dtype=tack.f64)
        data.fields["w"] = field
        centers = td.cell_centers(data).values.numpy()
        means = td.values_at_centers(data, "w").values.numpy()
        high = orders >= 2
        np.testing.assert_allclose(means[high], _quadratic_at(centers)[high], rtol=0,
                                   atol=1e-12)


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra"])
@pytest.mark.parametrize("curved", [False, True])
def test_integrals_and_gradients(cells, curved):
    """integrate takes points enough for the highest order; the gradient at an L2
    space's nodes of an order per cell is the quadratic's, exactly. (On curved cells
    a quadratic in space is of degree 4 in a cell's coordinates: orders 4 and 5.)"""
    data = _mesh(cells, curved)
    low = 4 if curved else 2
    space = td.L2(data, order=_orders(data, low))
    data.fields["w"] = td.interpolate(data, _quadratic, space, dtype=tack.f64)
    # The same quadratic at one order, the highest: integrated at the same points.
    data.fields["h"] = td.interpolate(data, _quadratic, space.at_order(5), dtype=tack.f64)
    got = td.integrate(data, fields=["w", "h"])
    np.testing.assert_allclose(got.values["w"], got.values["h"], rtol=1e-13)
    at = td.L2(data, order=_orders(data, low, seed=4), basis=space.basis)
    gradient = td.gradients(data, "w", at=at).values.numpy()
    positions = _node_positions(data, at)
    np.testing.assert_allclose(gradient, _quadratic_gradient(positions), rtol=0, atol=1e-10)


@tack.func
def _identity(x):
    return x


def _node_positions(data, space):
    return td.interpolate(data, _identity, space, dtype=tack.vec3(tack.f64)).values.numpy()


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra"])
@pytest.mark.parametrize("curved", [False, True])
def test_cuts(cells, curved):
    """contour, clip and threshold subdivide to the highest order, and the
    crossings are the field's roots, in each cell's own order: a sphere's
    contour lies on it; clip's two sides and threshold's parts fill the mesh."""
    data = _mesh(cells, curved)
    data.fields["w"] = td.interpolate(data, _ball, td.L2(data, order=_orders(data)),
                                      dtype=tack.f64)
    level = 0.09
    surface = td.contour(data, "w", [level])
    assert surface.num_cells > 50 and "w" in surface.fields
    r = surface.positions() - 0.5
    if not curved:
        np.testing.assert_allclose((r * r).sum(axis=1), level, rtol=0, atol=1e-12)
    np.testing.assert_allclose(surface.fields["w"].values.numpy(), level, rtol=0, atol=1e-12)
    inside = sum(td.integrate(p).measure for p in td.clip(data, "w", level, inside=True))
    outside = sum(td.integrate(p).measure for p in td.clip(data, "w", level))
    total = td.integrate(data).measure
    # Curved cells' cut parts are straight sub-cells: their volume is near the cells'.
    np.testing.assert_allclose(inside + outside, total, rtol=1e-5 if curved else 1e-12)
    if not curved:
        assert abs(inside - 4 / 3 * np.pi * 0.3 ** 3) < 0.01
    whole, part = td.threshold(data, "w", -1.0, 0.2)
    assert whole.num_cells and part.num_cells
    assert whole.fields["w"].space.varies           # whole cells keep their orders


def test_carry_and_project():
    """extract_cells keeps each cell's order and the basis; project into an order
    per cell puts each cell's nodes at its order; jump of a continuous field is zero
    at every order."""
    data = _mesh("hexahedra", curved=True)
    space = td.L2(data, order=_orders(data, 0, 5), basis="gauss_legendre")
    data.fields["w"] = td.interpolate(data, _quadratic, space, dtype=tack.f64)
    kept = td.extract_cells(data, np.arange(3, data.num_cells, 4))
    out = kept.fields["w"].space
    assert out.varies and out.basis == "gauss_legendre"
    np.testing.assert_array_equal(out.cell_keys().numpy(), space.cell_keys().numpy()[3::4])
    np.testing.assert_allclose(
        kept.fields["w"].values.numpy(),
        td.interpolate(kept, _quadratic, out, dtype=tack.f64).values.numpy(), rtol=0,
        atol=1e-13)
    # On the bent cells a quadratic in space is of degree 4 in a cell's coordinates.
    data.fields["h"] = td.interpolate(data, _quadratic, td.H1(data, order=4), dtype=tack.f64)
    target = td.L2(data, order=_orders(data, 4, 5, seed=9))
    projected = td.project(data, "h", target)
    np.testing.assert_allclose(projected.values.numpy(),
                               td.interpolate(data, _quadratic, target,
                                              dtype=tack.f64).values.numpy(),
                               rtol=0, atol=1e-12)
    high = td.L2(data, order=_orders(data, 4, 5, seed=2))
    data.fields["v"] = td.interpolate(data, _quadratic, high, dtype=tack.f64)
    np.testing.assert_allclose(td.jump(data, "v").values.numpy(), 0.0, rtol=0, atol=1e-12)


@tack.func
def _linear(x):
    return 1.0 + x[0] - 0.5 * x[1] + 0.25 * x[2]


def test_bernstein_orders_in_a_tack_array():
    """Orders given as a Tack array, not NumPy, reach the Bernstein conversion one
    subgroup at a time, through interpolate and project (orders 0 and 1 copied, the
    others converted), as NumPy orders do; orders 1 and up hold a linear function
    exactly."""
    data = _mesh("hexahedra")
    data.fields["h"] = td.interpolate(data, _linear, td.H1(data), dtype=tack.f64)
    points = np.random.default_rng(3).random((300, 3)) * 0.98 + 0.01
    exact = 1.0 + points[:, 0] - 0.5 * points[:, 1] + 0.25 * points[:, 2]
    for low, seed in ((0, 4), (1, 5)):
        host = _orders(data, low, 3, seed=seed).astype(np.int32)
        space = td.L2(data, order=tack.array(host), basis="bernstein")
        numpy_space = td.L2(data, order=host, basis="bernstein")
        assert space.orders() == list(range(low, 4))
        field = td.interpolate(data, _linear, space, dtype=tack.f64)
        np.testing.assert_array_equal(
            field.values.numpy(),
            td.interpolate(data, _linear, numpy_space, dtype=tack.f64).values.numpy())
        np.testing.assert_allclose(td.project(data, "h", space).values.numpy(),
                                   td.project(data, "h", numpy_space).values.numpy(),
                                   rtol=0, atol=1e-14)
        if low == 1:
            data.fields["w"] = field
            np.testing.assert_allclose(td.probe(data, points).fields["w"].values.numpy(),
                                       exact, rtol=0, atol=1e-12)
            np.testing.assert_allclose(td.project(data, "h", space).values.numpy(),
                                       field.values.numpy(), rtol=0, atol=1e-12)


def test_subdivide_evaluates_each_cell_at_its_order():
    data = _mesh("tetrahedra")
    data.fields["w"] = td.interpolate(data, _quadratic, td.L2(data, order=_orders(data)),
                                      dtype=tack.f64)
    refined = td.subdivide(data, 3)
    assert "w" in refined.fields
    np.testing.assert_allclose(refined.fields["w"].values.numpy(),
                               _quadratic_at(refined.positions()), rtol=0, atol=1e-12)


def test_what_an_order_per_cell_refuses():
    data = _mesh("tetrahedra")
    with pytest.raises(ValueError, match="0 to 15"):
        _ = td.L2(data, order=np.full(data.num_cells, 16)).size
    with pytest.raises(ValueError, match=r"whole number from 0 to 15, not \[2\.5\]"):
        td.L2(data, order=np.full(data.num_cells, 2.5)).cell_keys()
    with pytest.raises(ValueError, match="whole number"):
        td.L2(data, order=np.full(data.num_cells, np.nan)).cell_keys()
    assert td.L2(data, order=np.full(data.num_cells, 3.0),
                 basis="equispaced").orders() == [3]
    with pytest.raises(NotImplementedError, match="equispaced basis only"):
        _ = td.L2(data, order=np.full(data.num_cells, 3), basis="gauss_lobatto").size
    with pytest.raises(ValueError, match="blocks"):
        td.L2(data, order=np.full(data.num_cells, 3), layout="cells")
    with pytest.raises(ValueError, match="basis is one of"):
        td.L2(data, order=np.full(data.num_cells, 2), basis="serendipity")
