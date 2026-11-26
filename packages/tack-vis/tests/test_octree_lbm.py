# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The D2Q9 lattice Boltzmann prototype on a refined quadtree (study): a uniform
mesh against NumPy, the two-level scheme against its definition on a grid of
fine cells, conservation, and accuracy against a Taylor-Green vortex."""

import numpy as np
import pytest

import tack
from tack.dataset import octree_lbm as lb


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


_O = np.arange(9)
_CX, _CY = _O % 3 - 1, _O // 3 - 1
_W = np.where((_CX == 0) & (_CY == 0), 4 / 9, np.where((_CX == 0) | (_CY == 0), 1 / 9, 1 / 36))


def _feq(rho, ux, uy):
    cu = 3 * (_CX[:, None, None] * ux + _CY[:, None, None] * uy)
    return _W[:, None, None] * rho * (1 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy))


def _collide(f, omega):
    rho = f.sum(0)
    ux = (f * _CX[:, None, None]).sum(0) / rho
    uy = (f * _CY[:, None, None]).sum(0) / rho
    return f + omega * (_feq(rho, ux, uy) - f)


def _stream(f):
    return np.stack([np.roll(np.roll(f[q], _CY[q], 0), _CX[q], 1) for q in range(9)])


def _taylor_green(c, n, nu, t=0.0, u0=0.03):
    k = 2 * np.pi / n
    x, y = c[:, 0], c[:, 1]
    d = np.exp(-2 * nu * k * k * t)
    return (1 - 3 * u0 * u0 / 4 * (np.cos(2 * k * x) + np.cos(2 * k * y)) * d * d,
            -u0 * np.cos(k * x) * np.sin(k * y) * d, u0 * np.sin(k * x) * np.cos(k * y) * d)


def _square(n):
    return lambda x, y, h: abs(x - n / 2) < n / 4 and abs(y - n / 2) < n / 4


def _two_squares(n):
    return lambda x, y, h: (abs(x - n / 2) < n / 4 and abs(y - n / 2) < n / 4) if h > 0.75 \
        else (abs(x - n / 2) < n / 8 and abs(y - n / 2) < n / 8)


def test_the_mesh_is_balanced_and_layered():
    n = 16
    octree = lb.lbm_mesh((n, n), _two_squares(n), 2)
    s = lb.LBM2D(octree, 0.8)
    role, level = s.role, s.levels
    rows = s.rows.numpy().reshape(-1, 9)
    leaf = octree.is_leaf
    # A real leaf's neighbours at its level are nodes there that hold values.
    for nd in np.flatnonzero(role == lb.REAL):
        for q in rows[nd]:
            assert q >= 0 and level[q] == level[nd] and role[q] != lb.REFINED
    # Interface cells have four ghosts; no leaf is beside one two levels finer.
    number, table = (a.numpy() for a in (s.number, s.table))
    for nd in np.flatnonzero(role == lb.INTERFACE):
        assert (role[table[(number[nd] - 1) * 4:(number[nd] - 1) * 4 + 4]] == lb.GHOST).all()
    lo, size = (np.asarray(a.numpy()).reshape(-1, 3)[:, :2] for a in octree.geometry())
    leaves = np.flatnonzero(leaf)
    for a in leaves:
        near = np.all((lo[leaves] <= lo[a] + size[a] + 1e-9) &
                      (lo[leaves] + size[leaves] >= lo[a] - 1e-9), axis=1)
        assert np.abs(level[leaves[near]] - level[a]).max() <= 1


def test_a_uniform_mesh_is_numpys_d2q9():
    n, tau = 16, 0.8
    s = lb.LBM2D(lb.lbm_mesh((n, n), lambda x, y, h: False, 0), tau)
    c, _ = s.centres()
    s.initialize(*_taylor_green(c, n, 0.1))
    i, j = np.rint(c[:, 0] - 0.5).astype(int), np.rint(c[:, 1] - 0.5).astype(int)
    f = np.zeros((9, n, n))
    f[:, j, i] = s.f.numpy().reshape(-1, 9).T
    for _ in range(30):
        f = _stream(_collide(f, 1 / tau))
    s.step(30)
    np.testing.assert_allclose(s.f.numpy().reshape(-1, 9).T, f[:, j, i], rtol=0, atol=1e-15)


def test_two_levels_are_the_scheme_on_a_grid_of_fine_cells():
    # Without the corrections, the octree's run is the lattice Boltzmann method
    # on fine cells, each coarse cell uniform at the start of its step and its
    # mean at the end.
    n, tau = 16, 0.8
    s = lb.LBM2D(lb.lbm_mesh((n, n), _square(n), 1), tau, centred=False, linear=False)
    c, _ = s.centres()
    s.initialize(*_taylor_green(c, n, 0.1))
    keep, level = s.computational(), s.levels
    coarse = np.flatnonzero(keep & (level == 0))
    fine = np.flatnonzero(keep & (level == 1))
    ci, cj = (np.floor(c[coarse, a]).astype(int) for a in (0, 1))
    fi, fj = (np.floor(c[fine, a] * 2).astype(int) for a in (0, 1))
    is_fine = np.zeros((2 * n, 2 * n), bool)
    is_fine[fj, fi] = True
    f0 = s.f.numpy().reshape(-1, 9)
    big = np.ones((9, n, n)) / 9                  # coarse values over the fine cells: unused
    big[:, cj, ci] = f0[coarse].T
    small = np.ones((9, 2 * n, 2 * n)) / 9
    small[:, fj, fi] = f0[fine].T
    omega_c, omega_f = 1 / tau, 1 / (0.5 + 2 * (tau - 0.5))
    for _ in range(20):
        big = _collide(big, omega_c)
        small = np.where(is_fine, _collide(small, omega_f), np.repeat(np.repeat(big, 2, 1), 2, 2))
        small = _stream(small)
        small = _stream(np.where(is_fine, _collide(small, omega_f), small))
        big = small.reshape(9, n, 2, n, 2).mean(axis=(2, 4))
    s.step(20)
    f = s.f.numpy().reshape(-1, 9)
    np.testing.assert_allclose(f[coarse], big[:, cj, ci].T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(f[fine], small[:, fj, fi].T, rtol=0, atol=1e-14)


@pytest.mark.parametrize("finest", [1, 2])
def test_mass_and_momentum_are_conserved(finest):
    n = 16
    s = lb.LBM2D(lb.lbm_mesh((n, n), _two_squares(n), finest), 0.7)
    c, _ = s.centres()
    s.initialize(*_taylor_green(c, n, 0.1))
    m0, p0 = s.mass(), np.array(s.momentum())
    s.step(60)
    assert abs(s.mass() - m0) < 1e-12 * m0
    assert np.abs(np.array(s.momentum()) - p0).max() < 1e-12


@pytest.mark.parametrize("periodic", [(False, False), (True, False), (False, True)])
def test_walls_and_a_lid_conserve_mass_with_two_levels(periodic):
    n = 16
    s = lb.LBM2D(lb.lbm_mesh((n, n), lambda x, y, h: y > 12, 1, periodic=periodic), 0.7,
                 lid=0.05)
    c, _ = s.centres()
    s.initialize(1 + 0.01 * np.sin(c[:, 0]), 0.02 * np.cos(0.3 * c[:, 1]),
                 0.01 * np.sin(0.4 * c[:, 0]))
    m0 = s.mass()
    s.step(50)
    assert abs(s.mass() - m0) < 1e-12 * m0


def test_a_uniform_flow_stays_uniform_across_levels():
    n = 16
    s = lb.LBM2D(lb.lbm_mesh((n, n), _two_squares(n), 2), 0.8)
    s.initialize(1.0, 0.05, -0.02)
    s.step(40)
    rho, ux, uy = s.moments()
    keep = s.computational()
    assert np.abs(rho[keep] - 1).max() < 1e-13
    assert np.abs(ux[keep] - 0.05).max() < 1e-13
    assert np.abs(uy[keep] + 0.02).max() < 1e-13


def _error(refine, finest, n=32, tau=0.8, steps=200, **options):
    nu = (tau - 0.5) / 3
    s = lb.LBM2D(lb.lbm_mesh((n, n), refine, finest), tau, **options)
    c, side = s.centres()
    s.initialize(*_taylor_green(c, n, nu))
    s.step(steps)
    _, ux, uy = s.moments()
    _, ex, ey = _taylor_green(c, n, nu, steps)
    keep = s.computational()
    a = side[keep] ** 2
    return np.sqrt(np.sum(((ux - ex) ** 2 + (uy - ey) ** 2)[keep] * a) /
                   np.sum((ex ** 2 + ey ** 2)[keep] * a))


def test_refining_is_more_accurate_than_the_coarse_mesh():
    # While the vortex holds most of its energy: the staggered mode the
    # corrections leave is not damped, so once the flow has decayed a
    # hundredfold it is most of the relative error.
    coarse = _error(lambda x, y, h: False, 0)
    assert _error(_square(32), 1) < coarse
    assert _error(lambda x, y, h: not _square(32)(x, y, h), 1) < coarse
    # Without either correction the interfaces lose to the coarse mesh.
    assert _error(_square(32), 1, centred=False) > 2 * coarse
    assert _error(_square(32), 1, linear=False) > 2 * coarse
