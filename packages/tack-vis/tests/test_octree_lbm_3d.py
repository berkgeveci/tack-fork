# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The lattice Boltzmann prototype in 3D (D3Q27, study): a uniform mesh against
NumPy, two levels against the scheme's definition on a grid of fine cells,
conservation with three levels, walls and remeshes."""

import numpy as np
import pytest

from tack.dataset import octree_lbm as lb
from tack.dataset.octree_lbm_adapt import adapt


@pytest.fixture(autouse=True)
def _device(f64_backend):
    """Every device with f64 (the comparisons are to 1e-13 and closer)."""
    yield


_O = np.arange(27)
_C = np.stack([_O % 3 - 1, (_O // 3) % 3 - 1, _O // 9 - 1])
_W = np.array([8 / 27, 2 / 27, 1 / 54, 1 / 216])[np.abs(_C).sum(axis=0)]
N, TAU = 8, 0.8


def _feq(rho, u):
    cu = 3 * sum(_C[a][:, None, None, None] * u[a] for a in range(3))
    return _W[:, None, None, None] * rho * (1 + cu + 0.5 * cu * cu - 1.5 * sum(v * v for v in u))


def _collide(f, omega):
    rho = f.sum(0)
    u = [(f * _C[a][:, None, None, None]).sum(0) / rho for a in range(3)]
    return f + omega * (_feq(rho, u) - f)


def _stream(f):                                    # f[o, z, y, x]
    return np.stack([np.roll(np.roll(np.roll(f[k], _C[2, k], 0), _C[1, k], 1), _C[0, k], 2)
                     for k in range(27)])


def _field(c, u0=0.03):
    k = 2 * np.pi / N
    x, y, z = c.T
    return (1 + 0.01 * np.sin(k * z), -u0 * np.cos(k * x) * np.sin(k * y),
            u0 * np.sin(k * x) * np.cos(k * y), 0.01 * np.sin(k * x))


def _cube(x, y, z):
    return (np.abs(x - N / 2) < N / 4) & (np.abs(y - N / 2) < N / 4) & (np.abs(z - N / 2) < N / 4)


def _corner(x, y, z):
    return (np.abs(x - 3) < 1.2) & (np.abs(y - 4) < 1.2) & (np.abs(z - 5) < 1.2)


def _refined(pinned, finest, mesh=None, **options):
    s = lb.LBM(mesh if mesh is not None else lb.uniform_mesh((N, N, N)), TAU, **options)
    for _ in range(finest):
        s, _ = adapt(s, 1e9, -1.0, finest=finest, pinned=pinned)
    return s


def test_a_uniform_mesh_is_numpys_d3q27():
    s = lb.LBM(lb.uniform_mesh((N, N, N)), TAU)
    c, _ = s.centres()
    s.initialize(*_field(c))
    i, j, k = (np.floor(c[:, a]).astype(int) for a in range(3))
    f = np.zeros((27, N, N, N))
    f[:, k, j, i] = s.f.numpy().reshape(-1, 27).T
    for _ in range(15):
        f = _stream(_collide(f, 1 / TAU))
    s.step(15)
    np.testing.assert_allclose(s.f.numpy().reshape(-1, 27).T, f[:, k, j, i], rtol=0, atol=1e-15)


def test_two_levels_are_the_scheme_on_a_grid_of_fine_cells():
    s = _refined(_cube, 1, centred=False, linear=False)
    c, _ = s.centres()
    s.initialize(*_field(c))
    keep, level = s.computational(), s.levels
    coarse = np.flatnonzero(keep & (level == 0))
    fine = np.flatnonzero(keep & (level == 1))
    ci = [np.floor(c[coarse, a]).astype(int) for a in range(3)]
    fi = [np.floor(c[fine, a] * 2).astype(int) for a in range(3)]
    is_fine = np.zeros((2 * N,) * 3, bool)
    is_fine[fi[2], fi[1], fi[0]] = True
    f0 = s.f.numpy().reshape(-1, 27)
    big = np.ones((27, N, N, N)) / 27
    big[:, ci[2], ci[1], ci[0]] = f0[coarse].T
    small = np.ones((27,) + (2 * N,) * 3) / 27
    small[:, fi[2], fi[1], fi[0]] = f0[fine].T
    omega_c, omega_f = 1 / TAU, 1 / (0.5 + 2 * (TAU - 0.5))
    for _ in range(8):
        big = _collide(big, omega_c)
        spread = np.repeat(np.repeat(np.repeat(big, 2, 1), 2, 2), 2, 3)
        small = _stream(np.where(is_fine, _collide(small, omega_f), spread))
        small = _stream(np.where(is_fine, _collide(small, omega_f), small))
        big = small.reshape(27, N, 2, N, 2, N, 2).mean(axis=(2, 4, 6))
    s.step(8)
    f = s.f.numpy().reshape(-1, 27)
    np.testing.assert_allclose(f[coarse], big[:, ci[2], ci[1], ci[0]].T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(f[fine], small[:, fi[2], fi[1], fi[0]].T, rtol=0, atol=1e-14)


def test_three_levels_and_remeshes_conserve_mass_and_momentum():
    s = _refined(_corner, 2)
    assert s.octree.finest == 2
    c, _ = s.centres()
    s.initialize(*_field(c))
    m0, p0 = s.mass(), np.array(s.momentum())
    s.step(6)
    assert abs(s.mass() - m0) < 1e-13 * m0
    assert np.abs(np.array(s.momentum()) - p0).max() < 1e-13
    for _ in range(3):
        m0, p0 = s.mass(), np.array(s.momentum())
        s, _ = adapt(s, 0.003, 0.0015, finest=2)
        assert abs(s.mass() - m0) < 1e-13 * m0
        assert np.abs(np.array(s.momentum()) - p0).max() < 1e-13
        s.step(3)
    # 2:1 across faces, edges and corners.
    lo, size = (np.asarray(a.numpy()).reshape(-1, 3) for a in s.octree.geometry())
    leaves = np.flatnonzero(np.isin(s.role, (lb.REAL, lb.INTERFACE)))
    for a in leaves:
        near = np.all((lo[leaves] <= lo[a] + size[a] + 1e-9) &
                      (lo[leaves] + size[leaves] >= lo[a] - 1e-9), axis=1)
        assert np.abs(s.levels[leaves[near]] - s.levels[a]).max() <= 1


def test_walls_and_a_lid_conserve_mass_with_two_levels():
    mesh = lb.uniform_mesh((N, N, N), periodic=(True, False, False))
    s = _refined(lambda x, y, z: (y > 5) & (z > 4), 1, mesh=mesh, lid=0.05)
    c, _ = s.centres()
    s.initialize(*_field(c))
    m0 = s.mass()
    s.step(8)
    assert abs(s.mass() - m0) < 1e-13 * m0


def test_a_uniform_flow_stays_uniform_across_levels():
    s = _refined(_corner, 2)
    s.initialize(1.0, 0.05, -0.02, 0.03)
    s.step(5)
    rho, ux, uy, uz = s.moments()
    keep = s.computational()
    for v, expected in ((rho, 1.0), (ux, 0.05), (uy, -0.02), (uz, 0.03)):
        assert np.abs(v[keep] - expected).max() < 1e-13


def test_filling_ghosts_between_steps_changes_nothing_the_solver_computes():
    # A display fills the ghosts from their parents before every frame; the
    # solver must step exactly as if it had not.
    a, b = (_refined(_corner, 2, regularized=True) for _ in range(2))
    for s in (a, b):
        c, _ = s.centres()
        s.initialize(*_field(c))
    for _ in range(4):
        a.step(2)
        b.fill_ghosts()
        b.step(2)
    keep = a.computational()
    fa, fb = (s.f.numpy().reshape(-1, 27)[keep] for s in (a, b))
    assert np.array_equal(fa, fb)
    # The ghosts now hold their parents' values, linear in the parents' gradients:
    # a uniform flow's ghosts are the flow.
    s = _refined(_corner, 2)
    s.initialize(1.0, 0.05, -0.02, 0.03)
    s.step(3)
    s.fill_ghosts()
    rho, ux, uy, uz = s.moments()
    ghosts = s.role == lb.GHOST
    assert ghosts.any()
    for v, expected in ((rho, 1.0), (ux, 0.05), (uy, -0.02), (uz, 0.03)):
        assert np.abs(v[ghosts] - expected).max() < 1e-13
