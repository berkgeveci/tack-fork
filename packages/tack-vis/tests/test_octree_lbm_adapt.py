# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive refinement and coarsening for the lattice Boltzmann prototype
(study): a remesh that changes nothing changes nothing, remeshes conserve mass
and momentum, and the meshes they make are balanced and layered."""

import numpy as np
import pytest

from tack.dataset import octree_lbm as lb
from tack.dataset.octree_lbm_adapt import adapt


@pytest.fixture(autouse=True)
def _device(f64_backend):
    """Every device with f64 (the comparisons are to 1e-13 and closer)."""
    yield


N = 32


def _taylor_green(c, u0=0.03):
    k = 2 * np.pi / N
    x, y = c[:, 0], c[:, 1]
    return (1 - 3 * u0 * u0 / 4 * (np.cos(2 * k * x) + np.cos(2 * k * y)),
            -u0 * np.cos(k * x) * np.sin(k * y), u0 * np.sin(k * x) * np.cos(k * y))


def _square(x, y, h):
    return abs(x - N / 2) < N / 4 and abs(y - N / 2) < N / 4


def _solver(refine=_square, finest=1):
    s = lb.LBM2D(lb.lbm_mesh((N, N), refine, finest), 0.8)
    c, _ = s.centres()
    s.initialize(*_taylor_green(c))
    return s


def _check_mesh(s):
    """2:1 across faces and corners, real leaves' neighbours at their level hold
    values, interface cells have four ghosts."""
    role, level = s.role, s.levels
    rows = s.rows.numpy().reshape(-1, 9)
    for n in np.flatnonzero(role == lb.REAL):
        for q in rows[n]:
            assert q >= 0 and level[q] == level[n] and role[q] not in (lb.REFINED,)
    number, table = (a.numpy() for a in (s.number, s.table))
    for n in np.flatnonzero(role == lb.INTERFACE):
        first = (number[n] - 1) * 4
        assert (role[table[first:first + 4]] == lb.GHOST).all()
    lo, size = (np.asarray(a.numpy()).reshape(-1, 3)[:, :2] for a in s.octree.geometry())
    leaves = np.flatnonzero(np.isin(role, (lb.REAL, lb.INTERFACE, lb.SOLID)))
    for a in leaves:
        near = np.all((lo[leaves] <= lo[a] + size[a] + 1e-9) &
                      (lo[leaves] + size[leaves] >= lo[a] - 1e-9), axis=1)
        assert np.abs(level[leaves[near]] - level[a]).max() <= 1


def test_a_remesh_that_changes_nothing_changes_nothing():
    s = _solver()
    s.step(5)
    t, stats = adapt(s, refine_above=1e9, coarsen_below=-1.0)
    assert stats["refined"] == stats["merged"] == 0
    assert np.array_equal(t.octree._key, s.octree._key)
    assert np.array_equal(t.role, s.role)
    keep = s.computational()
    assert np.array_equal(t.f.numpy().reshape(-1, 9)[keep], s.f.numpy().reshape(-1, 9)[keep])


def test_remeshes_conserve_mass_and_momentum_and_keep_the_mesh_layered():
    s = _solver()
    for _ in range(5):
        m0, p0 = s.mass(), np.array(s.momentum())
        s, stats = adapt(s, refine_above=0.004, coarsen_below=0.003, finest=2)
        assert abs(s.mass() - m0) < 1e-13 * m0
        assert np.abs(np.array(s.momentum()) - p0).max() < 1e-13
        _check_mesh(s)
        s.step(8)
    assert stats["cells"] != 1792


def test_balance_ripples_out_from_a_deep_refinement():
    # Pinning one corner of the domain to level 3 from a uniform level-0 mesh
    # takes three remeshes, each refining its ring of coarser neighbours.
    s = _solver(lambda x, y, h: False, 0)
    def corner(x, y):
        return (np.abs(x - 10.3) < 0.6) & (np.abs(y - 20.7) < 0.6)

    for _ in range(3):
        s, stats = adapt(s, refine_above=1e9, coarsen_below=-1.0, finest=3, pinned=corner)
        _check_mesh(s)
    assert s.octree.finest == 3
    assert stats["rounds"] > 1


def test_a_uniform_flow_stays_uniform_through_remeshes():
    s = lb.LBM2D(lb.lbm_mesh((N, N), _square, 1), 0.8)
    s.initialize(1.0, 0.05, -0.02)
    pinned = (lambda x, y: (np.abs(x - 8) < 3) & (np.abs(y - 8) < 3))
    for _ in range(3):
        s, _ = adapt(s, refine_above=1e9, coarsen_below=1.0, finest=2, pinned=pinned)
        s.step(4)
    rho, ux, uy = s.moments()
    keep = s.computational()
    assert np.abs(rho[keep] - 1).max() < 1e-13
    assert np.abs(ux[keep] - 0.05).max() < 1e-13
    assert np.abs(uy[keep] + 0.02).max() < 1e-13


def _velocity_error(s, nu):
    k = 2 * np.pi / N
    c, side = s.centres()
    _, ux, uy = s.moments()
    decay = np.exp(-2 * nu * k * k * s.steps)
    _, ex, ey = _taylor_green(c)
    ex, ey = ex * decay, ey * decay
    keep = s.computational()
    a = side[keep] ** 2
    return np.sqrt(np.sum(((ux - ex) ** 2 + (uy - ey) ** 2)[keep] * a) /
                   np.sum((ex ** 2 + ey ** 2)[keep] * a))


def test_a_round_trip_through_a_coarser_level_costs_little_that_lasts():
    # Merging a uniform mesh a level and refining it back leaves an error that
    # dies away: what lasts is a few percent of the mesh's own.
    nu = (0.8 - 0.5) / 3
    everything = (lambda x, y: np.ones_like(x, bool))

    def fresh():
        s = _solver(lambda x, y, h: True, 1)
        s.step(50)
        return s

    plain, trip = fresh(), fresh()
    trip, _ = adapt(trip, 1e9, 1e9, finest=1)
    assert trip.computational().sum() == N * N
    trip, _ = adapt(trip, 1e9, -1.0, finest=1, pinned=everything)
    assert trip.computational().sum() == 4 * N * N
    plain.step(150)
    trip.step(150)
    assert _velocity_error(trip, nu) < 1.1 * _velocity_error(plain, nu)
