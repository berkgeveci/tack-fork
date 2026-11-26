# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A D2Q9 lattice Boltzmann solver on a refined quadtree (study, prototype): BGK
collisions, pull streaming through the octree's neighbourhood rows, and a
mass-conserving volumetric refinement (after Rohde et al. 2006), on a fixed
mesh.

**The mesh.** Cells are an octree's leaves, cell-centred, 2:1 balanced across
faces and corners. A leaf with a refined node among its 3x3 neighbours at its
level is an *interface* cell: it is refined in the octree, its four children
*ghosts*. Every node has a role (``REAL``, ``INTERFACE``, ``GHOST``,
``REFINED``); real leaves and interface cells are the computational cells,
ghosts are scratch, refined nodes hold nothing. ``lbm_mesh`` makes one from a
refinement predicate (on the host: a prototype's mesh).

**Time.** Acoustic scaling: a level's cells are half the size and step twice as
often as its parent level's, so the lattice velocities are the same at every
level and ``tau`` grows, ``tau_l = 1/2 + 2^l (tau_0 - 1/2)``, for one viscosity.
A level's step (``_step``): collide its computational cells; explode its
interface cells' values into their ghosts; two steps of the next level; stream
-- real leaves and ghosts pull from the node upstream at their level, interface
cells take the mean of their ghosts.

**Why ghosts conserve mass.** The scheme is the lattice Boltzmann method on a
virtual grid of the next level's cells, where each coarse cell is set at the
start of its step and its mean at the end. A real leaf's upstream neighbour at
its level always exists (balance and the interface layer see to it); a
ghost's may lie in a coarser cell, whose virtual value is the cell's own at
the first substep and, at the second, that of the cell two fine steps
upstream, which is the parent's upstream neighbour (``_stream``). The
octree's run matches that virtual grid to rounding, and mass and momentum are
conserved to rounding (``test_octree_lbm.py``).

**What makes it accurate.** Uniform coarse values in the ghosts (Rohde's
explosion) leave an error at the interfaces; two corrections, neither of which
moves mass between cells, remove most of it:

- *Linear explosion.* A coarse cell's virtual values are linear in its gradient
  (``_gradients``, central differences at its level); the four quarters'
  shares cancel, direction by direction.
- *Time-centred explosion.* A packet reaches the fine cells one or two
  substeps after the coarse collision, so its non-equilibrium part is half a
  fine step's stress off, one way or the other, every other substep. That
  drives D2Q9's staggered momentum mode (``sum (-1)^x j_x`` flips sign each
  step whatever the collision, so nothing damps it), a checkerboard of 1-2 % of
  the flow speed in a Taylor-Green vortex. A ghost on the downstream side of
  a direction gains half a fine step's stress, one on the upstream side
  loses it (``_collide``'s ``factor``, ``_explode``): 40 times less, and a refined mesh
  more accurate than the uniform coarse one. What is left of the mode is not
  damped either: once a decaying vortex has lost a hundredfold, it is most of
  the relative error. A filter of the staggered momentum would take it out.

Dupuis and Chopard's rescaling of the non-equilibrium parts does not fit: a
coarse cell sends some directions to coarse neighbours and others to fine
ones, and rescaled and unscaled shares do not add up to its mass.

**Walls.** Halfway bounce-back, a moving lid on the top wall (``lid``). A
diagonal retraces its line at a wall, so on the virtual grid part of a coarse
cell's bounced packet lands in its neighbour along the wall; ghosts see that,
and the plain rule at the coarse level would count those packets twice. A
level with finer ones bounces a diagonal as its four quarters would over two
steps of the next level (``_wall``). Two levels conserve mass to rounding;
where a wall crosses two interface layers (three levels or more) about
1e-9 of the mass is lost a step, since the rule would have to recurse
through the levels. Periodic axes wrap the roots' rows.

**Obstacles and open sides.** ``SOLID`` leaves (``lbm_mesh(solid=)``, at the
finest level) bounce packets back at their faces, a staircase. With
``inflow``, the left side is an equilibrium inlet and the right side a
zero-gradient outlet; keep interfaces off both, which the wall rule does not
cover.

Not here yet: refining and coarsening during the solve, the device's side of
the mesh (``lbm_mesh`` is NumPy and sets), 3D.
"""

import numpy as np

import tack
from tack.algorithms import argsort
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet, _compute_type
from tack.dataset.octree_topology import (
    _child_neighbourhoods,
    _root_neighbourhoods,
    children,
    grid_rows,
    refined_by_level,
)

__all__ = ["GHOST", "INTERFACE", "LBM", "LBM2D", "REAL", "REFINED", "SOLID", "lbm_mesh",
           "uniform_mesh"]

REAL, INTERFACE, GHOST, REFINED, SOLID = 0, 1, 2, 3, 4
_AROUND = [(a, b) for b in (-1, 0, 1) for a in (-1, 0, 1) if (a, b) != (0, 0)]


# -- the mesh ---------------------------------------------------------------


def _morton(i, j, level):
    key = np.zeros(len(i), np.uint64)
    for b in range(int(level.max()) if len(level) else 0):
        key |= (((i >> b) & 1) << (2 * b)).astype(np.uint64)
        key |= (((j >> b) & 1) << (2 * b + 1)).astype(np.uint64)
    return key


def lbm_mesh(shape, refine, finest, periodic=(True, True), spacing=1.0, solid=None,
             device=None):
    """A quadtree for ``LBM2D``: ``shape`` root cells of side ``spacing``, each leaf
    at level ``l < finest`` split while ``refine(x, y, side)`` (its centre and
    side) says so, then 2:1 balanced across faces and corners (``periodic``
    axes wrapping) and given its interface layer. Leaves whose centre
    ``solid(x, y)`` holds are ``SOLID`` (an obstacle, staircased): they must be
    at the finest level, and so must the cells around them. The nodes' roles
    are the field ``"role"``."""
    nx, ny = shape
    leaves = [set() for _ in range(finest + 2)]
    leaves[0] = {(i, j) for j in range(ny) for i in range(nx)}
    for level in range(finest):
        h = spacing / (1 << level)
        split = {c for c in leaves[level] if refine((c[0] + 0.5) * h, (c[1] + 0.5) * h, h)}
        leaves[level] -= split
        leaves[level + 1] |= {(2 * i + a, 2 * j + b) for i, j in split for a in (0, 1)
                              for b in (0, 1)}

    def wrapped(level, i, j):
        size = (nx << level, ny << level)
        out = []
        for v, n, p in ((i, size[0], periodic[0]), (j, size[1], periodic[1])):
            if p:
                v %= n
            elif not 0 <= v < n:
                return None
            out.append(v)
        return tuple(out)

    def holder(level, i, j):
        for m in range(level, -1, -1):
            c = (i >> (level - m), j >> (level - m))
            if c in leaves[m]:
                return m, c
        return None                                   # refined at this level

    # 2:1 balance: no leaf beside one more than a level finer.
    changed = True
    while changed:
        changed = False
        for level in range(finest, 1, -1):
            for i, j in list(leaves[level]):
                for a, b in _AROUND:
                    q = wrapped(level, i + a, j + b)
                    h = None if q is None else holder(level, *q)
                    if h is not None and h[0] < level - 1 and h[1] in leaves[h[0]]:
                        m, (p, r) = h
                        leaves[m].remove((p, r))
                        leaves[m + 1] |= {(2 * p + s, 2 * r + t) for s in (0, 1) for t in (0, 1)}
                        changed = True
    refined = [set() for _ in range(finest + 2)]
    for level in range(finest, 0, -1):
        refined[level - 1] = {(i >> 1, j >> 1) for i, j in leaves[level] | refined[level]}
    interface = [set() for _ in range(finest + 2)]
    for level in range(finest):
        for i, j in leaves[level]:
            if any((q := wrapped(level, i + a, j + b)) is not None and q in refined[level]
                   for a, b in _AROUND):
                interface[level].add((i, j))
    cells = []                                        # (level, i, j, role)
    for level in range(finest + 1):
        cells += [(level, i, j, REAL) for i, j in leaves[level] - interface[level]]
        cells += [(level, i, j, INTERFACE) for i, j in interface[level]]
        cells += [(level, i, j, REFINED) for i, j in refined[level]]
        if level:
            cells += [(level, 2 * i + s, 2 * j + t, GHOST) for i, j in interface[level - 1]
                      for s in (0, 1) for t in (0, 1)]
    level, i, j, role = (np.array(v, np.int64) for v in zip(*cells))
    if solid is not None:
        h = spacing / (1 << level)
        inside = np.array([bool(solid(x, y)) for x, y in zip((i + 0.5) * h, (j + 0.5) * h)])
        inside &= role != REFINED                     # refined nodes hold nothing
        if (inside & ((role != REAL) | (level != finest))).any():
            raise ValueError("solid cells must be real leaves at the finest level: refine "
                             "around the obstacle")
        role[inside] = SOLID
    tree = (i >> level) + nx * (j >> level)
    mask = (np.int64(1) << level) - 1
    key = _morton(i & mask, j & mask, level)
    from tack.dataset.dataset import uniform_grid

    roots = uniform_grid((nx + 1, ny + 1), spacing=(spacing, spacing), device=device)
    octree = OctreeDataSet(roots, tree, level, key, fields={"role": role.astype(np.int32)})
    octree.periodic = tuple(periodic)
    return octree


def uniform_mesh(shape, periodic=None, spacing=1.0, device=None):
    """An octree of root cells only, ``shape`` (two or three sizes) of side
    ``spacing``, every one a real leaf: where a 3D mesh starts, refined by
    ``octree_lbm_adapt.adapt`` (``pinned`` for the geometry)."""
    from tack.dataset.dataset import uniform_grid

    shape = tuple(int(s) for s in shape)
    d = len(shape)
    periodic = tuple(periodic) if periodic is not None else (True,) * d
    n = int(np.prod(shape))
    roots = uniform_grid(tuple(s + 1 for s in shape), spacing=(spacing,) * d, device=device)
    octree = OctreeDataSet(roots, np.arange(n), np.zeros(n, np.int64), np.zeros(n, np.uint64),
                           fields={"role": np.zeros(n, np.int32)}, _ordered=True)
    octree.periodic = periodic
    return octree


def _rows(octree, periodic):
    """Every node's 3^d neighbourhood at its level, as ``neighbourhoods`` makes
    it, with the roots' rows wrapping along ``periodic`` axes."""
    d = octree.dimension
    span = 3 ** d
    rows = grid_rows(octree, periodic)
    number, table = children(octree)
    with octree.device:
        out = tack.empty((octree.num_nodes * span,), tack.i32)
        _root_neighbourhoods(rows, len(rows) // span, span, out)
        for nodes in reversed(refined_by_level(octree)):
            _child_neighbourhoods(nodes, number, table, octree.marks, int(Ghosts.REFINEDCELL), d,
                                  span, out)
    return out


# -- the kernels ------------------------------------------------------------
# Direction o's components, (o % 3 - 1, (o // 3) % 3 - 1, o // 9 - 1), as
# tables for the kernels: a GPU has no integer division, and with the lattice's
# q and d compiled in (``D2Q9``, ``D3Q27``) the loops over directions unroll
# and the tables fold away (Metal's collision: 2.1 ms to 0.1).
_CX = tack.constant(tuple(o % 3 - 1 for o in range(27)), tack.i32)
_CY = tack.constant(tuple((o // 3) % 3 - 1 for o in range(27)), tack.i32)
_CZ = tack.constant(tuple(o // 9 - 1 for o in range(27)), tack.i32)


@tack.device_class
class D2Q9:
    """The lattice in 2D: q and d as compile-time constants of the kernels."""

    Q = 9
    DIM = 2


@tack.device_class
class D3Q27:
    """The lattice in 3D."""

    Q = 27
    DIM = 3


# DdQ3^d: a node's q = 3^d values are f[q n + o], direction o = (o % 3 - 1,
# (o // 3) % 3 - 1, o // 9 - 1 in 3D), the same numbering as the node's row,
# so the upstream entry for direction o is q - 1 - o, its opposite. Positions
# are (px, py, pz) (pz 0 in 2D); gradients are grad[(q n + o) d + axis]...
# laid out as grad[n q d + axis q + o].


@tack.func
def _cz(o, dim):
    c = 0
    if dim == 3:
        c = _CZ[o]
    return c


@tack.func
def _weight(cx, cy, cz, dim):
    moving = abs(cx) + abs(cy) + abs(cz)
    w = 1.0 / 36.0
    if dim == 2:
        if moving == 0:
            w = 4.0 / 9.0
        elif moving == 1:
            w = 1.0 / 9.0
    else:
        w = 1.0 / 216.0
        if moving == 0:
            w = 8.0 / 27.0
        elif moving == 1:
            w = 2.0 / 27.0
        elif moving == 2:
            w = 1.0 / 54.0
    return w


@tack.func
def _moments_at(f, at, q, dim):
    # Density and velocity of the q values from f[at].
    rho = f[at]
    mx = -f[at]
    my = -f[at]
    mz = f[at] * 0.0
    if dim == 3:
        mz = -f[at]
    for o in range(1, q):
        v = f[at + o]
        rho = rho + v
        mx = mx + v * _CX[o]
        my = my + v * _CY[o]
        mz = mz + v * _cz(o, dim)
    return rho, mx / rho, my / rho, mz / rho


@tack.func
def _feq(o, rho, ux, uy, uz, dim):
    cx = _CX[o]
    cy = _CY[o]
    cz = _cz(o, dim)
    cu = 3.0 * (cx * ux + cy * uy + cz * uz)
    return _weight(cx, cy, cz, dim) * rho * (1.0 + cu + 0.5 * cu * cu -
                                             1.5 * (ux * ux + uy * uy + uz * uz))


@tack.kernel
def _collide(nodes, f, omega, lat, regularized, px, sponge_from, sponge_to, sponge_tau, role, out,
             factor):
    # BGK; ``regularized``: the non-equilibrium part first projected onto its
    # stress (the second Hermite moment), which keeps low ``tau`` stable. In
    # the sponge (x from ``sponge_from`` to ``sponge_to``) ``tau`` rises
    # linearly by up to ``sponge_tau``, damping what leaves by the outlet.
    # With ``factor`` (1 / 4 tau; 0: none), an interface cell's
    # non-equilibrium part before collision, times ``factor``, goes into its
    # ``out`` row (free until the stream): half a fine step's stress, for the
    # explosion.
    q = lat.Q
    dim = lat.DIM
    for k in range(nodes.shape[0]):
        n = nodes[k]
        rate = omega
        if sponge_to > sponge_from and px[n] > sponge_from:
            ramp = min((px[n] - sponge_from) / (sponge_to - sponge_from), 1.0)
            rate = 1.0 / (1.0 / omega + sponge_tau * ramp)
        rho, ux, uy, uz = _moments_at(f, n * q, q, dim)
        if factor != 0.0 and role[n] == 1:
            for o in range(q):
                out[n * q + o] = factor * (f[n * q + o] - _feq(o, rho, ux, uy, uz, dim))
        if regularized == 1:
            pxx = rho * 0.0
            pyy = pxx
            pzz = pxx
            pxy = pxx
            pxz = pxx
            pyz = pxx
            for o in range(q):
                cx = _CX[o]
                cy = _CY[o]
                cz = _cz(o, dim)
                e = f[n * q + o] - _feq(o, rho, ux, uy, uz, dim)
                pxx = pxx + cx * cx * e
                pyy = pyy + cy * cy * e
                pzz = pzz + cz * cz * e
                pxy = pxy + cx * cy * e
                pxz = pxz + cx * cz * e
                pyz = pyz + cy * cz * e
            for o in range(q):
                cx = _CX[o]
                cy = _CY[o]
                cz = _cz(o, dim)
                trace = (pxx + pyy + pzz) / 3.0
                qp = (cx * cx * pxx + cy * cy * pyy + cz * cz * pzz +
                      2.0 * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz)) - trace
                if dim == 2:
                    qp = (cx * cx * pxx + cy * cy * pyy + 2.0 * cx * cy * pxy) - (pxx + pyy) / 3.0
                f[n * q + o] = (_feq(o, rho, ux, uy, uz, dim) +
                                (1.0 - rate) * 4.5 * _weight(cx, cy, cz, dim) * qp)
        else:
            for o in range(q):
                e = _feq(o, rho, ux, uy, uz, dim)
                f[n * q + o] = f[n * q + o] + rate * (e - f[n * q + o])


@tack.func
def _offset(d, length):
    # A coordinate difference, across a periodic axis of ``length`` (0: none)
    # the short way.
    if length > 0.0:
        d = d - length * tack.floor(d / length + 0.5)
    return d


@tack.func
def _value_at(r, o, x, y, z, px, py, pz, side, lx, ly, lz, q, dim, f, grad):
    # Node r's value of direction o at the point: its own plus its gradient's
    # share (zero but at the cells by the interfaces).
    g = r * q * dim
    v = (_offset(x - px[r], lx) * grad[g + o] + _offset(y - py[r], ly) * grad[g + q + o])
    if dim == 3:
        v = v + _offset(z - pz[r], lz) * grad[g + 2 * q + o]
    return f[r * q + o] + v / side[r]


@tack.func
def _outside(x, y, z, width, height, depth):
    # Beyond a wall: the domain is [0, width] x [0, height] x [0, depth], an axis
    # of size 0 periodic (or absent).
    out = 0
    if width > 0.0:
        if x < 0.0 or x > width:
            out = 1
    if height > 0.0:
        if y < 0.0 or y > height:
            out = 1
    if depth > 0.0:
        if z < 0.0 or z > depth:
            out = 1
    return out


@tack.func
def _source(n, d, x, y, z, rows, role, px, py, pz, side, lx, ly, lz, q, dim, f, grad, half):
    # The value of direction d at a point of node n's neighbourhood, at the
    # start of n's step, as the next level sees it: the holder's value at the
    # point, an interface cell's as its ghost there has it.
    h = side[n]
    a = tack.i32(tack.floor(_offset(x - px[n], lx) / h + 0.5))
    b = tack.i32(tack.floor(_offset(y - py[n], ly) / h + 0.5))
    entry = (a + 1) + 3 * (b + 1)
    if dim == 3:
        c = tack.i32(tack.floor(_offset(z - pz[n], lz) / h + 0.5))
        entry = entry + 9 * (c + 1)
    r = rows[n * q + entry]
    v = _value_at(r, d, x, y, z, px, py, pz, side, lx, ly, lz, q, dim, f, grad)
    if role[r] == 1:
        sx = 1
        if _offset(x - px[r], lx) < 0.0:
            sx = -1
        sy = 1
        if _offset(y - py[r], ly) < 0.0:
            sy = -1
        sz = 1
        if _offset(z - pz[r], lz) < 0.0:
            sz = -1
        along = sx * _CX[d] + sy * _CY[d] + sz * _cz(d, dim)
        if along > 0:
            v = v + half[r * q + d]
        elif along < 0:
            v = v - half[r * q + d]
    return v


@tack.func
def _wall(n, o, finest, rows, role, px, py, pz, side, lx, ly, lz, width, height, depth, q, dim,
          f, grad, half, lid, inflow, speed):
    # Direction o at node n, whose upstream neighbour is beyond the domain.
    # With ``inflow``, the low-x side is an inlet (equilibrium at density 1 and
    # ``speed`` along x) and the high-x side an outlet (equilibrium at density 1
    # and the cell's own velocity: a pressure outlet). Elsewhere a wall: halfway bounce-back, the opposite value,
    # with the lid's momentum (along x) at the high-y wall. A direction along
    # two or three axes, at a level with finer ones, takes what its 2^d
    # quarters get over two steps of the next level (it retraces its line,
    # into the neighbours along the wall), as the ghosts there do: the plain
    # rule would keep packets the ghosts take.
    cx = _CX[o]
    cy = _CY[o]
    cz = _cz(o, dim)
    v = f[n * q + q - 1 - o]
    upstream = px[n] - cx * side[n]
    if inflow == 1 and width > 0.0 and upstream < 0.0:
        cu = 3.0 * cx * speed
        v = _weight(cx, cy, cz, dim) * (1.0 + cu + 0.5 * cu * cu - 1.5 * speed * speed)
    elif inflow == 1 and width > 0.0 and upstream > width:
        rho, ux, uy, uz = _moments_at(f, n * q, q, dim)
        v = _feq(o, 1.0, ux, uy, uz, dim)
    elif abs(cx) + abs(cy) + abs(cz) < 2 or finest == 1:
        if cy == -1 and rows[n * q + (q - 1) // 2 + 3] < 0:
            v = v + 6.0 * _weight(cx, cy, cz, dim) * cx * lid
    else:
        h = side[n]
        v = v * 0.0
        kids = 1 << dim
        for j in range(kids):
            x = px[n] + 0.25 * h * (2 * (j & 1) - 1)
            y = py[n] + 0.25 * h * (2 * ((j >> 1) & 1) - 1)
            z = pz[n]
            if dim == 3:
                z = pz[n] + 0.25 * h * (2 * ((j >> 2) & 1) - 1)
            dx = cx
            dy = cy
            dz = cz
            gain = v * 0.0
            for step in range(2):
                ux = x - 0.5 * h * dx
                uy = y - 0.5 * h * dy
                uz = z - 0.5 * h * dz
                if _outside(ux, uy, uz, width, height, depth) == 1:
                    if height > 0.0 and uy > height and dy == -1:
                        gain = gain + 6.0 * _weight(dx, dy, dz, dim) * dx * lid
                    dx = -dx
                    dy = -dy
                    dz = -dz
                else:
                    x = ux
                    y = uy
                    z = uz
            back = (dx + 1) + 3 * (dy + 1)
            if dim == 3:
                back = back + 9 * (dz + 1)
            v = v + (_source(n, back, x, y, z, rows, role, px, py, pz, side, lx, ly, lz, q, dim,
                             f, grad, half) + gain) / kids
    return v


@tack.kernel
def _stream(nodes, parent, rows, level, role, at, finest, second, px, py, pz, side, lx, ly, lz,
            width, height, depth, lat, f, grad, out, lid, inflow, speed):
    # The pull, a thread per value, over a level's real leaves, interface cells
    # (skipped: ``_store`` averages their ghosts) and ghosts. A real leaf pulls
    # from the node upstream at its level, bouncing back from a solid one, or
    # from beyond the domain (``_wall``). A ghost pulls from the node upstream
    # at its level; from a coarser cell upstream, that cell's value at the
    # first substep and, at the second, the value of the cell two fine steps
    # upstream, the parent's upstream neighbour (bouncing there where that is
    # beyond a wall); a coarser cell's value is its own at the point, by its
    # gradient.
    q = lat.Q
    dim = lat.DIM
    for k, o in tack.ndindex(nodes.shape[0], q):
        n = nodes[k]
        kind = role[n]
        if kind != 1:
            r = rows[n * q + q - 1 - o]
            if r < 0:
                out[n * q + o] = _wall(n, o, finest, rows, role, px, py, pz, side, lx, ly, lz,
                                       width, height, depth, q, dim, f, grad, out, lid, inflow,
                                       speed)
            elif role[r] == 4:
                out[n * q + o] = f[n * q + q - 1 - o]
            elif kind == 0 or level[r] == at:
                out[n * q + o] = f[r * q + o]
            else:
                h = side[n]
                cx = _CX[o]
                cy = _CY[o]
                cz = _cz(o, dim)
                if second == 0:
                    out[n * q + o] = _value_at(r, o, px[n] - cx * h, py[n] - cy * h,
                                               pz[n] - cz * h, px, py, pz, side, lx, ly, lz, q,
                                               dim, f, grad)
                else:
                    s = rows[parent[n] * q + q - 1 - o]
                    if s >= 0:
                        out[n * q + o] = _value_at(s, o, px[n] - 2 * cx * h, py[n] - 2 * cy * h,
                                                   pz[n] - 2 * cz * h, px, py, pz, side, lx, ly,
                                                   lz, q, dim, f, grad)
                    else:
                        # Bounced at the upstream point, in the coarser cell.
                        v = _value_at(r, q - 1 - o, px[n] - cx * h, py[n] - cy * h,
                                      pz[n] - cz * h, px, py, pz, side, lx, ly, lz, q, dim, f,
                                      grad)
                        if cy == -1 and height > 0.0 and py[n] + h > height:
                            v = v + 6.0 * _weight(cx, cy, cz, dim) * cx * lid
                        out[n * q + o] = v


@tack.func
def _usable(r, level, role, at):
    # A neighbour to difference with: a node at the level that holds values.
    use = 0
    if r >= 0:
        if level[r] == at and role[r] < 3:
            use = 1
    return use


@tack.func
def _slope(c, lo, hi, use_lo, use_hi):
    g = c * 0.0
    if use_lo == 1 and use_hi == 1:
        g = 0.5 * (hi - lo)
    elif use_hi == 1:
        g = hi - c
    elif use_lo == 1:
        g = c - lo
    return g


@tack.kernel
def _gradients(nodes, rows, level, role, at, lat, f, grad):
    # Each direction's change across a cell along each axis, central where both
    # neighbours at the level hold values, one-sided where one does.
    q = lat.Q
    dim = lat.DIM
    for k in range(nodes.shape[0]):
        n = nodes[k]
        centre = (q - 1) // 2
        stride = 1
        for axis in range(dim):
            lo = rows[n * q + centre - stride]
            hi = rows[n * q + centre + stride]
            use_lo = _usable(lo, level, role, at)
            use_hi = _usable(hi, level, role, at)
            for o in range(q):
                grad[n * q * dim + axis * q + o] = _slope(f[n * q + o], f[max(lo, 0) * q + o],
                                                          f[max(hi, 0) * q + o], use_lo, use_hi)
            stride = stride * 3


@tack.kernel
def _gradients_explode(nodes, rows, level, role, at, lat, f, grad, number, table, half):
    # ``_gradients`` a thread per value, over a level's border cells, and an
    # interface cell's values then into its ghosts, as ``_explode`` does.
    q = lat.Q
    dim = lat.DIM
    for k, o in tack.ndindex(nodes.shape[0], q):
        n = nodes[k]
        centre = (q - 1) // 2
        c = f[n * q + o]
        g = c * 0.0
        gx = g
        gy = g
        gz = g
        stride = 1
        for axis in range(dim):
            lo = rows[n * q + centre - stride]
            hi = rows[n * q + centre + stride]
            g = _slope(c, f[max(lo, 0) * q + o], f[max(hi, 0) * q + o],
                       _usable(lo, level, role, at), _usable(hi, level, role, at))
            grad[n * q * dim + axis * q + o] = g
            if axis == 0:
                gx = g
            elif axis == 1:
                gy = g
            else:
                gz = g
            stride = stride * 3
        if role[n] == 1:
            kids = 1 << dim
            r = (number[n] - 1) * kids
            for j in range(kids):
                sx = 2 * (j & 1) - 1
                sy = 2 * ((j >> 1) & 1) - 1
                sz = 0
                if dim == 3:
                    sz = 2 * ((j >> 2) & 1) - 1
                along = sx * _CX[o] + sy * _CY[o] + sz * _cz(o, dim)
                lead = 0
                if along > 0:
                    lead = 1
                elif along < 0:
                    lead = -1
                v = c + 0.25 * sx * gx + 0.25 * sy * gy + lead * half[n * q + o]
                if dim == 3:
                    v = v + 0.25 * sz * gz
                f[table[r + j] * q + o] = v


@tack.kernel
def _explode(nodes, number, table, f, grad, half, lat):
    # An interface cell's values into its ghosts, linear in its gradient, and
    # time-centred: a ghost on the downstream side of a direction (whose packet
    # reaches the fine cells a substep early) gains half a fine step's stress
    # (``half``, from ``_collide``), one on the upstream side loses it. The
    # 2^d shares of each cancel, so each direction's total is the cell's.
    q = lat.Q
    dim = lat.DIM
    for k in range(nodes.shape[0]):
        n = nodes[k]
        kids = 1 << dim
        r = (number[n] - 1) * kids
        for j in range(kids):
            g = table[r + j]
            sx = 2 * (j & 1) - 1
            sy = 2 * ((j >> 1) & 1) - 1
            sz = 0
            if dim == 3:
                sz = 2 * ((j >> 2) & 1) - 1
            for o in range(q):
                along = sx * _CX[o] + sy * _CY[o] + sz * _cz(o, dim)
                lead = 0
                if along > 0:
                    lead = 1
                elif along < 0:
                    lead = -1
                v = (f[n * q + o] + 0.25 * sx * grad[n * q * dim + o] +
                     0.25 * sy * grad[n * q * dim + q + o] + lead * half[n * q + o])
                if dim == 3:
                    v = v + 0.25 * sz * grad[n * q * dim + 2 * q + o]
                f[g * q + o] = v


@tack.kernel
def _store(nodes, role, number, table, out, f, lat):
    # The stream's values into ``f``, a thread per value, over a level's real
    # leaves, interface cells and ghosts; an interface cell takes the mean of
    # its ghosts (a level finer, so none of these).
    q = lat.Q
    dim = lat.DIM
    for k, o in tack.ndindex(nodes.shape[0], q):
        n = nodes[k]
        if role[n] == 1:
            kids = 1 << dim
            r = (number[n] - 1) * kids
            s = f[table[r] * q + o]
            for j in range(1, kids):
                s = s + f[table[r + j] * q + o]
            f[n * q + o] = s / kids
        else:
            f[n * q + o] = out[n * q + o]


@tack.kernel
def _parents_of_ghosts(nodes, number, table, parent, dim):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        kids = 1 << dim
        r = (number[n] - 1) * kids
        for j in range(kids):
            parent[table[r + j]] = tack.i32(n)


@tack.kernel
def _list_keys(role, level, rows, q, groups, key, counts):
    # Node n's place among the solver's node lists, level * groups + group
    # (``LBM.__init__``), and each place's count.
    for n in range(role.shape[0]):
        r = role[n]
        g = 5                                   # solid
        if r == 0:                              # real: 1 beside an interface cell
            g = 0
            for o in range(q):
                m = rows[n * q + o]
                if m >= 0:
                    if role[m] == 1:
                        g = 1
        elif r == 1:                            # interface
            g = 2
        elif r == 2:                            # ghost
            g = 3
        elif r == 3:                            # refined
            g = 4
        k = level[n] * groups + g
        key[n] = k
        tack.atomic_add(counts, k, 1)


@tack.kernel
def _take(order, start, out):
    for k in range(out.shape[0]):
        out[k] = order[start + k]


@tack.kernel
def _centres(corner, sides, dimension, px, py, pz, side):
    # Each node's centre and side from the octree's geometry.
    for n in range(px.shape[0]):
        c = corner[n]
        h = sides[n]
        px[n] = c[0] + 0.5 * h[0]
        py[n] = c[1] + 0.5 * h[1]
        z = c[2] * 0.0
        if dimension == 3:
            z = c[2] + 0.5 * h[2]
        pz[n] = z
        side[n] = h[0]


@tack.kernel
def _moments(f, rho, ux, uy, uz, lat):
    q = lat.Q
    dim = lat.DIM
    for n in range(rho.shape[0]):
        r, vx, vy, vz = _moments_at(f, n * q, q, dim)
        rho[n] = r
        ux[n] = vx
        uy[n] = vy
        uz[n] = vz


# -- the solver ---------------------------------------------------------------


class LBM:
    """D2Q9 or D3Q27 BGK on ``octree`` (from ``lbm_mesh``, or ``uniform_mesh``
    refined by ``adapt``) with relaxation time ``tau`` at level 0, a lid moving
    at ``lid`` (lattice units, along x) on the high-y wall, and, with ``inflow``
    (a speed along x), an inlet on the low-x side and an outlet on the high-x
    one. ``regularized`` projects the non-equilibrium part onto its stress
    before relaxing (stable nearer ``tau`` 1/2). ``sponge = (x, extra)`` raises
    level 0's ``tau`` linearly by up to ``extra`` from ``x`` to the high-x side
    (each level's in proportion), damping what leaves by the outlet. One ``step`` is one step of
    level 0: ``2^l`` of level ``l``. Values are in lattice units of level 0: a
    root cell is 1 and so is its time step. ``centred`` and ``linear`` switch
    the explosion's corrections off, for comparison. ``dtype`` is what values
    are kept in: f64 where the device has it by default (f32 is about twice as
    fast on a GPU with f64, and the choice on most GPUs, whose f64 is slow)."""

    def __init__(self, octree, tau, lid=0.0, inflow=None, centred=True, linear=True,
                 regularized=False, sponge=None, dtype=None):
        self.octree = octree
        self.dim = d = octree.dimension
        self.q = q = 3 ** d
        self.lattice = D3Q27() if d == 3 else D2Q9()
        self.lid = float(lid)
        self.inflow = 0 if inflow is None else 1
        self.speed = 0.0 if inflow is None else float(inflow)
        self.regularized = bool(regularized)
        self.sponge = sponge
        self.finest = octree.finest
        device = octree.device
        self.dtype = dtype if dtype is not None else _compute_type(device, tack.f32)
        role = np.asarray(octree.fields["role"].numpy(), np.int64)
        level = octree._level
        self.role, self.levels = role, level
        self.tau = float(tau)
        self.taus = [0.5 + (1 << lv) * (tau - 0.5) for lv in range(self.finest + 1)]
        self.omega = [1.0 / t for t in self.taus]
        self.centred = centred
        self.linear = linear
        periodic = tuple(getattr(octree, "periodic", (False,) * d))
        self.rows = _rows(octree, periodic)
        self.number, self.table = children(octree)
        n = octree.num_nodes
        dims = [s - 1 for s in octree.roots.topology.point_dims]
        corner, sides = octree.geometry()
        side0 = float(sides[int(np.argmax(level == 0))][0])
        self.lengths = [dims[a] * side0 if a < d and periodic[a] else 0.0 for a in range(3)]
        self.walls = [0.0 if a >= d or periodic[a] else dims[a] * side0 for a in range(3)]
        levels = range(self.finest + 1)
        with device:
            self.role_array = tack.array(role.astype(np.int32), tack.i32)
            # The node lists, by a sort on (level, group): group 0 real leaves,
            # 1 real leaves beside an interface cell (which need gradients, as
            # interface cells do), 2 interface cells, 3 ghosts, 4 refined, 5
            # solid; so real, computed and border cells are each one run.
            key = tack.empty((n,), tack.i32)
            groups = 8
            counts = tack.zeros(((self.finest + 1) * groups,), tack.i32)
            if n:
                _list_keys(self.role_array, octree.level, self.rows, q, groups, key, counts)
            order = argsort(key, index_dtype=tack.i64)
            bounds = np.concatenate([[0], np.cumsum(counts.numpy())])

            def run(lv, first, last):
                lo, hi = int(bounds[lv * groups + first]), int(bounds[lv * groups + last + 1])
                out = tack.empty((hi - lo,), tack.i64)
                if hi > lo:
                    _take(order, lo, out)
                return out

            self.real = [run(lv, 0, 1) for lv in levels]
            self.interface = [run(lv, 2, 2) for lv in levels]
            self.ghosts = [run(lv, 3, 3) for lv in levels]
            self.computed = [run(lv, 0, 2) for lv in levels]
            self.border = [run(lv, 1, 2) for lv in levels]
            self.streamed = [run(lv, 0, 3) for lv in levels]
            self.px, self.py, self.pz, self.side = (tack.empty((n,), self.dtype) for _ in range(4))
            if n:
                _centres(corner, sides, d, self.px, self.py, self.pz, self.side)
            self.grad = tack.zeros((n * q * d,), self.dtype)
            self.parent = tack.full((n,), -1, tack.i32)
            for lv in levels:
                if len(self.interface[lv]):
                    _parents_of_ghosts(self.interface[lv], self.number, self.table, self.parent, d)
            self.f = tack.zeros((n * q,), self.dtype)
            self.out = tack.zeros((n * q,), self.dtype)
        self.steps = 0

    # -- state ----------------------------------------------------------------

    def centres(self):
        """Each node's centre and side, host arrays ``(n, d)`` and ``(n,)``."""
        lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in
                    self.octree.geometry())
        d = self.dim
        return lo[:, :d] + 0.5 * size[:, :d], size[:, 0]

    def initialize(self, rho, ux, uy, uz=0.0):
        """Every node at equilibrium for its density and velocity (one value a
        node, or scalars), ghosts included."""
        n, d, q = self.octree.num_nodes, self.dim, self.q
        rho, ux, uy, uz = (np.broadcast_to(np.asarray(v, np.float64), (n,)) for v in
                           (rho, ux, uy, uz))
        o = np.arange(q)
        cx, cy, cz = o % 3 - 1, (o // 3) % 3 - 1, (o // 9 - 1 if d == 3 else 0 * o)
        moving = np.abs(cx) + np.abs(cy) + np.abs(cz)
        w = (np.array([4 / 9, 1 / 9, 1 / 36]) if d == 2 else
             np.array([8 / 27, 2 / 27, 1 / 54, 1 / 216]))[moving]
        cu = 3.0 * (cx[None] * ux[:, None] + cy[None] * uy[:, None] + cz[None] * uz[:, None])
        feq = w[None] * rho[:, None] * (1 + cu + 0.5 * cu * cu -
                                        1.5 * (ux * ux + uy * uy + uz * uz)[:, None])
        with self.octree.device:
            self.f = tack.array(feq.reshape(-1).astype(self.dtype.numpy_dtype), self.dtype)
        self.steps = 0

    def moments(self):
        """Density and velocity at every node, host arrays ``rho, ux, uy`` (and
        ``uz`` in 3D); meaningful at the computational cells
        (``computational()``)."""
        n = self.octree.num_nodes
        with self.octree.device:
            out = [tack.empty((n,), self.dtype) for _ in range(4)]
            _moments(self.f, *out, self.lattice)
        return tuple(a.numpy() for a in out[:1 + self.dim])

    def computational(self):
        """Which nodes are cells of the solution: real leaves and interface cells."""
        return np.isin(self.role, (REAL, INTERFACE))

    def mass(self):
        """The total mass: density times volume over the computational cells."""
        rho = self.moments()[0]
        keep = self.computational()
        return float(np.sum(rho[keep] * (0.5 ** self.dim) ** self.levels[keep]))

    def momentum(self):
        rho, *u = self.moments()
        keep = self.computational()
        a = (0.5 ** self.dim) ** self.levels[keep]
        return tuple(float(np.sum(rho[keep] * v[keep] * a)) for v in u)

    def updates_per_step(self):
        """Cell updates in one step of level 0: each level's computational cells
        times its substeps."""
        return sum(len(c) << lv for lv, c in enumerate(self.computed))

    def fill_ghosts(self):
        """Give every ghost its parent's values, linear in the parent's gradient
        (as the next step's explosion will, without its half-step lead), for
        whatever looks at the state between steps: the vorticity, a contour.
        Between steps the solver itself reads no ghost, gradient or ``out``
        value before writing it, so this changes nothing it computes. Coarse
        levels first, since a finer interface cell's gradient may read the
        ghosts beside it."""
        with self.octree.device:
            self.out.fill(0.0)
            for lv in range(self.finest):
                if not len(self.interface[lv]):
                    continue
                if self.linear:
                    _gradients(self.interface[lv], self.rows, self.octree.level,
                               self.role_array, lv, self.lattice, self.f, self.grad)
                _explode(self.interface[lv], self.number, self.table, self.f, self.grad,
                         self.out, self.lattice)

    # -- stepping -------------------------------------------------------------

    def step(self, count=1):
        with self.octree.device:
            for _ in range(count):
                self._step(0, 0)
        self.steps += count

    def _step(self, lv, second):
        # Four launches a level's step: collide (and the interface cells' half
        # step), gradients and the explosion, the pull, the store (and the
        # interface cells' mean of their ghosts).
        f, out, lat = self.f, self.out, self.lattice
        factor = 0.0
        if len(self.interface[lv]):
            if self.centred:
                factor = 0.25 / self.taus[lv]
            else:
                out.fill(0.0)
        if len(self.computed[lv]):
            start, extra = self.sponge if self.sponge is not None else (0.0, 0.0)
            end = self.walls[0] if self.sponge is not None else 0.0
            _collide(self.computed[lv], f, self.omega[lv], lat, int(self.regularized), self.px,
                     float(start), float(end), float(extra * (1 << lv)), self.role_array, out,
                     factor)
        if lv < self.finest and len(self.interface[lv]):
            if self.linear:
                _gradients_explode(self.border[lv], self.rows, self.octree.level,
                                   self.role_array, lv, lat, f, self.grad, self.number,
                                   self.table, out)
            else:
                _explode(self.interface[lv], self.number, self.table, f, self.grad, out, lat)
        elif self.linear and len(self.border[lv]):
            _gradients(self.border[lv], self.rows, self.octree.level, self.role_array, lv, lat, f,
                       self.grad)
        if lv < self.finest:
            self._step(lv + 1, 0)
            self._step(lv + 1, 1)
        finest = int(lv == self.finest)
        geometry = (self.px, self.py, self.pz, self.side, *self.lengths, *self.walls, lat)
        if len(self.streamed[lv]):
            _stream(self.streamed[lv], self.parent, self.rows, self.octree.level,
                    self.role_array, lv, finest, second, *geometry, f, self.grad, out, self.lid,
                    self.inflow, self.speed)
            _store(self.streamed[lv], self.role_array, self.number, self.table, out, f, lat)

LBM2D = LBM
