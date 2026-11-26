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
upstream, which is the parent's upstream neighbour (``_pull_ghosts``). The
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
  loses it (``_half_step``, ``_explode``): 40 times less, and a refined mesh
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
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet, _compute_type
from tack.dataset.octree_topology import (
    _child_neighbourhoods,
    _root_neighbourhoods,
    children,
    refined_by_level,
)

__all__ = ["GHOST", "INTERFACE", "LBM2D", "REAL", "REFINED", "SOLID", "lbm_mesh"]

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


def _root_cells_side(octree):
    lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in octree.geometry())
    return size[octree._level == 0, 0].max()


def _rows(octree, periodic):
    """Every node's 3x3 neighbourhood at its level, as ``neighbourhoods`` makes
    it, with the roots' rows wrapping along ``periodic`` axes."""
    nx, ny = (n - 1 for n in octree.roots.topology.point_dims)
    roots = np.flatnonzero(octree._level == 0)
    root_of = np.full(nx * ny, -1, np.int64)
    root_of[octree._tree[roots]] = roots
    t = octree._tree[roots]
    rows = np.empty((len(roots), 9), np.int64)
    for o in range(9):
        a, b = o % 3 - 1, o // 3 - 1
        i, j = t % nx + a, t // nx + b
        inside = np.ones(len(t), bool)
        if periodic[0]:
            i %= nx
        else:
            inside &= (i >= 0) & (i < nx)
        if periodic[1]:
            j %= ny
        else:
            inside &= (j >= 0) & (j < ny)
        rows[:, o] = np.where(inside, root_of[np.clip(i, 0, nx - 1) + nx * np.clip(j, 0, ny - 1)],
                              -1)
    number, table = children(octree)
    with octree.device:
        out = tack.empty((octree.num_nodes * 9,), tack.i32)
        _root_neighbourhoods(tack.array(rows.reshape(-1).astype(np.int32), tack.i32), len(roots),
                             9, out)
        for nodes in reversed(refined_by_level(octree)):
            _child_neighbourhoods(nodes, number, table, octree.marks, int(Ghosts.REFINEDCELL), 2,
                                  9, out)
    return out


# -- the kernels ------------------------------------------------------------
# A node's nine values are f[9 n + o], direction o = (o % 3 - 1, o // 3 - 1);
# the upstream entry of a node's row for direction o is 8 - o, its opposite.


@tack.func
def _weight(cx, cy):
    w = 1.0 / 36.0
    if cx == 0 and cy == 0:
        w = 4.0 / 9.0
    elif cx == 0 or cy == 0:
        w = 1.0 / 9.0
    return w


@tack.kernel
def _collide(nodes, f, omega):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        rho = f[n * 9]
        mx = -f[n * 9]
        my = -f[n * 9]
        for o in range(1, 9):
            v = f[n * 9 + o]
            rho = rho + v
            mx = mx + v * (o % 3 - 1)
            my = my + v * (o // 3 - 1)
        ux = mx / rho
        uy = my / rho
        usq = ux * ux + uy * uy
        for o in range(9):
            cx = o % 3 - 1
            cy = o // 3 - 1
            cu = 3.0 * (cx * ux + cy * uy)
            feq = _weight(cx, cy) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
            f[n * 9 + o] = f[n * 9 + o] + omega * (feq - f[n * 9 + o])


@tack.func
def _offset(d, length):
    # A coordinate difference, across a periodic axis of ``length`` (0: none)
    # the short way.
    if length > 0.0:
        d = d - length * tack.floor(d / length + 0.5)
    return d


@tack.func
def _value_at(q, o, x, y, px, py, side, lx, ly, f, grad):
    # Node q's value of direction o at the point (x, y): its own plus its
    # gradient's share (zero but at the cells by the interfaces).
    return f[q * 9 + o] + (_offset(x - px[q], lx) * grad[q * 18 + o] +
                           _offset(y - py[q], ly) * grad[q * 18 + 9 + o]) / side[q]


@tack.func
def _outside(x, y, width, height):
    # Beyond a wall: the domain is [0, width] x [0, height], an axis of size 0
    # periodic.
    out = 0
    if width > 0.0:
        if x < 0.0 or x > width:
            out = 1
    if height > 0.0:
        if y < 0.0 or y > height:
            out = 1
    return out


@tack.func
def _source(n, d, x, y, rows, role, px, py, side, lx, ly, f, grad, half):
    # The value of direction d at the point (x, y) of node n's neighbourhood,
    # at the start of n's step, as the next level sees it: the holder's value
    # at the point, an interface cell's as its ghost there has it.
    h = side[n]
    a = tack.i32(tack.floor(_offset(x - px[n], lx) / h + 0.5))
    b = tack.i32(tack.floor(_offset(y - py[n], ly) / h + 0.5))
    r = rows[n * 9 + (a + 1) + 3 * (b + 1)]
    v = _value_at(r, d, x, y, px, py, side, lx, ly, f, grad)
    if role[r] == 1:
        sx = 1
        if _offset(x - px[r], lx) < 0.0:
            sx = -1
        sy = 1
        if _offset(y - py[r], ly) < 0.0:
            sy = -1
        along = sx * (d % 3 - 1) + sy * (d // 3 - 1)
        if along > 0:
            v = v + half[r * 9 + d]
        elif along < 0:
            v = v - half[r * 9 + d]
    return v


@tack.func
def _wall(n, o, finest, rows, role, px, py, side, lx, ly, width, height, f, grad, half, lid,
          inflow, speed):
    # Direction o at node n, whose upstream neighbour is beyond the domain.
    # With ``inflow``, the left side is an inlet (equilibrium at density 1 and
    # ``speed`` along x) and the right an outlet (the cell's own value: zero
    # gradient). Elsewhere a wall: halfway bounce-back, the opposite value,
    # with the lid's momentum at the top wall. A diagonal at a level with finer
    # ones takes what its four quarters get over two steps of the next level
    # (a diagonal retraces its line, into the neighbours along the wall), as
    # the ghosts there do: the plain rule would keep packets the ghosts take.
    cx = o % 3 - 1
    cy = o // 3 - 1
    v = f[n * 9 + 8 - o]
    upstream = px[n] - cx * side[n]
    if inflow == 1 and width > 0.0 and upstream < 0.0:
        cu = 3.0 * cx * speed
        v = _weight(cx, cy) * (1.0 + cu + 0.5 * cu * cu - 1.5 * speed * speed)
    elif inflow == 1 and width > 0.0 and upstream > width:
        v = f[n * 9 + o]
    elif cx == 0 or cy == 0 or finest == 1:
        if cy == -1 and rows[n * 9 + 7] < 0:
            v = v + 6.0 * _weight(cx, cy) * cx * lid
    else:
        h = side[n]
        v = v * 0.0
        for j in range(4):
            x = px[n] + 0.25 * h * (2 * (j & 1) - 1)
            y = py[n] + 0.25 * h * (2 * (j >> 1) - 1)
            dx = cx
            dy = cy
            gain = v * 0.0
            for step in range(2):
                ux = x - 0.5 * h * dx
                uy = y - 0.5 * h * dy
                if _outside(ux, uy, width, height) == 1:
                    if height > 0.0 and uy > height and dy == -1:
                        gain = gain + 6.0 * _weight(dx, dy) * dx * lid
                    dx = -dx
                    dy = -dy
                else:
                    x = ux
                    y = uy
            v = v + 0.25 * (_source(n, (dx + 1) + 3 * (dy + 1), x, y, rows, role, px, py, side,
                                    lx, ly, f, grad, half) + gain)
    return v


@tack.kernel
def _pull(nodes, rows, role, finest, px, py, side, lx, ly, width, height, f, grad, out, lid,
          inflow, speed):
    # A real leaf's pull: from the node upstream at its level, bouncing back
    # from a solid one, or from beyond the domain (``_wall``).
    for k in range(nodes.shape[0]):
        n = nodes[k]
        for o in range(9):
            q = rows[n * 9 + 8 - o]
            if q >= 0:
                if role[q] == 4:
                    out[n * 9 + o] = f[n * 9 + 8 - o]
                else:
                    out[n * 9 + o] = f[q * 9 + o]
            else:
                out[n * 9 + o] = _wall(n, o, finest, rows, role, px, py, side, lx, ly, width,
                                       height, f, grad, out, lid, inflow, speed)


@tack.kernel
def _pull_ghosts(nodes, parent, rows, level, role, at, finest, second, px, py, side, lx, ly, width,
                 height, f, grad, out, lid, inflow, speed):
    # A ghost's pull: from the node upstream at its level; from a coarser cell
    # upstream, that cell's value at the first substep and, at the second, the
    # value of the cell two fine steps upstream, the parent's upstream
    # neighbour (bouncing there where that is beyond a wall); a coarser cell's
    # value is its own at the point, by its gradient.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        h = side[n]
        for o in range(9):
            cx = o % 3 - 1
            cy = o // 3 - 1
            q = rows[n * 9 + 8 - o]
            if q < 0:
                out[n * 9 + o] = _wall(n, o, finest, rows, role, px, py, side, lx, ly, width,
                                       height, f, grad, out, lid, inflow, speed)
            elif role[q] == 4:
                out[n * 9 + o] = f[n * 9 + 8 - o]
            elif level[q] == at:
                out[n * 9 + o] = f[q * 9 + o]
            elif second == 0:
                out[n * 9 + o] = _value_at(q, o, px[n] - cx * h, py[n] - cy * h, px, py, side, lx,
                                           ly, f, grad)
            else:
                r = rows[parent[n] * 9 + 8 - o]
                if r >= 0:
                    out[n * 9 + o] = _value_at(r, o, px[n] - 2 * cx * h, py[n] - 2 * cy * h, px,
                                               py, side, lx, ly, f, grad)
                else:
                    # Bounced at the upstream point, in the coarser cell.
                    v = _value_at(q, 8 - o, px[n] - cx * h, py[n] - cy * h, px, py, side, lx, ly,
                                  f, grad)
                    if cy == -1 and height > 0.0 and py[n] + h > height:
                        v = v + 6.0 * _weight(cx, cy) * cx * lid
                    out[n * 9 + o] = v


@tack.func
def _usable(q, level, role, at):
    # A neighbour to difference with: a node at the level that holds values.
    use = 0
    if q >= 0:
        if level[q] == at and role[q] < 3:
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
def _gradients(nodes, rows, level, role, at, f, grad):
    # Each direction's change across a cell, central where both neighbours at
    # the level hold values, one-sided where one does.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        west = rows[n * 9 + 3]
        east = rows[n * 9 + 5]
        south = rows[n * 9 + 1]
        north = rows[n * 9 + 7]
        uw = _usable(west, level, role, at)
        ue = _usable(east, level, role, at)
        us = _usable(south, level, role, at)
        un = _usable(north, level, role, at)
        for o in range(9):
            c = f[n * 9 + o]
            grad[n * 18 + o] = _slope(c, f[max(west, 0) * 9 + o], f[max(east, 0) * 9 + o], uw, ue)
            grad[n * 18 + 9 + o] = _slope(c, f[max(south, 0) * 9 + o], f[max(north, 0) * 9 + o],
                                          us, un)


@tack.kernel
def _half_step(nodes, f, out, factor):
    # An interface cell's non-equilibrium part before collision, times
    # ``factor`` (1 / 4 tau), into its ``out`` row (free until the stream):
    # half a fine step's stress, for the explosion.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        rho = f[n * 9]
        mx = -f[n * 9]
        my = -f[n * 9]
        for o in range(1, 9):
            v = f[n * 9 + o]
            rho = rho + v
            mx = mx + v * (o % 3 - 1)
            my = my + v * (o // 3 - 1)
        ux = mx / rho
        uy = my / rho
        usq = ux * ux + uy * uy
        for o in range(9):
            cx = o % 3 - 1
            cy = o // 3 - 1
            cu = 3.0 * (cx * ux + cy * uy)
            feq = _weight(cx, cy) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
            out[n * 9 + o] = factor * (f[n * 9 + o] - feq)


@tack.kernel
def _coalesce(nodes, number, table, f, out):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        r = (number[n] - 1) * 4
        for o in range(9):
            out[n * 9 + o] = 0.25 * (f[table[r] * 9 + o] + f[table[r + 1] * 9 + o] +
                                     f[table[r + 2] * 9 + o] + f[table[r + 3] * 9 + o])


@tack.kernel
def _explode(nodes, number, table, f, grad, half):
    # An interface cell's values into its ghosts, linear in its gradient, and
    # time-centred: a ghost on the downstream side of a direction (whose packet
    # reaches the fine cells a substep early) gains half a fine step's stress
    # (``half``, from ``_half_step``), one on the upstream side loses it. The
    # four shares of each cancel, so each direction's total is the cell's.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        r = (number[n] - 1) * 4
        for j in range(4):
            g = table[r + j]
            sx = 2 * (j & 1) - 1
            sy = 2 * (j >> 1) - 1
            for o in range(9):
                along = sx * (o % 3 - 1) + sy * (o // 3 - 1)
                lead = 0
                if along > 0:
                    lead = 1
                elif along < 0:
                    lead = -1
                f[g * 9 + o] = (f[n * 9 + o] + 0.25 * sx * grad[n * 18 + o] +
                                0.25 * sy * grad[n * 18 + 9 + o] + lead * half[n * 9 + o])


@tack.kernel
def _store(nodes, out, f):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        for o in range(9):
            f[n * 9 + o] = out[n * 9 + o]


@tack.kernel
def _parents_of_ghosts(nodes, number, table, parent):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        r = (number[n] - 1) * 4
        for j in range(4):
            parent[table[r + j]] = tack.i32(n)


@tack.kernel
def _moments(f, rho, ux, uy):
    for n in range(rho.shape[0]):
        r = f[n * 9]
        mx = -f[n * 9]
        my = -f[n * 9]
        for o in range(1, 9):
            v = f[n * 9 + o]
            r = r + v
            mx = mx + v * (o % 3 - 1)
            my = my + v * (o // 3 - 1)
        rho[n] = r
        ux[n] = mx / r
        uy[n] = my / r


# -- the solver ---------------------------------------------------------------


class LBM2D:
    """D2Q9 BGK on ``octree`` (from ``lbm_mesh``) with relaxation time ``tau`` at
    level 0, a lid moving at ``lid`` (lattice units) on the top wall, and, with
    ``inflow`` (a speed along x), an inlet on the left side and an outlet on
    the right. One ``step`` is one step of level 0: ``2^l`` of level ``l``.
    Values are in lattice units of level 0: a root cell is 1 and so is its time
    step. ``centred`` and ``linear`` switch the explosion's corrections off,
    for comparison."""

    def __init__(self, octree, tau, lid=0.0, inflow=None, centred=True, linear=True):
        self.octree = octree
        self.lid = float(lid)
        self.inflow = 0 if inflow is None else 1
        self.speed = 0.0 if inflow is None else float(inflow)
        self.finest = octree.finest
        device = octree.device
        self.dtype = _compute_type(device, tack.f32)
        role = np.asarray(octree.fields["role"].numpy(), np.int64)
        level = octree._level
        self.role, self.levels = role, level
        self.tau = float(tau)
        self.taus = [0.5 + (1 << lv) * (tau - 0.5) for lv in range(self.finest + 1)]
        self.omega = [1.0 / t for t in self.taus]
        self.centred = centred
        self.linear = linear
        self.rows = _rows(octree, getattr(octree, "periodic", (False, False)))
        self.number, self.table = children(octree)
        n = octree.num_nodes
        periodic = getattr(octree, "periodic", (False, False))
        dims = [d - 1 for d in octree.roots.topology.point_dims]
        side0 = float(_root_cells_side(octree))
        self.lengths = [dims[a] * side0 if periodic[a] else 0.0 for a in range(2)]
        self.walls = [0.0 if periodic[a] else dims[a] * side0 for a in range(2)]
        # The cells that need gradients: computational cells with an interface
        # cell among their neighbours at their level (or one themselves).
        rows = self.rows.numpy().reshape(-1, 9)
        near = np.isin(role, (INTERFACE,))
        near_any = near | (np.where(rows >= 0, near[np.maximum(rows, 0)], False)).any(axis=1)
        centre, side = self.centres()

        def nodes(lv, *roles):
            return tack.array(np.flatnonzero((level == lv) & np.isin(role, roles)).astype(np.int64))

        with device:
            self.real = [nodes(lv, REAL) for lv in range(self.finest + 1)]
            self.interface = [nodes(lv, INTERFACE) for lv in range(self.finest + 1)]
            self.ghosts = [nodes(lv, GHOST) for lv in range(self.finest + 1)]
            self.computed = [nodes(lv, REAL, INTERFACE) for lv in range(self.finest + 1)]
            self.border = [tack.array(np.flatnonzero((level == lv) & np.isin(role, (REAL, INTERFACE))
                                                     & near_any).astype(np.int64))
                           for lv in range(self.finest + 1)]
            self.role_array = tack.array(role.astype(np.int32), tack.i32)
            self.px = tack.array(centre[:, 0].astype(self.dtype.numpy_dtype), self.dtype)
            self.py = tack.array(centre[:, 1].astype(self.dtype.numpy_dtype), self.dtype)
            self.side = tack.array(side.astype(self.dtype.numpy_dtype), self.dtype)
            self.grad = tack.zeros((n * 18,), self.dtype)
            self.parent = tack.full((n,), -1, tack.i32)
            for lv in range(self.finest + 1):
                if len(self.interface[lv]):
                    _parents_of_ghosts(self.interface[lv], self.number, self.table, self.parent)
            self.f = tack.zeros((n * 9,), self.dtype)
            self.out = tack.zeros((n * 9,), self.dtype)
        self.steps = 0

    # -- state ----------------------------------------------------------------

    def centres(self):
        """Each node's centre and side, host arrays ``(n, 2)`` and ``(n,)``."""
        lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in
                    self.octree.geometry())
        return lo[:, :2] + 0.5 * size[:, :2], size[:, 0]

    def initialize(self, rho, ux, uy):
        """Every node at equilibrium for its ``rho``, ``ux`` and ``uy`` (one value
        a node, or scalars), ghosts included."""
        n = self.octree.num_nodes
        rho, ux, uy = (np.broadcast_to(np.asarray(v, np.float64), (n,)) for v in (rho, ux, uy))
        o = np.arange(9)
        cx, cy = o % 3 - 1, o // 3 - 1
        w = np.where((cx == 0) & (cy == 0), 4 / 9, np.where((cx == 0) | (cy == 0), 1 / 9, 1 / 36))
        cu = 3.0 * (cx[None] * ux[:, None] + cy[None] * uy[:, None])
        feq = w[None] * rho[:, None] * (1 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy)[:, None])
        with self.octree.device:
            self.f = tack.array(feq.reshape(-1).astype(self.dtype.numpy_dtype), self.dtype)
        self.steps = 0

    def moments(self):
        """Density and velocity at every node (host arrays; meaningful at the
        computational cells, ``computational()``)."""
        n = self.octree.num_nodes
        with self.octree.device:
            rho, ux, uy = (tack.empty((n,), self.dtype) for _ in range(3))
            _moments(self.f, rho, ux, uy)
        return rho.numpy(), ux.numpy(), uy.numpy()

    def computational(self):
        """Which nodes are cells of the solution: real leaves and interface cells."""
        return np.isin(self.role, (REAL, INTERFACE))

    def mass(self):
        """The total mass: density times area over the computational cells."""
        rho = self.moments()[0]
        keep = self.computational()
        return float(np.sum(rho[keep] * 0.25 ** self.levels[keep]))

    def momentum(self):
        rho, ux, uy = self.moments()
        keep = self.computational()
        a = 0.25 ** self.levels[keep]
        return float(np.sum(rho[keep] * ux[keep] * a)), float(np.sum(rho[keep] * uy[keep] * a))

    def updates_per_step(self):
        """Cell updates in one step of level 0: each level's computational cells
        times its substeps."""
        return sum(len(c) << lv for lv, c in enumerate(self.computed))

    # -- stepping -------------------------------------------------------------

    def step(self, count=1):
        with self.octree.device:
            for _ in range(count):
                self._step(0, 0)
        self.steps += count

    def _step(self, lv, second):
        f, out = self.f, self.out
        if len(self.interface[lv]):
            if self.centred:
                _half_step(self.interface[lv], f, out, 0.25 / self.taus[lv])
            else:
                out.fill(0.0)
        if len(self.computed[lv]):
            _collide(self.computed[lv], f, self.omega[lv])
        if self.linear and len(self.border[lv]):
            _gradients(self.border[lv], self.rows, self.octree.level, self.role_array, lv, f,
                       self.grad)
        if lv < self.finest:
            if len(self.interface[lv]):
                _explode(self.interface[lv], self.number, self.table, f, self.grad, out)
            self._step(lv + 1, 0)
            self._step(lv + 1, 1)
        finest = int(lv == self.finest)
        if len(self.real[lv]):
            _pull(self.real[lv], self.rows, self.role_array, finest, self.px, self.py, self.side,
                  self.lengths[0], self.lengths[1], self.walls[0], self.walls[1], f, self.grad, out,
                  self.lid, self.inflow, self.speed)
        if len(self.ghosts[lv]):
            _pull_ghosts(self.ghosts[lv], self.parent, self.rows, self.octree.level,
                         self.role_array, lv, finest, second, self.px, self.py, self.side,
                         self.lengths[0], self.lengths[1], self.walls[0], self.walls[1], f,
                         self.grad, out, self.lid, self.inflow, self.speed)
        if len(self.interface[lv]):
            _coalesce(self.interface[lv], self.number, self.table, f, out)
        for nodes in (self.real[lv], self.ghosts[lv], self.interface[lv]):
            if len(nodes):
                _store(nodes, out, f)
