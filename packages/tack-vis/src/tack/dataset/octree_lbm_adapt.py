# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive refinement and coarsening for the lattice Boltzmann prototype
(study): ``adapt(solver, ...)`` returns a solver on a mesh fitted to the flow,
with the solution moved over conservatively.

**The logical tree** is the octree without its ghosts: interface cells are its
leaves. A remesh changes each leaf by at most one level:

1. *Mark* (``_vorticity``, ``_mark``): a leaf refines where ``|vorticity| h``
   (a speed: how much the velocity turns across the cell) is above
   ``refine_above``; it votes to merge with its siblings where twice that, its
   parent's, is below ``coarsen_below``. ``pinned`` cells (around an obstacle)
   go to the finest level and stay; refinement stays inside ``region``.
2. *Balance* (``_ripple``, repeated until nothing changes): a leaf that refines
   makes a coarser neighbour refine too, and keeps a neighbour at its level from
   merging, so no leaf ends up beside one two levels finer, across faces or
   corners; a leaf with five of its eight neighbours refined or refining
   refines too, so no coarse holes are left (each an interface cell, with
   ghosts and a one-sided view of its neighbours).
3. *Merge* (``_merges``): a refined node whose four children are leaves voting
   to merge, with no refined or refining neighbour at their level, becomes a
   leaf.
4. *The new tree by a scan* (``_counts``, ``_place``): each node's output count
   (0 for a ghost or a merged child, 5 for a leaf that refines: itself and its
   children right after it, 1 otherwise), an exclusive scan, a scatter. The
   pre-order survives: children follow their parent, merged ones drop out.
5. *The interface layer* on the new tree (``_interfaces``), and its ghosts by a
   second scan.
6. *The solution* (``_transfer``): a kept cell keeps its values; a new child
   takes its parent's, linear in the parent's gradient (the four shares
   cancel); a merged cell takes its children's mean. Each then has its
   non-equilibrium part rescaled to its level (``tau_f / 2 tau_c`` going down,
   its inverse going up): cell by cell, so mass and momentum are conserved
   to rounding.

All of it runs on the device; the octree's constructor and the solver's lists
are host NumPy, as for ``lbm_mesh``.
"""

import time

import numpy as np

import tack
from tack.algorithms import exclusive_scan
from tack.dataset import octree_lbm as lb
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet
from tack.dataset.octree_lbm import _feq, _gradients, _moments_at, _slope

__all__ = ["adapt"]


@tack.func
def _beside(r, at, level, role, number, table, q, dim, f):
    # A neighbour's velocity at its centre and whether it has one: a node at
    # the level holding values, or a refined one, by its children's mean.
    ux = f[0] * 0.0
    uy = ux
    uz = ux
    ok = 0
    if r >= 0:
        if level[r] == at and role[r] < 3:
            rho, ux, uy, uz = _moments_at(f, r * q, q, dim)
            ok = 1
        elif level[r] == at and role[r] == 3:
            kids = 1 << dim
            first = (number[r] - 1) * kids
            ok = 1
            for j in range(kids):
                if role[table[first + j]] > 1:
                    ok = 0
            if ok == 1:
                for j in range(kids):
                    rho, cx, cy, cz = _moments_at(f, table[first + j] * q, q, dim)
                    ux = ux + cx / kids
                    uy = uy + cy / kids
                    uz = uz + cz / kids
    return ux, uy, uz, ok


@tack.kernel
def _wanted_refined(nodes, want, out):
    # 1 where a leaf was marked to refine: `adapt`'s count of refined leaves, by
    # a sum on the device rather than a read of every node's mark.
    for k in range(nodes.shape[0]):
        out[k] = 1 if want[nodes[k]] == 1 else 0


@tack.kernel
def _vorticity(nodes, rows, level, role, number, table, side, lat, f, out):
    # |vorticity| at a computational cell, by differences with its neighbours
    # at its level (a refined one by its children's mean).
    q = lat.Q
    dim = lat.DIM
    for k in range(nodes.shape[0]):
        n = nodes[k]
        at = level[n]
        centre = (q - 1) // 2
        rho, ux, uy, uz = _moments_at(f, n * q, q, dim)
        ax, ay, az, ua = _beside(rows[n * q + centre - 1], at, level, role, number, table, q, dim, f)
        bx, by, bz, ub = _beside(rows[n * q + centre + 1], at, level, role, number, table, q, dim, f)
        cx, cy, cz, uc = _beside(rows[n * q + centre - 3], at, level, role, number, table, q, dim, f)
        dx, dy, dz, ud = _beside(rows[n * q + centre + 3], at, level, role, number, table, q, dim, f)
        wz = _slope(uy, ay, by, ua, ub) - _slope(ux, cx, dx, uc, ud)
        w2 = wz * wz
        if dim == 3:
            ex, ey, ez, ue = _beside(rows[n * q + centre - 9], at, level, role, number, table, q,
                                     dim, f)
            gx, gy, gz, ug = _beside(rows[n * q + centre + 9], at, level, role, number, table, q,
                                     dim, f)
            wx = _slope(uz, cz, dz, uc, ud) - _slope(uy, ey, gy, ue, ug)
            wy = _slope(ux, ex, gx, ue, ug) - _slope(uz, az, bz, ua, ub)
            w2 = w2 + wx * wx + wy * wy
        out[n] = tack.sqrt(w2) / side[n]


@tack.kernel
def _mark(nodes, measure, level, side, px, pinned, finest, refine_above, coarsen_below, xlo, xhi,
          want):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        m = measure[n] * side[n]
        w = 0
        if pinned[n] == 1:
            if level[n] < finest:
                w = 1
        elif m > refine_above and level[n] < finest and px[n] > xlo and px[n] < xhi:
            w = 1
        elif 2.0 * m < coarsen_below and level[n] > 0:
            w = -1
        want[n] = w


@tack.func
def _holder(r, role, parent):
    # The logical node at a row entry: a ghost stands for its parent.
    h = r
    if role[r] == 2:
        h = parent[r]
    return h


@tack.kernel
def _ripple(nodes, rows, role, level, parent, finest, holes, q, want, changed):
    # A leaf that refines: a coarser neighbour refines too, and a neighbour at
    # its level may not merge. A leaf with ``holes`` or more of its neighbours
    # at its level refined or refining refines too: no holes.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        centre = (q - 1) // 2
        if want[n] != 1 and level[n] < finest:
            around = 0
            for o in range(q):
                r = rows[n * q + o]
                if o != centre and r >= 0:
                    h = _holder(r, role, parent)
                    if level[h] == level[n] and (role[h] == 3 or want[h] == 1):
                        around = around + 1
            if around >= holes:
                want[n] = 1
                tack.atomic_add(changed, 0, 1)
        if want[n] == 1:
            at = level[n]
            for o in range(q):
                r = rows[n * q + o]
                if o != centre and r >= 0:
                    h = _holder(r, role, parent)
                    if level[h] < at and role[h] < 2 and want[h] != 1:
                        want[h] = 1
                        tack.atomic_add(changed, 0, 1)
                    elif level[h] == at and want[h] == -1:
                        want[h] = 0
                        tack.atomic_add(changed, 0, 1)


@tack.kernel
def _merges(nodes, number, table, rows, role, level, parent, want, q, dim, merge):
    # A refined node merges when its children are leaves voting to, and none
    # has a refined or refining neighbour at its level.
    for k in range(nodes.shape[0]):
        r = nodes[k]
        kids = 1 << dim
        first = (number[r] - 1) * kids
        centre = (q - 1) // 2
        ok = 1
        for j in range(kids):
            c = table[first + j]
            if role[c] > 1 or want[c] != -1:
                ok = 0
        if ok == 1:
            for j in range(kids):
                c = table[first + j]
                at = level[c]
                for o in range(q):
                    s = rows[c * q + o]
                    if o != centre and s >= 0:
                        h = _holder(s, role, parent)
                        if level[h] == at and (role[h] == 3 or want[h] == 1):
                            ok = 0
        merge[r] = ok


@tack.kernel
def _parents(nodes, number, table, dim, up):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        kids = 1 << dim
        first = (number[n] - 1) * kids
        for j in range(kids):
            up[table[first + j]] = tack.i32(n)


@tack.kernel
def _counts(role, want, merge, up, dim, count):
    for n in range(count.shape[0]):
        c = 1
        if role[n] == 2:
            c = 0
        elif up[n] >= 0 and merge[up[n]] == 1:
            c = 0
        elif want[n] == 1 and role[n] < 2:
            c = 1 + (1 << dim)
        count[n] = c


@tack.kernel
def _place(count, start, tree, level, key, role, merge, dim, out_tree, out_level, out_key, solid,
           src, how):
    # The logical tree's nodes: a kept node (``how`` 0), a merged one (-1),
    # and the children of one that refines (1 + octant).
    for n in range(count.shape[0]):
        c = count[n]
        if c > 0:
            kids = 1 << dim
            p = start[n]
            out_tree[p] = tree[n]
            out_level[p] = level[n]
            out_key[p] = key[n]
            src[p] = tack.i32(n)
            h = 0
            if role[n] == 3 and merge[n] == 1:
                h = -1
            how[p] = h
            s = 0
            if role[n] == 4:
                s = 1
            solid[p] = s
            if c > 1:
                for j in range(kids):
                    t = p + 1 + j
                    out_tree[t] = tree[n]
                    out_level[t] = level[n] + 1
                    out_key[t] = key[n] * tack.u64(kids) + tack.u64(j)
                    src[t] = tack.i32(n)
                    how[t] = 1 + j
                    solid[t] = 0


@tack.kernel
def _interfaces(rows, marks, refined, solid, q, dim, count):
    # A leaf beside a refined node at its level is an interface cell: it and
    # its 2^d ghosts.
    for n in range(count.shape[0]):
        c = 1
        centre = (q - 1) // 2
        if (marks[n] & refined) == 0 and solid[n] == 0:
            for o in range(q):
                r = rows[n * q + o]
                if o != centre and r >= 0:
                    if (marks[r] & refined) != 0:
                        c = 1 + (1 << dim)
        count[n] = c


@tack.kernel
def _place_ghosts(count, start, tree, level, key, marks, refined, solid, dim, out_tree, out_level,
                  out_key, role, logical):
    for n in range(count.shape[0]):
        kids = 1 << dim
        p = start[n]
        out_tree[p] = tree[n]
        out_level[p] = level[n]
        out_key[p] = key[n]
        logical[p] = tack.i32(n)
        r = 0
        if solid[n] == 1:
            r = 4
        elif (marks[n] & refined) != 0:
            r = 3
        elif count[n] > 1:
            r = 1
        role[p] = r
        if count[n] > 1:
            for j in range(kids):
                t = p + 1 + j
                out_tree[t] = tree[n]
                out_level[t] = level[n] + 1
                out_key[t] = key[n] * tack.u64(kids) + tack.u64(j)
                role[t] = 2
                logical[t] = tack.i32(n)


@tack.func
def _rescaled(o, f, at, scale, q, dim):
    # Value o of the q at ``at`` with its non-equilibrium part times scale.
    rho, ux, uy, uz = _moments_at(f, at, q, dim)
    e = _feq(o, rho, ux, uy, uz, dim)
    return e + scale * (f[at + o] - e)


@tack.kernel
def _transfer(role, logical, src, how, old_f, old_grad, old_number, old_table, old_level, down,
              up, q, dim, scratch, f):
    # Each new cell's values from the old mesh; ``scratch`` holds them before
    # the rescaling, q a node.
    for m in range(role.shape[0]):
        r = role[m]
        if r != 3:
            g = logical[m]
            n = src[g]
            h = how[g]
            kids = 1 << dim
            if r == 2:
                h = 0                        # a ghost: exploded from its parent below
            if h == 0:
                for o in range(q):
                    f[m * q + o] = old_f[n * q + o]
            elif h < 0:
                first = (old_number[n] - 1) * kids
                for o in range(q):
                    s = old_f[old_table[first] * q + o]
                    for j in range(1, kids):
                        s = s + old_f[old_table[first + j] * q + o]
                    scratch[m * q + o] = s / kids
                for o in range(q):
                    f[m * q + o] = _rescaled(o, scratch, m * q, up[old_level[n]], q, dim)
            else:
                j = h - 1
                sx = 2 * (j & 1) - 1
                sy = 2 * ((j >> 1) & 1) - 1
                sz = 2 * ((j >> 2) & 1) - 1
                g0 = n * q * dim
                for o in range(q):
                    v = (old_f[n * q + o] + 0.25 * sx * old_grad[g0 + o] +
                         0.25 * sy * old_grad[g0 + q + o])
                    if dim == 3:
                        v = v + 0.25 * sz * old_grad[g0 + 2 * q + o]
                    scratch[m * q + o] = v
                for o in range(q):
                    f[m * q + o] = _rescaled(o, scratch, m * q, down[old_level[n]], q, dim)


def adapt(solver, refine_above, coarsen_below, finest=None, pinned=None, region=None, solid=None):
    """A new solver on ``solver``'s mesh adapted to the flow, with its solution
    moved over (call between steps). A leaf refines where ``|vorticity| h``
    (lattice units of level 0) is above ``refine_above`` and its siblings and it
    merge where twice that is below ``coarsen_below``; ``finest`` caps the
    levels (the current finest by default); ``pinned(x, y[, z])`` (vectorized)
    picks cells that go to the finest level and stay; refinement keeps inside
    ``region = (xlo, xhi)``; ``solid(x, y[, z])`` (vectorized) makes the real
    leaves at the finest level whose centres it holds ``SOLID`` (any other
    leaf it holds is an error: pin around the obstacle first). Returns the new
    solver and a dict of what changed."""
    begin = time.perf_counter()
    octree = solver.octree
    d, q = solver.dim, solver.q
    finest = octree.finest if finest is None else int(finest)
    role, level = solver.role, solver.levels
    n = octree.num_nodes
    centre, _ = solver.centres()
    xlo, xhi = region if region is not None else (-np.inf, np.inf)
    computational = solver.computational()
    leaves = np.flatnonzero(computational).astype(np.int64)
    refined_nodes = np.flatnonzero(role == lb.REFINED).astype(np.int64)
    parents_of = np.flatnonzero((role == lb.REFINED) | (role == lb.INTERFACE)).astype(np.int64)
    taus = [0.5 + (1 << lv) * (solver.tau - 0.5) for lv in range(finest + 2)]
    holes = 5 if d == 2 else 14
    device = octree.device
    with device:
        dtype = solver.dtype
        leaves_d = tack.array(leaves)
        measure = tack.zeros((n,), dtype)
        if len(leaves):
            # Every level's leaves at once: each node's level is its own.
            _vorticity(leaves_d, solver.rows, octree.level, solver.role_array, solver.number,
                       solver.table, solver.side, solver.lattice, solver.f, measure)
            _gradients(leaves_d, solver.rows, octree.level, solver.role_array, -1,
                       solver.lattice, solver.f, solver.grad)
        pin = np.zeros(n, np.int32)
        if pinned is not None:
            pin[np.asarray(pinned(*centre.T), bool)] = 1
        want = tack.zeros((n,), tack.i32)
        if len(leaves):
            _mark(leaves_d, measure, octree.level, solver.side, solver.px, tack.array(pin),
                  finest, float(refine_above), float(coarsen_below), float(xlo), float(xhi), want)
        changed = tack.zeros((1,), tack.i32)
        rounds = 0
        while len(leaves):
            changed.fill(0)
            _ripple(leaves_d, solver.rows, solver.role_array, octree.level, solver.parent, finest,
                    holes, q, want, changed)
            rounds += 1
            if int(changed[0]) == 0:
                break
        merge = tack.zeros((n,), tack.i32)
        if len(refined_nodes):
            _merges(tack.array(refined_nodes), solver.number, solver.table, solver.rows,
                    solver.role_array, octree.level, solver.parent, want, q, d, merge)
        up = tack.full((n,), -1, tack.i32)
        if len(parents_of):
            _parents(tack.array(parents_of), solver.number, solver.table, d, up)
        count = tack.empty((n,), tack.i64)
        _counts(solver.role_array, want, merge, up, d, count)
        start = tack.empty((n,), tack.i64)
        total = int(exclusive_scan(count, start, n))
        tree, lev, key = (tack.empty((total,), t) for t in (tack.i64, tack.i32, tack.u64))
        solid_flag, src, how = (tack.empty((total,), tack.i32) for _ in range(3))
        _place(count, start, octree.tree, octree.level, octree.key, solver.role_array, merge, d,
               tree, lev, key, solid_flag, src, how)
    logical = OctreeDataSet(octree.roots, tree, lev, key, _ordered=True, _adopt=True)
    periodic = getattr(octree, "periodic", (False,) * d)
    rows = lb._rows(logical, periodic)
    refined_mark = int(Ghosts.REFINEDCELL)
    with device:
        count2 = tack.empty((total,), tack.i64)
        _interfaces(rows, logical.marks, refined_mark, solid_flag, q, d, count2)
        start2 = tack.empty((total,), tack.i64)
        final = int(exclusive_scan(count2, start2, total))
        tree2, lev2, key2 = (tack.empty((final,), t) for t in (tack.i64, tack.i32, tack.u64))
        role2, mapped = (tack.empty((final,), tack.i32) for _ in range(2))
        _place_ghosts(count2, start2, tree, lev, key, logical.marks, refined_mark, solid_flag, d,
                      tree2, lev2, key2, role2, mapped)
    mesh = OctreeDataSet(octree.roots, tree2, lev2, key2, fields={"role": role2}, _ordered=True,
                         _adopt=True)
    mesh.periodic = periodic
    if solid is not None:
        roles = role2.numpy()
        lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in mesh.geometry())
        inside = np.asarray(solid(*(lo[:, :d] + 0.5 * size[:, :d]).T), bool) & (roles != lb.REFINED)
        if (inside & (roles != lb.SOLID) & ((roles != lb.REAL) | (mesh._level != mesh.finest))).any():
            raise ValueError("solid cells must be real leaves at the finest level: pin around the "
                             "obstacle first")
        roles = np.where(inside, lb.SOLID, roles).astype(np.int32)
        with device:
            mesh.fields["role"] = tack.array(roles, tack.i32)
    new = lb.LBM(mesh, solver.tau, lid=solver.lid, inflow=solver.speed if solver.inflow else None,
                 centred=solver.centred, linear=solver.linear, regularized=solver.regularized,
                 sponge=solver.sponge, dtype=solver.dtype)
    with device:
        down = tack.array(np.array([taus[lv + 1] / (2 * taus[lv]) for lv in range(finest + 1)],
                                   dtype.numpy_dtype))
        upward = tack.array(np.array([2 * taus[lv] / taus[lv + 1] for lv in range(finest + 1)],
                                     dtype.numpy_dtype))
        scratch = tack.empty((final * q,), dtype)
        _transfer(role2, mapped, src, how, solver.f, solver.grad, solver.number, solver.table,
                  octree.level, down, upward, q, d, scratch, new.f)
    # Ghosts from their parents, as the next step's explosion would (the solver
    # reads none before that, but whoever looks at the state does).
    new.fill_ghosts()
    new.steps = solver.steps
    # The counts by sums on the device, not reads of every node's marks; the
    # computational cells are the new solver's computed runs (real and interface).
    refined = merged = 0
    with device:
        if len(leaves):
            marked = tack.empty((len(leaves),), tack.i32)
            _wanted_refined(leaves_d, want, marked)
            refined = int(marked.sum())
        if len(refined_nodes):
            merged = int(merge.sum())
    stats = {"refined": refined, "merged": merged,
             "cells": sum(len(nodes) for nodes in new.computed),
             "rounds": rounds, "ms": 1000 * (time.perf_counter() - begin)}
    return new, stats
