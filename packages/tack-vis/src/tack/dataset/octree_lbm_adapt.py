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
from tack.dataset.octree_lbm import _gradients, _slope, _weight

__all__ = ["adapt"]


@tack.func
def _velocity(q, f):
    rho = f[q * 9]
    mx = -f[q * 9]
    my = -f[q * 9]
    for o in range(1, 9):
        v = f[q * 9 + o]
        rho = rho + v
        mx = mx + v * (o % 3 - 1)
        my = my + v * (o // 3 - 1)
    return mx / rho, my / rho


@tack.func
def _beside(q, at, level, role, number, table, f):
    # A neighbour's velocity at its centre and whether it has one: a node at
    # the level holding values, or a refined one, by its children's mean.
    ux = f[0] * 0.0
    uy = ux
    ok = 0
    if q >= 0:
        if level[q] == at and role[q] < 3:
            ux, uy = _velocity(q, f)
            ok = 1
        elif level[q] == at and role[q] == 3:
            first = (number[q] - 1) * 4
            ok = 1
            for j in range(4):
                c = table[first + j]
                if role[c] > 1:
                    ok = 0
            if ok == 1:
                for j in range(4):
                    cx, cy = _velocity(table[first + j], f)
                    ux = ux + 0.25 * cx
                    uy = uy + 0.25 * cy
    return ux, uy, ok


@tack.kernel
def _vorticity(nodes, rows, level, role, number, table, side, f, out):
    # |vorticity| at a computational cell, by differences with its neighbours
    # at its level (a refined one by its children's mean).
    for k in range(nodes.shape[0]):
        n = nodes[k]
        at = level[n]
        ux, uy = _velocity(n, f)
        wx, wy, uw = _beside(rows[n * 9 + 3], at, level, role, number, table, f)
        ex, ey, ue = _beside(rows[n * 9 + 5], at, level, role, number, table, f)
        sx, sy, us = _beside(rows[n * 9 + 1], at, level, role, number, table, f)
        nx, ny, un = _beside(rows[n * 9 + 7], at, level, role, number, table, f)
        dxuy = _slope(uy, wy, ey, uw, ue)
        dyux = _slope(ux, sx, nx, us, un)
        out[n] = abs(dxuy - dyux) / side[n]


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
def _holder(q, role, parent):
    # The logical node at a row entry: a ghost stands for its parent.
    h = q
    if role[q] == 2:
        h = parent[q]
    return h


@tack.kernel
def _ripple(nodes, rows, role, level, parent, finest, want, changed):
    # A leaf that refines: a coarser neighbour refines too, and a neighbour at
    # its level may not merge. A leaf with five or more of its eight
    # neighbours at its level refined or refining refines too: no holes.
    for k in range(nodes.shape[0]):
        n = nodes[k]
        if want[n] != 1 and level[n] < finest:
            around = 0
            for o in range(9):
                q = rows[n * 9 + o]
                if o != 4 and q >= 0:
                    h = _holder(q, role, parent)
                    if level[h] == level[n] and (role[h] == 3 or want[h] == 1):
                        around = around + 1
            if around >= 5:
                want[n] = 1
                tack.atomic_add(changed, 0, 1)
        if want[n] == 1:
            at = level[n]
            for o in range(9):
                q = rows[n * 9 + o]
                if o != 4 and q >= 0:
                    h = _holder(q, role, parent)
                    if level[h] < at and role[h] < 2 and want[h] != 1:
                        want[h] = 1
                        tack.atomic_add(changed, 0, 1)
                    elif level[h] == at and want[h] == -1:
                        want[h] = 0
                        tack.atomic_add(changed, 0, 1)


@tack.kernel
def _merges(nodes, number, table, rows, role, level, parent, want, merge):
    # A refined node merges when its four children are leaves voting to, and
    # none has a refined or refining neighbour at its level.
    for k in range(nodes.shape[0]):
        r = nodes[k]
        first = (number[r] - 1) * 4
        ok = 1
        for j in range(4):
            c = table[first + j]
            if role[c] > 1 or want[c] != -1:
                ok = 0
        if ok == 1:
            for j in range(4):
                c = table[first + j]
                at = level[c]
                for o in range(9):
                    q = rows[c * 9 + o]
                    if o != 4 and q >= 0:
                        h = _holder(q, role, parent)
                        if level[h] == at and (role[h] == 3 or want[h] == 1):
                            ok = 0
        merge[r] = ok


@tack.kernel
def _parents(nodes, number, table, up):
    for k in range(nodes.shape[0]):
        n = nodes[k]
        first = (number[n] - 1) * 4
        for j in range(4):
            up[table[first + j]] = tack.i32(n)


@tack.kernel
def _counts(role, want, merge, up, count):
    for n in range(count.shape[0]):
        c = 1
        if role[n] == 2:
            c = 0
        elif up[n] >= 0 and merge[up[n]] == 1:
            c = 0
        elif want[n] == 1 and role[n] < 2:
            c = 5
        count[n] = c


@tack.kernel
def _place(count, start, tree, level, key, role, merge, out_tree, out_level, out_key, solid, src,
           how):
    # The logical tree's nodes: a kept node (``how`` 0), a merged one (5), and
    # the children of one that refines (1 + octant).
    for n in range(count.shape[0]):
        c = count[n]
        if c > 0:
            p = start[n]
            out_tree[p] = tree[n]
            out_level[p] = level[n]
            out_key[p] = key[n]
            src[p] = tack.i32(n)
            h = 0
            if role[n] == 3 and merge[n] == 1:
                h = 5
            how[p] = h
            s = 0
            if role[n] == 4:
                s = 1
            solid[p] = s
            if c == 5:
                for j in range(4):
                    q = p + 1 + j
                    out_tree[q] = tree[n]
                    out_level[q] = level[n] + 1
                    out_key[q] = key[n] * tack.u64(4) + tack.u64(j)
                    src[q] = tack.i32(n)
                    how[q] = 1 + j
                    solid[q] = 0


@tack.kernel
def _interfaces(rows, marks, refined, solid, count):
    # A leaf beside a refined node at its level is an interface cell: it and
    # its four ghosts.
    for n in range(count.shape[0]):
        c = 1
        if (marks[n] & refined) == 0 and solid[n] == 0:
            for o in range(9):
                q = rows[n * 9 + o]
                if o != 4 and q >= 0:
                    if (marks[q] & refined) != 0:
                        c = 5
        count[n] = c


@tack.kernel
def _place_ghosts(count, start, tree, level, key, marks, refined, solid, out_tree, out_level,
                  out_key, role, logical):
    for n in range(count.shape[0]):
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
        elif count[n] == 5:
            r = 1
        role[p] = r
        if count[n] == 5:
            for j in range(4):
                q = p + 1 + j
                out_tree[q] = tree[n]
                out_level[q] = level[n] + 1
                out_key[q] = key[n] * tack.u64(4) + tack.u64(j)
                role[q] = 2
                logical[q] = tack.i32(n)


@tack.func
def _rescaled(o, f, at, scale):
    # Value o of the nine at ``at`` with its non-equilibrium part times scale.
    rho = f[at]
    mx = -f[at]
    my = -f[at]
    for d in range(1, 9):
        v = f[at + d]
        rho = rho + v
        mx = mx + v * (d % 3 - 1)
        my = my + v * (d // 3 - 1)
    ux = mx / rho
    uy = my / rho
    cx = o % 3 - 1
    cy = o // 3 - 1
    cu = 3.0 * (cx * ux + cy * uy)
    e = _weight(cx, cy) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy))
    return e + scale * (f[at + o] - e)


@tack.kernel
def _transfer(role, logical, src, how, old_f, old_grad, old_number, old_table, old_level, down,
              up, scratch, f):
    # Each new cell's values from the old mesh; ``scratch`` holds them before
    # the rescaling, nine a node.
    for m in range(role.shape[0]):
        r = role[m]
        if r != 3:
            g = logical[m]
            n = src[g]
            h = how[g]
            if r == 2:
                h = 0                        # a ghost: exploded from its parent below
            if h == 0:
                for o in range(9):
                    f[m * 9 + o] = old_f[n * 9 + o]
            elif h == 5:
                first = (old_number[n] - 1) * 4
                for o in range(9):
                    scratch[m * 9 + o] = 0.25 * (old_f[old_table[first] * 9 + o] +
                                                 old_f[old_table[first + 1] * 9 + o] +
                                                 old_f[old_table[first + 2] * 9 + o] +
                                                 old_f[old_table[first + 3] * 9 + o])
                for o in range(9):
                    f[m * 9 + o] = _rescaled(o, scratch, m * 9, up[old_level[n]])
            else:
                j = h - 1
                sx = 2 * (j & 1) - 1
                sy = 2 * (j >> 1) - 1
                for o in range(9):
                    scratch[m * 9 + o] = (old_f[n * 9 + o] + 0.25 * sx * old_grad[n * 18 + o] +
                                          0.25 * sy * old_grad[n * 18 + 9 + o])
                for o in range(9):
                    f[m * 9 + o] = _rescaled(o, scratch, m * 9, down[old_level[n]])


def adapt(solver, refine_above, coarsen_below, finest=None, pinned=None, region=None):
    """A new ``LBM2D`` on ``solver``'s mesh adapted to the flow, with its solution
    moved over (call between steps). A leaf refines where ``|vorticity| h``
    (lattice units of level 0) is above ``refine_above`` and its siblings and it
    merge where twice that is below ``coarsen_below``; ``finest`` caps the
    levels (the current finest by default); ``pinned(x, y)`` (vectorized)
    picks cells that go to the finest level and stay; refinement keeps inside
    ``region = (xlo, xhi)``. Returns the new solver and a dict of what
    changed."""
    begin = time.perf_counter()
    octree = solver.octree
    finest = octree.finest if finest is None else int(finest)
    role, level = solver.role, solver.levels
    n = octree.num_nodes
    centre, _ = solver.centres()
    xlo, xhi = region if region is not None else (-np.inf, np.inf)
    leaves = np.flatnonzero(np.isin(role, (lb.REAL, lb.INTERFACE))).astype(np.int64)
    refined_nodes = np.flatnonzero(role == lb.REFINED).astype(np.int64)
    parents_of = np.flatnonzero(np.isin(role, (lb.REFINED, lb.INTERFACE))).astype(np.int64)
    taus = [0.5 + (1 << lv) * (solver.tau - 0.5) for lv in range(finest + 2)]
    device = octree.device
    with device:
        dtype = solver.dtype
        leaves_d = tack.array(leaves)
        measure = tack.zeros((n,), dtype)
        for lv in range(octree.finest + 1):
            if len(solver.computed[lv]):
                _vorticity(solver.computed[lv], solver.rows, octree.level, solver.role_array,
                           solver.number, solver.table, solver.side, solver.f, measure)
                _gradients(solver.computed[lv], solver.rows, octree.level, solver.role_array, lv,
                           solver.f, solver.grad)
        pin = np.zeros(n, np.int32)
        if pinned is not None:
            pin[np.asarray(pinned(centre[:, 0], centre[:, 1]), bool)] = 1
        want = tack.zeros((n,), tack.i32)
        if len(leaves):
            _mark(leaves_d, measure, octree.level, solver.side, solver.px, tack.array(pin),
                  finest, float(refine_above), float(coarsen_below), float(xlo), float(xhi), want)
        changed = tack.zeros((1,), tack.i32)
        rounds = 0
        while len(leaves):
            changed.fill(0)
            _ripple(leaves_d, solver.rows, solver.role_array, octree.level, solver.parent, finest,
                    want, changed)
            rounds += 1
            if int(changed[0]) == 0:
                break
        merge = tack.zeros((n,), tack.i32)
        if len(refined_nodes):
            _merges(tack.array(refined_nodes), solver.number, solver.table, solver.rows,
                    solver.role_array, octree.level, solver.parent, want, merge)
        up = tack.full((n,), -1, tack.i32)
        if len(parents_of):
            _parents(tack.array(parents_of), solver.number, solver.table, up)
        count = tack.empty((n,), tack.i64)
        _counts(solver.role_array, want, merge, up, count)
        start = tack.empty((n,), tack.i64)
        total = int(exclusive_scan(count, start, n))
        tree, lev, key = (tack.empty((total,), t) for t in (tack.i64, tack.i32, tack.u64))
        solid, src, how = (tack.empty((total,), tack.i32) for _ in range(3))
        _place(count, start, octree.tree, octree.level, octree.key, solver.role_array, merge,
               tree, lev, key, solid, src, how)
    logical = OctreeDataSet(octree.roots, tree, lev, key, _ordered=True)
    periodic = getattr(octree, "periodic", (False, False))
    rows = lb._rows(logical, periodic)
    refined_mark = int(Ghosts.REFINEDCELL)
    with device:
        count2 = tack.empty((total,), tack.i64)
        _interfaces(rows, logical.marks, refined_mark, solid, count2)
        start2 = tack.empty((total,), tack.i64)
        final = int(exclusive_scan(count2, start2, total))
        tree2, lev2, key2 = (tack.empty((final,), t) for t in (tack.i64, tack.i32, tack.u64))
        role2, mapped = (tack.empty((final,), tack.i32) for _ in range(2))
        _place_ghosts(count2, start2, tree, lev, key, logical.marks, refined_mark, solid, tree2,
                      lev2, key2, role2, mapped)
    mesh = OctreeDataSet(octree.roots, tree2, lev2, key2, fields={"role": role2.numpy()},
                         _ordered=True)
    mesh.periodic = periodic
    new = lb.LBM2D(mesh, solver.tau, lid=solver.lid,
                   inflow=solver.speed if solver.inflow else None, centred=solver.centred,
                   linear=solver.linear)
    with device:
        down = tack.array(np.array([taus[lv + 1] / (2 * taus[lv]) for lv in range(finest + 1)],
                                   dtype.numpy_dtype))
        upward = tack.array(np.array([2 * taus[lv] / taus[lv + 1] for lv in range(finest + 1)],
                                     dtype.numpy_dtype))
        scratch = tack.empty((final * 9,), dtype)
        _transfer(role2, mapped, src, how, solver.f, solver.grad, solver.number, solver.table,
                  octree.level, down, upward, scratch, new.f)
        # Ghosts from their parents, as the next step's explosion would (the
        # solver reads none before that, but whoever looks at the state does).
        for lv in range(new.finest + 1):
            if len(new.interface[lv]):
                lb._explode(new.interface[lv], new.number, new.table, new.f, new.grad, new.out)
    new.steps = solver.steps
    wanted = want.numpy()
    stats = {"refined": int(((wanted == 1) & np.isin(role, (lb.REAL, lb.INTERFACE))).sum()),
             "merged": int(merge.numpy().sum()), "cells": int(new.computational().sum()),
             "rounds": rounds, "ms": 1000 * (time.perf_counter() - begin)}
    return new, stats
