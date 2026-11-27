# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one-leaf ghost ring of a rank's forest, by ``tack.communication.Routing``
(study; the distributed-octrees proposal's step 2): every leaf a corner value
of the rank's own leaves reads, fetched from the ranks that own it.

- **Probes:** a leaf of level ``l`` at the region's edge probes the centres of
  the level ``l + 1`` cells in the shell around it (with 2:1 balance across faces,
  edges and corners, every leaf its corners reach holds one of them), those
  outside the rank's root cells and inside the domain, each point once.
- **Routing:** a probe goes to the rank owning its root cell; that rank finds the
  leaf holding it (``OctreeTents.leaf_at``) and replies with the leaf's tree,
  level, key and values.
- **The forest:** the rank's own trees and the ring's leaves; a ring leaf's tree
  is partial, so its ancestors are made and their missing children become
  placeholder leaves, hidden (the march and the corners skip them).

Owners here are by root cell (``owner``, a rank a root, -1 for none): the
shared blocks' finer split is not modelled yet."""

import numpy as np

import tack
from tack.communication import Routing
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet
from tack.dataset.octree_volume import OctreeTents


@tack.kernel
def _leaves_at(tents, xs, ys, zs, out):
    for k in range(out.shape[0]):
        n, size, depth, inside, fx, fy, fz = tents.leaf_at(tack.vec3(xs[k], ys[k], zs[k]))
        out[k] = n


def lattice(tree, level, key, roots, dimension=3):
    """Each node's cell on its level's lattice over the whole root grid, host
    ``(n, 3)`` int64."""
    nx, ny, _ = roots
    out = np.zeros((len(key), 3), np.int64)
    for b in range(int(level.max()) + 1 if len(level) else 0):
        for a in range(dimension):
            out[:, a] |= ((key >> np.uint64(dimension * b + a)) & np.uint64(1)).astype(np.int64) << b
    root = np.stack([tree % nx, (tree // nx) % ny, tree // (nx * ny)], axis=1)
    return (root << level[:, None]) + out


def probes(forest, own, owner, roots, origin, side, rank):
    """The ring's probe points of a rank's forest: centres of the finer cells around
    its own leaves at the region's edge, outside its root cells. Host ``(m, 3)``."""
    nx, ny, nz = roots
    leaf = (np.asarray(forest.marks.numpy()) & Ghosts.REFINEDCELL) == 0
    pick = np.flatnonzero(leaf & own)
    if not len(pick):
        return np.zeros((0, 3))
    level = forest._level[pick]
    cell = lattice(forest._tree[pick], level, forest._key[pick], roots)
    # Only leaves whose root cell has a neighbour root another rank owns.
    root = forest._tree[pick]
    ri, rj, rk = root % nx, (root // nx) % ny, root // (nx * ny)
    mine = owner[root]
    edge = np.zeros(len(pick), bool)
    for dk in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                i, j, k = ri + di, rj + dj, rk + dk
                inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
                other = np.where(inside, owner[np.clip(i + nx * (j + ny * k), 0, nx * ny * nz - 1)],
                                 mine)
                edge |= inside & (other != mine)
    cell, level = cell[edge], level[edge]
    shell = np.array([(a, b, c) for c in range(-1, 3) for b in range(-1, 3) for a in range(-1, 3)
                      if not (0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1)])   # 56 cells
    points = []
    for lv in np.unique(level):
        here = cell[level == lv]
        fine = (2 * here[:, None, :] + shell[None, :, :]).reshape(-1, 3)
        size = side / (1 << (int(lv) + 1))
        extent = np.array([nx, ny, nz]) << (int(lv) + 1)
        fine = fine[((fine >= 0) & (fine < extent)).all(axis=1)]
        # Each point once: as one integer a cell (a row-wise unique took 43 of 50 s).
        code = np.unique(fine[:, 0] + extent[0] * (fine[:, 1] + extent[1] * fine[:, 2]))
        p = np.stack([code % extent[0], (code // extent[0]) % extent[1],
                      code // (extent[0] * extent[1])], axis=1)
        rootof = p >> (int(lv) + 1)
        p = p[owner[rootof[:, 0] + nx * (rootof[:, 1] + ny * rootof[:, 2])] != rank]
        points.append(origin + (p + 0.5) * size)
    return np.concatenate(points) if points else np.zeros((0, 3))


def completed(tree, level, key):
    """The nodes that complete partial trees holding these leaves (host arrays): every
    ancestor, and each ancestor's children not among the leaves or the ancestors
    (placeholders). Returns ``(ancestors, placeholders)``, each ``(tree, level,
    key)``."""
    leaves = set(zip(tree.tolist(), level.tolist(), key.tolist()))
    ancestors = set()
    for t, lv, k in leaves:
        for up in range(1, lv + 1):
            ancestors.add((t, lv - up, k >> (3 * up)))
    placeholders = []
    for t, lv, k in ancestors:
        for j in range(8):
            child = (t, lv + 1, (k << 3) | j)
            if child not in leaves and child not in ancestors:
                placeholders.append(child)

    def arrays(nodes):
        nodes = sorted(nodes)
        return (np.array([n[0] for n in nodes], np.int64), np.array([n[1] for n in nodes], np.int64),
                np.array([n[2] for n in nodes], np.uint64))
    return arrays(ancestors), arrays(placeholders)


def ringed_forest(roots_grid, own_forest, ring_leaves, names):
    """A rank's forest: its own trees (whole) and the ring's leaves, their partial
    trees completed by their ancestors and hidden placeholders (fields NaN)."""
    t, lv, k, values = ring_leaves
    (at, al, ak), (pt, pl, pk) = completed(t, lv, k)
    own_hidden = (np.asarray(own_forest.marks.numpy()) & Ghosts.HIDDENCELL) != 0
    tree = np.concatenate([own_forest._tree, t, at, pt])
    level = np.concatenate([own_forest._level, lv, al, pl])
    key = np.concatenate([own_forest._key, k, ak, pk])
    fields = {}
    for n in names:
        own = np.asarray(own_forest.fields[n].numpy())
        fields[n] = np.concatenate([own, values[n].astype(own.dtype),
                                    np.full(len(at) + len(pt), np.nan, own.dtype)])
    mask = np.concatenate([own_hidden, np.zeros(len(t) + len(at), bool), np.ones(len(pt), bool)])
    return OctreeDataSet(roots_grid, tree, level, key, fields=fields, mask=mask)


def ring(blocks, forests, owns, owner, roots, origin, side, names):
    """Each block's ring leaves from the blocks owning them: ``{gid: (tree, level,
    key, {name: values})}`` host arrays, each leaf once; and each block's probe
    count."""
    nx, ny, _ = roots
    positions = {g: probes(forests[g], owns[g], owner, roots, origin, side, g)
                 for g in blocks.local}
    boxes = {}
    for g in blocks.local:
        mine = np.flatnonzero(owner == g)
        if not len(mine):
            boxes[g] = None
            continue
        i, j, k = mine % nx, (mine // nx) % ny, mine // (nx * ny)
        boxes[g] = (origin + side * np.array([i.min(), j.min(), k.min()]),
                    origin + side * np.array([i.max() + 1, j.max() + 1, k.max() + 1]))

    def holds(g, points):
        r = np.floor((points - origin) / side).astype(np.int64)
        return owner[r[:, 0] + nx * (r[:, 1] + ny * r[:, 2])] == g

    routing = Routing(blocks, positions, boxes, holds)
    arrived = routing.send()
    answers = {}
    for g in blocks.local:
        points, _ = arrived[g]
        forest = forests[g]
        tents = OctreeTents(forest, names[0], blend=1, balanced=1)
        with forest.device:
            at = tack.empty((len(points),), tack.i32)
            if len(points):
                _leaves_at(tents, *(tack.array(points[:, a].astype(np.float32), tack.f32)
                                    for a in range(3)), at)
            n = np.maximum(at.numpy(), 0)
            answers[g] = {"tree": tack.array(forest._tree[n], tack.i64),
                          "level": tack.array(forest._level[n].astype(np.int32), tack.i32),
                          "key": tack.array(forest._key[n], tack.u64)}
            for name in names:
                answers[g][name] = tack.array(
                    np.asarray(forest.fields[name].numpy())[n].astype(np.float32), tack.f32)
    like = {"tree": tack.i64, "level": tack.i32, "key": tack.u64, **dict.fromkeys(names, tack.f32)}
    back = routing.reply(answers, like)
    out = {}
    for g in blocks.local:
        found = routing.destination[g] >= 0
        t = back[g]["tree"].numpy()[found]
        lv = back[g]["level"].numpy()[found].astype(np.int64)
        k = back[g]["key"].numpy()[found]
        _, first = np.unique(np.stack([t, lv, k.astype(np.int64)], axis=1), axis=0,
                             return_index=True)
        out[g] = (t[first], lv[first], k[first],
                  {n: back[g][n].numpy()[found][first] for n in names})
    return out, {g: len(positions[g]) for g in blocks.local}
