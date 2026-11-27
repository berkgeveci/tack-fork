# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one-leaf ghost ring of a rank's forest, by ``tack.communication.Routing``
(study; the distributed-octrees proposal's step 2), made on the device: every
leaf a corner value of the rank's own leaves reads, fetched from the ranks that
own it.

- **Probes:** a leaf of level ``l`` at the region's edge probes the centres of
  the level ``l + 1`` cells in the shell around it (with 2:1 balance across faces,
  edges and corners, every leaf its corners reach holds one of them), those
  inside the domain and in another rank's root cells: counted, placed by a scan,
  each point once by a sort and ``unique``.
- **Routing:** a probe goes to the rank owning its root cell (``Routing`` takes the
  points on the host: its interface), which finds the leaf holding it
  (``OctreeTents.leaf_at``) and replies with the leaf's tree, level, key and values,
  gathered on the device.
- **The forest:** the ring's leaves once each; their ancestors and the ancestors'
  missing children (placeholders, hidden) by one sort of codes tagged leaf,
  ancestor, child; the rank's own nodes and these in pre-order by a sort of packed
  pre-order codes; the octree made from the device arrays (``_adopt``).

Owners here are by root cell (``owner``, a rank a root, -1 for none): the
shared blocks' finer split is not modelled yet. Codes pack (tree, level, key) in
64 bits: trees below 2^22, levels to 7."""

import numpy as np

import tack
from tack.algorithms import argsort, exclusive_scan, gather, unique
from tack.communication import Routing
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet, _unkey
from tack.dataset.octree_volume import OctreeTents

DEEPEST = tack.constant(7)                    # levels a code holds


@tack.func
def _cell(tree, level, key, n, nx, ny):
    # Node n's cell on its level's lattice over the whole root grid.
    lv = level[n]
    i, j, m = _unkey(key[n], lv, 3)
    t = tree[n]
    return (((t % nx) << lv) + tack.i64(i), (((t // nx) % ny) << lv) + tack.i64(j),
            ((t // (nx * ny)) << lv) + tack.i64(m))


@tack.func
def _owner_at(owner, i, j, k, lv, nx, ny, nz):
    # The rank owning the root cell of lattice cell (i, j, k) of level lv; -2
    # beyond the domain.
    r = tack.i32(-2)
    if i >= 0 and j >= 0 and k >= 0:
        ri = i >> lv
        rj = j >> lv
        rk = k >> lv
        if ri < nx and rj < ny and rk < nz:
            r = owner[ri + nx * (rj + ny * rk)]
    return r


@tack.kernel
def _probe_cells(tree, level, key, marks, refined, owner, rank, nx, ny, nz, offsets, write,
                 counts, codes):
    # The probes of each of the rank's leaves at the region's edge: with write 0,
    # how many; with 1, their codes (level + 1 above bit 58, the cell's flat index
    # on that level's lattice below), from offsets.
    for n in range(marks.shape[0]):
        c = 0
        if (marks[n] & refined) == 0:
            lv = level[n]
            ci, cj, ck = _cell(tree, level, key, n, nx, ny)
            edge = 0
            for o in range(27):
                r = _owner_at(owner, ci + ((o % 3 - 1) << lv), cj + (((o // 3) % 3 - 1) << lv),
                              ck + ((o // 9 - 1) << lv), lv, nx, ny, nz)
                if r != -2 and r != rank:
                    edge = 1
            if edge == 1:
                fine = lv + 1
                wx = tack.i64(nx) << fine
                wy = tack.i64(ny) << fine
                p = tack.i64(0)
                if write == 1:
                    p = offsets[n]
                for s in range(64):               # the 4^3 cells at -1 .. 2, less the leaf's 2^3
                    di = s % 4 - 1
                    dj = (s // 4) % 4 - 1
                    dk = s // 16 - 1
                    inner = 0 <= di <= 1 and 0 <= dj <= 1 and 0 <= dk <= 1
                    fi = 2 * ci + di
                    fj = 2 * cj + dj
                    fk = 2 * ck + dk
                    r = _owner_at(owner, fi, fj, fk, fine, nx, ny, nz)
                    if not inner and r != -2 and r != rank:
                        if write == 1:
                            codes[p + c] = (tack.u64(fine) << tack.u64(58)) | \
                                tack.u64(fi + wx * (fj + wy * fk))
                        c += 1
        if write == 0:
            counts[n] = c


@tack.kernel
def _points(codes, ox, oy, oz, side, nx, ny, x, y, z):
    for k in range(codes.shape[0]):
        fine = tack.i32(codes[k] >> tack.u64(58))
        flat = tack.i64(codes[k] & ((tack.u64(1) << tack.u64(58)) - tack.u64(1)))
        wx = tack.i64(nx) << fine
        wy = tack.i64(ny) << fine
        # f32 (Metal has no f64): a finest cell's half is 1250 cm in a domain of
        # 4.6e6 cm, f32 resolves 0.3 cm there.
        h = side / tack.f32(tack.i64(1) << fine)
        x[k] = ox + (tack.f32(flat % wx) + 0.5) * h
        y[k] = oy + (tack.f32((flat // wx) % wy) + 0.5) * h
        z[k] = oz + (tack.f32(flat // (wx * wy)) + 0.5) * h


@tack.kernel
def _leaves_at(tents, xs, ys, zs, out):
    for k in range(out.shape[0]):
        n, size, depth, inside, fx, fy, fz = tents.leaf_at(tack.vec3(xs[k], ys[k], zs[k]))
        out[k] = tack.i64(max(n, 0))


@tack.func
def _code(t, lv, k):
    return (tack.u64(t) << tack.u64(24)) | (tack.u64(lv) << tack.u64(21)) | k


@tack.kernel
def _tagged_leaves(tree, level, key, found, out):
    # A found ring leaf's code, tag 0 (leaf); a probe no rank held, all ones.
    for k in range(out.shape[0]):
        out[k] = ~tack.u64(0)
        if found[k] != 0:
            out[k] = _code(tree[k], level[k], key[k]) << tack.u64(2)


@tack.kernel
def _ancestors(codes, offsets, write, counts, out):
    # Each leaf's ancestors (tag 1) and each ancestor's eight children (tag 2):
    # with write 0, how many (9 an ancestor); with 1, the tagged codes.
    for k in range(codes.shape[0]):
        c = codes[k] >> tack.u64(2)
        lv = tack.i32((c >> tack.u64(21)) & tack.u64(7))
        if codes[k] == ~tack.u64(0):          # a probe no rank held
            lv = 0
        if write == 0:
            counts[k] = tack.i64(9 * lv)
        else:
            t = c >> tack.u64(24)
            key = c & ((tack.u64(1) << tack.u64(21)) - tack.u64(1))
            p = offsets[k]
            for up in range(1, lv + 1):
                a = key >> tack.u64(3 * up)
                al = lv - up
                out[p] = (_code(t, al, a) << tack.u64(2)) | tack.u64(1)
                for j in range(8):
                    out[p + 1 + j] = (_code(t, al + 1, (a << tack.u64(3)) | tack.u64(j))
                                      << tack.u64(2)) | tack.u64(2)
                p += 9


@tack.kernel
def _payload(leaves, out):
    for k in range(out.shape[0]):
        out[k] = tack.i64(k) if k < leaves else tack.i64(-1)


@tack.kernel
def _firsts(sorted_codes, flags):
    # 1 where a node's code first appears (its smallest tag, the sort's first).
    for k in range(flags.shape[0]):
        f = 1
        if k > 0:
            if (sorted_codes[k] >> tack.u64(2)) == (sorted_codes[k - 1] >> tack.u64(2)):
                f = 0
        if sorted_codes[k] == ~tack.u64(0):
            f = 0
        flags[k] = f


@tack.kernel
def _compact(flags, offsets, out):
    for k in range(flags.shape[0]):
        if flags[k] != 0:
            out[offsets[k]] = tack.i64(k)


@tack.kernel
def _ring_nodes(sorted_codes, picked, payload, leaf_values, nan, tree, level, key, values,
                hidden):
    # The ring's nodes from their tagged codes: a leaf its value, an ancestor
    # NaN, a placeholder NaN and hidden. (NaN comes in as ``nan``: Metal's
    # generator spells a NaN constant ``nanf``, which MSL does not declare.)
    for k in range(tree.shape[0]):
        v = sorted_codes[picked[k]]
        c = v >> tack.u64(2)
        tag = tack.i32(v & tack.u64(3))
        tree[k] = tack.i64(c >> tack.u64(24))
        level[k] = tack.i32((c >> tack.u64(21)) & tack.u64(7))
        key[k] = c & ((tack.u64(1) << tack.u64(21)) - tack.u64(1))
        value = nan
        if tag == 0:
            value = leaf_values[payload[picked[k]]]
        values[k] = value
        hidden[k] = tack.u8(1 if tag == 2 else 0)


@tack.kernel
def _preorder(tree, level, key, out):
    # A node's place in pre-order: its tree, its key at the deepest level (the
    # first descendant's), its level (a parent before its first child).
    for k in range(out.shape[0]):
        lv = level[k]
        out[k] = (tack.u64(tree[k]) << tack.u64(24)) | \
            (key[k] << tack.u64(3 * (DEEPEST - lv) + 3)) | tack.u64(lv)


@tack.kernel
def _own_hidden(marks, bit, out):
    for k in range(out.shape[0]):
        out[k] = tack.u8(1 if (marks[k] & bit) != 0 else 0)


@tack.kernel
def _mark_hidden(hidden, bit, marks):
    for k in range(marks.shape[0]):
        if hidden[k] != 0:
            marks[k] = marks[k] | bit


@tack.kernel
def _owned(tree, owner, rank, flags):
    for k in range(flags.shape[0]):
        flags[k] = 1 if owner[tree[k]] == rank else 0


def _scan(counts):
    offsets = tack.empty((len(counts),), tack.i64)
    total = int(exclusive_scan(counts, offsets, len(counts))) if len(counts) else 0
    return offsets, total


def own_forest(octree, owner_d, rank, name):
    """The rank's own trees of ``octree`` (a stand-in for its cells), made on the
    device: a flag a node, a scan, the nodes gathered in pre-order."""
    with octree.device:
        flags = tack.empty((octree.num_nodes,), tack.i64)
        _owned(octree.tree, owner_d, rank, flags)
        offsets, total = _scan(flags)
        index = tack.empty((total,), tack.i64)
        if total:
            _compact(flags, offsets, index)
        hidden = tack.empty((octree.num_nodes,), tack.u8)
        _own_hidden(octree.marks, int(Ghosts.HIDDENCELL), hidden)
        tree, level, key = (gather(a, index) for a in (octree.tree, octree.level, octree.key))
        values = gather(octree.fields[name], index)
        held = gather(hidden, index)
    forest = OctreeDataSet(octree.roots, tree, level, key, fields={name: values},
                           _ordered=True, _adopt=True)
    with octree.device:
        _mark_hidden(held, int(Ghosts.HIDDENCELL), forest.marks)
    return forest


def probes(forest, owner_d, rank, roots, origin, side):
    """The ring's probe points of a rank's forest, each once: host ``(m, 3)`` f64
    (``Routing`` takes host points)."""
    nx, ny, nz = roots
    with forest.device:
        n = forest.num_nodes
        counts = tack.empty((n,), tack.i64)
        nothing = tack.empty((1,), tack.u64)
        _probe_cells(forest.tree, forest.level, forest.key, forest.marks,
                     int(Ghosts.REFINEDCELL), owner_d, rank, nx, ny, nz, counts, 0, counts,
                     nothing)
        offsets, total = _scan(counts)
        codes = tack.empty((max(total, 1),), tack.u64)
        if total:
            _probe_cells(forest.tree, forest.level, forest.key, forest.marks,
                         int(Ghosts.REFINEDCELL), owner_d, rank, nx, ny, nz, offsets, 1,
                         counts, codes)
        if not total:
            return np.zeros((0, 3))
        each, _ = unique(gather(codes, argsort(codes, total)))
        xyz = [tack.empty((len(each),), tack.f32) for _ in range(3)]
        _points(each, *(float(v) for v in origin), float(side), nx, ny, *xyz)
        return np.stack([a.numpy() for a in xyz], axis=1).astype(np.float64)


def ring(blocks, forests, owner, owner_d, roots, origin, side, name):
    """Each block's ring from the blocks owning it: ``{gid: (tree, level, key,
    values, found)}`` device arrays, one entry a probe (``found`` 0 where no block
    held it); and each block's probe count."""
    nx, ny, _ = roots
    positions = {g: probes(forests[g], owner_d, g, roots, origin, side) for g in blocks.local}
    boxes = {}
    for g in blocks.local:
        mine = np.flatnonzero(owner == g)
        if not len(mine):
            boxes[g] = None
            continue
        i, j, k = mine % nx, (mine // nx) % ny, mine // (nx * ny)
        boxes[g] = (origin + side * np.array([i.min(), j.min(), k.min()]),
                    origin + side * np.array([i.max() + 1, j.max() + 1, k.max() + 1]))

    def holds(g, points):                     # Routing's question, on the host
        r = np.floor((points - origin) / side).astype(np.int64)
        return owner[r[:, 0] + nx * (r[:, 1] + ny * r[:, 2])] == g

    routing = Routing(blocks, positions, boxes, holds)
    arrived = routing.send()
    answers = {}
    for g in blocks.local:
        points, _ = arrived[g]
        forest = forests[g]
        tents = OctreeTents(forest, name, blend=1, balanced=1)
        with forest.device:
            at = tack.empty((len(points),), tack.i64)
            if len(points):
                _leaves_at(tents, *(tack.array(points[:, a].astype(np.float32), tack.f32)
                                    for a in range(3)), at)
            answers[g] = {"tree": gather(forest.tree, at), "level": gather(forest.level, at),
                          "key": gather(forest.key, at), name: gather(forest.fields[name], at),
                          "found": tack.full((len(points),), 1, tack.i32)}
    like = {"tree": tack.i64, "level": tack.i32, "key": tack.u64, name: tack.f32,
            "found": tack.i32}
    back = routing.reply(answers, like)
    return ({g: (back[g]["tree"], back[g]["level"], back[g]["key"], back[g][name],
                 back[g]["found"]) for g in blocks.local},
            {g: len(positions[g]) for g in blocks.local})


def ringed_forest(own, ring_leaves, name):
    """A rank's forest on the device: its own trees, the ring's leaves once each,
    their ancestors, and the ancestors' missing children as hidden placeholders,
    in pre-order. Returns the forest and the count of the ring's nodes (leaves,
    ancestors and placeholders)."""
    tree, level, key, values, found = ring_leaves
    m = len(tree)
    with own.device:
        tagged = tack.empty((max(m, 1),), tack.u64)
        if m:
            _tagged_leaves(tree, level, key, found, tagged)
        counts = tack.empty((max(m, 1),), tack.i64)
        if m:
            _ancestors(tagged, counts, 0, counts, tagged)
        offsets, extra = _scan(counts) if m else (counts, 0)
        more = tack.empty((max(extra, 1),), tack.u64)
        if extra:
            _ancestors(tagged, offsets, 1, counts, more)
        everything = tack.concat([tagged, more]) if m else tagged
        count = m + extra
        payload = tack.empty((count,), tack.i64)               # a leaf's index into the ring
        _payload(m, payload)
        order = argsort(everything, count)
        sorted_codes = gather(everything, order)
        sorted_payload = gather(payload, order)
        flags = tack.empty((count,), tack.i64)
        _firsts(sorted_codes, flags)
        offsets2, nodes = _scan(flags)
        picked = tack.empty((max(nodes, 1),), tack.i64)
        if nodes:
            _compact(flags, offsets2, picked)
        r_tree = tack.empty((nodes,), tack.i64)
        r_level = tack.empty((nodes,), tack.i32)
        r_key = tack.empty((nodes,), tack.u64)
        r_values = tack.empty((nodes,), tack.f32)
        r_hidden = tack.empty((nodes,), tack.u8)
        if nodes:
            _ring_nodes(sorted_codes, picked, sorted_payload, values, float("nan"), r_tree,
                        r_level, r_key, r_values, r_hidden)
        o_hidden = tack.empty((own.num_nodes,), tack.u8)
        _own_hidden(own.marks, int(Ghosts.HIDDENCELL), o_hidden)
        all_tree = tack.concat([own.tree, r_tree])
        all_level = tack.concat([own.level, r_level])
        all_key = tack.concat([own.key, r_key])
        all_values = tack.concat([own.fields[name], r_values])
        all_hidden = tack.concat([o_hidden, r_hidden])
        total = own.num_nodes + nodes
        codes = tack.empty((total,), tack.u64)
        _preorder(all_tree, all_level, all_key, codes)
        perm = argsort(codes, total, index_dtype=tack.i64)
        f_tree, f_level, f_key, f_values, f_hidden = (
            gather(a, perm) for a in (all_tree, all_level, all_key, all_values, all_hidden))
    forest = OctreeDataSet(own.roots, f_tree, f_level, f_key, fields={name: f_values},
                           _ordered=True, _adopt=True)
    with own.device:
        _mark_hidden(f_hidden, int(Ghosts.HIDDENCELL), forest.marks)
    return forest, nodes
