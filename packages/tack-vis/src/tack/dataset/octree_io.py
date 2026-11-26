# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A cell-centred octree in an ADIOS2 file (study, ``study/octree-filters``):
the linear octree as its own layout, in blocks.

A step holds, as ADIOS2 **local arrays** with one block a piece, in block
order: ``tree`` (i64), ``level`` (u8), ``key`` (u64), ``marks`` (u8) and one
array a field, the nodes of each block in pre-order; beside them, one entry a
block, ``block_nodes`` (the node count), ``block_trees`` (the first and last
tree), ``block_box`` (the block's bounds, relative to the roots' origin), and
ADIOS2's own per-block minima and maxima of every variable. The roots are
attributes: ``dimension``, ``roots/x``, ``roots/y``, ``roots/z`` (the grid's
lines, world coordinates), ``fields`` (the names). Nothing derived is stored:
children, neighbours, ranges and ghosts are made on reading, in milliseconds.

Local arrays over global ones: each rank writes its block with no collective,
a reader takes any subset of blocks by id at any rank count, and ADIOS2's
block statistics let it pick blocks by region or value. Whole trees go to one
block, and blocks are written in tree order, so reading every block in order
and concatenating is the global pre-order, which the octree's constructor
takes without a sort or a check (``_ordered``); blocks not in tree order (a
simulation's own pieces, say) are sorted on reading.

Timesteps are ADIOS2 steps; ensemble members are files. Compression is an
ADIOS2 operation on the variables, not used here.
"""

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import OctreeDataSet, _root_axes

__all__ = ["block_info", "read_octree", "write_octree"]

_PER_NODE = ("tree", "level", "key", "marks")


def _blocks_of_trees(octree, block_of_tree=None, blocks=None):
    """Each node's block: by ``block_of_tree`` (one block id a root cell, -1 for
    none), or ``blocks`` contiguous ranges of the pre-order of about equal node
    counts, cut at tree boundaries."""
    tree = octree._tree
    if block_of_tree is not None:
        block = np.asarray(block_of_tree, np.int64)[tree]
        if (block < 0).any():
            raise ValueError("every tree with nodes needs a block")
        return block
    blocks = int(blocks or 1)
    n = octree.num_nodes
    # The first node of each tree, and the cut points nearest to equal shares.
    starts = np.flatnonzero(np.diff(tree, prepend=-1) != 0)
    wanted = np.arange(1, blocks) * n / blocks
    cuts = starts[np.searchsorted(starts, wanted)]
    return np.searchsorted(np.unique(cuts), np.arange(n), side="right").astype(np.int64)


def write_octree(path, octree, block_of_tree=None, blocks=None, fields=None, append=False):
    """Write ``octree`` as a step of the ADIOS2 file at ``path`` (``append`` adds a
    step to an existing file), in blocks: ``block_of_tree`` gives each root
    cell's block, else ``blocks`` (default 1) contiguous ranges of the pre-order
    cut at tree boundaries. ``fields`` names the fields to write (all by
    default). Returns the number of blocks."""
    from adios2 import Stream

    names = list(octree.fields if fields is None else fields)
    block = _blocks_of_trees(octree, block_of_tree, blocks)
    order = np.argsort(block, kind="stable")           # blocks in order, pre-order within
    counts = np.bincount(block)
    ends = np.cumsum(counts)
    starts = ends - counts
    lo, size = (np.asarray(a.numpy(), np.float64).reshape(-1, 3) for a in octree.geometry())
    arrays = {"tree": octree._tree.astype(np.int64), "level": octree._level.astype(np.uint8),
              "key": octree._key.astype(np.uint64), "marks": np.asarray(octree.marks.numpy(), np.uint8)}
    for name in names:
        arrays[name] = np.asarray(octree.fields[name].numpy())
    axes = _root_axes(octree.roots)
    origin = np.asarray(octree.origin, np.float64)
    with Stream(str(path), "a" if append else "w") as s:
        s.begin_step()
        s.write_attribute("dimension", np.array([octree.dimension], np.int32))
        for a, name in enumerate("xyz"):
            line = np.asarray(axes[a], np.float64) + origin[a] if a < len(axes) else np.zeros(1)
            s.write_attribute(f"roots/{name}", line)
        s.write_attribute("fields", names if names else ["" ])
        for b in range(len(counts)):
            ids = order[starts[b]:ends[b]]
            for name, values in arrays.items():
                v = np.ascontiguousarray(values[ids])
                s.write(name, v, shape=[], start=[], count=list(v.shape))
            s.write("block_nodes", np.array([len(ids)], np.int64), shape=[], start=[], count=[1])
            trees = arrays["tree"][ids]
            s.write("block_trees", np.array([trees.min(), trees.max()] if len(ids) else [-1, -1],
                                            np.int64), shape=[], start=[], count=[2])
            box = np.concatenate([lo[ids].min(axis=0), (lo[ids] + size[ids]).max(axis=0)]) \
                if len(ids) else np.zeros(6)
            s.write("block_box", box.astype(np.float64), shape=[], start=[], count=[6])
        s.end_step()
    return len(counts)


def block_info(path, step=0):
    """Per block of ``step``: node count, first and last tree, and bounds -- host
    arrays ``(nodes, trees, boxes)``."""
    from adios2 import Stream

    with Stream(str(path), "r") as s:
        for k, _ in enumerate(s.steps()):
            if k == step:
                info = s.all_blocks_info("block_nodes")[0]
                nodes = np.array([int(i["Min"]) for i in info])
                trees = np.array([s.read("block_trees", block_id=b) for b in range(len(info))])
                boxes = np.array([s.read("block_box", block_id=b) for b in range(len(info))])
                return nodes, trees, boxes
    raise IndexError(f"no step {step}")


def read_octree(path, step=0, blocks=None, fields=None, device=None):
    """The octree of ``step`` of the ADIOS2 file at ``path``, from ``blocks`` (ids;
    all by default) with ``fields`` (names; all by default), on ``device``. The
    blocks are concatenated in the order given; when that is the file's
    order of whole trees, the nodes are in pre-order and the octree is made
    without a sort or a check."""
    from adios2 import Stream

    from tack.dataset.dataset import rectilinear_grid

    with Stream(str(path), "r") as s:
        for k, _ in enumerate(s.steps()):
            if k != step:
                continue
            dimension = int(s.read_attribute("dimension")[0])
            axes = [np.asarray(s.read_attribute(f"roots/{a}"), np.float64) for a in "xyz"]
            names = [n for n in s.read_attribute("fields") if n]
            if fields is not None:
                names = list(fields)
            info = s.all_blocks_info("block_nodes")[0]
            chosen = list(range(len(info))) if blocks is None else list(blocks)
            trees = np.array([s.read("block_trees", block_id=b) for b in chosen]).reshape(-1, 2)
            parts = {name: [] for name in (*_PER_NODE, *names)}
            for b in chosen:
                if int(info[b]["Min"]) == 0:
                    continue
                for name in parts:
                    parts[name].append(s.read(name, block_id=b))
            break
        else:
            raise IndexError(f"no step {step}")
    arrays = {name: np.concatenate(v) if v else np.zeros(0, np.int64) for name, v in parts.items()}
    # Whole trees in order: no two blocks share a tree, and each starts after the last.
    held = trees[:, 0] >= 0
    ordered = bool((trees[held, 0][1:] > trees[held, 1][:-1]).all()) if held.sum() > 1 else True
    from tack.dataset.octree import _compute_type
    from tack.runtime import dispatch

    dtype = _compute_type(dispatch.as_device(dispatch.get_backend()) if device is None else device,
                          tack.f32)                   # f64 where the device has it
    roots = rectilinear_grid(*axes[:dimension], *([(axes[2][0],)] if dimension == 2 else []),
                             dtype=dtype, device=device)
    hidden = (arrays["marks"] & Ghosts.HIDDENCELL) != 0
    field_arrays = {name: arrays[name] for name in names}
    return OctreeDataSet(roots, arrays["tree"], arrays["level"].astype(np.int64), arrays["key"],
                         fields=field_arrays, mask=hidden, _ordered=ordered)
