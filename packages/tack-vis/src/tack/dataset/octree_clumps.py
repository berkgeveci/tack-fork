# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Clumps of a cell-centred octree (study, ``study/octree-filters``): the
connected components of the visible leaves whose field is in a range, through
the faces between leaves. Union-find on the device, as ``connected_components``
does it: each face between two leaves in the range hooks the larger root under
the smaller (an atomic minimum, so concurrent hooks agree), then every leaf
jumps to its root, until nothing changes; the roots numbered in the nodes'
order. Astrophysics' clump finder, over the face list (``faces``).
"""


import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree_filters import _with
from tack.dataset.octree_topology import faces

__all__ = ["clumps"]


@tack.kernel
def _in_range(values, marks, refined_or_hidden, lower, upper, labels):
    # A visible leaf in the range starts as its own root; any other node -1.
    for n in range(labels.shape[0]):
        v = values[n]
        labels[n] = n if (marks[n] & refined_or_hidden) == 0 and lower <= v and v <= upper \
            else tack.i64(-1)


@tack.kernel
def _hook_faces(small, large, labels, changed):
    for f in range(small.shape[0]):
        a = small[f]
        b = large[f]
        if b >= 0:
            la = labels[a]
            lb = labels[b]
            if la >= 0 and lb >= 0 and la != lb:
                tack.atomic_min(labels, max(la, lb), min(la, lb))
                if changed[0] == 0:
                    tack.atomic_max(changed, 0, 1)


@tack.kernel
def _jump(labels):
    for v in range(labels.shape[0]):
        r = labels[v]
        if r >= 0:
            while labels[r] != r:
                r = labels[r]
            labels[v] = r


@tack.kernel
def _roots(labels, flags):
    for n in range(flags.shape[0]):
        flags[n] = 1 if labels[n] == n else 0


@tack.kernel
def _numbered(labels, numbers, out):
    for n in range(out.shape[0]):
        out[n] = numbers[labels[n]] - 1 if labels[n] >= 0 else tack.i64(-1)


def clumps(octree, name, lower, upper):
    """The octree with a field ``clump``: on each visible leaf whose ``name`` is in
    ``[lower, upper]``, the number of its connected component (through the
    faces between such leaves), components numbered from 0 in the nodes'
    order; -1 elsewhere. ``octree.fields["clump"].max() + 1`` is their count;
    a ``threshold`` of ``clump`` picks one."""
    from tack.algorithms import inclusive_scan

    n = octree.num_nodes
    found = faces(octree)
    with octree.device:
        labels = tack.empty((n,), tack.i64)
        if n:
            _in_range(octree._field(name), octree.marks,
                      int(Ghosts.REFINEDCELL | Ghosts.HIDDENCELL), float(lower), float(upper),
                      labels)
            changed = tack.zeros((1,), tack.i32)
            while True:
                changed.fill(0)
                _hook_faces(found.small, found.large, labels, changed)
                _jump(labels)
                if int(changed.numpy()[0]) == 0:
                    break
        flags = tack.empty((n,), tack.i32)
        numbers = tack.empty((n,), tack.i64)
        clump = tack.empty((n,), tack.i64)
        if n:
            _roots(labels, flags)
            inclusive_scan(flags, numbers, n)
            _numbered(labels, numbers, clump)
    fields = dict(octree.fields)
    fields["clump"] = clump
    return _with(octree, fields=fields)
