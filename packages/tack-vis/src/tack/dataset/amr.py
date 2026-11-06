# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Block-structured adaptive mesh refinement: ``AMRDataSet``, levels of structured
boxes, each finer than the one before, as VTK's ``vtkOverlappingAMR`` and
AMReX's hierarchy of ``BoxArray``s.

The hierarchy is defined in index space alone, as AMReX defines it. A box of
level ``l`` is a cell-index box ``[lo, hi)`` of its level; level ``l + 1``'s
indices are level ``l``'s times the refinement ratio between them (one per
axis). A box is an ordinary dataset whose ``StructuredTopology.offset`` is
``lo``. Its geometry is whatever the box's is -- an image's
``UniformCoordinates``, a rectilinear grid's axes, a curvilinear grid's
positions -- since which cells and points a finer level covers is a matter of
integer boxes, not of positions. That is what VTK's images-only AMR lacks.

Making the hierarchy marks what each box holds that another holds better
(``Ghosts``):

- ``REFINEDCELL``: a cell a finer level's box covers. Its data are valid but
  superseded: filters that make new cells leave it out, and reductions skip
  it, so each region of space counts once.
- ``REFINEDPOINT`` (a Tack bit; VTK has none): a point at whose location a
  finer level has a point. A coarse point on the edge of a refined region is
  one, so each location counts once, at the finest level holding it.
- ``DUPLICATEPOINT``: a point a sibling box with a lower global index also
  holds. Sibling boxes abut, each with its own points on their shared face.

The boxes of every rank are gathered when the hierarchy is made
(``boxes(level)``), as VTK's metadata and AMReX's ``BoxArray`` are replicated,
so the marks need no other communication.
"""

import contextlib
import itertools

import numpy as np

import tack
from tack import arrays
from tack.algorithms import exclusive_scan, segment_reduce
from tack.dataset import _box_pairs
from tack.dataset.dataset import DataSet
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import PartitionedDataSet, _one_time
from tack.dataset.topology import StructuredTopology

__all__ = ["AMRDataSet"]


class AMRDataSet:
    """Levels of structured boxes, each level finer than the one before: a
    block-structured AMR hierarchy, as VTK's ``vtkOverlappingAMR`` and AMReX's.

    ``levels`` holds this rank's boxes of each level, a ``PartitionedDataSet`` or a
    list of ``DataSet``s each. Every box has a ``StructuredTopology`` (all of one
    dimension) whose ``offset`` is its first cell's index in its level, and any
    structured geometry: an image, a rectilinear grid or a curvilinear grid.
    ``ratios[l]`` refines level ``l`` into ``l + 1``: a whole number, or one per
    axis. ``domain`` is level 0's cell-index box ``(lo, hi)``, ``hi`` exclusive;
    by default its boxes' bounding box. ``world`` is where the other ranks' boxes
    are (``tack.communication.World``; the levels' own, else this process).

    Refused when made: boxes of a level that overlap; a box of level ``l + 1``
    that, coarsened, is not within level ``l``'s boxes (proper nesting); image
    boxes of a level not on one lattice (one spacing, one origin, axis-aligned:
    VTK's model); boxes with ghost cells (``DUPLICATECELL``).

    The boxes it holds are copies of those given with their marks computed
    (``REFINEDCELL``, ``REFINEDPOINT``, siblings' ``DUPLICATEPOINT``; other bits
    kept), sharing their topology, geometry and fields. Iterating over it gives
    ``(level, box)`` for this rank's boxes, coarsest level first."""

    def __init__(self, levels, ratios=(), domain=None, world=None):
        levels = list(levels)
        if not levels:
            raise ValueError("an AMRDataSet has at least one level")
        if world is None:
            world = next((lvl.world for lvl in levels if isinstance(lvl, PartitionedDataSet)),
                         None)
        if world is None:
            from tack.communication import World
            world = World()
        self.world = world
        given = [list(lvl) for lvl in levels]
        self.dimension = self._dimension(given)
        self.ratios = self._ratios(ratios, len(given))
        boxes, self._first = [], []
        for level, pieces in enumerate(given):
            mine = np.array([_box(piece, self.dimension) for piece in pieces],
                            np.int64).reshape(-1, 2, 3)
            every = world._allgather(mine)
            self._first.append(int(sum(len(part) for part in every[:world.rank])))
            boxes.append(np.concatenate(every).reshape(-1, 2, 3))
        self._boxes = boxes
        #: Each box's arrays' cell-index box: the box itself, or grown by ghost
        #: layers (``ghost_cells``).
        self._extents = [b.copy() for b in boxes]
        self.layers = 0
        self._fills = None
        self.domain = self._domain(domain)
        self._check(given)
        self.levels = tuple(PartitionedDataSet(pieces, world)
                            for pieces in _with_marks(self, given))

    @classmethod
    def _grown(cls, base, levels, extents, layers, fills):
        """A hierarchy of ``base``'s boxes grown by ghost layers: ``levels`` the grown
        boxes (marks made), ``extents`` their cell-index boxes, ``fills`` what
        refreshes their ghosts."""
        out = cls.__new__(cls)
        out.world, out.dimension, out.ratios = base.world, base.dimension, base.ratios
        out._boxes, out._first, out.domain = base._boxes, base._first, base.domain
        out._extents, out.layers, out._fills = extents, layers, fills
        out.levels = tuple(PartitionedDataSet(pieces, base.world) for pieces in levels)
        return out

    def level_domain(self, level):
        """``domain`` refined to ``level``: its cell-index box there."""
        scale = np.ones(3, np.int64)
        for r in self.ratios[:level]:
            scale *= np.asarray(r, np.int64)
        lo, hi = (np.asarray(c, np.int64) for c in self.domain)
        return lo * scale, hi * scale

    def _refined(self, level, lo, hi):
        """Which cells and points of the cell-index box ``[lo, hi)`` of ``level`` a
        finer level covers: two bool arrays shaped z, y, x (``dimension`` axes)."""
        box = np.stack([np.asarray(lo, np.int64), np.asarray(hi, np.int64)])[None]
        return self._refined_boxes(level, box)[0]

    def _refined_boxes(self, level, boxes):
        """``_refined`` of each cell-index box of ``boxes`` (``(n, 2, 3)``), every box
        at once: the boxes and the finer level's that meet them are found by
        ``_meeting``, so the work is in the pairs that meet, not in every pair (a
        loop over all the children for each box took 129 s for AMR-Wind's 6,436
        boxes)."""
        d = self.dimension
        lo, hi = boxes[:, 0, :d], boxes[:, 1, :d]
        out = [(np.zeros(tuple(int(v) for v in (b - a)[::-1]), bool),
                np.zeros(tuple(int(v) + 1 for v in (b - a)[::-1]), bool))
               for a, b in zip(lo, hi)]
        if level + 1 < self.num_levels:
            ratio = np.asarray(self.ratios[level], np.int64)[:d]
            children = self._boxes[level + 1][:, :, :d]
            # A cell refined: a child, coarsened, covers it.
            clo, chi = children[:, 0] // ratio, -(-children[:, 1] // ratio)
            for i, a, b in _cut(lo, hi, clo, chi, closed=False):
                out[i][0][_slices(a - lo[i], b - lo[i])] = True
            # A point refined: its location p * ratio is one of the child's points.
            plo, phi = -(-children[:, 0] // ratio), children[:, 1] // ratio
            for i, a, b in _cut(lo, hi, plo, phi, closed=True):
                out[i][1][_slices(a - lo[i], b - lo[i] + 1)] = True
        return out

    # -- what it holds ---------------------------------------------------

    @property
    def num_levels(self):
        return len(self._boxes)

    def boxes(self, level):
        """Every rank's boxes of ``level``, an ``(n, 2, 3)`` integer array of cell-index
        boxes ``[lo, hi)``, in order of global index: rank 0's first, each rank's
        in the order given. Axes a box does not have are ``[0, 1)``."""
        return self._boxes[level].copy()

    def __iter__(self):
        for level, pieces in enumerate(self.levels):
            for piece in pieces:
                yield level, piece

    def __repr__(self):
        counts = [len(self._boxes[level]) for level in range(self.num_levels)]
        return f"AMRDataSet({self.num_levels} levels, boxes {counts})"

    def _as_pieces(self):
        """This rank's boxes of every level, as one ``PartitionedDataSet``: what a
        reduction over the hierarchy reduces, each box's marks saying what it
        owns."""
        return PartitionedDataSet([box for _, box in self], self.world)

    # -- checks ----------------------------------------------------------

    def _dimension(self, given):
        mine = {len(piece.topology.point_dims) for pieces in given for piece in pieces
                if isinstance(piece.topology, StructuredTopology)}
        for level, pieces in enumerate(given):
            for k, piece in enumerate(pieces):
                if not isinstance(piece, DataSet) or \
                        not isinstance(piece.topology, StructuredTopology):
                    raise TypeError(f"box {k} of level {level} is not a dataset of a "
                                    f"StructuredTopology")
        every = set().union(*self.world._allgather(sorted(mine)))
        if len(every) > 1:
            raise ValueError(f"an AMRDataSet's boxes are of one dimension, not "
                             f"{sorted(every)}")
        if not every:
            raise ValueError("an AMRDataSet needs a box somewhere")
        return every.pop()

    def _ratios(self, ratios, count):
        ratios = list(ratios)
        if len(ratios) == count:            # VTK gives the finest level one too
            ratios = ratios[:-1]
        if len(ratios) != count - 1:
            raise ValueError(f"{count} levels take {count - 1} refinement ratios, not "
                             f"{len(ratios)}")
        out = []
        for r in ratios:
            values = np.atleast_1d(np.asarray(r))
            if values.size == 1:
                values = np.full(self.dimension, values[0])
            elif values.size == 3 and (values[self.dimension:] == 1).all():
                values = values[:self.dimension]      # as ``ratios`` gives them back
            if values.size != self.dimension or values.dtype.kind not in "iu" or \
                    (values < 1).any():
                raise ValueError(f"a refinement ratio is a whole number of at least 1, or "
                                 f"one per axis ({self.dimension}), not {r!r}")
            out.append(tuple(int(v) for v in values) + (1,) * (3 - self.dimension))
        return tuple(out)

    def _domain(self, domain):
        level0 = self._boxes[0]
        if domain is None:
            return (tuple(int(v) for v in level0[:, 0].min(axis=0)),
                    tuple(int(v) for v in level0[:, 1].max(axis=0)))
        lo, hi = (np.asarray(corner, np.int64) for corner in domain)
        lo = np.concatenate([lo, np.zeros(3 - lo.size, np.int64)])
        hi = np.concatenate([hi, np.ones(3 - hi.size, np.int64)])
        if (level0[:, 0] < lo).any() or (level0[:, 1] > hi).any():
            raise ValueError(f"level 0's boxes reach outside the domain {domain!r}")
        return tuple(int(v) for v in lo), tuple(int(v) for v in hi)

    def _check(self, given):
        from tack.dataset import units

        spatial = self.world._allgather(sorted({piece.unit or "" for pieces in given
                                                for piece in pieces}))
        units.common("an AMRDataSet's boxes' geometry",
                     [unit or None for part in spatial for unit in part])
        _one_time("an AMRDataSet's boxes", [t for part in self.world._allgather(
            sorted({(piece.time, piece.time_unit) for pieces in given for piece in pieces},
                   key=repr)) for t in part])
        for level, pieces in enumerate(given):
            for k, piece in enumerate(pieces):
                marks = piece.ghosts
                if marks is not None and marks.cells is not None and \
                        (marks.cells.numpy() & Ghosts.DUPLICATECELL).any():
                    raise ValueError(f"box {k} of level {level} has ghost cells "
                                     "(DUPLICATECELL): give the boxes without them")
        device = next((pieces[0].device for pieces in given if pieces), None)
        with device if device is not None else contextlib.nullcontext():
            self._check_nesting()
        for level, pieces in enumerate(given):
            lattices = self.world._allgather([_lattice(piece, self.dimension)
                                              for piece in pieces])
            lattices = [entry for part in lattices for entry in part if entry is not None]
            if len(lattices) < 2:
                continue
            spacings = np.array([np.asarray(entry[0], np.float64) for entry in lattices])
            origins = np.array([np.asarray(entry[1], np.float64) for entry in lattices])
            ref_spacing, ref_origin = spacings[0], origins[0]
            tolerance = 1e-6 * float(np.max(ref_spacing))
            # ``np.allclose`` of each against the first, every box at once.
            same = (np.abs(spacings - ref_spacing) <= 1e-9 * np.abs(ref_spacing)).all(axis=1) \
                & (np.abs(origins - ref_origin) <= tolerance).all(axis=1)
            if not same.all():
                k = int(np.flatnonzero(~same)[0])
                spacing, origin = spacings[k], origins[k]
                raise ValueError(
                    f"the image boxes of level {level} are not on one lattice: spacing "
                    f"{tuple(spacing)} and origin {tuple(origin)} against "
                    f"{tuple(ref_spacing)} and {tuple(ref_origin)} (VTK's model: one "
                    "spacing per level and one origin)")

    def _check_nesting(self):
        """Refuse boxes of a level that overlap, and a box that, coarsened, is not
        within the coarser level's boxes: on the current device, from the pairs of
        boxes that meet (`_box_pairs`); only a count comes back, and the pairs only
        to name what is refused."""
        d = self.dimension
        for level, boxes in enumerate(self._boxes):
            found = _box_pairs.pairs(boxes, boxes, d, closed=False, distinct=True)
            if found.count:
                i, j = found.first.numpy(), found.second.numpy()
                lower = i < j
                k = np.lexsort((j[lower], i[lower]))[0]
                a, b = int(i[lower][k]), int(j[lower][k])
                raise ValueError(f"boxes {a} and {b} of level {level} overlap: {_text(boxes[a])} "
                                 f"and {_text(boxes[b])}")
        for level in range(1, self.num_levels):
            ratio = np.asarray(self.ratios[level - 1], np.int64)
            fine = self._boxes[level]
            coarsened = np.stack([fine[:, 0] // ratio, -(-fine[:, 1] // ratio)], 1)
            found = _box_pairs.pairs(coarsened, self._boxes[level - 1], d, closed=False)
            covered = tack.zeros(len(fine), tack.i64)
            if found.count:
                volumes = tack.empty(found.count, tack.i64)
                _met_volumes(found.first, found.second, found.lo, found.hi, found.blo,
                             found.bhi, volumes)
                segment_reduce(volumes, found.offsets, out=covered)
            short = tack.empty(len(fine), tack.i64)
            _short_boxes(covered, found.lo, found.hi, short)
            if exclusive_scan(short, tack.empty(len(fine), tack.i64), len(fine)):
                k = int(np.flatnonzero(short.numpy())[0])
                raise ValueError(f"box {k} of level {level} ({_text(fine[k])}) is not within "
                                 f"level {level - 1}'s boxes when coarsened by "
                                 f"{tuple(ratio[:d])}: the hierarchy is not properly nested")

    # -- the marks -------------------------------------------------------

    def _covers(self, level):
        """What marks the boxes of ``level``: ``[(bit, kind, boxes, closed, lower)]``,
        ``bit`` set on the cells or points (``kind``) where a box meets one of
        ``boxes`` (host ``(n, 2, 3)``; sharing a point where ``closed``; with
        ``lower``, only one of lower global index). ``REFINEDCELL`` where a child
        box, coarsened, covers cells; ``REFINEDPOINT`` where a child has a point at
        a point's location; ``DUPLICATEPOINT`` on the faces, edges and corners a
        sibling of lower global index shares."""
        out = []
        if level + 1 < self.num_levels:
            ratio = np.asarray(self.ratios[level], np.int64)
            children = self._boxes[level + 1]
            out.append((Ghosts.REFINEDCELL, "cells",
                        np.stack([children[:, 0] // ratio, -(-children[:, 1] // ratio)], 1),
                        False, False))
            out.append((Ghosts.REFINEDPOINT, "points",
                        np.stack([-(-children[:, 0] // ratio), children[:, 1] // ratio], 1),
                        True, False))
        out.append((Ghosts.DUPLICATEPOINT, "points", self._boxes[level], True, True))
        return out


@tack.kernel
def _met_volumes(first, second, lo, hi, blo, bhi, out):
    # The cells two boxes share, three axes (a box has extent 1 on axes it lacks).
    for p in range(first.shape[0]):
        i = first[p]
        j = second[p]
        v = tack.i64(1)
        for a in range(3):
            v *= min(hi[i, a], bhi[j, a]) - max(lo[i, a], blo[j, a])
        out[p] = v


@tack.kernel
def _short_boxes(covered, lo, hi, short):
    for k in range(covered.shape[0]):
        v = (hi[k, 0] - lo[k, 0]) * (hi[k, 1] - lo[k, 1]) * (hi[k, 2] - lo[k, 2])
        short[k] = 1 if covered[k] != v else 0


@tack.kernel
def _kept_bits(source, mask, out):
    for i in range(out.shape[0]):
        out[i] = source[i] & mask


@tack.kernel
def _mark_rows(rows, count, offsets, table, bit, marks):
    # One thread a row of a part: offsets[p] the first row of part p (count of
    # them, offsets[count] == rows); table[p] its box's first mark in ``marks``,
    # the box's row and plane strides, the part's start (x, y, z) in the box and
    # its extent along x and y. Parts of a launch may share points (two children
    # meeting on a face): both store the same byte, since a launch sets one bit.
    for r in range(rows):
        a = 0
        b = count - 1
        while a < b:
            m = (a + b + 1) // 2
            if offsets[m] <= r:
                a = m
            else:
                b = m - 1
        local = r - offsets[a]
        y = local % table[a, 7]
        z = local // table[a, 7]
        at = table[a, 0] + (table[a, 5] + z) * table[a, 2] + (table[a, 4] + y) * table[a, 1] + \
            table[a, 3]
        for x in range(table[a, 6]):
            marks[at + x] = marks[at + x] | bit


@tack.func
def _part_along(lo, hi, blo, bhi, i, j, a, d, closed):
    """Along axis ``a``: box ``i``'s marks' extent, and the start (relative to the
    box) and extent of where box ``j`` meets it; one mark on axes past ``d``."""
    extent = tack.i64(1)
    start = tack.i64(0)
    size = tack.i64(1)
    if a < d:
        low = max(lo[i, a], blo[j, a])
        extent = hi[i, a] - lo[i, a] + closed
        start = low - lo[i, a]
        size = max(min(hi[i, a], bhi[j, a]) - low + closed, 0)
    return extent, start, size


@tack.kernel
def _part_rows(first, second, lo, hi, blo, bhi, d, closed, lower, ids, starts, table, rows):
    # Pair p's row of the table _mark_rows reads, and its number of rows; none
    # where ``lower`` and the other box's global index is not below this one's.
    for p in range(first.shape[0]):
        i = first[p]
        j = second[p]
        e0, x0, n0 = _part_along(lo, hi, blo, bhi, i, j, 0, d, closed)
        e1, x1, n1 = _part_along(lo, hi, blo, bhi, i, j, 1, d, closed)
        e2, x2, n2 = _part_along(lo, hi, blo, bhi, i, j, 2, d, closed)
        table[p, 0] = starts[i]
        table[p, 1] = e0
        table[p, 2] = e0 * e1
        table[p, 3] = x0
        table[p, 4] = x1
        table[p, 5] = x2
        table[p, 6] = n0
        table[p, 7] = n1
        r = n1 * n2
        if n0 == 0 or (lower != 0 and j >= ids[i]):
            r = 0
        rows[p] = r


def _mark(marks, bit, found, d, closed, lower, ids, starts):
    """Set ``bit`` in ``marks`` where the boxes of the pairs ``found`` (``Pairs``)
    meet, relative to the first box, whose marks start at ``starts[i]``
    (``ids[i]`` its global index for ``lower``)."""
    count = found.count
    if not count:
        return
    table = tack.empty((count, 8), tack.i64)
    offsets = tack.empty(count + 1, tack.i64)
    _part_rows(found.first, found.second, found.lo, found.hi, found.blo, found.bhi, d,
               int(closed), int(lower), ids, starts, table, offsets)
    rows = exclusive_scan(offsets, offsets, count, close=True)
    if rows:
        _mark_rows(rows, count, offsets, table, int(bit), marks)


def _with_marks(amr, levels):
    """``levels``' boxes (this rank's, per level) with their marks, made on their
    devices: each kind of mark in one array per device for the whole hierarchy,
    the levels in order and each box's marks a view of it. The given marks' other
    bits are kept (one launch where they are one stretch of an array, as a
    hierarchy's are). Where boxes meet is found on the device (`_box_pairs`), and
    each kind of mark is set by one launch a level (`_mark_rows`), whatever the
    number of boxes. Only the box lists are on the host."""
    d = amr.dimension
    every = [box for pieces in levels for box in pieces]
    mine = [amr._boxes[level][amr._first[level]:amr._first[level] + len(pieces)]
            for level, pieces in enumerate(levels)]
    extents = np.concatenate(mine).reshape(-1, 2, 3)
    ids = np.concatenate([amr._first[level] + np.arange(len(pieces), dtype=np.int64)
                          for level, pieces in enumerate(levels)])
    level_of = np.repeat(np.arange(len(levels)), [len(pieces) for pieces in levels])
    cells = np.ones((len(every), 3), np.int64)
    cells[:, :d] = extents[:, 1, :d] - extents[:, 0, :d]
    points = cells.copy()
    points[:, :d] += 1
    dims = {"cells": cells, "points": points}
    covers = [amr._covers(level) for level in range(len(levels))]
    kept = {"cells": 0xFF & ~Ghosts.REFINEDCELL,
            "points": 0xFF & ~(Ghosts.REFINEDPOINT | Ghosts.DUPLICATEPOINT)}
    ghosts = [None] * len(every)
    for device, ks in _by_device(every):
        ks = np.asarray(ks, np.int64)
        views = {}
        for kind in ("cells", "points"):
            counts = dims[kind][ks].prod(axis=1)
            starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            given = [getattr(every[k].ghosts, kind, None) if every[k].ghosts is not None
                     else None for k in ks]
            with device:
                whole = _kept(given, counts, kept[kind])
                for level in range(len(levels)):
                    here = np.flatnonzero(level_of[ks] == level)
                    if not len(here):
                        continue
                    level_ids = tack.array(ids[ks[here]])
                    level_starts = tack.array(starts[here])
                    for bit, what, boxes, closed, lower in covers[level]:
                        if what != kind:
                            continue
                        found = _box_pairs.pairs(extents[ks[here]], boxes, d, closed)
                        _mark(whole, bit, found, d, closed, lower, level_ids, level_starts)
            views[kind] = whole._row_views(starts)
        for i, k in enumerate(ks):
            ghosts[k] = Ghosts(cells=views["cells"][i], points=views["points"][i])
    out, k = [], 0
    for pieces in levels:
        level = []
        for piece in pieces:
            level.append(piece._with_ghosts(ghosts[k]))
            k += 1
        out.append(level)
    return out


def _kept(given, counts, mask):
    """The marks ``given`` (each box's, or None) end to end in one new array on the
    current device, with only the bits of ``mask``; zeros for a box without
    (``counts`` its marks' number)."""
    n = int(sum(counts))
    if n and all(g is not None for g in given):
        joined = arrays._joined_view(given)
        if joined is not None:
            whole = tack.empty(n, tack.u8)
            _kept_bits(joined, mask, whole)
            return whole
    whole = tack.zeros(n, tack.u8)
    start = 0
    for g, m in zip(given, counts):
        if g is not None and m:
            _kept_bits(g, mask, whole[start:start + m])
        start += int(m)
    return whole


def _by_device(pieces):
    """``[(device, [k, ...])]``: the indices of ``pieces`` on each device, in order.
    A box's device is its marks' where it has them, as every box of a hierarchy
    does: ``DataSet.device`` searches the dataset's parts, about 15 us a box."""
    from tack.runtime.dispatch import as_device

    groups = []
    for k, piece in enumerate(pieces):
        ghosts = piece.ghosts
        marks = None if ghosts is None else (ghosts.cells if ghosts.cells is not None
                                             else ghosts.points)
        device = as_device(marks._backend) if marks is not None else piece.device
        for d, ks in groups:
            if d == device:
                ks.append(k)
                break
        else:
            groups.append((device, [k]))
    return groups


def in_device_buffers(levels):
    """Put a hierarchy's fields in one array per device, in place: each box's values
    become a view (``array[i:j]``) of its device's array, the levels in order and
    each level's boxes in order, so a level's boxes (statistics) and a device's
    boxes of any run of levels (``_Fill``'s groups: ghost cells, ``average_down``)
    are one stretch of it, taken without a copy. One copy now; the marks
    ``AMRDataSet`` makes are laid out the same way (`_with_marks`). A field is taken
    where every box on the device has it, at the same place among its fields, in
    one kind of space, as a stored array of one element type; any other is left
    as it is. ``levels`` are lists of ``DataSet``s, as ``AMRDataSet`` takes them;
    returned for chaining."""
    import tack
    from tack.dataset.dataset import _same_space
    from tack.lang.array import Array

    every = [box for pieces in levels for box in pieces]
    for device, ks in _by_device(every):
        boxes = [every[k] for k in ks]
        entries = [box.fields.entries() for box in boxes]
        for i, (name, first) in enumerate(entries[0]):
            fields = []
            for listed in entries:
                if i >= len(listed):
                    break
                other_name, field = listed[i]
                values = field.values
                if (other_name != name or not _same_space(field.space, first.space)
                        or type(values) is not Array or values.ndim != 1
                        or values.dtype != first.values.dtype):
                    break
                fields.append(field)
            if len(fields) != len(boxes):
                continue
            with device:
                whole = tack.concat([f.values for f in fields])
            views = whole._row_views(list(itertools.accumulate(
                (f.values.shape[0] for f in fields), initial=0)))
            for field, view in zip(fields, views):
                field.values = view
    return levels


def _box(piece, dimension):
    """A box's cell-index box in its level, three axes, ``[lo, hi)``."""
    topology = piece.topology
    lo = np.zeros(3, np.int64)
    hi = np.ones(3, np.int64)
    n = len(topology.point_dims)
    lo[:n] = topology.offset
    hi[:n] = np.asarray(topology.offset) + np.asarray(topology.point_dims) - 1
    return lo, hi


def _meeting(alo, ahi, blo, bhi, closed):
    """Every pair ``(i, j)`` of box ``i`` of ``a`` and box ``j`` of ``b`` that meet, as
    two arrays ordered by ``i``, then ``j``: ``[lo, hi)`` boxes sharing a cell, or
    with ``closed``, ``[lo, hi]`` boxes sharing a point. Arrays ``(n, axes)``.

    The boxes are put in bins of a lattice whose cells are the median box of ``b``,
    and only boxes sharing a bin are tested, so the work is in the pairs that are
    near each other. (A sweep along x tested every box of a slab through the whole
    domain: on AMR-Wind's five levels 15.3M candidates for 0.40M pairs that meet,
    against the bins' 0.92M.)"""
    empty = np.zeros(0, np.int64)
    if not len(alo) or not len(blo):
        return empty, empty
    alo, ahi, blo, bhi = (np.asarray(v, np.int64) for v in (alo, ahi, blo, bhi))
    size = np.maximum(np.median(bhi - blo, axis=0).astype(np.int64), 1)
    origin = np.minimum(alo.min(axis=0), blo.min(axis=0))
    last = 0 if closed else 1           # a box's last cell, or with closed its last point

    def binned(lo, hi):
        first = (lo - origin) // size
        return first, np.maximum((hi - last - origin) // size, first)

    afirst, alast = binned(alo, ahi)
    bfirst, blast = binned(blo, bhi)
    shape = np.maximum(alast.max(axis=0), blast.max(axis=0)) + 1
    ia, abin = _bin_entries(afirst, alast, shape)
    jb, bbin = _bin_entries(bfirst, blast, shape)
    order = np.argsort(bbin, kind="stable")
    bbin, jb = bbin[order], jb[order]
    first = np.searchsorted(bbin, abin, side="left")
    counts = np.searchsorted(bbin, abin, side="right") - first
    i = np.repeat(ia, counts)
    within = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    j = jb[np.repeat(first, counts) + within]
    pairs = np.unique(i * len(blo) + j)
    i, j = pairs // len(blo), pairs % len(blo)
    low = np.maximum(alo[i], blo[j])
    high = np.minimum(ahi[i], bhi[j])
    meet = (low <= high).all(axis=1) if closed else (low < high).all(axis=1)
    return i[meet], j[meet]


def _bin_entries(first, last, shape):
    """``(box, bin)`` for each bin of each box, a box covering the bins ``first`` to
    ``last`` along each axis of a lattice of ``shape`` bins."""
    extent = last - first + 1
    counts = extent.prod(axis=1)
    box = np.repeat(np.arange(len(first)), counts)
    local = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    index = np.zeros_like(local)
    stride = 1
    for axis in range(first.shape[1]):
        e = extent[box, axis]
        index += (first[box, axis] + local % e) * stride
        local //= e
        stride *= int(shape[axis])
    return box, index


def _cut(alo, ahi, blo, bhi, closed):
    """For each pair ``_meeting`` finds, ``(i, low, high)``: box ``i`` of ``a`` and
    where it meets the other box."""
    i, j = _meeting(alo, ahi, blo, bhi, closed)
    low = np.maximum(alo[i], blo[j])
    high = np.minimum(ahi[i], bhi[j])
    return zip(i.tolist(), low, high)


def _slices(a, b):
    """Index slices of an array shaped z, y, x for ``[a, b)`` given x first."""
    return tuple(slice(int(s), int(e)) for s, e in zip(a[::-1], b[::-1]))


def _lattice(piece, dimension):
    """An image box's lattice, ``(spacing, origin)`` in world coordinates, where its
    level's index 0 is; None for other geometry. Oriented images are refused:
    an AMR image is axis-aligned, as VTK's are."""
    from tack.arrays import UniformCoordinates

    coordinates = piece.geometry.values
    if not isinstance(coordinates, UniformCoordinates):
        return None
    if coordinates.ORIENTED:
        raise ValueError("an AMR image box is axis-aligned: give it no direction matrix")
    spacing = np.asarray(coordinates._spacing, np.float64)[:dimension]
    offset = np.asarray(piece.topology.offset, np.float64)
    corner = np.asarray(coordinates._origin, np.float64)[:dimension] + \
        np.asarray(piece.origin, np.float64)[:dimension]
    return spacing, corner - offset * spacing


def _text(box):
    return f"[{tuple(int(v) for v in box[0])}, {tuple(int(v) for v in box[1])})"

