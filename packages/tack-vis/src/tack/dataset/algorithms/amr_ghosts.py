# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Ghost cells of an AMR hierarchy, and fine data averaged down onto coarse.

Every value a box holds but does not own is a weighted sum of values other
boxes own: a ghost cell or point a sibling box holds is that box's value
(weight 1, AMReX's ``FillBoundary``); one no sibling holds takes the coarser
level's -- the coarse cell's value for a cell (injection), the coarse cell's
corners by linear weights for a point (``FillPatchTwoLevels``) -- and a coarse
cell a finer level covers takes its fine cells' average (``average_down``).
So each is a ``_Fill``: a ``tack.communication.Plan`` that brings the sources'
values next to each destination box, over devices and MPI ranks, and one
kernel that sums them by weight. A grid's positions are filled the same way,
so rectilinear and curvilinear boxes get their ghost layers' geometry from
their siblings, or interpolated in the coarse cells.
"""

import functools
import math

import numpy as np

import tack
from tack import arrays
from tack import communication as tc
from tack.algorithms import exclusive_scan
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.dataset import _box_pairs, _matching, ids
from tack.dataset.amr import AMRDataSet, _mark
from tack.dataset.carry import _point_values
from tack.dataset.dataset import DataSet, Field, _split_key
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import common_origin
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import StructuredTopology
from tack.runtime.dispatch import backend_of

__all__ = ["average_down"]


@tack.kernel
def _apply(staged, offsets, entries, weights, targets, values):
    # Each target's value: its sources' staged values, summed by weight.
    for j in range(targets.shape[0]):
        first = offsets[j]
        total = staged[entries[first]] * weights[first]
        for e in range(first + 1, offsets[j + 1]):
            total += staged[entries[e]] * weights[e]
        values[targets[j]] = total


@tack.kernel
def _pick(staged, picks, targets, values):
    # An integer value: its weightiest source's, unchanged.
    for j in range(targets.shape[0]):
        values[targets[j]] = staged[picks[j]]


@tack.kernel
def _divide(sums, measures, targets, values):
    for j in range(targets.shape[0]):
        values[targets[j]] = sums[targets[j]] / measures[targets[j]]


class _Fill:
    """Values of some entries of each box as weighted sums of other boxes' entries,
    one group at a time (``_Groups``: a rank's boxes on one device, their arrays
    end to end): a ``Plan`` between groups stages the sources other groups hold
    after the group's own values, and one kernel sums each target's sources by
    weight. Rows of one box per block, and a plan with a pair for every two boxes
    that touch, made AMR-Wind's 3,328 boxes 326,480 launches a fill."""

    def __init__(self, groups, kind, parts):
        """``parts[g]``, for each of this rank's groups: its ``_Part``, on its device.
        The rows stay there; only the sources other groups hold come to the host,
        to make the plan that stages them (none where a rank has one device)."""
        self.groups, self.kind = groups, kind
        blocks = groups.blocks
        self.boxes, receives = {}, {}
        for g in blocks.local:
            part = parts[g]
            n = groups.size(g, kind)
            count = int(part.groups.shape[0])
            with blocks.device(g):
                entries = tack.empty((count,), tack.i64)
                remote = tack.empty((count,), tack.i64)
                if count:
                    _local_entries(part.groups, part.places, g, entries, remote)
                held = exclusive_scan(remote, remote, count)
                keys = np.zeros((0, 2), np.int64)
                if held:
                    at = tack.empty((held,), tack.i64)
                    pairs = tack.empty((held, 2), tack.i64)
                    _remote_sources(part.groups, part.places, g, remote, at, pairs)
                    keys, slot = np.unique(pairs.numpy(), axis=0, return_inverse=True)
                    _put(at, tack.array(n + slot.reshape(-1).astype(np.int64), tack.i64),
                         entries)
                picks = tack.empty((part.rows,), tack.i64)
                if part.rows:
                    _heaviest(part.offsets, part.numerators, entries, picks)
            receives[g] = {int(source): (keys[keys[:, 0] == source, 1],
                                         n + np.flatnonzero(keys[:, 0] == source))
                           for source in np.unique(keys[:, 0])}
            self.boxes[g] = {"n": n, "m": len(keys), "rows": part.rows, "den": part.den,
                             "numerators": part.numerators,
                             "device": {"targets": part.targets, "offsets": part.offsets,
                                        "entries": entries, "picks": picks}}
        self.plan = _matching.plan(blocks, receives)

    def box_targets(self):
        """Each local box's targets as its own indices, on the host: for the
        fallbacks that work box by box."""
        groups, out = self.groups, {}
        for g in groups.blocks.local:
            members = groups.members[g]
            if not members:
                continue
            at = self.boxes[g]["device"]["targets"].numpy()
            starts = groups.starts(g, self.kind)
            bounds = np.searchsorted(at, np.append(starts, np.iinfo(np.int64).max))
            for n, i in enumerate(members):
                out[i] = at[bounds[n]:bounds[n + 1]] - starts[n]
        return out

    def run(self, values, combine=True):
        """Fill the targets of ``values`` (this rank's boxes' arrays of one kind, in
        the order the hierarchy holds them), in place. With ``combine`` false,
        integer arrays included, each target takes its weightiest source's
        value."""
        groups, blocks = self.groups, self.groups.blocks
        combined, in_place = {}, set()
        for g in blocks.local:
            box = self.boxes[g]
            parts = [values[i] for i in groups.members[g]]
            with blocks.device(g):
                # The group's values where they are one stretch of memory with room
                # for the staged sources after them (`in_device_buffers`, and
                # `ghost_cells`' grown arrays): filled where they are, nothing
                # copied in or back.
                joined = arrays._joined_view(parts, box["m"]) if parts else None
                if joined is not None:
                    combined[g] = joined
                    in_place.add(g)
                    continue
                if box["m"] or not parts:
                    # A device none of whose boxes take part still joins the exchange.
                    dtype = values[0].dtype if values else tack.f32
                    parts = [*parts, tack.zeros((box["m"],), dtype)]
                combined[g] = tack.concat(parts) if len(parts) > 1 else parts[0].copy()
        self.plan.run(combined)
        self.fill_groups(combined, combine)
        for g in blocks.local:
            if g in in_place or not self.boxes[g]["rows"]:
                continue
            with blocks.device(g):
                for i, start in zip(groups.members[g], groups.starts(g, self.kind)):
                    _copy_into(combined[g], int(start), values[i])

    def fill_groups(self, combined, combine=True):
        """The fill's kernel on each group's buffer (``combined[g]``: the group's
        values, then its staged sources, exchanged already)."""
        blocks = self.groups.blocks
        for g in blocks.local:
            box, buffer = self.boxes[g], combined[g]
            if not box["rows"]:
                continue
            with blocks.device(g):
                scalar = getattr(buffer.dtype, "scalar_dtype", buffer.dtype)
                tables = self._tables(g, scalar)
                if combine and scalar.numpy_dtype.kind == "f":
                    _apply(buffer, tables["offsets"], tables["entries"], tables["weights"],
                           tables["targets"], buffer)
                else:
                    _pick(buffer, tables["picks"], tables["targets"], buffer)

    def _tables(self, g, scalar):
        """Group ``g``'s rows on its device; the weights per floating type, made at
        its first run (``update_ghosts`` runs a fill again and again)."""
        box = self.boxes[g]
        tables = box["device"]
        key = ("weights", scalar)
        if key not in tables:
            weights = tack.empty((box["numerators"].shape[0],), scalar)
            if weights.shape[0]:
                _weights_of(box["numerators"], tack.full((1,), box["den"], scalar), weights)
            tables[key] = weights
        return {**tables, "weights": tables[key]}


@tack.kernel
def _local_entries(groups, places, g, entries, remote):
    # A source the group holds is its place; another group's is staged (remote).
    for e in range(groups.shape[0]):
        entries[e] = places[e]
        remote[e] = 1 if groups[e] != g else 0


@tack.kernel
def _remote_sources(groups, places, g, positions, at, pairs):
    # positions: the remote flags scanned. Each remote source's entry and (group,
    # place), in entry order.
    for e in range(groups.shape[0]):
        if groups[e] != g:
            k = positions[e]
            at[k] = e
            pairs[k, 0] = groups[e]
            pairs[k, 1] = places[e]


@tack.kernel
def _put(at, values, out):
    for k in range(at.shape[0]):
        out[at[k]] = values[k]


@tack.kernel
def _heaviest(offsets, numerators, entries, picks):
    # Each row's first source of the largest weight: an integer takes its value.
    for j in range(picks.shape[0]):
        best = offsets[j]
        for e in range(offsets[j] + 1, offsets[j + 1]):
            if numerators[e] > numerators[best]:
                best = e
        picks[j] = entries[best]


@tack.kernel
def _weights_of(numerators, den, weights):
    # A weight in the values' type: the integer numerator over the denominator,
    # one division, so the same rational rounds alike in every type.
    for e in range(weights.shape[0]):
        weights[e] = numerators[e] / den[0]


@tack.kernel
def _copy_into(source, start, values):
    for i in range(values.shape[0]):
        values[i] = source[start + i]


# -- where boxes are ------------------------------------------------------------------

class _Groups:
    """This rank's boxes by device: each device's boxes a group, a block of
    ``tack.communication``, its boxes' arrays end to end in hierarchy order; and,
    gathered over ranks, every box's group and the start of its arrays (of
    ``extents``, cells and points) in its group's."""

    def __init__(self, amr, extents, levels=None):
        """``levels``: the levels whose boxes are members (all by default); the groups
        are the devices of every box, so every rank makes the same blocks."""
        d = amr.dimension
        # Each box's device, looked up once: ``DataSet.device`` searches the dataset's
        # parts on every call (27 us; 3.6 s of average_down's per-box calls).
        every = [box.device for _, box in amr]
        devices = []
        for device in every:
            if device not in devices:
                devices.append(device)
        self.local, self.pieces, self.devices, of_box = [], [], [], []
        n = 0
        for level, pieces in enumerate(amr.levels):
            for k, box in enumerate(pieces):
                device, n = every[n], n + 1
                if levels is not None and level not in levels:
                    continue
                self.local.append((level, amr._first[level] + k))
                self.pieces.append(box)
                self.devices.append(device)
                of_box.append(devices.index(device))
        self.blocks = tc.Blocks(devices, amr.world)
        gids = list(self.blocks.local)
        self.members = {g: [i for i, k in enumerate(of_box) if k == n] for n, g in enumerate(gids)}
        sizes = {"cells": [], "points": []}
        for level, k in self.local:
            ext = extents[level][k]
            sizes["cells"].append(int(np.prod(ext[1, :d] - ext[0, :d])))
            sizes["points"].append(int(np.prod(ext[1, :d] - ext[0, :d] + 1)))
        self._sizes = {kind: np.asarray(v, np.int64) for kind, v in sizes.items()}
        base = {kind: np.zeros(len(self.local), np.int64) for kind in sizes}
        for g, members in self.members.items():
            for kind in sizes:
                lengths = self._sizes[kind][members]
                base[kind][members] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        # Per level, over every rank's boxes in the hierarchy's order: group and starts.
        self.group, self.cell_base, self.point_base = [], [], []
        for level in range(amr.num_levels):
            mine = [i for i, (lv, _) in enumerate(self.local) if lv == level]
            row = np.stack([np.asarray([gids[of_box[i]] for i in mine], np.int64),
                            base["cells"][mine], base["points"][mine]], axis=1) \
                if mine else np.zeros((0, 3), np.int64)
            every = np.concatenate(amr.world._allgather(row)).reshape(-1, 3)
            self.group.append(every[:, 0])
            self.cell_base.append(every[:, 1])
            self.point_base.append(every[:, 2])

    def size(self, g, kind):
        return int(self._sizes[kind][self.members[g]].sum())

    def starts(self, g, kind):
        lengths = self._sizes[kind][self.members[g]]
        return np.concatenate([[0], np.cumsum(lengths)[:-1]])

    def boxes(self, g, level):
        """The group's boxes of ``level``: their places in the hierarchy's order of this
        rank's boxes, and their global indices in the level."""
        places = [i for i in self.members[g] if self.local[i][0] == level]
        return places, np.asarray([self.local[i][1] for i in places], np.int64)


@tack.device_class
class _LevelBoxes:
    """A level's boxes on a device, found by the bins of a uniform grid over the
    level (each bin the boxes that meet it, closed, in index order), with each
    box's arrays' extent and its place in its group's: what the source kernels
    look up."""

    def __init__(self, amr, groups, extents, level, device):
        d = amr.dimension
        boxes, ext = amr._boxes[level], extents[level]
        lo, hi = amr.level_domain(level)
        n = len(boxes)
        size = np.ones(3, np.int64)
        if n:
            size[:d] = np.maximum((boxes[:, 1, :d] - boxes[:, 0, :d]).max(axis=0), 1)
        dims = np.ones(3, np.int64)
        dims[:d] = (hi[:d] - lo[:d]) // size[:d] + 2
        first = (boxes[:, 0, :] - lo) // size
        last = (boxes[:, 1, :] - lo) // size
        first[:, d:], last[:, d:] = 0, 0
        last = np.minimum(last, dims - 1)
        span = (last - first + 1).max(axis=0) if n else np.ones(3, np.int64)
        bins, ids = [], []
        for dz in range(int(span[2])):
            for dy in range(int(span[1])):
                for dx in range(int(span[0])):
                    at = first + np.array([dx, dy, dz])
                    ok = (at <= last).all(axis=1)
                    bins.append(at[ok, 0] + dims[0] * (at[ok, 1] + dims[1] * at[ok, 2]))
                    ids.append(np.flatnonzero(ok))
        bins = np.concatenate(bins) if bins else np.zeros(0, np.int64)
        ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        order = np.lexsort((ids, bins))
        offsets = np.concatenate([[0], np.cumsum(np.bincount(bins, minlength=int(np.prod(dims))))])
        def put(a):
            return tack.array(np.ascontiguousarray(a, np.int64).reshape(-1), tack.i64)

        with device:
            self.lo, self.hi = put(boxes[:, 0]), put(boxes[:, 1])
            self.elo, self.ehi = put(ext[:, 0]), put(ext[:, 1])
            self.group = put(groups.group[level])
            self.cbase, self.pbase = put(groups.cell_base[level]), put(groups.point_base[level])
            self.origin, self.size, self.dims = put(lo), put(size), put(dims)
            self.offsets, self.ids = put(offsets), put(ids[order])
        self.d = int(d)

    @tack.func
    def holds(self, j, q0, q1, q2, closed):
        """Whether box ``j`` holds index ``q``: ``[lo, hi)``, or ``[lo, hi]`` closed."""
        inside = 1
        for a in range(self.d):
            q = q0 if a == 0 else (q1 if a == 1 else q2)
            low = self.lo[3 * j + a]
            high = self.hi[3 * j + a]
            if q < low or q > high or (closed == 0 and q == high):
                inside = 0
        return inside

    @tack.func
    def find(self, q0, q1, q2, closed, skip, below):
        """The lowest box below ``below``, other than ``skip``, that holds ``q``, or -1."""
        found = -1
        b0 = (q0 - self.origin[0]) // self.size[0]
        b1 = (q1 - self.origin[1]) // self.size[1] if self.d > 1 else 0
        b2 = (q2 - self.origin[2]) // self.size[2] if self.d > 2 else 0
        if (b0 >= 0 and b1 >= 0 and b2 >= 0 and b0 < self.dims[0] and b1 < self.dims[1]
                and b2 < self.dims[2]):
            b = b0 + self.dims[0] * (b1 + self.dims[1] * b2)
            m = self.offsets[b]
            while m < self.offsets[b + 1] and found < 0:
                j = self.ids[m]
                if j != skip and j < below and self.holds(j, q0, q1, q2, closed) == 1:
                    found = j
                m += 1
        return found

    @tack.func
    def row_boxes(self, q1, q2, x0, x1, closed, skip, most, found):
        """The boxes other than ``skip`` that hold ``(q1, q2)`` along y and z
        (``[lo, hi)``, or ``[lo, hi]`` closed) and meet ``[x0, x1]`` along x, in
        ``found`` in increasing order, from the bins the row crosses: their number,
        or -1 where there are more than ``most`` (at most ``found``'s size)."""
        count = 0 * q1
        b1 = (q1 - self.origin[1]) // self.size[1] if self.d > 1 else 0 * q1
        b2 = (q2 - self.origin[2]) // self.size[2] if self.d > 2 else 0 * q2
        first = max((x0 - self.origin[0]) // self.size[0], 0)
        last = min((x1 - self.origin[0]) // self.size[0], self.dims[0] - 1)
        if b1 >= 0 and b2 >= 0 and b1 < self.dims[1] and b2 < self.dims[2]:
            for b0 in range(first, last + 1):
                b = b0 + self.dims[0] * (b1 + self.dims[1] * b2)
                for m in range(self.offsets[b], self.offsets[b + 1]):
                    j = self.ids[m]
                    meets = 1 if j != skip and count >= 0 else 0
                    if meets == 1 and (self.lo[3 * j] > x1 or self.hi[3 * j] < x0
                                       or (closed == 0 and self.hi[3 * j] == x0)):
                        meets = 0
                    for a in range(1, self.d):
                        q = q1 if a == 1 else q2
                        if meets == 1 and (q < self.lo[3 * j + a] or q > self.hi[3 * j + a]
                                           or (closed == 0 and q == self.hi[3 * j + a])):
                            meets = 0
                    at = count
                    if meets == 1:
                        # In order, once: a box is in every bin it reaches.
                        while at > 0 and found[at - 1] > j:
                            at -= 1
                        if at > 0 and found[at - 1] == j:
                            meets = 0
                    if meets == 1:
                        if count == most:
                            count = -1
                        else:
                            k = count
                            while k > at:
                                found[k] = found[k - 1]
                                k -= 1
                            found[at] = j
                            count += 1
        return count

    @tack.func
    def pick(self, found, count, q0, closed):
        """The first of ``found`` (``count`` of them, `row_boxes`') whose x extent
        holds ``q0``, or -1."""
        s = -1 + 0 * q0
        m = 0 * q0
        while m < count and s < 0:
            j = found[m]
            if q0 >= self.lo[3 * j] and (q0 < self.hi[3 * j]
                                         or (closed == 1 and q0 == self.hi[3 * j])):
                s = j
            m += 1
        return s

    @tack.func
    def find_scaled(self, q0, q1, q2, r0, r1, r2):
        """The lowest box that holds ``q / r`` (a finer level's point), closed, or -1."""
        found = -1
        b0 = (q0 // r0 - self.origin[0]) // self.size[0]
        b1 = (q1 // r1 - self.origin[1]) // self.size[1] if self.d > 1 else 0
        b2 = (q2 // r2 - self.origin[2]) // self.size[2] if self.d > 2 else 0
        if (b0 >= 0 and b1 >= 0 and b2 >= 0 and b0 < self.dims[0] and b1 < self.dims[1]
                and b2 < self.dims[2]):
            b = b0 + self.dims[0] * (b1 + self.dims[1] * b2)
            m = self.offsets[b]
            while m < self.offsets[b + 1] and found < 0:
                j = self.ids[m]
                inside = 1
                for a in range(self.d):
                    q = q0 if a == 0 else (q1 if a == 1 else q2)
                    r = r0 if a == 0 else (r1 if a == 1 else r2)
                    if q < self.lo[3 * j + a] * r or q > self.hi[3 * j + a] * r:
                        inside = 0
                if inside == 1:
                    found = j
                m += 1
        return found

    @tack.func
    def place(self, j, q0, q1, q2, points):
        """Where index ``q`` of box ``j``'s arrays is in its group's: its start plus the
        index's place in the box's extent, x fastest (points one more a side)."""
        extra = 1 if points == 1 else 0
        nx = self.ehi[3 * j] - self.elo[3 * j] + extra
        ny = self.ehi[3 * j + 1] - self.elo[3 * j + 1] + extra
        flat = q2 - self.elo[3 * j + 2] if self.d > 2 else q2 - q2
        if self.d > 1:
            flat = flat * ny + (q1 - self.elo[3 * j + 1])
        flat = flat * nx + (q0 - self.elo[3 * j])
        start = self.pbase[j] if points == 1 else self.cbase[j]
        return start + flat


@tack.func
def _entry(offsets, n, e):
    """The box (an index into ``offsets``) whose entries hold entry ``e``."""
    low = 0
    high = n
    while high - low > 1:
        mid = (low + high) // 2
        if offsets[mid] <= e:
            low = mid
        else:
            high = mid
    return low


@tack.func
def _index(same, j, r, points, use_box):
    """Entry ``r`` of box ``j``'s extent (or of the box itself), x fastest: ``(q0, q1,
    q2)``."""
    extra = 1 if points == 1 else 0
    lo0 = same.lo[3 * j] if use_box == 1 else same.elo[3 * j]
    lo1 = same.lo[3 * j + 1] if use_box == 1 else same.elo[3 * j + 1]
    lo2 = same.lo[3 * j + 2] if use_box == 1 else same.elo[3 * j + 2]
    hi0 = same.hi[3 * j] if use_box == 1 else same.ehi[3 * j]
    hi1 = same.hi[3 * j + 1] if use_box == 1 else same.ehi[3 * j + 1]
    nx = hi0 - lo0 + extra
    ny = (hi1 - lo1 + extra) if same.d > 1 else hi1 - hi1 + 1
    q0 = lo0 + r % nx
    q1 = lo1 + (r // nx) % ny if same.d > 1 else lo1
    q2 = lo2 + r // (nx * ny) if same.d > 2 else lo2
    return q0, q1, q2


@tack.func
def _hole(same, j, a, points):
    """Axis ``a`` of box ``j``'s grown extent, for ``_row_of``: its size, and the
    start and size of the hole in it, the entries a sibling never holds (the box's
    own cells, or its own points off its boundary); an axis the boxes lack is one
    entry, inside the hole."""
    n = 1 + 0 * j
    start = 0 * j
    size = 1 + 0 * j
    if a < same.d:
        lo = same.lo[3 * j + a]
        hi = same.hi[3 * j + a]
        elo = same.elo[3 * j + a]
        n = same.ehi[3 * j + a] - elo + points
        start = lo - elo + points
        size = hi - lo - points
    return n, start, size


def _shell_sizes(boxes, extents, d, points):
    """Each box's shell entries (``_row_of``): its grown extent's cells or points
    less the hole."""
    lo, hi = boxes[:, 0, :d], boxes[:, 1, :d]
    full = np.prod(extents[:, 1, :d] - extents[:, 0, :d] + points, axis=1)
    hole = np.prod(np.maximum(hi - lo - points, 0), axis=1)
    return full - hole


_ROW_MOST = tack.constant(32)
# The boxes a row keeps (at most _ROW_MOST): a launch's argument, so a test can make
# rows overflow onto the per-entry search.
_row_most = int(_ROW_MOST)


@tack.func
def _row_of(same, ids, row_starts, n, r, points):
    """Row ``r`` (along x) of the boxes' grown extents, box ``b``'s rows from
    ``row_starts[b]``: its box ``j``, its ``(q1, q2)``, its extent along x and
    where the hole (``_hole``) starts in it and its size there (0 where the row
    misses the hole)."""
    b = _entry(row_starts, n, r)
    j = ids[b]
    local = r - row_starts[b]
    nx, ax, hx = _hole(same, j, 0, points)
    ny, ay, hy = _hole(same, j, 1, points)
    nz, az, hz = _hole(same, j, 2, points)
    y = local % ny
    z = local // ny
    skip = 0 * hx
    if hx > 0 and hy > 0 and hz > 0 and y >= ay and y < ay + hy and z >= az and z < az + hz:
        skip = hx
    return j, same.elo[3 * j + 1] + y, same.elo[3 * j + 2] + z, nx, ax, skip


@tack.kernel
def _shell_rows(same, ids, row_starts, n, points, sizes):
    # Each row's entries in the boxes' shells: its extent less the hole.
    for r in range(sizes.shape[0]):
        j, q1, q2, nx, ax, skip = _row_of(same, ids, row_starts, n, r, points)
        sizes[r] = nx - skip


def _shell_layout(same, ids, extents, d, points, device):
    """The rows of boxes ``ids``' shells (``_hole``): each box's first row (host
    and device) and, on the device, each row's first entry, x fastest, in the
    order the shells were enumerated entry by entry."""
    spans = np.ones((len(ids), 3), np.int64)
    spans[:, :d] = extents[:, 1, :d] - extents[:, 0, :d] + points
    row_starts = np.concatenate([[0], np.cumsum(spans[:, 1] * spans[:, 2])]).astype(np.int64)
    rows = int(row_starts[-1])
    with device:
        row_starts_d = tack.array(row_starts, tack.i64)
        entries = tack.empty((rows + 1,), tack.i64)
        _shell_rows(same, tack.array(ids, tack.i64), row_starts_d, len(ids), points, entries[:rows])
        exclusive_scan(entries, entries, rows, close=True, total=False)
    return rows, row_starts_d, entries


@tack.kernel
def _ghost_cells_of(same, coarse, has_coarse, r0, r1, r2, ids, row_starts, n, rows,
                    entry_starts, most, targets, groups, places):
    # One thread a row (along x) of the boxes' shells (`_row_of`): each entry a
    # ghost cell's target place and its one source, a sibling's (the lowest
    # holding it) else the coarse cell's; -1 targets for the boxes' own cells, -1
    # groups for ghosts nothing holds (hidden). The siblings meeting the row are
    # found once (`row_boxes`), so an entry only scans them; per entry, decoding
    # and the bins' search were 0.5 s of AMR-Wind's 47M shell cells.
    for r in range(rows):
        j, q1, q2, nx, ax, skip = _row_of(same, ids, row_starts, n, r, 0)
        x0 = same.elo[3 * j]
        big = same.ids.shape[0] + same.lo.shape[0]
        found = tack.local_array(tack.i64, _ROW_MOST)
        count = same.row_boxes(q1, q2, x0, x0 + nx - 1, 0, j, most, found)
        e = entry_starts[r]
        for x in range(nx):
            if skip == 0 or x < ax or x >= ax + skip:
                q0 = x0 + x
                target = -1
                group = -1
                place = 0 * j
                if same.holds(j, q0, q1, q2, 0) == 0:
                    target = same.place(j, q0, q1, q2, 0)
                    s = same.pick(found, count, q0, 0)
                    if count < 0:
                        s = same.find(q0, q1, q2, 0, j, big)
                    if s >= 0:
                        group = same.group[s]
                        place = same.place(s, q0, q1, q2, 0)
                    elif has_coarse == 1:
                        c0 = q0 // r0
                        c1 = q1 // r1
                        c2 = q2 // r2
                        t = coarse.find(c0, c1, c2, 0, -1,
                                        coarse.ids.shape[0] + coarse.lo.shape[0])
                        if t >= 0:
                            group = coarse.group[t]
                            place = coarse.place(t, c0, c1, c2, 0)
                targets[e] = target
                groups[e] = group
                places[e] = place
                e += 1


@tack.kernel
def _ghost_points_of(same, coarse, has_coarse, r0, r1, r2, den, scale, ids, row_starts, n,
                     rows, entry_starts, most, write, starts, counts, targets, groups, places,
                     weights):
    # One thread a row (along x) of the boxes' shells (`_row_of`), each point its
    # sources' count (-1 not a target, 0 hidden) and, with ``write``, the sources
    # at ``starts``. A ghost point is a sibling's (the lowest holding it), else
    # interpolated in the coarse cell holding it by its corners' owners (the
    # lowest coarse box holding each); an own point a lower sibling also holds is
    # that sibling's. Weights are integer numerators over ``den`` (the ratios'
    # product) times ``scale``, so a group's levels share one denominator and no
    # device needs f64. The siblings meeting the row are found once.
    for r in range(rows):
        j, q1, q2, nx, ax, skip = _row_of(same, ids, row_starts, n, r, 1)
        x0 = same.elo[3 * j]
        big = same.ids.shape[0] + same.lo.shape[0]
        found = tack.local_array(tack.i64, _ROW_MOST)
        count = same.row_boxes(q1, q2, x0, x0 + nx - 1, 1, j, most, found)
        e = entry_starts[r]
        for x in range(nx):
            if skip == 0 or x < ax or x >= ax + skip:
                q0 = x0 + x
                points = -1
                at = starts[e] if write == 1 else 0 * e
                s = same.pick(found, count, q0, 1)
                if same.holds(j, q0, q1, q2, 1) == 1:
                    if count < 0:
                        s = same.find(q0, q1, q2, 1, j, j)
                    elif s >= j:
                        s = -1                  # only a lower sibling's
                    if s >= 0:
                        points = 1
                        if write == 1:
                            targets[e] = same.place(j, q0, q1, q2, 1)
                            groups[at] = same.group[s]
                            places[at] = same.place(s, q0, q1, q2, 1)
                            weights[at] = den * scale
                else:
                    points = 0
                    if write == 1:
                        targets[e] = same.place(j, q0, q1, q2, 1)
                    if count < 0:
                        s = same.find(q0, q1, q2, 1, j, big)
                    if s >= 0:
                        points = 1
                        if write == 1:
                            groups[at] = same.group[s]
                            places[at] = same.place(s, q0, q1, q2, 1)
                            weights[at] = den * scale
                    elif has_coarse == 1:
                        points = _coarse_corners(coarse, q0, q1, q2, r0, r1, r2, scale, big,
                                                 write, at, groups, places, weights)
                if write == 0:
                    counts[e] = points
                e += 1


@tack.func
def _coarse_corners(coarse, q0, q1, q2, r0, r1, r2, scale, big, write, at, groups, places,
                    weights):
    """A ghost point no sibling holds: interpolated in the coarse cell holding it
    (the lowest coarse box's), each corner with a weight by its owner (the lowest
    coarse box holding it); written at ``at`` with ``write``. Its number of
    sources, 0 where no coarse box holds it."""
    count = 0 * q0
    t = coarse.find_scaled(q0, q1, q2, r0, r1, r2)
    if t >= 0:
        # The coarse cell: q // r, kept within the box; q's place in it.
        c0 = min(max(q0 // r0, coarse.lo[3 * t]), coarse.hi[3 * t] - 1)
        c1 = q1 - q1
        c2 = q2 - q2
        if coarse.d > 1:
            c1 = min(max(q1 // r1, coarse.lo[3 * t + 1]), coarse.hi[3 * t + 1] - 1)
        if coarse.d > 2:
            c2 = min(max(q2 // r2, coarse.lo[3 * t + 2]), coarse.hi[3 * t + 2] - 1)
        t0 = q0 - c0 * r0
        t1 = q1 - c1 * r1
        t2 = q2 - c2 * r2
        corners = 1
        for a in range(coarse.d):
            corners = corners * 2
        for c in range(corners):
            a0 = c & 1
            a1 = (c >> 1) & 1
            a2 = (c >> 2) & 1
            w = t0 if a0 == 1 else r0 - t0
            if coarse.d > 1:
                w = w * (t1 if a1 == 1 else r1 - t1)
            if coarse.d > 2:
                w = w * (t2 if a2 == 1 else r2 - t2)
            if w > 0:
                k0 = c0 + a0
                k1 = c1 + a1 if coarse.d > 1 else c1
                k2 = c2 + a2 if coarse.d > 2 else c2
                if write == 1:
                    # The corner's owner: only the writing pass needs it.
                    owner = coarse.find(k0, k1, k2, 1, -1, big + coarse.lo.shape[0])
                    groups[at + count] = coarse.group[owner]
                    places[at + count] = coarse.place(owner, k0, k1, k2, 1)
                    weights[at + count] = w * scale
                count += 1
    return count


_MOST_FINE = tack.constant(64)


@tack.kernel
def _down_cells_of(coarse, fine, r0, r1, r2, ids, offsets, n, write, starts, counts, targets,
                   groups, places):
    # Each own cell of the coarse boxes: the fine cells over it, by fine box (the
    # lowest holding each), then x fastest within; -1 for a cell no fine box covers.
    for e in range(counts.shape[0]):
        b = _entry(offsets, n, e)
        j = ids[b]
        q0, q1, q2 = _index(coarse, j, e - offsets[b], 0, 1)
        big = fine.ids.shape[0] + fine.lo.shape[0]
        boxes = tack.local_array(tack.i64, _MOST_FINE)
        found = tack.local_array(tack.i64, _MOST_FINE)
        count = 0
        n1 = r1 if fine.d > 1 else 1
        n2 = r2 if fine.d > 2 else 1
        for k2 in range(n2):
            for k1 in range(n1):
                for k0 in range(r0):
                    f0 = q0 * r0 + k0
                    f1 = q1 * r1 + k1 if fine.d > 1 else q1
                    f2 = q2 * r2 + k2 if fine.d > 2 else q2
                    s = fine.find(f0, f1, f2, 0, -1, big)
                    if s >= 0:
                        # Insert after every entry of a box at most s: by box, stable.
                        at = count
                        while at > 0 and boxes[at - 1] > s:
                            boxes[at] = boxes[at - 1]
                            found[at] = found[at - 1]
                            at -= 1
                        boxes[at] = s
                        found[at] = fine.place(s, f0, f1, f2, 0)
                        count += 1
        if write == 1:
            if count > 0:
                targets[e] = coarse.place(j, q0, q1, q2, 0)
            for m in range(count):
                groups[starts[e] + m] = fine.group[boxes[m]]
                places[starts[e] + m] = found[m]
        else:
            counts[e] = count if count > 0 else -1


@tack.kernel
def _down_points_of(coarse, fine, r0, r1, r2, ids, offsets, n, targets, groups, places):
    # Each own point of the coarse boxes: the fine point at its location, the lowest
    # fine box's holding it; -1 targets where none does.
    for e in range(targets.shape[0]):
        b = _entry(offsets, n, e)
        j = ids[b]
        q0, q1, q2 = _index(coarse, j, e - offsets[b], 1, 1)
        f0 = q0 * r0
        f1 = q1 * r1 if fine.d > 1 else q1
        f2 = q2 * r2 if fine.d > 2 else q2
        s = fine.find(f0, f1, f2, 1, -1, fine.ids.shape[0] + fine.lo.shape[0])
        target = -1
        group = -1
        place = 0 * e
        if s >= 0:
            target = coarse.place(j, q0, q1, q2, 1)
            group = fine.group[s]
            place = fine.place(s, f0, f1, f2, 1)
        targets[e] = target
        groups[e] = group
        places[e] = place


def _ratio3(amr, level):
    r = np.ones(3, np.int64)
    r[:amr.dimension] = np.asarray(amr.ratios[level], np.int64)[:amr.dimension]
    return r


def _tables(amr, groups, extents, g):
    """Each level's ``_LevelBoxes`` on group ``g``'s device, made when first asked."""
    made = {}

    def table(level):
        if level not in made:
            made[level] = _LevelBoxes(amr, groups, extents, level, groups.blocks.device(g))
        return made[level]
    return table


def _entries(sizes, device):
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    with device:
        return offsets, tack.array(offsets, tack.i64)


class _Part:
    """A group's rows of a fill, on the group's device: ``targets`` (places in the
    group's arrays), CSR ``offsets`` into the sources, each a source ``groups``
    and ``places`` (that group's arrays') with weight ``numerators / den``,
    integers, so no device needs f64 to hold them."""

    def __init__(self, targets, offsets, groups, places, numerators, den):
        self.targets, self.offsets = targets, offsets
        self.groups, self.places, self.numerators = groups, places, numerators
        self.den = int(den)
        self.rows = int(targets.shape[0])

    @classmethod
    def empty(cls):
        none = tack.zeros((0,), tack.i64)
        return cls(none, tack.zeros((1,), tack.i64), none, none, none, 1)


@tack.kernel
def _one_source_flags(targets, groups, kept, gone):
    # A target with its source (kept), or with none (gone: hidden).
    for e in range(targets.shape[0]):
        kept[e] = 1 if targets[e] >= 0 and groups[e] >= 0 else 0
        gone[e] = 1 if targets[e] >= 0 and groups[e] < 0 else 0


@tack.kernel
def _one_source_rows(targets, groups, places, kept, gone, rows, sources, at, hidden):
    # kept and gone scanned: each kept target and its source at its row, each gone
    # target among the hidden.
    for e in range(targets.shape[0]):
        if targets[e] >= 0:
            if groups[e] >= 0:
                k = kept[e]
                rows[k] = targets[e]
                sources[k] = groups[e]
                at[k] = places[e]
            else:
                hidden[gone[e]] = targets[e]


@tack.kernel
def _counted_flags(counts, kept, gone, sizes):
    # A target with sources (kept), with none (gone: hidden), or none (-1); and
    # its number of sources, to scan into their starts.
    for e in range(counts.shape[0]):
        kept[e] = 1 if counts[e] > 0 else 0
        gone[e] = 1 if counts[e] == 0 else 0
        sizes[e] = max(counts[e], 0)


@tack.kernel
def _counted_rows(counts, targets, starts, kept, gone, rows, offsets, hidden):
    for e in range(counts.shape[0]):
        if counts[e] > 0:
            rows[kept[e]] = targets[e]
            offsets[kept[e]] = starts[e]
        elif counts[e] == 0:
            hidden[gone[e]] = targets[e]


def _one_source(targets, groups, places):
    """The rows of targets with one source each (``targets`` -1 where an entry is
    none, ``groups`` -1 where nothing holds it), compacted on the current device:
    a ``_Part`` of weights 1, and the targets nothing holds."""
    n = int(targets.shape[0])
    if not n:
        return _Part.empty(), tack.zeros((0,), tack.i64)
    kept = tack.empty((n,), tack.i64)
    gone = tack.empty((n,), tack.i64)
    _one_source_flags(targets, groups, kept, gone)
    rows = exclusive_scan(kept, kept, n)
    lost = exclusive_scan(gone, gone, n)
    out = [tack.empty((rows,), tack.i64) for _ in range(3)]
    hidden = tack.empty((lost,), tack.i64)
    _one_source_rows(targets, groups, places, kept, gone, *out, hidden)
    return _Part(out[0], tack.arange(rows + 1, dtype=tack.i64), out[1], out[2],
                 tack.full((rows,), 1, tack.i64), 1), hidden


def _counted(counts, targets, starts, m, sources):
    """The rows of targets with ``counts`` sources each (-1 not a target, 0 hidden),
    their sources at ``starts`` (``m`` in all) of ``sources`` (``(groups, places,
    numerators, den)``), compacted on the current device: a ``_Part``, and the
    targets nothing holds."""
    n = int(counts.shape[0])
    if not n:
        return _Part.empty(), tack.zeros((0,), tack.i64)
    kept = tack.empty((n,), tack.i64)
    gone = tack.empty((n,), tack.i64)
    sizes = tack.empty((n,), tack.i64)
    _counted_flags(counts, kept, gone, sizes)
    rows = exclusive_scan(kept, kept, n)
    lost = exclusive_scan(gone, gone, n)
    targets_out = tack.empty((rows,), tack.i64)
    offsets = tack.full((rows + 1,), m, tack.i64)
    hidden = tack.empty((lost,), tack.i64)
    _counted_rows(counts, targets, starts, kept, gone, targets_out, offsets, hidden)
    groups, places, numerators, den = sources
    return _Part(targets_out, offsets, groups[:m] if m else groups[:0],
                 places[:m] if m else places[:0], numerators[:m] if m else numerators[:0],
                 den), hidden


@tack.kernel
def _clamped(counts, sizes):
    for e in range(counts.shape[0]):
        sizes[e] = max(counts[e], 0)


def _ghost_parts(amr, groups, extents):
    """Every group's ghost-cell and ghost-point fills' parts (``_Part``s) and its
    hidden cells and points (places in its arrays), on the group's device: each
    level's boxes' shells are found by `_ghost_cells_of` and `_ghost_points_of` (a
    thread a row of the shells)
    into views of arrays for the whole group, then compacted there. Nothing is
    read back but totals: reading the sources back, assembling them in NumPy and
    uploading them again was 4.2 s of AMR-Wind's five levels."""
    blocks, d = groups.blocks, amr.dimension
    cell_parts, point_parts, hidden = {}, {}, {}
    for g in blocks.local:
        device = blocks.device(g)
        table = _tables(amr, groups, extents, g)
        runs = []
        for level in range(amr.num_levels):
            places, ids = groups.boxes(g, level)
            if places:
                own, ext = amr._boxes[level][ids], extents[level][ids]
                runs.append((level, ids, _shell_sizes(own, ext, d, 0),
                             _shell_sizes(own, ext, d, 1)))
        # The point weights' common denominator: each level's ratios' product
        # divides it, so a weight is the same rational, rounded alike.
        dens = [int(np.prod(_ratio3(amr, level - 1)[:d])) if level else 1
                for level, *_ in runs]
        den = int(np.lcm.reduce(dens)) if dens else 1
        ncells = sum(int(r[2].sum()) for r in runs)
        npoints = sum(int(r[3].sum()) for r in runs)
        with device:
            targets = tack.empty((ncells,), tack.i64)
            sources = tack.empty((ncells,), tack.i64)
            places = tack.empty((ncells,), tack.i64)
            counts = tack.empty((npoints,), tack.i64)
            point_targets = tack.full((npoints,), -1, tack.i64)
            one = tack.zeros((1,), tack.i64)
            launches = []
            a = b = 0
            for (level, ids, cell_sizes, point_sizes), level_den in zip(runs, dens):
                same = table(level)
                coarse = table(level - 1) if level else same
                r = _ratio3(amr, level - 1) if level else np.ones(3, np.int64)
                ids_d = tack.array(ids, tack.i64)
                ext = extents[level][ids]
                k = int(cell_sizes.sum())
                if k:
                    rows, row_starts, entries = _shell_layout(same, ids, ext, d, 0, device)
                    _ghost_cells_of(same, coarse, int(level > 0), *map(int, r), ids_d,
                                    row_starts, len(ids), rows, entries, _row_most,
                                    targets[a:a + k],
                                    sources[a:a + k], places[a:a + k])
                a += k
                k = int(point_sizes.sum())
                rows, row_starts, entries = 0, None, None
                if k:
                    rows, row_starts, entries = _shell_layout(same, ids, ext, d, 1, device)
                    _ghost_points_of(same, coarse, int(level > 0), *map(int, r), level_den,
                                     den // level_den, ids_d, row_starts, len(ids), rows,
                                     entries, _row_most, 0, one, counts[b:b + k], one, one,
                                     one, one)
                launches.append((level, same, coarse, r, level_den, ids_d, row_starts, len(ids),
                                 rows, entries, b, k))
                b += k
            cell_parts[g], hidden_cells = _one_source(targets, sources, places)
            starts = tack.empty((max(npoints, 1),), tack.i64)
            if npoints:
                _clamped(counts, starts)
            m = exclusive_scan(starts, starts, npoints)
            point_sources = [tack.empty((max(m, 1),), tack.i64) for _ in range(3)]
            for (level, same, coarse, r, level_den, ids_d, row_starts, nb, rows, entries, b,
                 k) in launches:
                if k:
                    _ghost_points_of(same, coarse, int(level > 0), *map(int, r), level_den,
                                     den // level_den, ids_d, row_starts, nb, rows, entries,
                                     _row_most, 1,
                                     starts[b:b + k], counts[b:b + k],
                                     point_targets[b:b + k], *point_sources)
            point_parts[g], hidden_points = _counted(counts, point_targets, starts, m,
                                                     (*point_sources, den))
        hidden[g] = (hidden_cells, hidden_points)
    return cell_parts, point_parts, hidden


def _hidden_by_box(groups, hidden, kind):
    """Each local box's hidden entries of ``kind`` (``hidden``, `_ghost_parts`'), as
    its own indices on the host: only curvilinear boxes' positions need them."""
    c = 0 if kind == "cells" else 1
    out = {}
    for g in groups.blocks.local:
        members = groups.members[g]
        if not members:
            continue
        at = hidden[g][c].numpy()
        starts = groups.starts(g, kind)
        bounds = np.searchsorted(at, np.append(starts, np.iinfo(np.int64).max))
        for n, i in enumerate(members):
            out[i] = at[bounds[n]:bounds[n + 1]] - starts[n]
    return out


def _down_parts(amr, groups, level, kind):
    """Every group's ``average_down`` part for ``level``'s boxes (``_Part``s): each
    covered coarse cell (point) and the fine cells (point) over it."""
    blocks, extents, d = groups.blocks, amr._extents, amr.dimension
    r = _ratio3(amr, level)
    if kind == "cells" and int(np.prod(r[:d])) > int(_MOST_FINE):
        raise ValueError(f"average_down takes refinement ratios of at most {int(_MOST_FINE)} "
                         f"fine cells a coarse cell, not {tuple(r[:d])}")
    parts = {}
    for g in blocks.local:
        device = blocks.device(g)
        places, ids = groups.boxes(g, level)
        if not places:
            with device:
                parts[g] = _Part.empty()
            continue
        table = _tables(amr, groups, extents, g)
        coarse, fine = table(level), table(level + 1)
        box = amr._boxes[level][ids]
        nb = len(ids)
        extra = 1 if kind == "points" else 0
        offsets, offsets_d = _entries(np.prod(box[:, 1, :d] - box[:, 0, :d] + extra, axis=1),
                                      device)
        total = int(offsets[-1])
        with device:
            ids_d = tack.array(ids, tack.i64)
            targets = tack.full((total,), -1, tack.i64)
            if kind == "points":
                grp = tack.empty((total,), tack.i64)
                where = tack.empty((total,), tack.i64)
                if total:
                    _down_points_of(coarse, fine, *map(int, r), ids_d, offsets_d, nb, targets,
                                    grp, where)
                parts[g], _ = _one_source(targets, grp, where)
                continue
            counts = tack.empty((total,), tack.i64)
            one = tack.zeros((1,), tack.i64)
            if total:
                _down_cells_of(coarse, fine, *map(int, r), ids_d, offsets_d, nb, 0, one, counts,
                               one, one, one)
            starts = tack.empty((max(total, 1),), tack.i64)
            if total:
                _clamped(counts, starts)
            m = exclusive_scan(starts, starts, total)
            grp = tack.empty((max(m, 1),), tack.i64)
            where = tack.empty((max(m, 1),), tack.i64)
            if total:
                _down_cells_of(coarse, fine, *map(int, r), ids_d, offsets_d, nb, 1, starts,
                               counts, targets, grp, where)
            parts[g], _ = _counted(counts, targets, starts, m,
                                   (grp, where, tack.full((max(m, 1),), 1, tack.i64), 1))
    return parts


# -- ghost cells ----------------------------------------------------------------------

def amr_ghost_cells(amr, layers, fields):
    """``ghost_cells`` of an ``AMRDataSet``: each box grown by ``layers`` cells within
    its level's domain, its ghosts filled from its siblings or the coarser level."""
    from tack.dataset.algorithms.ghost_cells import _carried, _kind

    if amr.layers:
        raise ValueError("the hierarchy already has ghost cells; give ghost_cells the "
                         "hierarchy without them")
    d = amr.dimension
    pieces = [box for _, box in amr]
    names = _carried(pieces, fields, "ghost_cells")
    common_origin(amr._as_pieces(), "ghost_cells of an AMRDataSet")
    extents = []
    for level in range(amr.num_levels):
        lo, hi = amr.level_domain(level)
        grown = amr._boxes[level].copy()
        grown[:, 0, :d] = np.maximum(grown[:, 0, :d] - layers, lo[:d])
        grown[:, 1, :d] = np.minimum(grown[:, 1, :d] + layers, hi[:d])
        extents.append(grown)
    groups = _Groups(amr, extents)
    cell_parts, point_parts, hidden = _ghost_parts(amr, groups, extents)
    fills = {"cells": _Fill(groups, "cells", cell_parts),
             "points": _Fill(groups, "points", point_parts)}

    # The grown boxes: own values placed, ghosts filled; positions too, for grids
    # whose positions are not an image's.
    # Only grids other than images read positions (an image keeps its lattice), but
    # an image box can be a source of their ghosts: all or none, over every rank.
    need = any(amr.world._allgather(
        any(not isinstance(piece.geometry.values, UniformCoordinates) for piece in pieces)))
    grown_boxes, arrays_by_kind, positions = [], [], []
    grown = _GroupArrays(groups, fills)
    # A device's boxes' values in one launch where they are views of one array.
    placed = [{} for _ in pieces]
    for name in names:
        for g in groups.blocks.local:
            members = groups.members[g]
            if members:
                with groups.blocks.device(g):
                    views = grown.placed(g, name, _kind(pieces[members[0]].fields[name]),
                                         [pieces[i].fields[name].values for i in members],
                                         amr, extents)
                for i, view in zip(members, views or ()):
                    placed[i][name] = view
    for i, ((level, k), piece) in enumerate(zip(groups.local, pieces)):
        box, ext = amr._boxes[level][k], extents[level][k]
        with groups.devices[i]:
            assembled = placed[i]
            for name in names:
                if name in assembled:
                    continue
                kind = _kind(piece.fields[name])
                assembled[name] = _placed(piece.fields[name].values, kind, box, ext, d,
                                          functools.partial(grown.view, i, name, kind))
            positions.append(tack.array(_placed_host(piece.positions(), "points", box, ext, d),
                                        tack.vec3(piece.dtype)) if need else None)
        arrays_by_kind.append(assembled)
        grown_boxes.append((i, level, k, piece, box, ext))
    # Each field's kind, as a rank with boxes knows it: a rank without any runs the
    # same fills, with nothing of its own to put in them.
    mine = [_kind(pieces[0].fields[name]) for name in names] if pieces else None
    kinds = next((k for k in amr.world._allgather(mine) if k is not None), [])
    for i, kind in enumerate(kinds):
        fills[kind].run([arrays[names[i]] for arrays in arrays_by_kind])
    if need:
        fills["points"].run(positions)

    marks = _grown_ghosts(amr, groups, extents, hidden)
    # Only curvilinear boxes' positions need their hidden points, on the host.
    hidden_points = _hidden_by_box(groups, hidden, "points") if need else {}
    none = np.zeros(0, np.int64)
    out = [[] for _ in range(amr.num_levels)]
    for i, level, k, piece, box, ext in grown_boxes:
        out[level].append(_grown_box(piece, groups.devices[i], box, ext, d, arrays_by_kind[i],
                                     names, positions[i], hidden_points.get(i, none), marks[i]))
    return AMRDataSet._grown(amr, out, extents, int(layers), fills)




def _placed_host(values, kind, box, ext, d):
    """A box's host values (``(n,)`` or ``(n, w)``) placed in arrays of the grown
    extent, zeros elsewhere."""
    values = np.asarray(values)
    extra = values.shape[1:]
    if kind == "cells":
        own, grown = box[1, :d] - box[0, :d], ext[1, :d] - ext[0, :d]
    else:
        own, grown = box[1, :d] - box[0, :d] + 1, ext[1, :d] - ext[0, :d] + 1
    out = np.zeros(tuple(int(v) for v in grown[::-1]) + extra, values.dtype)
    start = box[0, :d] - ext[0, :d]
    region = tuple(slice(int(s), int(s + n)) for s, n in zip(start[::-1], own[::-1]))
    out[region] = values.reshape(tuple(int(v) for v in own[::-1]) + extra)
    return out.reshape((-1, *extra))


class _GroupArrays:
    """The grown boxes' arrays, a group's (``_Groups``: a rank's boxes on a device) in
    one buffer per field, each box's a view of it in the group's order, with room
    after them for the sources the fill stages from other groups: the form
    ``_Fill.run`` fills in place, and ``update_ghosts`` again after."""

    def __init__(self, groups, fills):
        self.groups, self.fills, self.buffers = groups, fills, {}
        self.group_of = {i: g for g in groups.blocks.local for i in groups.members[g]}
        self.starts = {kind: {i: int(start) for g in groups.blocks.local
                              for i, start in zip(groups.members[g], groups.starts(g, kind))}
                       for kind in ("cells", "points")}

    def buffer(self, g, name, kind, dtype):
        """Group ``g``'s zeroed buffer for ``name``, made when first asked, with room
        for the fill's staged sources after the boxes'."""
        key = (g, name)
        if key not in self.buffers:
            extra = self.fills[kind].boxes[g]["m"]
            self.buffers[key] = tack.zeros((self.groups.size(g, kind) + extra,), dtype)
        return self.buffers[key]

    def view(self, i, name, kind, values, n):
        """Box ``i``'s zeroed array of ``n`` entries for ``name``: its stretch of the
        group's buffer, made at the group's first box; a new array where a box's
        type differs from the group's."""
        buffer = self.buffer(self.group_of[i], name, kind, values.dtype)
        if buffer.dtype != values.dtype:
            return tack.zeros((n,), values.dtype)
        start = self.starts[kind][i]
        return buffer[start:start + n]

    def placed(self, g, name, kind, values, amr, extents):
        """Group ``g``'s boxes' ``values`` (of ``name``, its members' in order) placed
        in their grown extents, zeros elsewhere, by one launch (`_place_rows`), as
        views of the group's buffer: where the values are one stretch of an array
        (`arrays._joined_view`, as `vtk_to_dataset` and `in_device_buffers` lay a
        hierarchy out), each box's its own entries; else None, for `_placed` box by
        box (one launch a box was 1.3 s of AMR-Wind's five levels)."""
        if kind is None or any(type(v) is not tack.Array or v.ndim != 1 for v in values):
            return None
        groups, d = self.groups, amr.dimension
        members = groups.members[g]
        where = [groups.local[i] for i in members]
        own = np.stack([amr._boxes[level][k] for level, k in where])
        grown = np.stack([extents[level][k] for level, k in where])
        extra = int(kind == "points")
        sizes = np.ones((len(where), 3), np.int64)
        sizes[:, :d] = own[:, 1, :d] - own[:, 0, :d] + extra
        spans = np.ones((len(where), 3), np.int64)
        spans[:, :d] = grown[:, 1, :d] - grown[:, 0, :d] + extra
        if any(int(v.shape[0]) != int(n) for v, n in zip(values, sizes.prod(axis=1))):
            return None                     # an order-2 field's values go past its points
        joined = arrays._joined_view(values)
        if joined is None:
            return None
        buffer = self.buffer(g, name, kind, joined.dtype)
        if buffer.dtype != joined.dtype:
            return None
        offset = np.zeros((len(where), 3), np.int64)
        offset[:, :d] = own[:, 0, :d] - grown[:, 0, :d]
        starts = groups.starts(g, kind).astype(np.int64)
        sources = np.concatenate([[0], np.cumsum(sizes.prod(axis=1))]).astype(np.int64)
        row_starts = np.concatenate([[0], np.cumsum(sizes[:, 1] * sizes[:, 2])]).astype(np.int64)
        table = np.ascontiguousarray(np.concatenate([spans[:, :2], offset, sizes[:, :2]],
                                                    axis=1))
        if row_starts[-1]:
            _place_rows(int(row_starts[-1]), len(where), tack.array(row_starts, tack.i64),
                        tack.array(starts, tack.i64), tack.array(table, tack.i64),
                        tack.array(sources[:-1], tack.i64), joined, buffer)
        return buffer._row_views(np.append(starts, groups.size(g, kind)))


def _placed(values, kind, box, ext, d, out_of=None):
    """A box's values placed in an array of the grown extent, zeros elsewhere, on the
    device (through the host it was 3.6 s of AMR-Wind's five levels); the array
    ``out_of(values, n)`` gives, zeroed, where given."""
    extra = 1 if kind == "points" else 0
    own = np.ones(3, np.int64)
    grown = np.ones(3, np.int64)
    start = np.zeros(3, np.int64)
    own[:d] = box[1, :d] - box[0, :d] + extra
    grown[:d] = ext[1, :d] - ext[0, :d] + extra
    start[:d] = box[0, :d] - ext[0, :d]
    values = arrays.materialize(_point_values(values, int(np.prod(own))))
    n = int(np.prod(grown))
    out = tack.zeros((n,), values.dtype) if out_of is None else out_of(values, n)
    if values.shape[0]:
        _place_into(values, out, *map(int, own[:2]), *map(int, grown[:2]), *map(int, start))
    return out


@tack.kernel
def _place_rows(rows, count, row_starts, starts, table, sources, values, out):
    # One thread a row (along x) of the boxes' own entries: box b's rows from
    # row_starts[b] (count boxes, rows in all), its values from sources[b] of
    # ``values``, its grown array from starts[b] of ``out``. table[b]: the grown
    # extent along x and y, where the box starts in it (x, y, z), and the box's
    # extent along x and y.
    for r in range(rows):
        a = 0
        b = count - 1
        while a < b:
            m = (a + b + 1) // 2
            if row_starts[m] <= r:
                a = m
            else:
                b = m - 1
        local = r - row_starts[a]
        nx = table[a, 5]
        y = local % table[a, 6]
        z = local // table[a, 6]
        at = starts[a] + table[a, 2] + table[a, 0] * ((y + table[a, 3]) +
                                                      table[a, 1] * (z + table[a, 4]))
        source = sources[a] + local * nx
        for x in range(nx):
            out[at + x] = values[source + x]


@tack.kernel
def _place_into(values, out, nx, ny, gx, gy, sx, sy, sz):
    for k in range(values.shape[0]):
        i = k % nx
        j = (k // nx) % ny
        m = k // (nx * ny)
        out[(i + sx) + gx * ((j + sy) + gy * (m + sz))] = values[k]


def _grown_box(piece, device, box, ext, d, assembled, names, positions, hidden_points, ghosts):
    """The box grown to ``ext``: its topology, geometry and fields, with its marks
    (``ghosts``, `_grown_ghosts`'); ``hidden_points`` its points nothing filled."""
    from tack.dataset.algorithms.ghost_cells import _kind

    # Plain Python numbers, not NumPy's: a box is a few dozen of them, and NumPy's
    # small-array calls were a good part of a box's cost.
    own, (low, high) = box[0].tolist(), ext.tolist()
    dims = tuple(high[a] - low[a] + 1 for a in range(d))
    # Every part is made here, consistent: assembled without the constructors'
    # checks of each (AMR-Wind's grown boxes are 15,598).
    num_points, num_cells = math.prod(dims), math.prod(n - 1 for n in dims)
    topology = StructuredTopology._assembled(
        dims, tuple(low[:d]), ids.choose(piece.topology.id_dtype, max(num_points, 12 * num_cells)),
        backend_of(device))
    coordinates = piece.geometry.values
    if isinstance(coordinates, UniformCoordinates):          # axis-aligned, as boxes are
        spacing = coordinates._spacing
        origin = tuple(o - (own[a] - low[a]) * spacing[a] if a < d else o - 0.0
                       for a, o in enumerate(coordinates._origin))
        geometry = UniformCoordinates._assembled(dims + (1,) * (3 - d), origin, spacing,
                                                 coordinates.dtype, device)
    else:
        grid = positions.numpy().reshape(-1, 3).astype(np.float64)
        grid = _extrapolated(grid, hidden_points, dims)
        if isinstance(coordinates, CartesianProduct):
            shaped = grid.reshape((*reversed(dims), 3))
            axes = []
            for a in range(3):
                if a < d:
                    moved = np.moveaxis(shaped[..., a], d - 1 - a, -1)
                    axes.append(np.median(moved.reshape(-1, dims[a]), axis=0))
                else:
                    axes.append(np.asarray(
                        arrays.to_host((coordinates.x, coordinates.y, coordinates.z)[a]),
                        np.float64))
            geometry = CartesianProduct(*axes, dtype=coordinates.dtype, device=device)
        else:
            with device:
                geometry = tack.array(grid.astype(piece.dtype.numpy_dtype), tack.vec3(piece.dtype))
    entries = []
    for name in names:
        field = piece.fields[name]
        if _kind(field) == "points":
            space = H1(topology) if isinstance(field.space, H1) else Values(topology, "points")
        else:
            space = Constant(topology) if isinstance(field.space, Constant) else \
                Values(topology, "cells")
        entries.append((_split_key(name)[0], Field._assembled(space, assembled[name],
                                                              field.unit)))
    return DataSet._assembled(topology, Field._assembled(H1(topology), geometry, piece.unit),
                              entries,
                              ghosts, origin=piece.origin, time=piece.time,
                              time_unit=piece.time_unit)


_DUPLICATE = {"cells": Ghosts.DUPLICATECELL, "points": Ghosts.DUPLICATEPOINT}
_HIDDEN = {"cells": Ghosts.HIDDENCELL, "points": Ghosts.HIDDENPOINT}


def _grown_ghosts(amr, groups, extents, hidden):
    """Each local box's grown marks, ``{place: Ghosts}``, made on its group's device:
    each kind of mark one array a group, its boxes' views of it in the group's
    order, as the constructor lays them out. One launch a kind places the boxes'
    own marks and marks their ghosts duplicates (`_grown_own_marks`); the refined
    bits are set where a finer box, coarsened, meets a grown box, by the
    constructor's pairs and rows (`_box_pairs.pairs`, `amr._mark`; on a box's own
    entries they are set already, so setting them again changes nothing); the
    hidden bits where the fills found no source (``hidden[g]``, `_ghost_parts`'
    places in the group's arrays, which are laid out as the marks are). Only the
    box lists are on the host."""
    d = amr.dimension
    out = {}
    for g in groups.blocks.local:
        members = groups.members[g]
        if not members:
            continue
        where = [groups.local[i] for i in members]
        own = np.stack([amr._boxes[level][k] for level, k in where])
        grown = np.stack([extents[level][k] for level, k in where])
        levels = np.asarray([level for level, _ in where])
        views = {}
        with groups.blocks.device(g):
            for c, kind in enumerate(("cells", "points")):
                extra = int(kind == "points")
                sizes = np.ones((len(where), 3), np.int64)
                sizes[:, :d] = own[:, 1, :d] - own[:, 0, :d] + extra
                spans = np.ones((len(where), 3), np.int64)
                spans[:, :d] = grown[:, 1, :d] - grown[:, 0, :d] + extra
                offset = np.zeros((len(where), 3), np.int64)
                offset[:, :d] = own[:, 0, :d] - grown[:, 0, :d]
                starts = np.concatenate([[0], np.cumsum(spans.prod(axis=1))]).astype(np.int64)
                sources = np.concatenate([[0], np.cumsum(sizes.prod(axis=1))]).astype(np.int64)
                whole = tack.empty(int(starts[-1]), tack.u8)
                if len(whole):
                    given = _joined_marks([getattr(groups.pieces[i].ghosts, kind, None)
                                           for i in members], sources)
                    table = np.ascontiguousarray(np.concatenate([spans[:, :2], offset, sizes],
                                                                axis=1))
                    row_starts = np.concatenate([[0], np.cumsum(spans[:, 1] * spans[:, 2])])
                    _grown_own_marks(int(row_starts[-1]), len(where),
                                     tack.array(row_starts.astype(np.int64), tack.i64),
                                     tack.array(starts, tack.i64), tack.array(table, tack.i64),
                                     tack.array(sources, tack.i64), given, _DUPLICATE[kind],
                                     whole)
                for level in np.unique(levels):
                    here = np.flatnonzero(levels == level)
                    level_starts = tack.array(starts[here], tack.i64)
                    for bit, what, boxes, closed, lower in amr._covers(int(level)):
                        if what != kind or lower:
                            continue        # the siblings' DUPLICATEPOINT: a ghost has it
                        found = _box_pairs.pairs(grown[here], boxes, d, closed)
                        # ``lower`` is false: _part_rows reads no global ids.
                        _mark(whole, bit, found, d, closed, False, level_starts, level_starts)
                at = hidden[g][c]
                if at.shape[0]:
                    _set_bits(at, _HIDDEN[kind], whole)
                views[kind] = whole._row_views(starts)
        for n, i in enumerate(members):
            out[i] = Ghosts._assembled(views["cells"][n], views["points"][n])
    return out


def _joined_marks(given, sources):
    """The boxes' own marks of a kind end to end: one stretch of the array they are
    views of, as the constructor makes them, without a copy; else joined in a
    new array, zeros for a box without (``sources`` their CSR starts)."""
    if all(g is not None for g in given):
        joined = arrays._joined_view(given)
        if joined is not None:
            return joined
    parts = [g if g is not None else tack.zeros(int(b - a), tack.u8)
             for g, a, b in zip(given, sources[:-1], sources[1:])]
    return tack.concat(parts) if len(parts) > 1 else parts[0]


@tack.kernel
def _grown_own_marks(rows, count, row_starts, starts, table, sources, own, bit, marks):
    # One thread a row (along x) of the grown boxes' marks: box b's rows from
    # row_starts[b] (count boxes, rows in all), its marks from starts[b]. Each
    # mark is the box's own where it holds the entry, else ``bit``. table[b]: the
    # grown box's extent along x and y, where the box starts in it (x, y, z) and
    # its extent (x, y, z); its own marks start at sources[b] of ``own``. (A
    # thread a mark searched for its box 280M times on AMR-Wind: 2.2 s.)
    for r in range(rows):
        a = 0
        b = count - 1
        while a < b:
            m = (a + b + 1) // 2
            if row_starts[m] <= r:
                a = m
            else:
                b = m - 1
        local = r - row_starts[a]
        gx = table[a, 0]
        nx = table[a, 5]
        x0 = table[a, 2]
        y = local % table[a, 1] - table[a, 3]
        z = local // table[a, 1] - table[a, 4]
        at = starts[a] + local * gx
        if y >= 0 and y < table[a, 6] and z >= 0 and z < table[a, 7]:
            source = sources[a] + nx * (y + table[a, 6] * z) - x0
            for x in range(gx):
                if x >= x0 and x < x0 + nx:
                    marks[at + x] = own[source + x]
                else:
                    marks[at + x] = bit
        else:
            for x in range(gx):
                marks[at + x] = bit


@tack.kernel
def _set_bits(at, bit, marks):
    for i in range(at.shape[0]):
        marks[at[i]] = marks[at[i]] | bit


def _extrapolated(grid, hidden, dims):
    """Positions of the points nothing filled, extrapolated linearly from their
    neighbours along each axis in turn, so a hidden ghost cell has a shape."""
    if not len(hidden):
        return grid
    shape = tuple(reversed(dims))
    known = np.ones(len(grid), bool)
    known[hidden] = False
    grid, known = grid.reshape((*shape, 3)).copy(), known.reshape(shape)
    for _ in range(2 * max(dims)):
        if known.all():
            break
        for axis in range(len(shape)):
            for step in (1, -1):
                near = np.roll(known, -step, axis)
                far = np.roll(known, -2 * step, axis)
                valid = ~known & near & far
                edge = [slice(None)] * len(shape)
                edge[axis] = slice(-2, None) if step == 1 else slice(None, 2)
                valid[tuple(edge)] = False
                if valid.any():
                    a = np.roll(grid, -step, axis)
                    b = np.roll(grid, -2 * step, axis)
                    grid[valid] = 2 * a[valid] - b[valid]
                    known |= valid
    return grid.reshape(-1, 3)


def amr_update_ghosts(amr, names):
    """``update_ghosts`` of a hierarchy ``ghost_cells`` made."""
    from tack.dataset.algorithms.ghost_cells import _kind

    if amr._fills is None:
        raise ValueError("update_ghosts needs the hierarchy ghost_cells made, which keeps "
                         "the exchanges")
    layout_boxes = [box for _, box in amr]
    for name in names:
        values, kind = [], None
        for box in layout_boxes:
            field = box.fields[name]
            kind = _kind(field)
            if kind is None or not isinstance(field.values, tack.Array):
                raise TypeError(f"update_ghosts refreshes point and cell fields of explicit "
                                f"values; {name!r} is not one")
            values.append(field.values)
        amr._fills[kind].run(values)


# -- fine onto coarse -----------------------------------------------------------------

def average_down(amr, field):
    """Put the finer levels' data into the coarse cells and points they cover, in
    place, as AMReX's ``average_down``: each refined coarse cell (``REFINEDCELL``)
    takes the volume-weighted average of the fine cells over it, each refined
    coarse point (``REFINEDPOINT``) the value of the fine point at its location.
    ``field`` names a floating-point point or cell field every box has; finest
    level first, so a value reaches the coarsest level in one call."""
    from tack.dataset.algorithms.ghost_cells import _kind

    if not isinstance(amr, AMRDataSet):
        raise TypeError(f"average_down takes an AMRDataSet, not {type(amr).__name__}")
    boxes = [box for _, box in amr]
    # Every rank's boxes' say: a rank without boxes knows nothing of the field.
    kinds = set().union(*amr.world._allgather({_kind(box.fields[field]) for box in boxes}))
    if len(kinds) != 1 or None in kinds:
        raise TypeError(f"average_down averages a point or cell field; {field!r} is not one")
    kind = kinds.pop()
    measures = {}                       # (level, index): the box's cell measures
    for level in range(amr.num_levels - 2, -1, -1):
        if not len(amr._boxes[level + 1]) or not len(amr._boxes[level]):
            continue                         # nothing finer to average onto this level
        # Only this level's boxes and the next finer level's take part: a fill over
        # every box of the hierarchy copied all of them at every level.
        groups = _Groups(amr, amr._extents, levels=(level, level + 1))
        fill = _Fill(groups, kind, _down_parts(amr, groups, level, kind))
        values = [box.fields[field].values for box in groups.pieces]
        for array in values:
            scalar = getattr(array.dtype, "scalar_dtype", array.dtype)
            if not isinstance(array, tack.Array) or scalar.numpy_dtype.kind != "f":
                raise TypeError(f"average_down averages floating-point values of explicit "
                                f"arrays; {field!r} is not one")
        if kind == "points":
            fill.run(values)
            continue
        if _average_cells_in_place(fill, groups, values):
            continue
        # Cells: sums of measure times value over sums of measure. The measures are
        # the boxes' own, each found once.
        weighted, sums = [], []
        for key, box, device, array in zip(groups.local, groups.pieces, groups.devices, values):
            if key not in measures:
                measures[key] = _measures(box, device, array.dtype)
            with device:
                weighted.append(_times(array, measures[key]))
                sums.append(measures[key].copy())       # the fill sums into them
        fill.run(weighted)
        fill.run(sums)
        for i, targets in fill.box_targets().items():
            if len(targets):
                with groups.devices[i]:
                    _divide(weighted[i], sums[i], tack.array(targets, tack.i64), values[i])


def _average_cells_in_place(fill, groups, values):
    """``average_down``'s cells where each group's values are one stretch of memory
    (`in_device_buffers`): per group, the values times their measures and the
    measures in buffers with room for the staged sources, both filled, and the
    targets divided into the values where they are -- a few launches a group, not
    three a box. False, having done nothing, where a group's values are not one
    stretch."""
    blocks = groups.blocks
    joined = {}
    for g in blocks.local:
        parts = [values[i] for i in groups.members[g]]
        joined[g] = arrays._joined_view(parts) if parts else None
        if parts and joined[g] is None:
            return False
    weighted, sums = {}, {}
    for g in blocks.local:
        box = fill.boxes[g]
        n, m = box["n"], box["m"]
        dtype = values[0].dtype if values else tack.f32
        with blocks.device(g):
            weighted[g] = tack.zeros((n + m,), dtype)
            sums[g] = tack.zeros((n + m,), dtype)
            if n:
                measures = _group_measures([groups.pieces[i] for i in groups.members[g]],
                                           dtype)
                _scaled(joined[g], measures, weighted[g])
                _copy_offset(measures, sums[g])
    for buffers in (weighted, sums):
        fill.plan.run(buffers)
        fill.fill_groups(buffers)
    for g in blocks.local:
        box = fill.boxes[g]
        if box["rows"]:
            with blocks.device(g):
                _divide(weighted[g], sums[g], box["device"]["targets"], joined[g])
    return True


@tack.kernel
def _copy_offset(source, out):
    for i in range(source.shape[0]):
        out[i] = source[i]


def _group_measures(boxes, dtype):
    """The boxes' cells' measures end to end, in ``dtype``: an image box's all its
    spacing's product (one value a box, spread by ``segment_broadcast``), any other
    box's by `_measures`."""
    from tack.algorithms.segmented import segment_broadcast

    if not all(isinstance(b.geometry.values, UniformCoordinates) for b in boxes):
        return tack.concat([_measures(b, b.device, dtype) for b in boxes])
    volumes = np.array([np.prod(np.asarray(b.geometry.values._spacing, np.float64)
                                [:len(b.topology.point_dims)]) for b in boxes])
    offsets = np.concatenate([[0], np.cumsum([b.num_cells for b in boxes])]).astype(np.int64)
    return segment_broadcast(tack.array(volumes.astype(dtype.numpy_dtype), dtype),
                             tack.array(offsets, tack.i64), int(offsets[-1]))


def _measures(box, device, dtype):
    """A box's cells' measures in ``dtype``: an image's, all alike, the product of its
    spacing (``cell_geometry`` of each of AMR-Wind's 15,598 boxes was 10 s)."""
    from tack.dataset.algorithms.cell_geometry import cell_geometry

    coordinates = box.geometry.values
    with device:
        if isinstance(coordinates, UniformCoordinates):
            d = len(box.topology.point_dims)
            volume = float(np.prod(np.asarray(coordinates._spacing, np.float64)[:d]))
            return tack.full((box.num_cells,), volume, dtype)
        return arrays.materialize(cell_geometry(box).measure.values)


@tack.kernel
def _scaled(values, m, out):
    for i in range(out.shape[0]):
        out[i] = values[i] * m[i]


def _times(values, m):
    out = tack.empty(values.shape, values.dtype)
    if values.shape[0]:
        _scaled(values, m, out)
    return out


def _extent_size(ext, kind, d):
    extra = 1 if kind == "points" else 0
    return int(np.prod(ext[1, :d] - ext[0, :d] + extra))

