# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-cell data in blocks: a field's values kept as one array per block of cells
(``UnstructuredTopology.from_cell_blocks``), as solvers and VTK's cell grids keep them.

``cell_block_values(parts, cells)`` joins them without copying (``tack.arrays.Concatenate``),
so whatever reads a field's values whole reads them; a field view of a shape group
that is one block reads its block's array directly, shifted to its first value
(``for_group``), with no search through the blocks on each read."""

import tack
from tack import arrays
from tack.arrays import _Implicit, _metadata, dtype_of, size_of, width_of

__all__ = ["cell_block_values"]


@tack.device_class
class _Rebased(_Implicit):
    """``part`` read at the indices of the whole it is a part of: value ``k`` of the
    whole is ``part[k - base]``, for the ``k`` in the part's range."""

    def __init__(self, part, base, size):
        self.part = part
        self.base = int(base)
        _metadata(self, size, width_of(part), dtype_of(part))

    @tack.func
    def get(self, k):
        return self.part[k - self.base]


def cell_block_values(parts, cells):
    """Per-cell values in blocks, one array each (``parts``), the blocks' cells
    ``cells`` (``(first cell, count)`` each, block after block, as the topology's
    blocks): their values joined, holding the parts."""
    if len(parts) != len(cells):
        raise ValueError(f"{len(parts)} blocks of values for {len(cells)} blocks of cells")
    parts = list(parts)
    joined = arrays.Concatenate(*parts)
    starts, at = [], 0
    for part in parts:
        starts.append(at)
        at += size_of(part)
    joined._cell_block_parts = (parts, starts, [first for first, _ in cells])
    return joined


def for_group(values, group):
    """``values`` as a view of ``group`` reads them: a block's own array, rebased, when
    ``values`` are in blocks and the group's cells are one block's; else ``values``."""
    held = getattr(values, "_cell_block_parts", None)
    if held is None or not isinstance(values, _Implicit):
        return values
    parts, starts, firsts = held
    first = group.root.start
    for part, start, cell in zip(parts, starts, firsts, strict=True):
        if cell == first:
            return _Rebased(part, start, size_of(values))
    return values
