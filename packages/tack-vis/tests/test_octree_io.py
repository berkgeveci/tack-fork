# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An octree written to and read from an ADIOS2 file (study): the same nodes,
marks and fields back, in blocks of whole trees, any subset of them an octree."""

import numpy as np
import pytest
from test_octree_filters import _with_field
from test_octree_topology import CASES

import tack
from tack.dataset import octree_filters as of
from tack.dataset import octree_io as oio

adios2 = pytest.importorskip("adios2")


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _same(a, b, fields=("s",)):
    for name in ("_tree", "_level", "_key"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.marks.numpy(), b.marks.numpy())
    for name in fields:
        np.testing.assert_array_equal(a.fields[name].numpy(), b.fields[name].numpy())


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("blocks", [1, 5])
def test_a_round_trip_gives_the_octree_back(name, make, blocks, tmp_path):
    octree = _with_field(make())
    s = octree.fields["s"].numpy()
    octree = of.threshold(octree, "s", np.nanpercentile(s[octree.is_leaf], 10), np.inf)
    path = tmp_path / "octree.bp"
    assert 1 <= oio.write_octree(path, octree, blocks=blocks) <= blocks
    back = oio.read_octree(path)
    _same(back, octree)
    nodes, trees, boxes = oio.block_info(path)
    assert nodes.sum() == octree.num_nodes
    assert (trees[1:, 0] > trees[:-1, 1]).all()        # whole trees, in order
    # A subset of blocks is an octree of their trees alone.
    part = oio.read_octree(path, blocks=[0])
    assert part.num_nodes == nodes[0]
    assert set(part._tree) <= set(range(trees[0, 0], trees[0, 1] + 1))
    # Two steps in one file.
    second = of.average_up(octree, ["s"])
    oio.write_octree(path, second, blocks=blocks, append=True)
    _same(oio.read_octree(path, step=1), second)
    _same(oio.read_octree(path, step=0), octree)


@pytest.mark.parametrize("name,make", CASES, ids=[c[0] for c in CASES])
def test_a_simulations_own_pieces_come_back_sorted(name, make, tmp_path):
    # Blocks from a partition of the root cells that is not in tree order.
    octree = _with_field(make())
    rng = np.random.default_rng(0)
    block_of_tree = rng.integers(0, 4, octree.roots.num_cells)
    path = tmp_path / "pieces.bp"
    oio.write_octree(path, octree, block_of_tree=block_of_tree)
    back = oio.read_octree(path)
    _same(back, octree)
    nodes, trees, _ = oio.block_info(path)
    assert not (trees[1:, 0] > trees[:-1, 1]).all()
