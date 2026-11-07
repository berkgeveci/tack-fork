# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``test_mixed_cells``' pieces across MPI ranks: ``mpiexec -n 2 python
_mixed_pieces_mpi.py``. Every rank makes ghost cells, global ids and the merge of
its share of a mixed mesh's slabs; rank 0 compares them with what one process
makes of all of them, and prints ``ok`` or fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_mixed_cells as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    root = comm.size - 1                  # the merge sent on to another rank
    ghosts, ids, merged = case.mixed_pieces_across(tc.World(comm), root)
    gathered = comm.gather((ghosts, ids, merged), root=0)
    if comm.rank == 0:
        alone = case.mixed_pieces_across(tc.World())
        for k, what in enumerate(("ghost cells", "global ids")):
            across = {g: d for part in gathered for g, d in part[k].items()}
            assert sorted(across) == sorted(alone[k]), what
            for g in alone[k]:
                assert across[g] == alone[k][g], f"{what}: piece {g} differs"
        merges = [part[2] for part in gathered]
        assert all(m is None for r, m in enumerate(merges) if r != root)
        assert merges[root] == alone[2], "the merge differs"
        print(f"ok: mixed pieces on {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
