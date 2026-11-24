# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A 3D octree's leaves as a conforming polyhedral mesh (study,
``study/octree-filters``).

``leaves()`` gives voxels whose corners hang where a leaf meets smaller ones:
a coarse voxel's face is one square, and its finer neighbours' four squares do
not share it. Here every face of ``octree_topology.faces`` is one polygon,
shared by the two leaves it lies between, so a coarse leaf is a box whose side
is split where its neighbour is finer. And every point that lies on an edge of
a face is a point of that face's walk, so no edge hangs either (a coarse
face's edge that a finer leaf's corner lies on goes through that corner).

So the mesh is watertight face by face and edge by edge, and the finite-volume
algorithms (``face_geometry``, ``net_flux``, ``upwind_flux``, ``jump``,
``boundary_faces``, ``perot``) and those through shared points
(``connected_components``) take it as it is. Built on the host (a study).
"""

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree import _root_axes
from tack.dataset.octree_topology import faces, lattice_boxes

__all__ = ["leaves_as_polyhedra"]


def _squares(octree, lo, hi, small, direction):
    """Each face's four corners on the lattice, ``(f, 4, 3)``, wound out of its
    small side: counterclockwise seen from the outside."""
    axis = direction // 2
    upper = (direction % 2).astype(bool)
    rows = np.arange(len(small))
    corners = np.zeros((len(small), 4, 3), np.int64)
    u, v = (axis + 1) % 3, (axis + 2) % 3
    plane = np.where(upper, hi[small, axis], lo[small, axis])
    u0, u1 = lo[small, u], hi[small, u]
    v0, v1 = lo[small, v], hi[small, v]
    # Out along +axis: (u0, v0), (u1, v0), (u1, v1), (u0, v1), since u x v is +axis.
    walk_u = np.stack([u0, u1, u1, u0], axis=1)
    walk_v = np.stack([v0, v0, v1, v1], axis=1)
    flip = ~upper                                   # out along -axis: the other way round
    walk_u[flip] = walk_u[flip][:, ::-1]
    walk_v[flip] = walk_v[flip][:, ::-1]
    for k in range(4):
        corners[rows, k, axis] = plane
        corners[rows, k, u] = walk_u[:, k]
        corners[rows, k, v] = walk_v[:, k]
    return corners


class _Lines:
    """The points on each axis-parallel line, sorted along it, to find the points
    strictly inside an edge."""

    def __init__(self, points):
        self.scale = int(points.max()) + 2
        if self.scale ** 3 >= 2 ** 62:
            raise ValueError("the lattice is too fine for one int64 key a point")
        self.by_axis = []
        for e in range(3):
            k1, k2 = [a for a in range(3) if a != e]
            key = self._key(points[:, k1], points[:, k2], points[:, e])
            order = np.argsort(key, kind="stable")
            self.by_axis.append((key[order], order))

    def _key(self, a, b, c):
        return (a * self.scale + b) * self.scale + c

    def inside(self, start, stop):
        """The ids of the points strictly between lattice points ``start`` and
        ``stop`` (which differ along one axis), in order from ``start``."""
        e = int(np.flatnonzero(start != stop)[0])
        k1, k2 = [a for a in range(3) if a != e]
        keys, order = self.by_axis[e]
        a, b = sorted((int(start[e]), int(stop[e])))
        first = np.searchsorted(keys, self._key(start[k1], start[k2], a), side="right")
        last = np.searchsorted(keys, self._key(start[k1], start[k2], b), side="left")
        ids = order[first:last]
        return ids if start[e] < stop[e] else ids[::-1]


def _positions(octree, points):
    """Lattice points as positions relative to the origin, ``(m, 3)`` f64."""
    finest = octree.finest
    axes = _root_axes(octree.roots)
    out = np.zeros((len(points), 3))
    for a in range(3):
        lines = np.asarray(axes[a], np.float64)
        c = np.minimum(points[:, a] >> finest, len(lines) - 2)
        f = (points[:, a] - (c << finest)) / float(1 << finest)
        out[:, a] = lines[c] + f * (lines[c + 1] - lines[c])
    return out


def leaves_as_polyhedra(octree):
    """The leaves of a 3D octree as a ``DataSet`` of polyhedra, cell ``c`` the
    ``c``-th leaf (as ``leaves()`` numbers them), every field as cell data and
    hidden leaves ``HIDDENCELL``: one polygon a face between two leaves (the
    smaller one's square, wound out of it), with every point that lies on its
    edges, so the mesh conforms where leaves of different sizes meet."""
    from tack.dataset.dataset import DataSet, Field
    from tack.dataset.polyhedra import PolyhedralTopology
    from tack.dataset.spaces import Constant

    if octree.dimension != 3:
        raise ValueError("leaves_as_polyhedra takes a 3D octree")
    found = faces(octree)
    small, large, direction = (np.asarray(a.numpy()) for a in found)
    lo, hi = lattice_boxes(octree)
    corners = _squares(octree, lo, hi, small, direction.astype(np.int64))
    points, ids = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    ids = ids.reshape(-1, 4)
    lines = _Lines(points)

    face_offsets = [0]
    face_points = []
    for f in range(len(small)):
        for k in range(4):
            face_points.append(ids[f, k])
            face_points.extend(lines.inside(corners[f, k], corners[f, (k + 1) % 4]).tolist())
        face_offsets.append(len(face_points))

    leaf_number = np.cumsum(octree.is_leaf) - 1
    rows = np.arange(len(small))
    inner = large >= 0
    cell = np.concatenate([leaf_number[small], leaf_number[large[inner]]])
    face = np.concatenate([rows, rows[inner]])
    side = np.concatenate([np.zeros(len(small), np.uint8), np.ones(int(inner.sum()), np.uint8)])
    order = np.argsort(cell, kind="stable")
    count = int(octree.is_leaf.sum())
    cell_offsets = np.concatenate([[0], np.cumsum(np.bincount(cell, minlength=count))])

    leaf = np.flatnonzero(octree.is_leaf)
    with octree.device:
        topology = PolyhedralTopology(
            tack.array(np.asarray(face_offsets, np.int64)),
            tack.array(np.asarray(face_points, np.int64)),
            tack.array(cell_offsets.astype(np.int64)),
            tack.array(face[order].astype(np.int64)),
            tack.array(side[order], tack.u8), num_points=len(points))
        dtype = octree.roots.dtype
        positions = tack.array(_positions(octree, points).astype(dtype.numpy_dtype),
                               tack.vec3(dtype))
        fields = {name: Field(Constant(topology), tack.array(np.asarray(values.numpy())[leaf]),
                              octree.units.get(name))
                  for name, values in octree.fields.items()}
        hidden = octree.marks.numpy()[leaf] & Ghosts.HIDDENCELL
        ghosts = Ghosts(cells=tack.array(hidden.astype(np.uint8), tack.u8)) \
            if hidden.any() else None
    return DataSet(topology, positions, fields=fields, dtype=dtype, ghosts=ghosts,
                   **octree._frame())
