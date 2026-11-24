# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An octree's dual grid (study, ``study/octree-filters``): its points the
leaves' centres with the cells' values as point data, exactly as given, and a
cell for every corner of the leaves inside the domain, made of the 2^d leaves
around it -- a hexahedron (a quad in 2D) whose corners repeat where a coarse
leaf is around the corner more than once, at a corner that hangs on a coarse
leaf's face or edge.

The leaves around a corner are found as ``probe`` finds a leaf: the cell of the
finest lattice on each side of the corner, then the last node at or before its
code. Corners are those of ``leaves()``, every one once, hanging ones included.
A corner on the domain's boundary has a side outside, so no cell: the dual grid
covers the domain less a half-leaf layer, as VTK's does. A cell around a hidden
leaf is left out.

Every algorithm on hexahedra with point data then runs on it (``contour``,
``slice``, ``probe`` with interpolation, ``streamlines``).
"""

import numpy as np

import tack
from tack.dataset.ghosts import Ghosts
from tack.dataset.octree_topology import _holder, _search_arrays, lattice_boxes

__all__ = ["dual_grid"]


@tack.kernel
def _around(px, py, pz, nx, ny, nz, finest, dimension, tree, carried, level, count, out):
    # Corner q's cell: the leaf holding each lattice cell around the corner, in a
    # hexahedron's (a quad's) order: x as (v ^ v >> 1) & 1, y as v >> 1 & 1, z as
    # v >> 2 & 1.
    for q in range(px.shape[0]):
        corners = 1 << dimension
        for v in range(corners):
            gi = px[q] + tack.i64((v ^ (v >> 1)) & 1) - 1
            gj = py[q] + tack.i64((v >> 1) & 1) - 1
            gk = pz[q]
            if dimension == 3:
                gk = pz[q] + tack.i64((v >> 2) & 1) - 1
            out[q * corners + v] = _holder(gi, gj, gk, nx, ny, nz, finest, dimension, tree,
                                           carried, level, count)


def _corners(octree):
    """The leaves' corners strictly inside the domain, on the finest lattice,
    each once: host ``(m, 3)`` int64."""
    d = octree.dimension
    lo, hi = lattice_boxes(octree)
    leaf = octree.is_leaf
    lo, hi = lo[leaf], hi[leaf]
    picks = []
    for v in range(1 << d):
        p = lo.copy()
        for a in range(d):
            if (v >> a) & 1:
                p[:, a] = hi[:, a]
        picks.append(p)
    points = np.unique(np.concatenate(picks), axis=0)
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    top = np.array([nx, ny, nz]) << octree.finest
    inside = ((points[:, :d] > 0) & (points[:, :d] < top[:d])).all(axis=1)
    return points[inside]


def dual_grid(octree):
    """The dual grid of ``octree``'s visible leaves: a ``DataSet`` of hexahedra
    (quads in 2D), point ``c`` the ``c``-th leaf's centre (as ``leaves()``
    numbers them), every field as point data."""
    from tack.dataset import ids, shapes
    from tack.dataset.dataset import DataSet, Field
    from tack.dataset.spaces import H1
    from tack.dataset.topology import UnstructuredTopology, uniform_cells

    d = octree.dimension
    corners = 1 << d
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    points = _corners(octree)
    with octree.device:
        around = tack.empty((len(points) * corners,), tack.i64)
        if len(points):
            px, py, pz = (tack.array(np.ascontiguousarray(points[:, a])) for a in range(3))
            _around(px, py, pz, nx, ny, nz, octree.finest, d, *_search_arrays(octree), around)
    nodes = np.asarray(around.numpy()).reshape(-1, corners)
    visible = (octree.marks.numpy() & Ghosts.HIDDENCELL) == 0
    keep = (nodes >= 0).all(axis=1) & visible[np.maximum(nodes, 0)].all(axis=1)
    number = np.cumsum(octree.is_leaf) - 1
    connectivity = number[nodes[keep]].reshape(-1)
    leaf = np.flatnonzero(octree.is_leaf)
    count = int(keep.sum())
    idt = ids.choose(None, max(len(leaf), len(connectivity)))
    dtype = octree.roots.dtype
    with octree.device:
        types, offsets = uniform_cells(shapes.HEXAHEDRON if d == 3 else shapes.QUAD, count, idt)
        topology = UnstructuredTopology(types, offsets,
                                        tack.array(connectivity.astype(idt.numpy_dtype), idt),
                                        num_points=len(leaf))
        centres = octree.centers()[leaf]
        positions = tack.array(centres.astype(dtype.numpy_dtype), tack.vec3(dtype))
        data = DataSet(topology, positions, dtype=dtype, **octree._frame())
        for name, values in octree.fields.items():
            data.fields[name] = Field(H1(data), tack.array(np.asarray(values.numpy())[leaf]),
                                      octree.units.get(name))
    return data
