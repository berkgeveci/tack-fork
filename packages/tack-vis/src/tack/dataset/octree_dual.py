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

Built on the device: 21 ms for 77,000 leaves and 67 ms for 300,000 on yavin's
CPU, against 0.48 and 2.1 s for the host build it replaced (``_dual_grid_host``,
kept to check it).
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


def _dual_grid_host(octree):
    """``dual_grid`` built on the host, the first form, kept to check the
    device's: the dual grid of ``octree``'s visible leaves, a ``DataSet`` of hexahedra
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


# -- on the device -------------------------------------------------------------------


@tack.kernel
def _interior_runs(si, sj, sk, flags, ex, ey, ez, dimension, keep):
    # A run's first entry (a distinct corner) strictly inside the domain.
    for s in range(keep.shape[0]):
        inside = si[s] > 0 and si[s] < ex and sj[s] > 0 and sj[s] < ey
        if dimension == 3:
            inside = inside and sk[s] > 0 and sk[s] < ez
        keep[s] = 1 if flags[s] == 1 and inside else 0


@tack.kernel
def _compact_corners(si, sj, sk, keep, numbers, px, py, pz):
    for s in range(keep.shape[0]):
        if keep[s] == 1:
            k = numbers[s] - 1
            px[k] = si[s]
            py[k] = sj[s]
            pz[k] = sk[s]


@tack.kernel
def _whole_cells(nodes, marks, hidden, corners, keep):
    # A corner makes a cell when every leaf around it is inside and visible.
    for q in range(keep.shape[0]):
        ok = 1
        for v in range(corners):
            n = nodes[q * corners + v]
            if n < 0:
                ok = 0
            elif (marks[n] & hidden) != 0:
                ok = 0
        keep[q] = ok


@tack.kernel
def _cell_points(nodes, keep, numbers, number, corners, connectivity):
    # A kept corner's cell: its leaves' numbers, the dual grid's point ids.
    for q in range(keep.shape[0]):
        if keep[q] == 1:
            k = numbers[q] - 1
            for v in range(corners):
                connectivity[k * corners + v] = number[nodes[q * corners + v]] - 1


@tack.kernel
def _leaf_flags(marks, refined, out):
    for n in range(out.shape[0]):
        out[n] = 1 if (marks[n] & refined) == 0 else 0


@tack.kernel
def _leaf_centres(leaf, lo, size, out):
    for c in range(leaf.shape[0]):
        n = leaf[c]
        out[c] = lo[n] + size[n] * 0.5


def dual_grid(octree):
    """The dual grid of ``octree``'s visible leaves: a ``DataSet`` of hexahedra
    (quads in 2D), point ``c`` the ``c``-th leaf's centre (as ``leaves()``
    numbers them), every field as point data. On the device: the leaves'
    corners sorted and numbered by runs (as ``leaves()`` finds its points), the
    interior ones kept, the leaves around each found by a search of the keys,
    and the cells around hidden leaves or the boundary dropped."""
    from tack.algorithms import argsort, gather, inclusive_scan
    from tack.dataset import ids, shapes
    from tack.dataset.dataset import DataSet, Field
    from tack.dataset.octree import _leaf_corners, _run_starts
    from tack.dataset.spaces import H1
    from tack.dataset.topology import UnstructuredTopology, uniform_cells

    d = octree.dimension
    corners = 1 << d
    finest = octree.finest
    nx, ny, nz = ([n - 1 for n in octree.roots.topology.point_dims] + [1])[:3]
    count = int(octree.is_leaf.sum())
    entries = count * corners
    dtype = octree.roots.dtype
    with octree.device:
        leaf = tack.array(np.flatnonzero(octree.is_leaf).astype(np.int64), tack.i64)
        columns = [tack.empty((entries,), tack.i64) for _ in range(3)]
        m = 0
        if entries:
            _leaf_corners(leaf, octree.tree, octree.level, octree.key, nx, ny, d, finest,
                          *columns)
            order = argsort(columns[0], index_dtype=tack.i64)
            for column in columns[1:d]:
                order = gather(order, argsort(gather(column, order), index_dtype=tack.i64))
            si, sj, sk = (gather(c, order) for c in columns)
            flags = tack.empty((entries,), tack.i32)
            _run_starts(si, sj, sk, flags)
            keep = tack.empty((entries,), tack.i32)
            _interior_runs(si, sj, sk, flags, nx << finest, ny << finest, nz << finest, d, keep)
            numbers = tack.empty((entries,), tack.i64)
            m = int(inclusive_scan(keep, numbers, entries))
        px, py, pz = (tack.empty((m,), tack.i64) for _ in range(3))
        nodes = tack.empty((m * corners,), tack.i64)
        cells = 0
        if m:
            _compact_corners(si, sj, sk, keep, numbers, px, py, pz)
            _around(px, py, pz, nx, ny, nz, finest, d, *_search_arrays(octree), nodes)
            whole = tack.empty((m,), tack.i32)
            _whole_cells(nodes, octree.marks, int(Ghosts.HIDDENCELL), corners, whole)
            cell_numbers = tack.empty((m,), tack.i64)
            cells = int(inclusive_scan(whole, cell_numbers, m))
        idt = ids.choose(None, max(count, cells * corners))
        connectivity = tack.empty((cells * corners,), idt)
        if cells:
            is_leaf = tack.empty((octree.num_nodes,), tack.i64)
            _leaf_flags(octree.marks, int(Ghosts.REFINEDCELL), is_leaf)
            number = tack.empty((octree.num_nodes,), tack.i64)
            inclusive_scan(is_leaf, number, octree.num_nodes)
            _cell_points(nodes, whole, cell_numbers, number, corners, connectivity)
        types, offsets = uniform_cells(shapes.HEXAHEDRON if d == 3 else shapes.QUAD, cells, idt)
        topology = UnstructuredTopology(types, offsets, connectivity, num_points=count)
        lo, size = octree.geometry()
        positions = tack.empty((count,), tack.vec3(dtype))
        if count:
            _leaf_centres(leaf, lo, size, positions)
        data = DataSet(topology, positions, dtype=dtype, **octree._frame())
        for name, values in octree.fields.items():
            data.fields[name] = Field(H1(data), gather(values, leaf), octree.units.get(name))
    return data
