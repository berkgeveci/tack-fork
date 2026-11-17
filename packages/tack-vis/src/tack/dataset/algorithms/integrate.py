# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``integrate``: the measure of a dataset and the integrals of its fields over it, by
quadrature in each cell's reference element through its own map -- curved cells and
fields of any order exactly, to rounding (VTK's vtkIntegrateAttributes)."""

import math
from typing import NamedTuple

import numpy as np

import tack
from tack import arrays
from tack.dataset import shapes, units, views
from tack.dataset.algorithms._common import _field, _with_basis
from tack.dataset.algorithms.cell_geometry import _signed_twice_area
from tack.dataset.dataset import Field, is_field_key
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['Integrals', 'integrate']


class Integrals(NamedTuple):
    """``integrate``'s answer: ``measure`` (the total volume, area or length, in
    ``measure_unit``) and, per field, its integral (a float, or a tuple for a vector
    field) in ``values`` and that integral's unit in ``units``."""

    measure: float
    measure_unit: object
    values: dict
    units: dict


@tack.func
def _element(m, dimension):
    # The measure element of a cell of ``dimension`` mapped by Jacobian m: a volume's
    # |det J|, a surface's |J0 x J1|, a line's |J0| (columns past the cell's
    # dimension are zero).
    out = abs(m.determinant())
    if dimension == 2:
        a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
        b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
        out = a.cross(b).norm()
    elif dimension == 1:
        out = tack.vec3(m[0, 0], m[1, 0], m[2, 0]).norm()
    return out


@tack.kernel
def _cell_measures(cells, points, weights, measures, centroids):
    # Each cell's measure and centroid: the quadrature's weights times the measure
    # element, and the positions those weigh.
    for c in cells:
        e = cells.entity_id(c)
        m = weights[0] * 0.0
        x = cells.position(c, points[0]) * 0.0
        for q in range(weights.shape[0]):
            d = weights[q] * _element(cells.geometry_jacobian(c, points[q]), cells.DIMENSION)
            m += d
            x += d * cells.position(c, points[q])
        measures[e] = m
        centroids[e] = x / m if m > 0.0 else cells.position(c, cells.parametric_center())


@tack.func
def _cell_integral(cells, u, c, points, weights):
    # Cell c's integral of the field and its measure, ``(integral, measure)``: the
    # quadrature's weights times the measure element (through the cell's own map)
    # times the field's value by its own basis, and the same without the value.
    # integrate's and project's one quadrature of a cell.
    m = weights[0] * 0.0
    total = u.value(c, points[0]) * 0.0
    for q in range(weights.shape[0]):
        d = weights[q] * _element(cells.geometry_jacobian(c, points[q]), cells.DIMENSION)
        m += d
        total += d * u.value(c, points[q])
    return total, m


@tack.kernel
def _cell_integrals(cells, u, points, weights, out):
    # Each cell's integral of the field (_cell_integral).
    for c in cells:
        total, _ = _cell_integral(cells, u, c, points, weights)
        out[cells.entity_id(c)] = total


@tack.device_class
class _Rule:
    """A Gauss rule of ``N`` points along an axis on [0, 1] (``points``, ``weights``),
    for a field of ``W`` components (1 a scalar), for ``_tensor_integrals``: ``N``
    and ``W`` class constants, so its local arrays have a size (``_rule``, one class
    per pair)."""

    N = 1
    W = 1

    def __init__(self, points, weights):
        self.points = points
        self.weights = weights


_rules = {}


def _rule(data, n, width=1):
    """The ``n``-point Gauss-Legendre rule on [0, 1] as a ``_Rule`` for ``width``
    components, in the geometry's float type."""
    key = (int(n), int(width))
    if key not in _rules:
        _rules[key] = type(f"_Rule{n}x{width}", (_Rule,), {"N": key[0], "W": key[1]})
    x, w = np.polynomial.legendre.leggauss(int(n))
    dtype = data.dtype.numpy_dtype
    return _rules[key](tack.array(((x + 1) / 2).astype(dtype)), tack.array((w / 2).astype(dtype)))


@tack.func
def _basis_at_rule(u, rule, line, b):
    # The 1D basis at the rule's points: b[q p + i], p = u.ORDER + 1.
    p = u.ORDER + 1
    for q in range(rule.N):
        u.lagrange_values(rule.points[q], line)
        for i in range(p):
            b[q * p + i] = line[i]


@tack.func
def _along_x_and_y(b, coeffs, xs, xys, n, p, d):
    # Component d's cell values (coeffs from d p^3, node i + p (j + p k)) contracted
    # along x and y, into xys from d n^2 p: xys[(qx n + qy) p + k] = sum_ij b[qx, i]
    # b[qy, j] coeffs[i + p (j + p k)] (xs, n p^2, between).
    at = d * p * p * p
    for qx in range(n):
        for jk in range(p * p):
            v = b[qx * p] * coeffs[at + jk * p]
            for i in range(1, p):
                v += b[qx * p + i] * coeffs[at + jk * p + i]
            xs[qx * p * p + jk] = v
    to = d * n * n * p
    for qx in range(n):
        for qy in range(n):
            for k in range(p):
                v = b[qy * p] * xs[qx * p * p + k * p]
                for j in range(1, p):
                    v += b[qy * p + j] * xs[qx * p * p + k * p + j]
                xys[to + (qx * n + qy) * p + k] = v


@tack.func
def _along_z(b, xys, n, p, d, qx, qy, qz):
    # Component d's value at point (qx, qy, qz): xys contracted along z.
    at = d * n * n * p + (qx * n + qy) * p
    v = b[qz * p] * xys[at]
    for k in range(1, p):
        v += b[qz * p + k] * xys[at + k]
    return v


@tack.device_class
class _PointElements:
    """The measure element at each of a rule's points from the geometry's Jacobian
    there, a point at a time: any geometry of hexahedra or voxels."""

    @tack.func
    def fill(self, cells, c, rule, out):
        # out[(qz N + qy) N + qx]: |det J| at point (qx, qy, qz).
        n = rule.N
        for qz in range(n):
            for qy in range(n):
                for qx in range(n):
                    pc = tack.vec3(rule.points[qx], rule.points[qy], rule.points[qz])
                    out[(qz * n + qy) * n + qx] = _element(cells.geometry_jacobian(c, pc), 3)


@tack.device_class
class _TensorElements:
    """The measure element at each of a rule's points by sum factorization, for a
    curved geometry in a tensor basis (``views._TensorGeometry``, ORDER + 1 = g
    nodes an axis): its node positions (less node 0's, as its own Jacobian takes
    them, so f32 keeps thin cells) contracted along x by the 1D basis' values and
    derivatives, then along y, keeping the three products a column needs; each
    column of J at a point is then one contraction along z."""

    @tack.func
    def fill(self, cells, c, rule, out):
        g = cells.ORDER + 1
        n = rule.N
        line = tack.local_array_like(cells.basis_nodes, cells.ORDER + 1)
        vb = tack.local_array_like(cells.basis_nodes, rule.N * (cells.ORDER + 1))
        db = tack.local_array_like(cells.basis_nodes, rule.N * (cells.ORDER + 1))
        for q in range(n):
            cells.lagrange_values(rule.points[q], line)
            for i in range(g):
                vb[q * g + i] = line[i]
            cells.lagrange_derivatives(rule.points[q], line)
            for i in range(g):
                db[q * g + i] = line[i]
        cube = g * g * g
        pos = tack.local_array_like(cells.basis_nodes, 3 * (cells.ORDER + 1) * (cells.ORDER + 1)
                                    * (cells.ORDER + 1))
        first = cells.get(cells.dof_index(c, 0))
        for m in range(cube):
            x = cells.get(cells.dof_index(c, m)) - first
            pos[m] = x[0]
            pos[cube + m] = x[1]
            pos[2 * cube + m] = x[2]
        # Along x, by values (xv) and derivatives (xd): [qx g^2 + j + g k] a coordinate.
        xv = tack.local_array_like(cells.basis_nodes, rule.N * (cells.ORDER + 1) * (cells.ORDER + 1))
        xd = tack.local_array_like(cells.basis_nodes, rule.N * (cells.ORDER + 1) * (cells.ORDER + 1))
        # Along y: vv (value, value), vd (x value, y derivative), dv (x derivative, y
        # value), each [((a n + qx) n + qy) g + k] for coordinate a.
        vv = tack.local_array_like(cells.basis_nodes, 3 * rule.N * rule.N * (cells.ORDER + 1))
        vd = tack.local_array_like(cells.basis_nodes, 3 * rule.N * rule.N * (cells.ORDER + 1))
        dv = tack.local_array_like(cells.basis_nodes, 3 * rule.N * rule.N * (cells.ORDER + 1))
        for a in range(3):
            for qx in range(n):
                for jk in range(g * g):
                    at = a * cube + jk * g
                    sv = vb[qx * g] * pos[at]
                    sd = db[qx * g] * pos[at]
                    for i in range(1, g):
                        sv += vb[qx * g + i] * pos[at + i]
                        sd += db[qx * g + i] * pos[at + i]
                    xv[qx * g * g + jk] = sv
                    xd[qx * g * g + jk] = sd
            for qx in range(n):
                for qy in range(n):
                    for k in range(g):
                        at = qx * g * g + k * g
                        s_vv = vb[qy * g] * xv[at]
                        s_vd = db[qy * g] * xv[at]
                        s_dv = vb[qy * g] * xd[at]
                        for j in range(1, g):
                            s_vv += vb[qy * g + j] * xv[at + j]
                            s_vd += db[qy * g + j] * xv[at + j]
                            s_dv += vb[qy * g + j] * xd[at + j]
                        to = ((a * n + qx) * n + qy) * g + k
                        vv[to] = s_vv
                        vd[to] = s_vd
                        dv[to] = s_dv
        for qz in range(n):
            for qy in range(n):
                for qx in range(n):
                    m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                    for a in range(3):
                        at = ((a * n + qx) * n + qy) * g
                        jx = vb[qz * g] * dv[at]
                        jy = vb[qz * g] * vd[at]
                        jz = db[qz * g] * vv[at]
                        for k in range(1, g):
                            jx += vb[qz * g + k] * dv[at + k]
                            jy += vb[qz * g + k] * vd[at + k]
                            jz += db[qz * g + k] * vv[at + k]
                        m[a, 0] = jx
                        m[a, 1] = jy
                        m[a, 2] = jz
                    out[(qz * n + qy) * n + qx] = abs(m.determinant())


@tack.kernel
def _tensor_integrals(cells, u, rule, geo, out):
    # Each hexahedron's integral of a scalar field in a tensor basis (u, ORDER + 1 = p
    # nodes an axis, x fastest) on the rule's N^3 points, by sum factorization: the
    # 1D basis at the rule's points once, the cell's values contracted one axis at a
    # time (p^3 -> N p^2 -> N^2 p -> one value a point), so a point costs p
    # multiply-adds rather than p^3; then the weights times the measure element at
    # each point, as _cell_integrals.
    for c in cells:
        p = u.ORDER + 1
        n = rule.N
        line = tack.local_array_like(u.basis_nodes, u.ORDER + 1)
        b = tack.local_array_like(u.basis_nodes, rule.N * (u.ORDER + 1))
        coeffs = tack.local_array_like(u.basis_nodes, (u.ORDER + 1) * (u.ORDER + 1) * (u.ORDER + 1))
        xs = tack.local_array_like(u.basis_nodes, rule.N * (u.ORDER + 1) * (u.ORDER + 1))
        xys = tack.local_array_like(u.basis_nodes, rule.N * rule.N * (u.ORDER + 1))
        elements = tack.local_array_like(u.basis_nodes, rule.N * rule.N * rule.N)
        geo.fill(cells, c, rule, elements)
        _basis_at_rule(u, rule, line, b)
        u.dofs_into(c, coeffs)
        _along_x_and_y(b, coeffs, xs, xys, n, p, 0)
        total = rule.weights[0] * 0.0
        for qz in range(n):
            for qy in range(n):
                for qx in range(n):
                    w = (rule.weights[qx] * rule.weights[qy] * rule.weights[qz]
                         * elements[(qz * n + qy) * n + qx])
                    total += w * _along_z(b, xys, n, p, 0, qx, qy, qz)
        out[cells.entity_id(c)] = total


@tack.kernel
def _tensor_vector_integrals(cells, u, rule, geo, out):
    # _tensor_integrals for a field of rule.W components: each component's values
    # gathered into its own block and contracted alone; the measure elements once
    # for all of them.
    for c in cells:
        p = u.ORDER + 1
        n = rule.N
        cube = p * p * p
        line = tack.local_array_like(u.basis_nodes, u.ORDER + 1)
        b = tack.local_array_like(u.basis_nodes, rule.N * (u.ORDER + 1))
        coeffs = tack.local_array_like(
            u.basis_nodes, rule.W * (u.ORDER + 1) * (u.ORDER + 1) * (u.ORDER + 1))
        xs = tack.local_array_like(u.basis_nodes, rule.N * (u.ORDER + 1) * (u.ORDER + 1))
        xys = tack.local_array_like(u.basis_nodes, rule.W * rule.N * rule.N * (u.ORDER + 1))
        totals = tack.local_array_like(u.basis_nodes, rule.W)
        elements = tack.local_array_like(u.basis_nodes, rule.N * rule.N * rule.N)
        geo.fill(cells, c, rule, elements)
        _basis_at_rule(u, rule, line, b)
        for m in range(cube):
            v = u.dof(c, m)
            for d in range(rule.W):
                coeffs[d * cube + m] = v[d]
        for d in range(rule.W):
            _along_x_and_y(b, coeffs, xs, xys, n, p, d)
            totals[d] = rule.weights[0] * 0.0
        for qz in range(n):
            for qy in range(n):
                for qx in range(n):
                    w = (rule.weights[qx] * rule.weights[qy] * rule.weights[qz]
                         * elements[(qz * n + qy) * n + qx])
                    for d in range(rule.W):
                        totals[d] += w * _along_z(b, xys, n, p, d, qx, qy, qz)
        e = cells.entity_id(c)
        for d in range(rule.W):
            out[e][d] = totals[d]


@tack.kernel
def _drop_cells(cells, values):
    # Cells of a lower dimension than the mesh's: no part of its measure.
    for c in cells:
        values[cells.entity_id(c)] = 0.0


def _top_cells(data, values):
    """``values`` (one per cell) with the cells below the mesh's dimension zeroed."""
    top = data.topology.dimension
    for group in data.topology.groups():
        if group.count and group.shape.DIMENSION < top:
            _drop_cells(data.domain_view("cells", group), values)
    return values


@tack.kernel
def _polyhedron_integrals(cells, u, out):
    # A point field is linear on a polyhedron's sub-tetrahedra (each face fanned
    # from its points' mean, the fans from the cell's points' mean, values too):
    # each one's volume times the mean of its corners' values. Positions from the
    # cell's first point, as cell_geometry takes them.
    for c in cells:
        origin = cells.point_position(cells.point_id(c, 0))
        xc = origin - origin
        uc = u[cells.point_id(c, 0)] * 0.0
        for j in range(cells.num_points(c)):
            xc += cells.point_position(cells.point_id(c, j)) - origin
            uc += u[cells.point_id(c, j)]
        xc = xc / cells.num_points(c)
        uc = uc / cells.num_points(c)
        total = uc * 0.0
        for k in range(cells.num_faces(c)):
            size = cells.side_num_points(c, k)
            xf = origin - origin
            uf = uc * 0.0
            for j in range(size):
                xf += cells.side_position(c, k, j) - origin
                uf += u[cells.side_point_id(c, k, j)]
            xf = xf / size
            uf = uf / size
            for j in range(size):
                nxt = j + 1 if j + 1 < size else 0
                a = cells.side_position(c, k, j) - origin - xc
                b = cells.side_position(c, k, nxt) - origin - xc
                six = a.dot(b.cross(xf - xc))
                total += six * (uc + uf + u[cells.side_point_id(c, k, j)]
                                + u[cells.side_point_id(c, k, nxt)])
        out[cells.entity_id(c)] = total / 24.0


@tack.kernel
def _polygon_integrals(cells, u, out):
    # Linear on the triangles fanning a polygon from its points' mean: each one's
    # own area (signed by the polygon's normal, Newell's, so a planar polygon not
    # star-shaped from its mean cancels as it should) times its corners' mean
    # value. A non-planar polygon is that fan, as Tack interpolates and locates in
    # it, not its projection onto the normal, which measured a contour's wrinkled
    # free surface 1% small.
    for c in cells:
        origin = cells.side_position(c, 0, 0)
        n = origin - origin
        for k in range(cells.num_faces(c)):
            n += (cells.side_position(c, k, 0) - origin).cross(
                cells.side_position(c, k, 1) - origin)
        length = n.norm()
        unit = n / length if length > 0.0 else n
        xc = origin - origin
        uc = u[cells.point_id(c, 0)] * 0.0
        for j in range(cells.num_points(c)):
            xc += cells.point_position(cells.point_id(c, j)) - origin
            uc += u[cells.point_id(c, j)]
        xc = xc / cells.num_points(c)
        uc = uc / cells.num_points(c)
        total = uc * 0.0
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin - xc
            b = cells.side_position(c, k, 1) - origin - xc
            twice = _signed_twice_area(a, b, unit)
            total += twice * (uc + u[cells.side_point_id(c, k, 0)]
                              + u[cells.side_point_id(c, k, 1)])
        out[cells.entity_id(c)] = total / 6.0


@tack.kernel
def _polyline_integrals(cells, u, out):
    # Linear on each segment: its length times its ends' mean value.
    for c in cells:
        total = u[cells.line_point_id(c, 0)] * 0.0
        for j in range(cells.line_num_points(c) - 1):
            p, q = cells.line_point_id(c, j), cells.line_point_id(c, j + 1)
            total += (cells.point_position(q) - cells.point_position(p)).norm() * (u[p] + u[q])
        out[cells.entity_id(c)] = total / 2.0


# ── Integrals as weighted sums ──────────────────────────────────────
#
# A field linear in its points' values (order-1 point data) has the integral
# sum_p u_p W_p, W_p the integral of point p's shape function over its cells; cell
# data has sum_c u_c |c|. The weights depend on the mesh alone: one quadrature
# makes them, and each field is then a dot product (``_weights``), rather than a
# quadrature of its own per field (14 of them took 13.6 s on AhmedML).

@tack.kernel
def _corner_weights(cells, points, weights, start, mask, out):
    # Each corner's share of its cell's integral of a field linear in its points'
    # values: its shape function's integral, the quadrature's weights times the
    # measure element times the shape function; 0 for a cell ``mask`` drops.
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            out[first + j] = weights[0] * 0.0
        if mask[cells.entity_id(c)] != 0:
            for q in range(weights.shape[0]):
                d = weights[q] * _element(cells.geometry_jacobian(c, points[q]), cells.DIMENSION)
                for j in range(cells.NUM_POINTS):
                    out[first + j] += d * cells.shape_function(j, points[q])


@tack.func
def _local(cells, c, p):
    """Point ``p``'s place among cell ``c``'s points."""
    at = 0
    for j in range(cells.num_points(c)):
        if cells.point_id(c, j) == p:
            at = j
    return at


@tack.kernel
def _polyhedron_corner_weights(cells, corners, start, mask, out):
    # ``_polyhedron_integrals``' weight on each of the cell's points: each
    # sub-tetrahedron's volume over 4 to each of its corners, the cell point's
    # share spread evenly over the cell's points and a face point's over its face's.
    for c in cells:
        first = start + corners[cells.index(c)]
        n = cells.num_points(c)
        origin = cells.point_position(cells.point_id(c, 0))
        zero = origin[0] - origin[0]
        for j in range(n):
            out[first + j] = zero
        if mask[cells.entity_id(c)] != 0:
            xc = origin - origin
            for j in range(n):
                xc += cells.point_position(cells.point_id(c, j)) - origin
            xc = xc / n
            centre = zero
            for k in range(cells.num_faces(c)):
                size = cells.side_num_points(c, k)
                xf = origin - origin
                for j in range(size):
                    xf += cells.side_position(c, k, j) - origin
                xf = xf / size
                face = zero
                for j in range(size):
                    nxt = j + 1 if j + 1 < size else 0
                    a = cells.side_position(c, k, j) - origin - xc
                    b = cells.side_position(c, k, nxt) - origin - xc
                    w = a.dot(b.cross(xf - xc)) / 24.0
                    centre += w
                    face += w
                    out[first + _local(cells, c, cells.side_point_id(c, k, j))] += w
                    out[first + _local(cells, c, cells.side_point_id(c, k, nxt))] += w
                for j in range(size):
                    out[first + _local(cells, c, cells.side_point_id(c, k, j))] += face / size
            for j in range(n):
                out[first + j] += centre / n


@tack.kernel
def _polygon_corner_weights(cells, corners, start, mask, out):
    # ``_polygon_integrals``' weight on each of the polygon's points: each fan
    # triangle's signed area over 3 to each of its corners, the cell point's share
    # spread evenly over the polygon's points.
    for c in cells:
        first = start + corners[cells.index(c)]
        n = cells.num_points(c)
        origin = cells.side_position(c, 0, 0)
        zero = origin[0] - origin[0]
        for j in range(n):
            out[first + j] = zero
        if mask[cells.entity_id(c)] != 0:
            normal = origin - origin
            for k in range(cells.num_faces(c)):
                normal += (cells.side_position(c, k, 0) - origin).cross(
                    cells.side_position(c, k, 1) - origin)
            length = normal.norm()
            unit = normal / length if length > 0.0 else normal
            xc = origin - origin
            for j in range(n):
                xc += cells.point_position(cells.point_id(c, j)) - origin
            xc = xc / n
            centre = zero
            for k in range(cells.num_faces(c)):
                a = cells.side_position(c, k, 0) - origin - xc
                b = cells.side_position(c, k, 1) - origin - xc
                w = _signed_twice_area(a, b, unit) / 6.0
                centre += w
                out[first + _local(cells, c, cells.side_point_id(c, k, 0))] += w
                out[first + _local(cells, c, cells.side_point_id(c, k, 1))] += w
            for j in range(n):
                out[first + j] += centre / n


@tack.kernel
def _owned_cells(marks, skipped, mask):
    for c in range(mask.shape[0]):
        mask[c] = 1 if (marks[c] & skipped) == 0 else 0


@tack.kernel
def _masked(values, mask, out):
    # A literal zero where a cell is left out: its value times 0 is NaN where the
    # value is NaN or infinite.
    for c in range(out.shape[0]):
        out[c] = values[c] if mask[c] != 0 else 0.0


@tack.kernel
def _weighted_chunks(values, weights, chunk, n, out):
    # Each chunk of ``chunk`` entries' sum of value times weight, accumulated in
    # ``out``'s type (f64 where the device has it). An entry of weight 0 adds
    # nothing: a cell left out, or a point only such cells use, may hold NaN or an
    # infinity, which times 0 is NaN.
    for k in range(out.shape[0]):
        total = out[k] * 0.0
        for i in range(k * chunk, min((k + 1) * chunk, n)):
            if weights[i] != 0:
                total += values[i] * weights[i]
        out[k] = total


#: Entries a chunk of ``_weighted_chunks``: its sums go to the host, summed there in f64.
_CHUNK = 4096


def _weighted_sum(values, weights):
    """``sum_i values[i] weights[i]`` as a float (or an array, for vector values): by
    chunks on the device, in f64 where it has it, then in f64 on the host."""
    from tack.runtime.dispatch import get_backend

    n = arrays.size_of(values)
    width = arrays.width_of(values)
    acc = tack.f64 if get_backend().supports_f64 else arrays.dtype_of(values)
    chunks = max(-(-n // _CHUNK), 1)
    out = tack.zeros((chunks,), tack.vec(width, acc) if width else acc)
    if n:
        _weighted_chunks(values, weights, _CHUNK, n, out)
    return np.asarray(out.numpy(), np.float64).sum(axis=0)


class _Weights:
    """The weights ``integrate`` takes fields' integrals by, made on first use: each
    cell's measure for cell data, each point's shape function's integral for point
    data; cells not owned (ghost marks) or below the mesh's dimension weigh 0.
    ``None`` from ``points`` where a group's cells are not ones it weighs (a
    polyhedral topology's: their integrals take ``_no_reference_values``)."""

    def __init__(self, data, measures):
        self.data = data
        self.measures = measures            # top cells' measures (_top_cells)
        self._mask = self._points = self._cells = None

    def mask(self):
        if self._mask is None:
            data = self.data
            mask = tack.empty((data.num_cells,), tack.u8)
            ghosts = data.ghosts
            if ghosts is not None and ghosts.cells is not None:
                from tack.dataset.ghosts import Ghosts

                if data.num_cells:
                    _owned_cells(ghosts.cells, int(Ghosts.SKIPPED_CELLS), mask)
            else:
                mask.fill(1)
            self._mask = _top_cells(data, mask)
        return self._mask

    def cells(self):
        if self._cells is None:
            measures = self.measures
            self._cells = tack.empty((arrays.size_of(measures),), arrays.dtype_of(measures))
            if self.data.num_cells:
                _masked(measures, self.mask(), self._cells)
        return self._cells

    def points(self):
        if self._points is None:
            self._points = self._point_weights()
        return self._points

    def _point_weights(self):
        from tack.algorithms import segment_reduce
        from tack.arrays import Permutation
        from tack.dataset.topology import (
            StructuredTopology,
            UnstructuredTopology,
            corner_layout,
            polyhedral_corners,
        )

        data = self.data
        topology = data.topology
        if (not isinstance(topology, (UnstructuredTopology, StructuredTopology))
                or not isinstance(data.geometry.space, H1)):
            return None
        top = topology.dimension
        start_of, total = corner_layout(topology)
        corners = tack.zeros((max(total, 1),), data.dtype)
        mask = self.mask()
        for group in data.launch_groups("cells", []):
            if not group.count or group.shape.DIMENSION != top:
                continue                                  # weighs nothing
            cells = data.domain_view("cells", group)
            start = start_of[group.root]
            if group.shape.REFERENCE:
                n = _per_axis(group.shape, _points_per_axis(data, 1, group.shape))
                _corner_weights(cells, *_tables(data, group.shape, n), start, mask, corners)
            elif group.shape.DIMENSION == 3:
                _polyhedron_corner_weights(cells, polyhedral_corners(topology)[0], start, mask,
                                           corners)
            elif group.shape.DIMENSION == 2:
                _polygon_corner_weights(cells, polyhedral_corners(topology)[0], start, mask,
                                        corners)
            else:
                return None
        weights = tack.zeros((data.num_points,), data.dtype)
        if total and data.num_points:
            offsets, entries = topology.point_links()
            segment_reduce(Permutation(entries, corners), offsets, op="sum", out=weights,
                           empty="keep")
        return weights


def quadrature(shape, n):
    """``(points, weights)``: ``n``-point Gauss-Legendre rules along each axis of
    ``shape``'s reference element, collapsed onto a simplex (Duffy: the triangle
    ``(u, v (1 - u))`` weighted by ``1 - u``, the tetrahedron likewise) and a wedge
    (the triangle's rule times the segment's). A pyramid's reference element is the
    unit cube -- its map collapses the top itself -- so it takes the cube's.
    ``points`` are ``(m, 3)``, the weights sum to the element's measure, and a
    polynomial of degree ``2 n - 1`` along each axis of the cube (fewer collapsed)
    integrates exactly."""
    d = shape.DIMENSION
    if d == 0:
        return np.zeros((1, 3)), np.ones(1)     # a vertex: its point, measure 0
    x, w = np.polynomial.legendre.leggauss(int(n))
    x, w = (x + 1) / 2, w / 2
    if shape in (shapes.Triangle, shapes.Tetra, shapes.Wedge):
        u, v = np.meshgrid(x, x, indexing="ij")
        wu, wv = np.meshgrid(w, w, indexing="ij")
        if shape is shapes.Tetra:
            u, v, t = np.meshgrid(x, x, x, indexing="ij")
            wu, wv, wt = np.meshgrid(w, w, w, indexing="ij")
            points = np.stack([u, v * (1 - u), t * (1 - u) * (1 - v)], -1).reshape(-1, 3)
            return points, (wu * wv * wt * (1 - u) ** 2 * (1 - v)).reshape(-1)
        tri = np.stack([u, v * (1 - u)], -1).reshape(-1, 2)
        tri_w = (wu * wv * (1 - u)).reshape(-1)
        if shape is shapes.Triangle:
            return np.c_[tri, np.zeros(len(tri))], tri_w
        points = np.array([[a, b, z] for a, b in tri for z in x])
        return points, np.outer(tri_w, w).reshape(-1)
    grids = np.meshgrid(*([x] * d), indexing="ij")
    weights = np.prod(np.meshgrid(*([w] * d), indexing="ij"), axis=0).reshape(-1)
    points = np.stack([g.reshape(-1) for g in grids], -1)
    return np.c_[points, np.zeros((len(points), 3 - d))], weights


def _order(space):
    """The polynomial order a space's values carry along an axis: an evaluated
    space's ``integration_order``."""
    if space.evaluated:
        if space.integration_order is None:
            raise TypeError(f"integrate needs {space!r} to give its integration_order")
        return int(space.integration_order)
    if isinstance(space, (Constant, Values)):
        return 0 if isinstance(space, Constant) or space.on == "cells" else 1
    if getattr(space, "varies", False):
        return space.max_order                  # an order per cell: the highest
    return max(int(getattr(space, "order", 1)), 0)


def _tables(data, shape, n):
    """The rule of ``n`` points an axis for ``shape`` as device arrays in the
    geometry's float type."""
    points, weights = quadrature(shape, n)
    dtype = data.dtype
    p = tack.empty((len(points),), tack.vec(3, dtype))
    tack.copyto(p, points.astype(dtype.numpy_dtype))
    w = tack.empty((len(weights),), dtype)
    tack.copyto(w, weights.astype(dtype.numpy_dtype))
    return p, w


_TENSOR = (shapes.Line, shapes.Pixel, shapes.Quad, shapes.Voxel, shapes.Hexahedron)

#: Gauss points an axis on a quad, at least: a quad out of plane has an area element
#: (|J0 x J1|, a square root) no rule integrates exactly; four points an axis give
#: its bilinear patch's area within rounding unless it is twisted far out of plane
#: (a saddle of unit height on a unit square: 6e-5).
_QUAD_POINTS = 4


def _per_axis(shape, n):
    """``n`` points an axis, or ``_QUAD_POINTS`` on a quad if more."""
    return max(n, _QUAD_POINTS) if shape is shapes.Quad else n


def _points_per_axis(data, field_order, shape=None):
    """Enough Gauss points an axis for a field of ``field_order`` on ``data``'s
    geometry: the integrand's degree along an axis is at most the field's plus the
    measure element's (the dimension times the geometry's order), and a collapsed
    rule loses one more an axis. Given a ``shape`` that is a tensor product (no
    collapse), the fewest that are exact there: the measure element's degree along
    an axis is one less, and a rule of n points is exact to degree 2 n - 1."""
    g = _order(data.geometry.space)
    if shape in _TENSOR:
        return max(1, (field_order + shape.DIMENSION * g + 1) // 2)
    dimension = data.topology.dimension
    return max(1, (field_order + dimension * g + dimension) // 2 + 1)


def cell_measures(data, n=None):
    """``(measures, centroids)`` of every cell with a reference element, by quadrature
    (``n`` points an axis, by default enough for the geometry): the curved cells'
    ``cell_geometry``."""
    vec = tack.vec(3, data.dtype)
    measures = tack.empty((data.num_cells,), data.dtype)
    centroids = tack.empty((data.num_cells,), vec)
    given = n
    n = _points_per_axis(data, 0) if n is None else n
    for group in data.launch_groups("cells", []):
        if group.count:
            m = n if given is not None else _per_axis(group.shape, n)
            _cell_measures(data.domain_view("cells", group), *_tables(data, group.shape, m),
                           measures, centroids)
    return measures, centroids


def _group_integrals(data, field, group, order, out):
    """``group``'s cells' integrals of ``field`` into ``out``, on the fewest Gauss
    points exact for the shape: by sum factorization (``_tensor_integrals``,
    ``_tensor_vector_integrals``) for a field in a tensor basis on hexahedra or
    voxels, its measure element too on a curved tensor geometry (``_TensorElements``);
    by a value a point (``_cell_integrals``) otherwise."""
    cells = data.domain_view("cells", group)
    view = field.view(group)
    n = _per_axis(group.shape, _points_per_axis(data, order, group.shape))
    width = arrays.width_of(field.values)
    if group.shape in _TENSOR and group.shape.DIMENSION == 3 and isinstance(
            view, views._TensorBasis):
        geo = _TensorElements() if isinstance(cells, views._TensorGeometry) else \
            _PointElements()
        if width:
            _tensor_vector_integrals(cells, view, _rule(data, n, width), geo, out)
        else:
            _tensor_integrals(cells, view, _rule(data, n), geo, out)
    else:
        _cell_integrals(cells, view, *_tables(data, group.shape, n), out)


def _cell_values(data, field):
    """Each cell's integral of ``field`` (a field with a basis), by quadrature."""
    field = _with_basis(data, field)
    width = arrays.width_of(field.values)
    dtype = arrays.dtype_of(field.values)
    out = tack.zeros((data.num_cells,), tack.vec(width, dtype) if width else dtype)
    order = _order(field.space)
    top = data.topology.dimension
    for group in data.launch_groups("cells", [field]):
        if group.count and group.shape.DIMENSION == top:
            # The fewest points exact for the shape (2 an axis on a hexahedron for
            # order 1, not the 4 a collapsed rule needs).
            _group_integrals(data, field, group, order, out)
    return out


def _on_points(space):
    """Whether ``space`` holds one value a point: linear on polyhedra's pieces."""
    return ((isinstance(space, H1) and space.order == 1 and space.dofs is None
             and space.basis is None)
            or (isinstance(space, Values) and space.on == "points"))


def _no_reference_values(data, field):
    """Each polyhedron's, polygon's or polyline's integral of a point field, exact
    on the pieces it is linear on."""
    values = arrays.materialize(field.values)
    width = arrays.width_of(values)
    dtype = arrays.dtype_of(values)
    out = tack.zeros((data.num_cells,), tack.vec(width, dtype) if width else dtype)
    for group in data.launch_groups("cells", []):
        if not group.count or group.shape.REFERENCE:        # shapes beside polyhedra
            continue
        cells = data.domain_view("cells", group)
        if group.shape.DIMENSION == 3:
            _polyhedron_integrals(cells, values, out)
        elif group.shape.DIMENSION == 2:
            _polygon_integrals(cells, values, out)
        elif group.shape is shapes.PolyLine:
            _polyline_integrals(cells, values, out)
    return out


@tack.kernel
def _times_measures(cells, values, measures, out):
    for c in cells:
        e = cells.entity_id(c)
        out[e] = values[e] * measures[e]


def _mixed_cell_values(data, field, measures):
    """Each cell's integral of ``field`` on a topology with polyhedra beside its
    shapes: a shape's by quadrature, a polyhedron's exactly on its sub-tetrahedra
    (point data) or as its value times its measure (cell data, ``measures``)."""
    field = _with_basis(data, field)
    space = field.space
    if not (_on_points(space) or isinstance(space, Constant)):
        raise NotImplementedError(
            f"integrating {space!r} needs a reference element (polyhedra integrate cell "
            "data and point data)")
    width = arrays.width_of(field.values)
    dtype = arrays.dtype_of(field.values)
    out = tack.zeros((data.num_cells,), tack.vec(width, dtype) if width else dtype)
    values = arrays.materialize(field.values)
    order = _order(space)
    for group in data.launch_groups("cells", [field]):
        if not group.count:
            continue
        cells = data.domain_view("cells", group)
        if group.shape.REFERENCE:
            _group_integrals(data, field, group, order, out)
        elif isinstance(space, Constant):
            _times_measures(cells, values, measures, out)
        elif group.shape.DIMENSION == 3:
            _polyhedron_integrals(cells, values, out)
        else:
            _polygon_integrals(cells, values, out)
    return out


def _piece(data, names):
    """One dataset's owned cells' measure and integrals, summed in cell order (f64):
    ``names`` as ``_names`` gives them. Each sum is of ``owned_values``, which
    leaves duplicate, hidden and refined cells out (``Ghosts.SKIPPED_CELLS``), so
    the dataset is taken as it is, its cell geometry kept, not extracted (#404)."""
    from tack.dataset.algorithms.cell_geometry import cell_geometry
    from tack.dataset.ghosts import owned_values

    given = [field for _, field in names]
    measures = Field(Values(data, "cells"), _top_cells(data, arrays.materialize(
        cell_geometry(data).measure.values)))
    weights = _Weights(data, measures.values)
    from tack.runtime.dispatch import get_backend

    if get_backend().supports_f64:
        # The owned cells' measures summed as the fields are, by chunks in f64 on the
        # device, not copied to the host (21.5M of AhmedML's: 0.1 s of 0.17 on CUDA);
        # refused, as owned_values refuses, where the cells' marks were lost.
        from tack.dataset.ghosts import refuse_lost

        refuse_lost(data, measures.space)
        measure = float(_weighted_sum(measures.values, weights.mask()))
    else:                                       # the host's f64 sum, not the device's f32
        measure = float(np.sum(np.asarray(owned_values(data, measures).numpy(), np.float64)))
    values = {}
    field_units = {}
    for (name, _), field in zip(names, given):
        field = _field(data, field)
        field_units[name] = field.unit
        if not getattr(data.topology, "reference_cells", True):
            field = field.space.stored(field, data)
        # Point data and cell data: a weighted sum of their values.
        space = _with_basis(data, field).space
        if _on_points(space) and weights.points() is not None:
            values[name] = _weighted_sum(arrays.materialize(field.values), weights.points())
            continue
        if isinstance(space, Constant) or (isinstance(space, Values) and space.on == "cells"):
            values[name] = _weighted_sum(arrays.materialize(field.values), weights.cells())
            continue
        if data.topology.mixed:
            # Polyhedra beside the shapes: each group its own way, cell by cell.
            per_cell = _top_cells(data, _mixed_cell_values(data, field, measures.values))
            values[name] = np.asarray(owned_values(data, Field(Values(data, "cells"), per_cell))
                                      .numpy(), np.float64).sum(axis=0)
            continue
        if not getattr(data.topology, "reference_cells", True):
            space = _with_basis(data, field).space
            if _on_points(space):
                per_cell = _top_cells(data, _no_reference_values(data, field))
                values[name] = np.asarray(owned_values(data, Field(Values(data, "cells"),
                                                                   per_cell)).numpy(),
                                          np.float64).sum(axis=0)
                continue
            if not isinstance(space, Constant):
                raise NotImplementedError(
                    f"integrating {space!r} needs a reference element (polygons and "
                    "polyhedra integrate cell data and point data)")
            # Constant in each cell: its value times the cell's measure.
            cell_values = owned_values(data, Field(Values(data, "cells"),
                                                   arrays.materialize(field.values))).numpy()
            owned = np.asarray(owned_values(data, measures).numpy(), np.float64)
            cell_values = np.asarray(cell_values, np.float64)
            values[name] = (cell_values * (owned if cell_values.ndim == 1
                                           else owned[:, None])).sum(axis=0)
            continue
        per_cell = _cell_values(data, field)
        values[name] = np.asarray(owned_values(data, Field(Values(data, "cells"), per_cell))
                                  .numpy(), np.float64).sum(axis=0)
    return measure, values, field_units


@on_inputs_device
def integrate(data, fields=None):
    """The measure of ``data`` -- its cells' total volume, area or length -- and the
    integral over it of each of ``fields`` (names; a mapping of names to field
    arguments, each integral under its name; by default every field with a basis
    or on the cells), as VTK's ``vtkIntegrateAttributes`` gives them:
    ``Integrals(measure, measure_unit, values, units)``, ``values[name]`` a float or
    a tuple for a vector field.

    Each cell is integrated by Gauss quadrature in its reference element through
    its own map (``quadrature``: collapsed onto simplices and wedges), with enough
    points for the field's order and the geometry's, so a curved geometry and a
    field of any order (``H1``, ``L2``, point and cell data) integrate exactly, to
    rounding. An evaluated field (an expression's) is integrated through its view
    in the same way, with points for its space's ``integration_order``: exact where
    the field is a polynomial of that order, a quadrature approximation where it
    is not. Polyhedra, polygons and polylines take their measure from
    ``cell_geometry`` and integrate cell data and point data, which is linear on
    their pieces (sub-tetrahedra, triangles, segments), exactly. Only the cells of the mesh's dimension count, as in VTK: a volume's
    integrals leave out its surfaces, lines and vertices. Only owned cells count
    (ghost marks), and a ``PartitionedDataSet`` sums its pieces over every rank, in
    the pieces' order."""
    from tack.dataset.algorithms._common import _measure_unit
    from tack.dataset.partitioned import PartitionedDataSet, agree

    if isinstance(data, PartitionedDataSet):
        from tack.dataset.algorithms.statistics import _over_pieces

        pieces = list(data)
        # Refused alike on every rank, a rank without pieces too: from the arguments.
        given = (list(fields.values()) if hasattr(fields, "items") else
                 [] if fields is None else
                 [fields] if is_field_key(fields) or isinstance(fields, Field) else list(fields))
        if any(isinstance(field, Field) for field in given):
            raise TypeError("integrate over a PartitionedDataSet takes fields by name or "
                            "as objects whose on(piece) gives the piece's: a Field belongs "
                            "to one partition's topology")
        if not hasattr(fields, "items") and any(not is_field_key(f) for f in given):
            raise TypeError("integrate names each Field it is given: pass {name: field}")

        def mine():
            # This rank's names, measure unit and fields' units, every piece's
            # checked: the same names on each, one unit each, so that a piece
            # without a field, or in another unit, fails on every rank.
            if not pieces:
                return None
            return _schema_over(pieces, fields, "integrate over a PartitionedDataSet's "
                                "partitions")

        # Every rank's, the same on every rank with pieces, so that sums of values
        # in two units are refused on every rank; a rank without pieces gets them too.
        held = [m for m in agree(data.world, mine) if m is not None]
        labels, measure_unit = _schema_across(
            held, "integrate over a PartitionedDataSet's ranks") if held else ([], None)
        names = _names(pieces[0], fields) if pieces else [(name, name) for name in labels]
        parts = _over_pieces(data, lambda piece: _piece(piece, names))
        # Every piece's fields' units, every rank's (so every rank refuses alike):
        # one each, or the sums would add values in two units.
        field_units = {name: units.common(f"integrate over a PartitionedDataSet's field "
                                          f"{name!r}", [u[name] for _, _, u in parts])
                       for name in labels}
        measure = math.fsum(m for m, _, _ in parts)
        values = {name: np.sum([v[name] for _, v, _ in parts], axis=0) for name in labels}
        out_units = {name: units.derived((field_units[name], 1), (measure_unit, 1))
                     for name in labels}
    else:
        names = _names(data, fields)
        measure, values, field_units = _piece(data, names)
        measure_unit = _measure_unit(data)
        out_units = {name: units.derived((field_units[name], 1), (measure_unit, 1))
                     for name, _ in names}
    return Integrals(measure=measure, measure_unit=measure_unit,
                     values={name: (float(v) if np.ndim(v) == 0 else tuple(map(float, v)))
                             for name, v in values.items()},
                     units=out_units)


def _schema_over(pieces, fields, what):
    """``(names, measure unit)`` of ``pieces``, which must have the same fields to
    integrate (``_names``, in any order) and one unit for the measure
    (``units.common``: unknown where one is). The fields' units are compared
    where every piece's integrals are, with no field evaluated twice."""
    from tack.dataset.algorithms._common import _measure_unit

    found = [_names(piece, fields) for piece in pieces]
    labels = [name for name, _ in found[0]]
    for i, names in enumerate(found[1:], 1):
        if sorted(name for name, _ in names) != sorted(labels):
            raise ValueError(f"{what} need the same fields to integrate: 0 has {labels}, {i} "
                             f"{[name for name, _ in names]} (choose fields=)")
    return labels, units.common(f"{what}' geometry", [_measure_unit(p) for p in pieces])


def _schema_across(schemas, what):
    """``_schema_over``'s of every rank with pieces as one: the same names, one
    measure unit."""
    labels, _ = schemas[0]
    for other, _ in schemas[1:]:
        if sorted(other) != sorted(labels):
            raise ValueError(f"{what} need the same fields to integrate, not {labels} and "
                             f"{other} (choose fields=)")
    return labels, units.common(f"{what}' geometry", [m for _, m in schemas])


def _names(data, fields):
    """``[(name, field)]``, each field a field argument (``_field``): every stored
    field with a basis or on the cells by default, the names given, or a mapping's
    entries. Names and ``Field``s are checked here; an object whose ``on(data)``
    gives the field is resolved once a piece, where it is integrated."""
    if fields is None:
        return [(name, name) for name, field in data.fields.items()
                if isinstance(field.space, (H1, L2, Constant)) or field.space.evaluated
                or (isinstance(field.space, Values) and field.space.on in ("points", "cells"))]
    if hasattr(fields, "items"):
        names = list(fields.items())
    else:
        entries = [fields] if is_field_key(fields) or isinstance(fields, Field) else list(fields)
        if any(not is_field_key(entry) for entry in entries):
            raise TypeError("integrate names each Field it is given: pass {name: field}")
        names = [(entry, entry) for entry in entries]
    for _, field in names:
        if is_field_key(field) or isinstance(field, Field):
            _field(data, field)
    return names
