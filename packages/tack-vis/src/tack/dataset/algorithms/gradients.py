# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``gradients``: the gradient of ``field`` -- ``H1``, ``L2`` or ``Constant``, scalar or vector -- at each cell's parametric center (``at="cells"``) or at the points (``at="points"``: each cell's gradient at the point, averaged over the cells around it, as VTK's vtkGradientFilter and Viskores' point gradients do). On polyhedra, which have no reference element, each cell's gradient is Green–Gauss over its faces fanned from their points' means."""

import tack
from tack import arrays
from tack.arrays import materialize, size_of, width_of
from tack.dataset import shapes, units
from tack.dataset.algorithms._common import _field, _vectors, _with_basis
from tack.dataset.algorithms._links import _average_links, _cell_values_at_corners
from tack.dataset.carry import _take
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.dataset.topology import corner_layout, polyhedral_corners
from tack.runtime.dispatch import on_inputs_device

__all__ = ['gradients', 'world_gradient', 'world_hessian', 'world_jet']


@tack.func
def world_gradient(cells, u, c, pc):
    """`u`'s gradient in the world at parametric `pc` of cell `c`, as its view gives it:
    from its basis by default (`_BasisGradient`), the gradient it was given
    (`Field(..., gradient=)`), or a space's own. For kernels that need a field's
    gradient where `gradients` does not put it."""
    return u.world_gradient(cells, c, pc)


@tack.func
def world_hessian(cells, u, c, pc):
    """`u`'s second derivatives in the world at parametric `pc` of solid cell `c`, a
    symmetric 3x3 matrix, as its view gives them (`world_jet`). From the basis: by the
    chain rule through the geometry, the parametric Hessian less the geometry's
    curvature along the world gradient, taken to world axes on both sides by the
    inverse Jacobian; analytic on every shape at orders 1 and 2, and at any order on
    tensor shapes, simplices and wedges; zero on surfaces, lines and vertices. NaN for
    a field given only its gradient."""
    _, _, h = world_jet(cells, u, c, pc)
    return h


@tack.func
def world_jet(cells, u, c, pc):
    """Scalar `u`'s value, world gradient and world Hessian at parametric `pc` of cell
    `c`, as its view gives them, from one pass over the cell's values where it can:
    what a kernel needing several of them reads once. The gradient is
    `world_gradient`'s, and the Hessian `world_hessian`'s."""
    return u.world_jet(cells, c, pc)


@tack.func
def _in_plane(m, d):
    """The world vector in the plane of `m`'s first two columns whose dots with them
    are `d[0]` and `d[1]`."""
    a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
    b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
    aa = a.dot(a)
    ab = a.dot(b)
    bb = b.dot(b)
    det = aa * bb - ab * ab
    return ((bb * d[0] - ab * d[1]) * a + (aa * d[1] - ab * d[0]) * b) / det


class _BasisGradient:
    """A field view's world gradient from its basis: on every field view with a
    ``parametric_gradient`` and no other source (``_derivative_mixins``)."""

    @tack.func
    def world_gradient(self, cells, c, pc):
        # d(u)/d(x) from d(u)/d(pc), through the geometry's Jacobian J. A solid's is
        # J^-T d(u)/d(pc). A 2D cell's lies in its tangent plane, spanned by J's
        # columns a and b: g = alpha a + beta b with g . a and g . b the parametric
        # derivatives, VTK's and Viskores' in-plane gradient (no area: NaN, as a
        # solid of no volume). Lines and vertices: zero.
        g = tack.vec3(0.0, 0.0, 0.0)
        if cells.DIMENSION == 3:
            g = cells.geometry_jacobian(c, pc).inverse().transpose() @ self.parametric_gradient(c, pc)
        elif cells.DIMENSION == 2:
            g = _in_plane(cells.geometry_jacobian(c, pc), self.parametric_gradient(c, pc))
        return g


class _BasisJet:
    """A field view's value, world gradient and world Hessian from its basis, in one
    pass: on every field view with a ``parametric_jet``."""

    @tack.func
    def world_jet(self, cells, c, pc):
        v, pg, ph = self.parametric_jet(c, pc)
        g = tack.vec3(0.0, 0.0, 0.0)
        h = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        if cells.DIMENSION == 3:
            inverse = cells.geometry_jacobian(c, pc).inverse()
            g = inverse.transpose() @ pg
            h = inverse.transpose() @ (ph - cells.geometry_curvature(c, pc, g)) @ inverse
        elif cells.DIMENSION == 2:
            g = _in_plane(cells.geometry_jacobian(c, pc), pg)
        return v, g, h


class _GivenGradient:
    """A field given its gradient (``Field(..., gradient=)``): ``world_gradient`` is that
    field's value at ``pc`` of cell ``c``, and ``world_jet`` pairs it with the field's
    value and a Hessian of NaN, which a gradient alone does not determine."""

    @tack.func
    def world_gradient(self, cells, c, pc):
        return self.given_gradient.value(c, pc)

    @tack.func
    def world_jet(self, cells, c, pc):
        n = self.no_hessian
        return (self.value(c, pc), self.given_gradient.value(c, pc),
                tack.mat3([n, n, n], [n, n, n], [n, n, n]))


@tack.kernel
def _gradients(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = world_gradient(cells, u, c, cells.parametric_center())


@tack.kernel
def _node_gradients(cells, u, out):
    # The gradient at each of the cell's nodes of the target space.
    for c in cells:
        for j in range(cells.node_count(c)):
            out[cells.node_dof(c, j)] = world_gradient(cells, u, c, cells.node_parametric_point(c, j))


#: A corner whose scaled Jacobian (the determinant over its columns' lengths) is
#: within this of 0 is flat, two of the cell's faces meeting there in one plane,
#: whatever rounding left of the 0 -- in f32, positions of size 6 round a cell of
#: size 1e-3 by 4e-4 of it, and AhmedML's flat corners came out at 1e-4 to 2e-4 --
#: its angle under a tenth of a degree; below its negative, the cell is tangled.
_FLAT = tack.constant(1e-3)


@tack.func
def _scaled(m):
    """A solid's Jacobian ``m``, scaled: its determinant over the product of its
    columns' lengths (at a hexahedron's corner, its three edges there), 1 for a
    right angle, 0 flat, negative folded; 0 for a column of no length."""
    a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
    b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
    d = tack.vec3(m[0, 2], m[1, 2], m[2, 2])
    lengths = a.norm() * b.norm() * d.norm()
    scaled = lengths - lengths
    if lengths > 0.0:
        scaled = m.determinant() / lengths
    return scaled


@tack.func
def _corner_kind(cells, c, pc):
    """``(singular, folded)`` where the geometry's Jacobian at parametric `pc` of cell
    `c` is: singular 1 where it has no inverse -- a solid's flat (its scaled
    Jacobian within ``_FLAT`` of 0), a 2D cell's in-plane determinant
    (`_in_plane`'s) exactly 0 -- and folded 1 where a solid's is below ``-_FLAT``
    (a tetrahedron, whose Jacobian is one, never: inside out is ``orient``'s)."""
    m = cells.geometry_jacobian(c, pc)
    s = 0
    folded = 0
    if cells.DIMENSION == 3:
        scaled = _scaled(m)
        if abs(scaled) <= _FLAT:
            s = 1
        elif scaled < -_FLAT and cells.ID != shapes.TETRA:
            folded = 1
    elif cells.DIMENSION == 2:
        a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
        b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
        aa = a.dot(a)
        ab = a.dot(b)
        bb = b.dot(b)
        if aa * bb - ab * ab == 0.0:
            s = 1
    return s, folded


@tack.func
def _shape_green_gauss(cells, u, c):
    """A solid cell's gradient as a polyhedron's (``_polyhedral_gradients``): Green-Gauss
    over its faces from the shape's tables, each fanned from its points' mean, so a
    linear field's is exact, without the map's Jacobian. Positions and values from
    the cell's first corner. Faces wound all outward or all inward give the same
    gradient, the volume's sign cancelling."""
    origin = cells.corner_position(c, 0)
    u0 = u.corner_value(c, 0)
    six = origin[0] - origin[0]
    total = origin - origin
    for k in range(cells.NUM_FACES):
        size = cells.face_num_points(k)
        first = cells.corner_position(c, cells.face_corner(k, 0)) - origin
        u_first = u.corner_value(c, cells.face_corner(k, 0)) - u0
        a = first
        ua = u_first
        points = first
        values = u_first
        newell = origin - origin
        spun = origin - origin
        along = origin - origin
        for j in range(1, size + 1):
            b = first
            ub = u_first
            if j < size:
                b = cells.corner_position(c, cells.face_corner(k, j)) - origin
                ub = u.corner_value(c, cells.face_corner(k, j)) - u0
                points += b
                values += ub
            n = a.cross(b)
            both = ua + ub
            newell += n
            spun += n * both
            along += (b - a) * both
            a = b
            ua = ub
        apex = points / size
        six += apex.dot(newell)
        total += spun + along.cross(apex) + newell * (values / size)
    return total / six


#: A hexahedron's vertex ``j``'s neighbours along r, s and t (``3 j`` to ``3 j + 2``), in
#: VTK's order of its corners, and the sign taking each edge from r (s, t) = 0 to 1.
_HEXAHEDRON_NEXT = tack.constant((1, 3, 4, 0, 2, 5, 3, 1, 6, 2, 0, 7,
                                  5, 7, 0, 4, 6, 1, 7, 5, 2, 6, 4, 3), tack.i32)
_HEXAHEDRON_SIGN = tack.constant((1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1,
                                  1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1), tack.i32)


@tack.func
def _hexahedron_corner(cells, u, c, j):
    """A straight hexahedron's Jacobian at its vertex ``j`` and an order-1 field's
    parametric gradient there: at a vertex only it and its three neighbours along the
    axes have a shape gradient (of 1), so the Jacobian's columns are those three
    edges and the gradient the values' differences along them -- what the
    trilinear map gives there, without evaluating its eight shape gradients. Returns
    the three edges (the Jacobian's columns) and the gradient."""
    x = cells.corner_position(c, j)
    v = u.corner_value(c, j)
    a = _HEXAHEDRON_NEXT[3 * j]
    b = _HEXAHEDRON_NEXT[3 * j + 1]
    d = _HEXAHEDRON_NEXT[3 * j + 2]
    sa = tack.f32(_HEXAHEDRON_SIGN[3 * j])
    sb = tack.f32(_HEXAHEDRON_SIGN[3 * j + 1])
    sd = tack.f32(_HEXAHEDRON_SIGN[3 * j + 2])
    ea = (cells.corner_position(c, a) - x) * sa
    eb = (cells.corner_position(c, b) - x) * sb
    ed = (cells.corner_position(c, d) - x) * sd
    g = tack.vec3((u.corner_value(c, a) - v) * sa, (u.corner_value(c, b) - v) * sb,
                  (u.corner_value(c, d) - v) * sd)
    return ea, eb, ed, g


@tack.func
def _edges_frame(ea, eb, ed):
    """For a Jacobian of columns ``ea``, ``eb``, ``ed``: the cross products that make
    its inverse transpose -- J^-T g = (g0 (eb x ed) + g1 (ed x ea) + g2 (ea x eb)) / det
    -- its determinant, and its scaled Jacobian (``_scaled``'s), without inverting
    a matrix."""
    bc = eb.cross(ed)
    ca = ed.cross(ea)
    ab = ea.cross(eb)
    det = ea.dot(bc)
    lengths = ea.norm() * eb.norm() * ed.norm()
    scaled = lengths - lengths
    if lengths > 0.0:
        scaled = det / lengths
    return bc, ca, ab, det, scaled


@tack.func
def _hexahedron_corner_rows(cells, u, c, j):
    """``_hexahedron_corner`` for a 3-vector field: the edges, and the field's
    differences along them (each a 3-vector, one component a component)."""
    x = cells.corner_position(c, j)
    v = u.corner_value(c, j)
    a = _HEXAHEDRON_NEXT[3 * j]
    b = _HEXAHEDRON_NEXT[3 * j + 1]
    d = _HEXAHEDRON_NEXT[3 * j + 2]
    sa = tack.f32(_HEXAHEDRON_SIGN[3 * j])
    sb = tack.f32(_HEXAHEDRON_SIGN[3 * j + 1])
    sd = tack.f32(_HEXAHEDRON_SIGN[3 * j + 2])
    ea = (cells.corner_position(c, a) - x) * sa
    eb = (cells.corner_position(c, b) - x) * sb
    ed = (cells.corner_position(c, d) - x) * sd
    da = (u.corner_value(c, a) - v) * sa
    db = (u.corner_value(c, b) - v) * sb
    dd = (u.corner_value(c, d) - v) * sd
    return ea, eb, ed, da, db, dd


@tack.func
def _green_gauss_component(cells, u, c, k):
    """``_shape_green_gauss`` of component ``k`` of a vector field."""
    origin = cells.corner_position(c, 0)
    u0 = u.corner_value(c, 0)[k]
    six = origin[0] - origin[0]
    total = origin - origin
    for f in range(cells.NUM_FACES):
        size = cells.face_num_points(f)
        first = cells.corner_position(c, cells.face_corner(f, 0)) - origin
        u_first = u.corner_value(c, cells.face_corner(f, 0))[k] - u0
        a = first
        ua = u_first
        points = first
        values = u_first
        newell = origin - origin
        spun = origin - origin
        along = origin - origin
        for j in range(1, size + 1):
            b = first
            ub = u_first
            if j < size:
                b = cells.corner_position(c, cells.face_corner(f, j)) - origin
                ub = u.corner_value(c, cells.face_corner(f, j))[k] - u0
                points += b
                values += ub
            n = a.cross(b)
            both = ua + ub
            newell += n
            spun += n * both
            along += (b - a) * both
            a = b
            ua = ub
        apex = points / size
        six += apex.dot(newell)
        total += spun + along.cross(apex) + newell * (values / size)
    return total / six


@tack.kernel
def _vector_edge_gradients(cells, u, start, out, singular):
    # ``_corner_gradients`` of a 3-vector field on straight hexahedra, every
    # component from one frame a corner (``_edges_frame``): component i's gradient
    # is (da_i (eb x ed) + db_i (ed x ea) + dd_i (ea x eb)) / det, row i.
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        tangled = 0
        for j in range(cells.NUM_POINTS):
            singular[first + j] = 0
            ea, eb, ed, da, db, dd = _hexahedron_corner_rows(cells, u, c, j)
            bc, ca, ab, det, scaled = _edges_frame(ea, eb, ed)
            if scaled < -_FLAT:
                tangled = 1
            if abs(scaled) <= _FLAT:
                out[first + j] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                singular[first + j] = 1
            else:
                g0 = (da[0] * bc + db[0] * ca + dd[0] * ab) / det
                g1 = (da[1] * bc + db[1] * ca + dd[1] * ab) / det
                g2 = (da[2] * bc + db[2] * ca + dd[2] * ab) / det
                out[first + j] = [g0[0], g0[1], g0[2], g1[0], g1[1], g1[2],
                                  g2[0], g2[1], g2[2]]
        if tangled == 1:
            g0 = _green_gauss_component(cells, u, c, 0)
            g1 = _green_gauss_component(cells, u, c, 1)
            g2 = _green_gauss_component(cells, u, c, 2)
            for j in range(cells.NUM_POINTS):
                out[first + j] = [g0[0], g0[1], g0[2], g1[0], g1[1], g1[2],
                                  g2[0], g2[1], g2[2]]
                singular[first + j] = 0


@tack.kernel
def _corner_gradients(cells, u, start, edges, out, singular):
    # ``edges`` 1: a straight hexahedron's order-1 field, its corners' Jacobians and
    # gradients from their edges (``_hexahedron_corner``), as the general path's are.
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        # Tangled: the map folds over (a corner's scaled Jacobian below -_FLAT), so
        # its inverse, which the corner gradients take, is wrong near the fold and
        # unbounded at it. The cell's gradient is then its own as a polyhedron's,
        # at every corner.
        tangled = 0
        for j in range(cells.NUM_POINTS):
            singular[first + j] = 0
            if cells.ID == shapes.PYRAMID and j == 4:
                # The Jacobian is singular at a pyramid's apex. Extrapolate from
                # two points just below it on the axis, as vtkPyramid does.
                ga = world_gradient(cells, u, c, tack.vec3(0.5, 0.5, 0.998))
                gb = world_gradient(cells, u, c, tack.vec3(0.5, 0.5, 0.996))
                out[first + j] = 2.0 * ga - gb
            elif edges == 1:
                ea, eb, ed, pg = _hexahedron_corner(cells, u, c, j)
                bc, ca, ab, det, scaled = _edges_frame(ea, eb, ed)
                if scaled < -_FLAT:
                    tangled = 1
                if abs(scaled) <= _FLAT:
                    out[first + j] = tack.vec3(0.0, 0.0, 0.0)
                    singular[first + j] = 1
                else:
                    out[first + j] = (pg[0] * bc + pg[1] * ca + pg[2] * ab) / det
            else:
                pc = cells.parametric_point(j)
                flat, folded = _corner_kind(cells, c, pc)
                if folded == 1:
                    tangled = 1
                if flat == 1:
                    # Singular at this corner too, where two of the cell's faces
                    # meet flat (a corner on a straight edge of a face), or all but
                    # by rounding (AhmedML's hexahedra, which gave gradients to
                    # 1e9): its gradient grows without bound toward the corner and
                    # has no value there, so the cell is left out of the point's
                    # average.
                    out[first + j] = tack.vec3(0.0, 0.0, 0.0)
                    singular[first + j] = 1
                else:
                    out[first + j] = world_gradient(cells, u, c, pc)
        if cells.DIMENSION == 3 and tangled == 1:
            g = _shape_green_gauss(cells, u, c)
            for j in range(cells.NUM_POINTS):
                out[first + j] = g
                singular[first + j] = 0


def _by_edges(data, field, group):
    """1 where ``_corner_gradients`` takes a group's corners from their edges: straight
    hexahedra (an order-1 geometry) and a field of their corners' values (order-1
    point data) with its basis' gradient; else 0."""
    geometry = data.geometry.space
    space = field.space
    return int(group.shape is shapes.Hexahedron and field.gradient is None
               and isinstance(geometry, H1) and geometry.order == 1
               and isinstance(space, H1) and space.order == 1
               and space.dofs is None and space.basis is None)


@tack.kernel
def _without_singular(offsets, entries, singular, out, nan):
    # A point's average over the corners that have a gradient: the corners left out
    # added zero to the mean over all, which is scaled to the rest; NaN if none is left.
    for p in range(out.shape[0]):
        n = offsets[p + 1] - offsets[p]
        left_out = 0
        for k in range(offsets[p], offsets[p + 1]):
            left_out += singular[entries[k]]
        if left_out > 0:
            if left_out == n:
                out[p] = out[p] * nan
            else:
                # In the gradients' own type: Metal has no f64.
                out[p] = out[p] * n / (n - left_out)


@tack.kernel
def _polyhedral_gradients(cells, u, out):
    for c in cells:
        # Green-Gauss: (1/V) times the sum over the cell's boundary of each triangle's
        # outward area vector times the mean of its three values, the faces walked
        # outward and fanned from their points' means -- in position and in value,
        # so a linear field's gradient is exact. The triangles are cell_geometry's,
        # so V is its volume. Positions are taken from the cell's first point and
        # values from its value there (the area vectors of a closed surface sum to
        # zero), so cells far from the origin keep their precision.
        origin = cells.side_position(c, 0, 0)
        u0 = u[cells.side_point_id(c, 0, 0)]
        six = 0.0                                  # six times the volume
        total = tack.vec3(0.0, 0.0, 0.0)           # 2 A_t times 3 mean(u) on each triangle
        for k in range(cells.num_faces(c)):
            # One walk round the face. With F its points' mean, triangle j's
            # 2 A_j = p_j x p_j+1 + (p_j+1 - p_j) x F, so the face's share is
            # S + W x F + u_F N, with S = sum (p_j x p_j+1)(u_j + u_j+1),
            # W = sum (p_j+1 - p_j)(u_j + u_j+1) and N = sum p_j x p_j+1 (the
            # Newell vector, sum 2 A_j); its share of 6V is F . N. F and u_F are
            # known only at the end, and none of the sums needs them.
            size = cells.side_num_points(c, k)
            first = cells.side_position(c, k, 0) - origin
            u_first = u[cells.side_point_id(c, k, 0)] - u0
            a = first
            ua = u_first
            points = first
            values = u_first
            newell = tack.vec3(0.0, 0.0, 0.0)
            spun = tack.vec3(0.0, 0.0, 0.0)
            along = tack.vec3(0.0, 0.0, 0.0)
            for j in range(1, size + 1):
                b = first
                ub = u_first
                if j < size:
                    b = cells.side_position(c, k, j) - origin
                    ub = u[cells.side_point_id(c, k, j)] - u0
                    points += b
                    values += ub
                n = a.cross(b)
                both = ua + ub
                newell += n
                spun += n * both
                along += (b - a) * both
                a = b
                ua = ub
            apex = points / size
            six += apex.dot(newell)
            total += spun + along.cross(apex) + newell * (values / size)
        # (1/V) sum A_t mean(u)_t = sum (2 A_t)(3 mean(u)_t) / 6V. No volume: NaN.
        out[cells.entity_id(c)] = total / six


@tack.kernel
def _polygon_gradients(cells, u, out):
    for c in cells:
        # In-plane Green-Gauss: (1/A) times the sum over the edges of each edge's
        # outward in-plane normal (its length long) times its mean value, with N
        # the Newell vector (2A along the normal): sum (u_k + u_k+1) ((p_k+1 - p_k)
        # x N) / |N|^2. Exact for a linear field on a planar polygon. Positions
        # and values from the first point, as the polyhedral gradient takes them.
        origin = cells.side_position(c, 0, 0)
        u0 = u[cells.side_point_id(c, 0, 0)]
        newell = tack.vec3(0.0, 0.0, 0.0)
        total = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin
            b = cells.side_position(c, k, 1) - origin
            newell += a.cross(b)
            total += (u[cells.side_point_id(c, k, 0)] + u[cells.side_point_id(c, k, 1)]
                      - 2.0 * u0) * (b - a)
        out[cells.entity_id(c)] = total.cross(newell) / newell.dot(newell)


@tack.kernel
def _zero_gradients(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = [0.0, 0.0, 0.0]


@tack.kernel
def _entry_cells(offsets, cells):
    # Which cell each of cell_points()' entries belongs to.
    for c in range(offsets.shape[0] - 1):
        for k in range(offsets[c], offsets[c + 1]):
            cells[k] = c


def _no_reference_gradients(field, view, values, dimension, out):
    """The gradients of the cells of ``view``, a group without a reference element:
    Green-Gauss over a 3D cell's fanned faces, a polygon's in its plane over its
    edges (both exact for a linear field), zero for polylines and for cell data
    (constant in each cell, as on the shape path). ``values`` are the field's,
    materialized, for point data."""
    if isinstance(field.space, H1) and dimension == 3:
        _polyhedral_gradients(view, values, out)
    elif isinstance(field.space, H1) and dimension == 2:
        _polygon_gradients(view, values, out)
    else:
        _zero_gradients(view, out)


def _polyhedral_scalar_gradients(data, field, at):
    """A scalar field's gradient on a topology without reference elements: each
    cell's by `_no_reference_gradients`, and at the points the mean of the cells'
    around each point."""
    cells = _vectors(data.num_cells, data.dtype)
    values = materialize(field.values) if isinstance(field.space, H1) else None
    for group in data.launch_groups("cells", []):
        if group.count:
            _no_reference_gradients(field, data.domain_view("cells", group), values,
                                    group.shape.DIMENSION, cells)
    if at == "cells":
        return cells
    out = tack.empty((data.num_points,), tack.vec(3, data.dtype))
    out.fill(0)
    point_offsets, point_ids = data.topology.cell_points()
    entries = point_ids.shape[0]
    if entries:
        owners = tack.empty((entries,), data.id_dtype)
        _entry_cells(point_offsets, owners)
        offsets, links = data.topology.point_links()
        _average_links(offsets, links, _take(cells, owners), out)
    return out


@tack.kernel
def _row(values, k, out):
    for i in range(out.shape[0]):
        v = values[i]
        out[i] = tack.vec3(v[3 * k], v[3 * k + 1], v[3 * k + 2])


def _gradient_row(gradient, k):
    """Row ``k`` of a vector field's given gradient: its component ``k``'s."""
    if gradient is None:
        return None
    values = materialize(gradient.values)
    out = tack.empty((size_of(values),), tack.vec(3, arrays.dtype_of(values)))
    if out.shape[0]:
        _row(values, k, out)
    return Field(gradient.space, out, gradient.unit)


@tack.kernel
def _component(values, k, out):
    for i in range(out.shape[0]):
        out[i] = values[i][k]


@tack.kernel
def _place_rows_from(rows, k, first, n, out):
    # As _place_rows, over the n entries from first.
    for i in range(n):
        g = rows[first + i]
        out[first + i][3 * k] = g[0]
        out[first + i][3 * k + 1] = g[1]
        out[first + i][3 * k + 2] = g[2]


@tack.kernel
def _place_rows(rows, k, out):
    for i in range(out.shape[0]):
        g = rows[i]
        out[i][3 * k] = g[0]
        out[i][3 * k + 1] = g[1]
        out[i][3 * k + 2] = g[2]


def _scalar_gradients(data, field, at):
    if isinstance(at, L2):
        # At an L2 space's nodes: each cell's own, from the cell alone.
        out = _vectors(at.size, data.dtype)
        groups = data.launch_groups("cells", [field], keys=(at,) if at.varies else ())
        for group, view in data._cell_node_views(at, groups):
            _node_gradients(view, field.view(group), out)
        return out
    if not getattr(data.topology, "reference_cells", True) and not data.topology.mixed:
        return _polyhedral_scalar_gradients(data, field, at)
    # Polyhedra beside the shapes: theirs as on a polyhedral topology, each
    # shape's from its basis.
    values = (materialize(field.values) if data.topology.mixed
              and isinstance(field.space, H1) else None)
    if at == "cells":
        out = _vectors(data.num_cells, data.dtype)
        if not data.topology.mixed:
            for_each(_gradients, data, "cells", field, out)
            return out
        for group in data.launch_groups("cells", [field]):
            if not group.count:
                continue
            view = data.domain_view("cells", group)
            if group.shape.REFERENCE:
                _gradients(view, field.view(group), out)
            else:
                _no_reference_gradients(field, view, values, group.shape.DIMENSION, out)
        return out
    # At the points: each cell's gradient at each of its corners, averaged over
    # the cells around each point, in layout order (VTK's and Viskores' point
    # gradients). Lines and vertices contribute zero; a polyhedron gives its
    # cell's gradient at each of its points, as on a polyhedral topology.
    start_of, total = corner_layout(data.topology)
    corners = _vectors(total, data.dtype)
    # Each shape's corners say; a polyhedron's (its cell's gradient) never are.
    singular = (tack.zeros if data.topology.mixed else tack.empty)((total,), tack.i32)
    _corner_rows(data, field, values, start_of, corners, singular)
    return _at_points(data, corners, singular, 3)


def _corner_rows(data, field, values, start_of, corners, singular, skip=None):
    """A scalar field's gradient at each corner of each cell, into ``corners`` (in
    ``corner_layout``'s order), and ``singular`` where a corner has none; groups
    ``skip`` says (a function of the group) are left alone."""
    for group in data.launch_groups("cells", [field]):
        if not group.count or (skip is not None and skip(group)):
            continue
        view = data.domain_view("cells", group)
        if group.shape.REFERENCE:
            _corner_gradients(view, field.view(group), start_of[group.root],
                              _by_edges(data, field, group), corners,
                              singular)
        else:
            cells = _vectors(data.num_cells, data.dtype)
            _no_reference_gradients(field, view, values, group.shape.DIMENSION, cells)
            _cell_values_at_corners(view, cells, start_of[group.root],
                                    polyhedral_corners(data.topology)[0], corners)


def _at_points(data, corners, singular, size):
    """The corners' gradients (``size`` components) averaged over the cells around each
    point, the corners ``singular`` says left out."""
    out = tack.empty((data.num_points,), tack.vec(size, data.dtype))
    out.fill(0)
    if arrays.size_of(corners):
        offsets, entries = data.topology.point_links()
        _average_links(offsets, entries, corners, out)
        if singular.max() > 0:              # rare: only then is any point rescaled
            _without_singular(offsets, entries, singular, out, float("nan"))
    return out


def _vector_point_gradients(data, field):
    """A 3-vector point field's gradient at the points, ``d(u_i)/d(x_j)`` at ``3 i + j``:
    straight hexahedra's corners (``_by_edges``) all three components at once
    (``_vector_edge_gradients``, one Jacobian inverse a corner), every other group's
    a component at a time; then one average over each point's cells."""
    start_of, total = corner_layout(data.topology)
    corners = tack.empty((total,), tack.vec(9, data.dtype))
    singular = (tack.zeros if data.topology.mixed else tack.empty)((total,), tack.i32)
    values = materialize(field.values)
    groups = [g for g in data.launch_groups("cells", [field]) if g.count]
    edges = {id(g) for g in groups if g.shape.REFERENCE and _by_edges(data, field, g)}
    if len(edges) < len(groups):
        # The others a component at a time, each component's rows placed in turn;
        # the one-pass groups' entries, written over here, are written after.
        rows = _vectors(total, data.dtype)
        for k in range(3):
            part = tack.empty((arrays.size_of(values),), arrays.dtype_of(values))
            if part.shape[0]:
                _component(values, k, part)
            component = Field(field.space, part)
            _corner_rows(data, component, part, start_of, rows, singular,
                         skip=lambda g: id(g) in edges)
            for group in groups:                 # each group's own range of corners
                if id(group) not in edges:
                    size = (group.root.count * group.shape.NUM_POINTS if group.shape.REFERENCE
                            else polyhedral_corners(data.topology)[1])
                    if size:
                        _place_rows_from(rows, k, start_of[group.root], size, corners)
    for group in groups:
        if id(group) in edges:
            _vector_edge_gradients(data.domain_view("cells", group), field.view(group),
                                   start_of[group.root], corners, singular)
    return _at_points(data, corners, singular, 9)


@on_inputs_device
def gradients(data, field, at="cells"):
    """The gradient of ``field`` -- ``H1``, ``L2`` or ``Constant``, scalar or vector --
    at each cell's parametric center (``at="cells"``) or at the points
    (``at="points"``: each cell's gradient at the point, averaged over the cells
    around it, as VTK's vtkGradientFilter and Viskores' point gradients do).

    A scalar gives a 3-vector per entity; a vector of ``k`` components ``3k``
    values, row by row -- ``d(u_i)/d(x_j)`` at ``3 * i + j``, the layout of
    VTK's and Viskores' gradient tensors.

    ``at`` may instead be an ``L2`` space of the dataset: the gradient at each of
    its nodes in each cell -- a discontinuous field, exact at the nodes, so an
    ``H1`` field of order ``p`` keeps its order's detail in ``L2`` of order ``p``,
    where a gradient at the points averages it away. The field's basis gives its
    derivative in parametric coordinates and the geometry's Jacobian turns that
    into a world gradient, so the two need not share a space. A 2D cell's
    gradient lies in its plane (VTK's and Viskores' in-plane gradient), lines
    and vertices get zero, and a cell of no volume or area NaN (its Jacobian is
    singular). At the points, a pyramid's apex gives its gradient extrapolated from
    just below it, as vtkPyramid does; a cell whose Jacobian is singular at any
    other corner -- where two of its faces meet flat, as in a hexahedron with a
    straight corner -- has no gradient there (it grows without bound toward the
    corner) and is left out of that point's average, which is NaN only if no
    cell around the point is left (VTK leaves its inverse Jacobian unset there;
    Viskores adds nothing but still counts the cell). ``flow_quantities`` derives divergence, vorticity and
    the Q-criterion from a vector field's gradient.

    A field given its gradient (``Field(..., gradient=)``) gives that instead of its
    basis', evaluated at the same points and averaged the same way.

    On a polyhedral topology there is no reference element. A cell's gradient
    of point data is then Green-Gauss over its boundary: the faces walked
    outward and fanned from their points' means (the triangles ``cell_geometry``
    measures the volume by), each triangle's area vector times the mean of its
    values, over the volume. That is exact for a linear field and needs no
    interpolant. Cell data, constant in each cell, has a zero gradient there,
    as on shapes. A polygon's is the same in its plane, over its edges; polylines
    get zero. At the points, the cells' gradients are averaged over the cells
    around each point."""
    field = _with_basis(data, _field(data, field))
    if not getattr(data.topology, "reference_cells", True) and field.gradient is not None:
        raise NotImplementedError("gradients of a field given its gradient on a mesh without "
                                  "reference elements")
    if (not getattr(data.topology, "reference_cells", True)
            and not isinstance(field.space, (H1, Constant))):
        raise NotImplementedError(f"gradients of a field on {field.space!r} on a mesh "
                                  "without reference elements: point or cell data only")
    if isinstance(at, L2):
        if at.topology is not data.topology:
            raise ValueError("gradients at the nodes of a space of another topology")
        if not getattr(data.topology, "reference_cells", True):
            raise NotImplementedError("gradients at an L2 space's nodes take cells with a "
                                      "reference element")
        space = at
    elif at not in ("cells", "points"):
        raise ValueError(f"gradients are at 'cells' or 'points', or an L2 space's nodes, "
                         f"not {at!r}")
    else:
        space = Values(data, at)
    # Per unit length; a difference of degrees Celsius is in kelvin.
    unit = units.derived(units.delta(field.unit), (data.unit, -1))
    width = width_of(field.values)
    if not width:
        return Field(space, _scalar_gradients(data, field, at), unit)
    if (at == "points" and width == 3 and field.gradient is None
            and (getattr(data.topology, "reference_cells", True) or data.topology.mixed)):
        return Field(space, _vector_point_gradients(data, field), unit)
    n = space.size if isinstance(at, L2) else data.num_cells if at == "cells" \
        else data.num_points
    out = tack.empty((n,), tack.vec(3 * width, data.dtype))
    values = materialize(field.values)
    for k in range(width):
        part = tack.empty((size_of(field.values),), arrays.dtype_of(field.values))
        if part.shape[0]:
            _component(values, k, part)
        rows = _scalar_gradients(data, Field(field.space, part, gradient=_gradient_row(field.gradient, k)), at)
        if n:
            _place_rows(rows, k, out)
    return Field(space, out, unit)
