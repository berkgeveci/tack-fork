# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``contour`` of a structured grid by flying edges (issue #316), as VTK's
``vtkFlyingEdges3D`` and Viskores' flying edges: crossings found on the grid's
edges, counted along each row of points, numbered by a scan and written once,
so triangles share their points with no per-corner arrays and no sort.

Its triangles, points and the values carried are the general path's
(``contour._contour_points`` and its merge), up to the order of points and
triangles: every crossing is read through the same views (an image's and a
rectilinear grid's positions by ``get_ijk``, as its cells read them), placed
from the end first by position, and named by its end where it lands on one
(``_end_hit``); every cell takes the same triangles of the hexahedron's case
table; a triangle two of whose points are one point is dropped.

Each point owns the crossings of its edges along +x, +y and +z, and the point
itself where a crossing lands on it; per value, its edges' codes (none, between
the ends, at this end, at the far end; two bits an edge) are found once, then
each row of points counts what it emits and each row of cells its triangles,
in the order (row, value, along x). The triangles' pass walks each row of
cells along x with a running count for each of the four rows of points around
it, as VTK's does, so a crossing's number is found without a table a point.

Each (row, value) has a window of points, ``[lo, hi]``, outside which none of
its points' edges is crossed, and only the windows' flags are kept (at
``offset + i - lo``): a contour's are whole rows; a plane's (``cut``, as
``vtkFlyingEdgesPlaneCutter``) are found from the plane's equation, the
stretch of each row within the largest change of distance along an edge of the
plane, so a slice visits only the cells near it, its values computed where
they are read rather than stored for every point; a curvilinear grid's planes
take whole rows, the values still computed where read.

A sheet (a 2D grid) takes the same points' passes, one layer of cells along
z, and lines instead of triangles: each cell's crossings, going round its
sides through its view, paired as ``contour._level_lines`` pairs them, each
rising crossing with the next falling one, so the values above lie on a
line's right; a line whose two crossings land on one point is dropped.

Hidden and refined cells (``Ghosts.MASKED_CELLS``, an AMR box's) make no
triangles, and crossings only they use are dropped afterwards, as for cells
with a NaN corner: the general path's output, which extracts the other cells
first, without the extraction.
"""

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids, shapes
from tack.dataset.algorithms._common import _crossing, _end_hit, _first_end

# An edge's code, two bits an axis in a point's flag (bits 2a, 2a + 1).
_NONE = tack.constant(0, tack.i32)
_BETWEEN = tack.constant(1, tack.i32)
_AT_START = tack.constant(2, tack.i32)
_AT_END = tack.constant(3, tack.i32)


@tack.func
def _corner_of(cells, i, j, k):
    """A cell holding point ``(i, j, k)`` and the point's corner in it, so the
    point is read through the cells' views, as the general path reads it."""
    ci = min(i, cells.cx - 1)
    cj = min(j, cells.cy - 1)
    ck = min(k, cells.cz - 1)
    a = i - ci
    b = j - cj
    d = k - ck
    return [ci, cj, ck], 4 * d + 2 * b + (a ^ b)


@tack.device_class
class _Solid:
    """A solid grid's points read through its cells' views (``_corner_of``): an
    (i, j, k) cell and its corner, as the general path reads them."""

    def __init__(self, cells, u):
        self.cells = cells
        self.u = u

    @tack.func
    def at(self, i, j, k):
        """``(position, value)`` of point ``(i, j, k)``."""
        c, corner = _corner_of(self.cells, i, j, k)
        return self.cells.corner_position(c, corner), self.u.corner_value(c, corner)

    @tack.func
    def position(self, i, j, k):
        c, corner = _corner_of(self.cells, i, j, k)
        return self.cells.corner_position(c, corner)


@tack.device_class
class _Sheet(_Solid):
    """A sheet's (a 2D grid's) points, through its (i, j) cells' views: a quad's
    corners go (0, 0), (1, 0), (1, 1), (0, 1)."""

    @tack.func
    def at(self, i, j, k):
        c, corner = self._corner(i, j)
        return self.cells.corner_position(c, corner), self.u.corner_value(c, corner)

    @tack.func
    def position(self, i, j, k):
        c, corner = self._corner(i, j)
        return self.cells.corner_position(c, corner)

    @tack.func
    def _corner(self, i, j):
        ci = min(i, self.cells.cx - 1)
        cj = min(j, self.cells.cy - 1)
        a = i - ci
        b = j - cj
        return [ci, cj], 2 * b + (a ^ b)


@tack.func
def _crosses(v0, v1, level):
    # A cell with a NaN corner has no surface, so an edge to a NaN is in none of
    # the triangles.
    return v0 == v0 and v1 == v1 and (v0 >= level) != (v1 >= level)


@tack.func
def _codes_of(values, p, i, j, k, nx, ny, nz, level, v):
    """A point's flag for ``level``: its edges crossed (``_BETWEEN``) or not;
    ``_landings`` finds those that land on an end."""
    v0 = values[p]
    vx = values[p + 1] if i + 1 < nx else v0
    vy = values[p + nx] if j + 1 < ny else v0
    vz = values[p + nx * ny] if k + 1 < nz else v0
    f = 0
    if _crosses(v0, vx, level):
        f = 1
    if _crosses(v0, vy, level):
        f |= 4
    if _crosses(v0, vz, level):
        f |= 16
    if v == 0 and v0 != v0:
        f |= _MISSING
    return f


@tack.kernel
def _window_codes(values, levels, nlevels, nx, ny, nz, windows, flags):
    # Each window's points' flags, one (row, value) a thread.
    for q in range(windows.shape[0]):
        w = windows[q]
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        level = levels[v]
        for i in range(w[0], w[1] + 1):
            flags[w[2] + i - w[0]] = tack.u8(_codes_of(values, i + nx * r, i, j, k, nx, ny, nz,
                                                       level, v))


@tack.kernel
def _edge_codes(values, levels, nlevels, nx, ny, nz, flags):
    # Each point's edges along +x, +y and +z, per value: two bits each, here
    # crossed or not (_BETWEEN); _landings finds those that land on an end. The
    # contour's windows are whole rows, its flags at (row * nlevels + v) * nx + i.
    for k, j, i in tack.ndindex(nz, ny, nx):
        p = i + nx * (j + ny * k)
        v0 = values[p]
        vx = values[p + 1] if i + 1 < nx else v0
        vy = values[p + nx] if j + 1 < ny else v0
        vz = values[p + nx * ny] if k + 1 < nz else v0
        for v in range(nlevels):
            level = levels[v]
            f = 0
            if _crosses(v0, vx, level):
                f = 1
            if _crosses(v0, vy, level):
                f |= 4
            if _crosses(v0, vz, level):
                f |= 16
            if v == 0 and v0 != v0:
                f |= _MISSING
            flags[((j + ny * k) * nlevels + v) * nx + i] = tack.u8(f)


@tack.kernel
def _landings(grid, values, levels, nlevels, nx, ny, nz, windows, flags):
    # Each crossed edge whose crossing lands on an end (_end_hit), recoded
    # _AT_START or _AT_END; positions, which an image computes, are read through
    # the cells' views as the general path reads them, and only here. One
    # (row of points, value) a thread, each writing only its own points' flags.
    for q in range(windows.shape[0]):
        w = windows[q]
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        level = levels[v]
        for i in range(w[0], w[1] + 1):
            p = i + nx * r
            base = w[2] + i - w[0]
            f = tack.i32(flags[base])
            if f != 0:
                for axis in range(3):
                    if ((f >> (2 * axis)) & 3) == _BETWEEN:
                        ii = i + (1 if axis == 0 else 0)
                        jj = j + (1 if axis == 1 else 0)
                        kk = k + (1 if axis == 2 else 0)
                        b = ii + nx * (jj + ny * kk)
                        hit = _end_hit(level, p, b, values[p], values[b],
                                       grid.position(i, j, k), grid.position(ii, jj, kk))
                        if hit == p:
                            f += (_AT_START - _BETWEEN) << (2 * axis)
                        elif hit == b:
                            f += (_AT_END - _BETWEEN) << (2 * axis)
                flags[base] = tack.u8(f)


@tack.func
def _flag(flags, windows, q, i):
    """Point ``i``'s flag in (row, value) ``q``: 0 outside its window."""
    w = windows[q]
    f = 0
    if i >= w[0] and i <= w[1]:
        f = tack.i32(flags[w[2] + i - w[0]])
    return f


@tack.func
def _code(f, axis):
    return (f >> (2 * axis)) & 3


@tack.func
def _is_hit(flags, windows, q, i, j, k, nlevels, ny):
    """Whether a crossing lands on point ``i`` of (row, value) ``q`` (row ``(j,
    k)``): one of its own edges' at its start, or a lower neighbour's at its end."""
    f = _flag(flags, windows, q, i)
    hit = 0
    for axis in range(3):
        if _code(f, axis) == _AT_START:
            hit = 1
    if i > 0 and _code(_flag(flags, windows, q, i - 1), 0) == _AT_END:
        hit = 1
    if j > 0 and _code(_flag(flags, windows, q - nlevels, i), 1) == _AT_END:
        hit = 1
    if k > 0 and _code(_flag(flags, windows, q - ny * nlevels, i), 2) == _AT_END:
        hit = 1
    return hit


#: A point's flag bit saying a crossing lands on it (``_point_counts`` sets it).
_LANDED = tack.constant(64, tack.i32)
#: The first value's flag bit saying the point's value is NaN.
_MISSING = tack.constant(128, tack.i32)
#: Cells left out, as ``ghosts.visible`` leaves them out: hidden or refined
#: (``Ghosts.MASKED_CELLS``).
_MASKED = tack.constant(40, tack.i32)


@tack.func
def _landed(f):
    return 1 if (f & _LANDED) != 0 else 0


@tack.func
def _emitted(f):
    """How many points a point emits (its flag ``f``, with ``_LANDED`` set):
    itself if a crossing lands on it, then each of its edges crossed between its
    ends."""
    count = 1 if (f & _LANDED) != 0 else 0
    for axis in range(3):
        if ((f >> (2 * axis)) & 3) == _BETWEEN:
            count += 1
    return count


@tack.kernel
def _point_counts(flags, windows, nlevels, ny, counts, nans):
    # Each (row of points, value)'s points, and each row's NaNs, which can leave a
    # crossing no cell uses.
    for q in range(windows.shape[0]):
        w = windows[q]
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        total = 0
        missing = 0
        for i in range(w[0], w[1] + 1):
            # Marked here, by the one thread that walks this row for this value;
            # others read only the bits of the point's own edges.
            base = w[2] + i - w[0]
            if _is_hit(flags, windows, q, i, j, k, nlevels, ny) == 1:
                flags[base] = flags[base] | tack.u8(_LANDED)
            f = tack.i32(flags[base])
            total += _emitted(f)
            if v == 0 and (f & _MISSING) != 0:
                missing += 1
        counts[q] = total
        if v == 0:
            nans[r] = missing


@tack.func
def _edge_of(hexa, e, ci, cj, ck):
    """Local edge ``e`` of cell ``(ci, cj, ck)``: its axis and its lower point's
    ``(i, j, k)``."""
    ja = hexa.edge_corner(e, 0)
    jb = hexa.edge_corner(e, 1)
    ab = (ja >> 1) & 1
    bb = (jb >> 1) & 1
    a0 = (ja & 1) ^ ab
    a1 = (jb & 1) ^ bb
    d0 = ja >> 2
    d1 = jb >> 2
    axis = 0 if a0 != a1 else (1 if ab != bb else 2)
    return axis, ci + min(a0, a1), cj + min(ab, bb), ck + min(d0, d1)


@tack.func
def _landing(flags, windows, v, nlevels, hexa, e, ci, cj, ck, nx, ny):
    """The point a crossing of local edge ``e`` lands on (its global id), or -1
    for one between the edge's ends."""
    axis, i, j, k = _edge_of(hexa, e, ci, cj, ck)
    p = i + nx * (j + ny * k)
    code = _code(_flag(flags, windows, (j + ny * k) * nlevels + v, i), axis)
    out = tack.i64(-1)
    if code == _AT_START:
        out = p
    elif code == _AT_END:
        out = p + (1 if axis == 0 else (nx if axis == 1 else nx * ny))
    return out


@tack.func
def _kept(flags, windows, v, nlevels, hexa, case, t, ci, cj, ck, nx, ny):
    """Whether triangle ``t`` of the case keeps three distinct points (two crossings
    landing on one point leave none)."""
    a = _landing(flags, windows, v, nlevels, hexa, hexa.contour_edge(case, 3 * t), ci, cj,
                 ck, nx, ny)
    b = _landing(flags, windows, v, nlevels, hexa, hexa.contour_edge(case, 3 * t + 1), ci, cj,
                 ck, nx, ny)
    c = _landing(flags, windows, v, nlevels, hexa, hexa.contour_edge(case, 3 * t + 2), ci, cj,
                 ck, nx, ny)
    return not ((a >= 0 and (a == b or a == c)) or (b >= 0 and b == c))


@tack.func
def _cell_case(cells, u, c, level):
    # As contour._cell_case: the corners at or above the level, as bits; none
    # where a corner is NaN.
    case = 0
    nan = 0
    for j in range(cells.NUM_POINTS):
        x = u.corner_value(c, j)
        if x >= level:
            case |= 1 << j
        if x != x:
            nan = 1
    return 0 if nan == 1 else case


@tack.func
def _cell_window(windows, nlevels, v, cj, ck, ny, cx):
    """The cells ``[lo, hi]`` of row of cells ``(cj, ck)`` that the windows of its four
    rows of points reach for value ``v`` (a point ``i`` is a corner of cells
    ``i - 1`` and ``i``); ``hi < lo`` if none."""
    r0 = cj + ny * ck
    lo = cx
    hi = tack.i64(-1)
    for s in range(4):
        w = windows[(r0 + (s & 1) + ny * (s >> 1)) * nlevels + v]
        if w[1] >= w[0]:
            lo = min(lo, max(w[0] - 1, 0))
            hi = max(hi, min(w[1], cx - 1))
    return lo, hi


@tack.func
def _crossed(point_counts, nlevels, v, cj, ck, ny):
    """Whether any of the four rows of points around row of cells ``(cj, ck)``
    emits a point for value ``v``: if none does, none of its cells' edges is
    crossed, and it has no triangles."""
    r0 = cj + ny * ck
    return (point_counts[r0 * nlevels + v] + point_counts[(r0 + 1) * nlevels + v]
            + point_counts[(r0 + ny) * nlevels + v]
            + point_counts[(r0 + ny + 1) * nlevels + v]) > 0


@tack.kernel
def _triangle_counts(cells, u, hexa, flags, windows, point_counts, levels, nlevels, nx, ny,
                     nz, marks, hidden, counts):
    # Each (row of cells, value)'s triangles.
    cx = nx - 1
    cy = ny - 1
    for q in range(cy * (nz - 1) * nlevels):
        r = q // nlevels
        v = q - r * nlevels
        cj = r % cy
        ck = r // cy
        level = levels[v]
        total = 0
        lo, hi = _cell_window(windows, nlevels, v, cj, ck, ny, cx)
        if not _crossed(point_counts, nlevels, v, cj, ck, ny):
            hi = lo - 1
        for ci in range(lo, hi + 1):
            case = _cell_case(cells, u, [ci, cj, ck], level)
            if hidden == 1 and (tack.i32(marks[ci + cx * (cj + cy * ck)]) & _MASKED) != 0:
                case = 0                  # a hidden or refined cell has no surface
            for t in range(hexa.contour_count(case)):
                if _kept(flags, windows, v, nlevels, hexa, case, t, ci, cj, ck, nx, ny):
                    total += 1
        counts[q] = total


@tack.kernel
def _points(grid, flags, windows, levels, nlevels, nx, ny, starts, total, positions,
            edges, weights):
    # Each (row of points, value)'s points, in its order: a point a crossing lands
    # on, then its edges' crossings between their ends along x, y and z.
    for q in range(windows.shape[0]):
        w = windows[q]
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        level = levels[v]
        at = starts[q]
        end = starts[q + 1] if q + 1 < windows.shape[0] else total
        for i in range(w[0], w[1] + 1 if end > at else w[0]):
            p = i + nx * r
            f = tack.i32(flags[w[2] + i - w[0]])
            x0, v0 = grid.at(i, j, k)
            if _landed(f) == 1:
                positions[at] = x0
                edges[at] = [p, p]
                weights[at] = 0.0
                at += 1
            for axis in range(3):
                if _code(f, axis) == _BETWEEN:
                    ii = i + (1 if axis == 0 else 0)
                    jj = j + (1 if axis == 1 else 0)
                    kk = k + (1 if axis == 2 else 0)
                    b = ii + nx * (jj + ny * kk)
                    x1, v1 = grid.at(ii, jj, kk)
                    a = p
                    xa = x0
                    xb = x1
                    va = v0
                    vb = v1
                    # From the end first by position, as the general path.
                    if not _first_end(x0, x1, p, b):
                        a = b
                        b = p
                        xa = x1
                        xb = x0
                        va = v1
                        vb = v0
                    w = _crossing(level, va, vb)
                    positions[at] = xa + w * (xb - xa)
                    edges[at] = [a, b]
                    weights[at] = w
                    at += 1


@tack.func
def _pick(s, a, b, c, d):
    out = a
    if s == 1:
        out = b
    elif s == 2:
        out = c
    elif s == 3:
        out = d
    return out


@tack.func
def _slot(f, axis, code, hit, start, far_start):
    """The number of a crossing of the edge along ``axis`` of a point whose first
    number is ``start`` (``hit`` 1 if a crossing lands on it, which comes first):
    its own between its ends, the point itself, or the far point (whose first
    number is ``far_start``, its own landing)."""
    out = start
    if code == _BETWEEN:
        out = start + hit
        for a in range(3):
            if a < axis and _code(f, a) == _BETWEEN:
                out += 1
    elif code == _AT_END:
        out = far_start
    return out


@tack.kernel
def _triangles(cells, u, hexa, flags, windows, levels, nlevels, nx, ny, nz, point_starts,
               triangle_starts, triangles, marks, hidden, rows, sources, which):
    # Each (row of cells, value)'s triangles, walking along x with the next
    # number of each of the four rows of points around it.
    cx = nx - 1
    cy = ny - 1
    for q in range(cy * (nz - 1) * nlevels):
        r = q // nlevels
        v = q - r * nlevels
        cj = r % cy
        ck = r // cy
        level = levels[v]
        at = triangle_starts[q]
        end = triangle_starts[q + 1] if q + 1 < cy * (nz - 1) * nlevels else triangles
        # Rows (cj, ck), (cj + 1, ck), (cj, ck + 1), (cj + 1, ck + 1): the first
        # number of each one's point at ci.
        r0 = cj + ny * ck
        r1 = r0 + 1
        r2 = r0 + ny
        r3 = r2 + 1
        q0 = r0 * nlevels + v
        q1 = r1 * nlevels + v
        q2 = r2 * nlevels + v
        q3 = r3 * nlevels + v
        # Points before the cells' window emit nothing, so each row's count
        # starts at its first number.
        n0 = point_starts[q0]
        n1 = point_starts[q1]
        n2 = point_starts[q2]
        n3 = point_starts[q3]
        lo, hi = _cell_window(windows, nlevels, v, cj, ck, ny, cx)
        if end <= at:
            hi = lo - 1
        for ci in range(lo, hi + 1):
            # The next point's first number in each row.
            m0 = n0 + _emitted(_flag(flags, windows, q0, ci))
            m1 = n1 + _emitted(_flag(flags, windows, q1, ci))
            m2 = n2 + _emitted(_flag(flags, windows, q2, ci))
            m3 = n3 + _emitted(_flag(flags, windows, q3, ci))
            case = _cell_case(cells, u, [ci, cj, ck], level)
            cell = ci + cx * (cj + cy * ck)
            if hidden == 1 and (tack.i32(marks[cell]) & _MASKED) != 0:
                case = 0
            for t in range(hexa.contour_count(case)):
                if _kept(flags, windows, v, nlevels, hexa, case, t, ci, cj, ck, nx, ny):
                    for corner in range(3):
                        e = hexa.contour_edge(case, 3 * t + corner)
                        axis, i, j, k = _edge_of(hexa, e, ci, cj, ck)
                        s = (j - cj) + 2 * (k - ck)
                        start = _pick(s, n0, n1, n2, n3) if i == ci else _pick(s, m0, m1, m2,
                                                                                 m3)
                        f = _flag(flags, windows, (j + ny * k) * nlevels + v, i)
                        code = _code(f, axis)
                        # The far point: along x the next point of the row; along y
                        # or z the row above at the same i.
                        far = _pick(s + (2 if axis == 2 else 1), n0, n1, n2, n3)
                        if axis == 0:
                            far = _pick(s, m0, m1, m2, m3)
                        elif i != ci:
                            far = _pick(s + (2 if axis == 2 else 1), m0, m1, m2, m3)
                        rows[3 * at + corner] = _slot(f, axis, code, _landed(f), start, far)
                    sources[at] = cell
                    which[at] = v
                    at += 1
            n0 = m0
            n1 = m1
            n2 = m2
            n3 = m3


# ── A sheet's lines ─────────────────────────────────────────────────

@tack.func
def _side_number(flags, windows, v, nlevels, a, b, nx, cj, ci, n0, n1, m0, m1):
    """``(number, landing)``: the output number of side ``(a, b)``'s crossing (a
    quad's side, its points' global ids), and the point it lands on (-1 if
    between the ends). ``n0``/``n1`` are rows ``cj``/``cj + 1``'s first numbers
    at point ``ci``, ``m0``/``m1`` at ``ci + 1``."""
    owner = min(a, b)
    axis = 0 if max(a, b) - owner == 1 else 1
    oi = owner % nx
    oj = owner // nx
    s = oj - cj
    f = _flag(flags, windows, oj * nlevels + v, oi)
    code = _code(f, axis)
    start = (n0 if s == 0 else n1) if oi == ci else (m0 if s == 0 else m1)
    number = start
    landing = tack.i64(-1)
    if code == _BETWEEN:
        number = start + _landed(f) + (1 if axis == 1 and _code(f, 0) == _BETWEEN else 0)
    elif code == _AT_START:
        landing = owner
    elif code == _AT_END:
        landing = owner + (1 if axis == 0 else nx)
        # The far point: along x the next point of the row, along y the row above.
        number = (m0 if s == 0 else m1) if axis == 0 else (n1 if oi == ci else m1)
    return number, landing


@tack.func
def _sheet_lines(cells, c, values, level, v, nlevels, flags, windows, nx, cj, ci, n0, n1, m0,
                 m1, rows, sources, which, at, write):
    """Value ``v``'s lines of cell ``c``, as the general path pairs them
    (``contour._level_lines``): going round the cell from its first rising
    crossing, each rising one with the next falling one, so the values above lie
    on the line's right; a line whose two crossings land on one point is
    dropped. Written from ``at`` when ``write``; returns the next ``at``."""
    n = cells.num_faces(c)
    first = -1
    for k in range(n):
        if first < 0:
            if (not values[cells.side_point_id(c, k, 0)] >= level
                    and values[cells.side_point_id(c, k, 1)] >= level):
                first = k
    if first >= 0:
        rising = tack.i64(0)
        rising_landing = tack.i64(-1)
        for i in range(n):
            k = (first + i) % n
            a = cells.side_point_id(c, k, 0)
            b = cells.side_point_id(c, k, 1)
            above_a = values[a] >= level
            above_b = values[b] >= level
            if above_a != above_b:
                number, landing = _side_number(flags, windows, v, nlevels, a, b, nx, cj, ci,
                                               n0, n1, m0, m1)
                if not above_a:
                    rising = number
                    rising_landing = landing
                elif not (landing >= 0 and landing == rising_landing):
                    if write == 1:
                        rows[2 * at] = rising
                        rows[2 * at + 1] = number
                        sources[at] = cells.entity_id(c)
                        which[at] = v
                    at += 1
    return at


@tack.func
def _cell_nan(cells, c, values):
    # As contour._side_nan: a cell with a NaN corner has no lines.
    nan = 0
    for k in range(cells.num_faces(c)):
        x = values[cells.side_point_id(c, k, 0)]
        if x != x:
            nan = 1
    return nan == 1


@tack.kernel
def _sheet_line_runs(cells, values, flags, windows, points, levels, nlevels, nx, ny, marks,
                     hidden, line_starts, total, write, counts, rows, sources, which):
    # Each (row of cells, value)'s lines: counted (write 0; ``points`` the rows'
    # point counts), or written from their start (write 1; ``points`` the rows'
    # first numbers), walking along x with the first number of each of the two
    # rows of points around it.
    cx = nx - 1
    cy = ny - 1
    for q in range(cy * nlevels):
        cj = q // nlevels
        v = q - cj * nlevels
        level = levels[v]
        q0 = cj * nlevels + v
        q1 = q0 + nlevels
        at = tack.i64(0)
        n0 = tack.i64(0)
        n1 = tack.i64(0)
        lo, hi = _sheet_window(windows, nlevels, v, cj, cx)
        if write == 1:
            at = line_starts[q]
            n0 = points[q0]
            n1 = points[q1]
            end = line_starts[q + 1] if q + 1 < cy * nlevels else total
            if end <= at:
                hi = lo - 1
        elif points[q0] + points[q1] == 0:
            hi = lo - 1                # no crossing in its rows of points: no lines
        f00 = _flag(flags, windows, q0, lo)
        for ci in range(lo, hi + 1):
            f01 = _flag(flags, windows, q1, ci)
            f10 = _flag(flags, windows, q0, ci + 1)
            m0 = n0 + _emitted(f00)
            m1 = n1 + _emitted(f01)
            # A cell none of whose four sides is crossed has no lines: its sides are
            # its corner (ci, cj)'s two edges, (ci, cj + 1)'s along x and
            # (ci + 1, cj)'s along y.
            crossed = (_code(f00, 0) | _code(f00, 1) | _code(f01, 0) | _code(f10, 1)) != 0
            c = [ci, cj]
            e = ci + cx * cj
            if crossed and not _cell_nan(cells, c, values) and not (
                    hidden == 1 and (tack.i32(marks[e]) & _MASKED) != 0):
                at = _sheet_lines(cells, c, values, level, v, nlevels, flags, windows, nx, cj,
                                  ci, n0, n1, m0, m1, rows, sources, which, at, write)
            n0 = m0
            n1 = m1
            f00 = f10
        if write == 0:
            counts[q] = at


@tack.func
def _sheet_window(windows, nlevels, v, cj, cx):
    """The cells ``[lo, hi]`` of row of cells ``cj`` its two rows of points' windows
    reach for value ``v``; ``hi < lo`` if none."""
    lo = cx
    hi = tack.i64(-1)
    for s in range(2):
        w = windows[(cj + s) * nlevels + v]
        if w[1] >= w[0]:
            lo = min(lo, max(w[0] - 1, 0))
            hi = max(hi, min(w[1], cx - 1))
    return lo, hi


@tack.kernel
def _mark_used(rows, used):
    for t in range(rows.shape[0]):
        used[rows[t]] = 1


@tack.kernel
def _renumbered(rows, new_ids):
    for t in range(rows.shape[0]):
        rows[t] = new_ids[rows[t]]


@tack.kernel
def _taken(kept, positions, edges, weights, out_positions, out_edges, out_weights):
    for n in range(kept.shape[0]):
        out_positions[n] = positions[kept[n]]
        out_edges[n] = edges[kept[n]]
        out_weights[n] = weights[kept[n]]


@tack.kernel
def _whole_rows(nx, windows):
    for q in range(windows.shape[0]):
        windows[q] = [0, nx - 1, q * nx]


def serves(data, field):
    """Whether ``data``'s contour of ``field`` can go by flying edges: a structured
    grid of solid cells (triangles) or of a sheet's (lines) with an H1 geometry
    and a point field, both of order 1."""
    from tack.dataset.spaces import H1

    return ((_solid_grid(data) or _sheet(data)) and isinstance(field.space, H1)
            and field.space.order == 1)


def _sheet(data):
    from tack.dataset.spaces import H1
    from tack.dataset.topology import StructuredTopology

    dims = getattr(data.topology, "point_dims", ())
    return (isinstance(data.topology, StructuredTopology) and len(dims) == 2
            and min(dims) >= 2 and isinstance(data.geometry.space, H1)
            and data.geometry.space.order == 1)


def _masks(data):
    """``(marks, 1)``: the cells' ghost marks where some are hidden or refined;
    else ``(a stand-in, 0)``."""
    from tack.dataset.ghosts import Ghosts, _has

    ghosts = data.ghosts
    if ghosts is not None and ghosts.cells is not None and _has(ghosts.cells,
                                                                Ghosts.MASKED_CELLS):
        return ghosts.cells, 1
    return tack.zeros((1,), tack.u8), 0


def _solid_grid(data):
    from tack.dataset.spaces import H1
    from tack.dataset.topology import StructuredTopology

    dims = getattr(data.topology, "point_dims", ())
    return (isinstance(data.topology, StructuredTopology) and len(dims) == 3
            and min(dims) >= 2 and isinstance(data.geometry.space, H1)
            and data.geometry.space.order == 1)


def surface(data, field, levels, nlevels):
    """``contour``'s surface of ``field`` (``_surface``), every row's window whole: a
    solid grid's triangles, or a sheet's lines."""
    nx, ny, nz = (int(d) for d in (*data.topology.point_dims, 1)[:3])
    windows = tack.empty((ny * nz * nlevels,), tack.vec(3, tack.i64))
    _whole_rows(nx, windows)
    group = next(g for g in data.launch_groups("cells", [field]) if g.count)
    return _surface(data, field.values, field.view(group), levels, nlevels,
                    arrays.dtype_of(field.values), windows, nlevels * nx * ny * nz,
                    contour=True)


# ── A plane through an image or a rectilinear grid ──────────────────

@tack.device_class
class _PlaneValues:
    """A plane's signed distance at a grid's points, computed where it is read:
    ``function.value`` of each point's position, as ``implicit_values`` stores it,
    so the same bits. Read as an array (``values[p]``) and as a field's view
    (``corner_value``)."""

    def __init__(self, positions, function, nx, ny):
        self.positions = positions
        self.function = function
        self.nx = nx
        self.ny = ny

    @tack.func
    def value_at(self, p):
        return self.function.value(self.positions[p])

    @tack.func
    def __getitem__(self, p):
        return self.value_at(p)

    @tack.func
    def corner_value(self, c, j):
        # By the corner's (i, j, k), as get reads it after dividing its index.
        a = (j ^ (j >> 1)) & 1
        b = (j >> 1) & 1
        d = j >> 2
        return self.value_at((c[0] + a) + ((c[1] + b) + (c[2] + d) * self.ny) * self.nx)


@tack.device_class
class _PlaneValuesByIJK(_PlaneValues):
    """``_PlaneValues`` of an image or a rectilinear grid, whose corners are read by
    their (i, j, k), as ``get`` reads them after dividing an index: the same bits,
    without the divisions."""

    @tack.func
    def corner_value(self, c, j):
        a = (j ^ (j >> 1)) & 1
        b = (j >> 1) & 1
        d = j >> 2
        return self.function.value(self.positions.get_ijk([c[0] + a, c[1] + b, c[2] + d]))


def cuts(data, function):
    """Whether ``slice`` of ``data`` by ``function`` can go by the plane cutter: a
    ``Plane`` through a structured grid of solid cells. On an image or a
    rectilinear grid only the rows' stretches near the plane are visited
    (``_plane_windows``); on a curvilinear grid every row, the distance computed
    where read all the same. Hidden and refined cells are left out."""
    from tack.dataset.implicit import Plane

    return isinstance(function, Plane) and _solid_grid(data)


def _by_arithmetic(data):
    from tack.arrays import CartesianProduct, UniformCoordinates

    return type(data.geometry.values) in (UniformCoordinates, CartesianProduct)


@tack.func
def _last_at_most(axis, x, n):
    """The last of ``n`` increasing coordinates ``axis`` at most ``x``; -1 if none."""
    low = tack.i64(-1)
    high = tack.i64(n)
    while high - low > 1:
        mid = (low + high) // 2
        if axis[mid] <= x:
            low = mid
        else:
            high = mid
    return low


@tack.kernel
def _image_windows(start, sx, sy, sz, reach, levels, nlevels, nx, ny, windows, lengths):
    # An image's rows: the distance at point i of row (j, k) is about
    # start + sx i + sy j + sz k, so its points within ``reach`` of a value are an
    # interval of i, widened by a point each way.
    for q in range(lengths.shape[0]):
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        target = levels[v] - (start + sy * j + sz * k)
        lo = tack.i64(0)
        hi = tack.i64(-1)
        if abs(sx) > 1e-30:
            a = (target - reach) / sx
            b = (target + reach) / sx
            t0 = min(a, b)
            t1 = max(a, b)
            if t1 >= -1.0 and t0 <= nx:
                lo = max(tack.i64(floor(max(t0, -2.0))) - 1, 0)
                hi = min(tack.i64(ceil(min(t1, nx + 1.0))) + 1, nx - 1)
        elif abs(target) <= reach:
            hi = nx - 1
        windows[q] = [lo, hi, 0]
        lengths[q] = max(hi - lo + 1, 0)


@tack.kernel
def _grid_windows(xs, ys, zs, ox, oy, oz, nx_, ny_, nz_, reach, levels, nlevels, nx, ny,
                  windows, lengths):
    # A rectilinear grid's rows: the distance at point i of row (j, k) is
    # nx_ (xs[i] - ox) + nx_ (ys[j] - oy) + nz_ (zs[k] - oz), monotone in xs[i], so
    # its points within ``reach`` of a value are an interval, found by two
    # binary searches and widened by a point each way.
    for q in range(lengths.shape[0]):
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        target = levels[v] - (ny_ * (ys[j] - oy) + nz_ * (zs[k] - oz))
        lo = tack.i64(0)
        hi = tack.i64(-1)
        if abs(nx_) > 1e-30:
            a = ox + (target - reach) / nx_
            b = ox + (target + reach) / nx_
            last = _last_at_most(xs, max(a, b), nx)
            if last >= 0:
                lo = max(_last_at_most(xs, min(a, b), nx), 0)
                hi = min(last + 1, nx - 1)
        elif abs(target) <= reach:
            hi = nx - 1
        windows[q] = [lo, hi, 0]
        lengths[q] = max(hi - lo + 1, 0)


@tack.kernel
def _window_offsets(offsets, windows):
    for q in range(windows.shape[0]):
        w = windows[q]
        windows[q] = [w[0], w[1], offsets[q]]


def _plane_windows(data, plane, levels, nlevels):
    """Each (row, value)'s window, ``(lo, hi, offset)``, of the points within the
    reach of the plane at that value, and the flags' total. The reach is the
    largest change of distance along an edge of the grid, and rounding, so every
    crossed edge's points are inside."""
    import numpy as np

    from tack.arrays import UniformCoordinates

    values = data.geometry.values
    nx, ny, nz = (int(d) for d in data.topology.point_dims)
    normal = np.array([plane.nx, plane.ny, plane.nz])
    origin = np.array([plane.ox, plane.oy, plane.oz])
    eps = float(np.finfo(data.dtype.numpy_dtype).eps)
    windows = tack.empty((ny * nz * nlevels,), tack.vec(3, tack.i64))
    lengths = tack.empty((ny * nz * nlevels,), tack.i64)
    if type(values) is UniformCoordinates:
        edges = values._matrix @ np.diag(values._spacing)
        steps = normal @ edges                     # distance per step along i, j, k
        corners = np.array([[a * (nx - 1), b * (ny - 1), c * (nz - 1)] for a in (0, 1)
                            for b in (0, 1) for c in (0, 1)], float) @ edges.T
        size = float(np.abs(np.asarray(values._origin)).max() + np.abs(corners).max()
                     + np.abs(origin).max())
        reach = float(np.abs(steps).max()) * (1.0 + 1e-6) + 64.0 * eps * size
        start = float(normal @ (np.asarray(values._origin) - origin))
        _image_windows(start, *map(float, steps), reach, levels, nlevels, nx, ny, windows,
                       lengths)
    else:
        xs, ys, zs = (arrays.to_host(a).astype(float).reshape(-1)
                      for a in (values.xs, values.ys, values.zs))
        steps = [abs(normal[0]) * np.diff(xs).max(),
                 abs(normal[1]) * (np.diff(ys).max() if ny > 1 else 0.0),
                 abs(normal[2]) * (np.diff(zs).max() if nz > 1 else 0.0)]
        size = float(max(np.abs(a).max() for a in (xs, ys, zs)) + np.abs(origin).max())
        reach = float(max(steps)) * (1.0 + 1e-6) + 64.0 * eps * size
        _grid_windows(values.xs, values.ys, values.zs, *map(float, origin), *map(float, normal),
                      reach, levels, nlevels, nx, ny, windows, lengths)
    total = exclusive_scan(lengths, lengths, lengths.shape[0])
    _window_offsets(lengths, windows)
    return windows, int(total)


def cut(data, function, levels, nlevels):
    """``slice``'s surface of a ``Plane`` (``cuts``): ``contour`` of its distance by
    flying edges over each row's window near the plane, the distance computed
    where read (``_PlaneValues``)."""
    from tack.dataset.implicit import relative_to

    nx, ny, nz = (int(d) for d in data.topology.point_dims)
    plane = relative_to(function, data.origin)
    kind = _PlaneValuesByIJK if _by_arithmetic(data) else _PlaneValues
    values = kind(data.geometry.values, plane, nx, ny)
    if not _by_arithmetic(data):
        # A curvilinear grid's points may lie anywhere: every row, its values
        # computed where read, nothing stored for every point but its edges' byte.
        windows = tack.empty((ny * nz * nlevels,), tack.vec(3, tack.i64))
        _whole_rows(nx, windows)
        return _surface(data, values, values, levels, nlevels, data.dtype, windows,
                        nlevels * nx * ny * nz, contour=True)
    windows, total = _plane_windows(data, plane, levels, nlevels)
    return _surface(data, values, values, levels, nlevels, data.dtype, windows, total,
                    contour=False)


def _surface(data, values, u, levels, nlevels, weight_dtype, windows, nflags, contour):
    """``(points, edges, weights, rows, sources, which, count, triangles, odt)``: the
    merged points with the edges and weights they interpolate along, the
    triangles' rows of points, their cells and their values' indices (see the
    module's docstring), from ``values`` (read ``values[p]``) and ``u`` (read as a
    field's view) over ``windows``."""
    from tack.dataset.algorithms._common import _compact

    nx, ny, nz = (int(d) for d in (*data.topology.point_dims, 1)[:3])
    npoints, nrows = nx * ny * nz, ny * nz
    sheet = len(data.topology.point_dims) == 2
    ncrows = (ny - 1) * (1 if sheet else nz - 1)
    group = next(g for g in data.launch_groups("cells", []) if g.count)
    cells = data.domain_view("cells", group)
    grid = (_Sheet if sheet else _Solid)(cells, u)
    hexa = shapes.Hexahedron()
    idt = data.id_dtype
    odt = ids.for_output(data, max(3 * nlevels * (npoints + data.num_cells * 5), 1))
    flags = tack.empty((nflags,), tack.u8)
    if contour:
        _edge_codes(values, levels, nlevels, nx, ny, nz, flags)
    elif nflags:
        _window_codes(values, levels, nlevels, nx, ny, nz, windows, flags)
    _landings(grid, values, levels, nlevels, nx, ny, nz, windows, flags)
    point_counts = tack.empty((nrows * nlevels,), odt)
    nans = tack.empty((nrows,), tack.i64)
    _point_counts(flags, windows, nlevels, ny, point_counts, nans)
    # Hidden and refined cells make no triangles, as visible() leaves them out.
    marks, hidden = _masks(data)
    triangle_counts = tack.empty((ncrows * nlevels,), odt)
    width = 2 if sheet else 3                  # a line's points, or a triangle's
    stand_in = tack.empty((1,), odt)
    if sheet:
        _sheet_line_runs(cells, values, flags, windows, point_counts, levels, nlevels, nx, ny,
                         marks, hidden, triangle_counts, 0, 0, triangle_counts, stand_in,
                         tack.empty((1,), idt), tack.empty((1,), tack.i32))
    else:
        _triangle_counts(cells, u, hexa, flags, windows, point_counts, levels, nlevels, nx,
                         ny, nz, marks, hidden, triangle_counts)
    count = exclusive_scan(point_counts, point_counts, nrows * nlevels)
    triangles = exclusive_scan(triangle_counts, triangle_counts, ncrows * nlevels)
    positions = tack.empty((count,), tack.vec(3, data.dtype))
    edges = tack.empty((count,), tack.vec(2, idt))
    weights = tack.empty((count,), weight_dtype)
    if count:
        _points(grid, flags, windows, levels, nlevels, nx, ny, point_counts, count,
                positions, edges, weights)
    rows = tack.empty((width * triangles,), odt)
    sources = tack.empty((triangles,), idt)
    which = tack.empty((triangles,), tack.i32)
    if triangles and sheet:
        _sheet_line_runs(cells, values, flags, windows, point_counts, levels, nlevels, nx, ny,
                         marks, hidden, triangle_counts, triangles, 1, stand_in, rows, sources,
                         which)
    elif triangles:
        _triangles(cells, u, hexa, flags, windows, levels, nlevels, nx, ny, nz, point_counts,
                   triangle_counts, triangles, marks, hidden, rows, sources, which)
    del flags, point_counts, triangle_counts
    if count and (hidden or int(nans.sum())):
        # A crossing whose cells all have a NaN corner, or are hidden, is in no
        # triangle: kept out, as the general path never makes it.
        used = tack.zeros((count,), tack.i32)
        if triangles:
            _mark_used(rows, used)
        new_ids, kept, kept_count = _compact(used, odt)
        if kept_count < count:
            if triangles:
                _renumbered(rows, new_ids)
            out = (tack.empty((kept_count,), positions.dtype),
                   tack.empty((kept_count,), edges.dtype), tack.empty((kept_count,), weights.dtype))
            if kept_count:
                _taken(kept, positions, edges, weights, *out)
            positions, edges, weights = out
            count = kept_count
    return positions, edges, weights, rows, sources, which, count, triangles, odt


__all__ = []
