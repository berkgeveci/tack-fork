"""``contour`` of a whole AMR hierarchy by flying edges, every box in each launch, one
surface out.

``hierarchy_surface(amr, field, level)``: PR #401's flying edges (``_flying_edges``,
the contour path: one value, whole rows, solid image boxes) run over the hierarchy's
rows -- a row of points or of cells is ``(box, j, k)``, numbered across every box --
instead of one grid's. Each pass is one launch for all boxes; one scan numbers every
box's points and triangles, so the output is one surface: positions, triangles (rows
of point ids) and each triangle's box. Within a box its points and triangles are the
per-box path's (crossings read, placed and named alike: positions ``origin + index *
spacing`` as an image's ``get_ijk``, ``_end_hit``, ``_first_end``, ``_crossing``, the
hexahedron's case table, degenerate triangles dropped); across boxes nothing is merged,
as the box-by-box contour does not merge.

Hidden and refined cells (``Ghosts.MASKED_CELLS``) make no triangles, and crossings only
they use are dropped afterwards, as in #401. The boxes' point values and ghost marks are
read as one array each (views of one buffer, end to end); the boxes must be images.
"""
import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import shapes
from tack.dataset.algorithms._common import _compact, _crossing, _end_hit, _first_end
from tack.dataset.algorithms._flying_edges import (
    _AT_END,
    _AT_START,
    _BETWEEN,
    _LANDED,
    _MASKED,
    _MISSING,
    _code,
    _crosses,
    _edge_of,
    _emitted,
    _landed,
    _mark_used,
    _pick,
    _renumbered,
    _slot,
)


@tack.func
def _position(origins, spacings, b, i, j, k):
    # An image's get_ijk: origin + index * spacing, axis by axis.
    o = origins[b]
    s = spacings[b]
    return [o[0] + i * s[0], o[1] + j * s[1], o[2] + k * s[2]]


@tack.func
def _row_of(row_box, row_start, dims, rr):
    """Point row ``rr``'s box, its (j, k), its first point's local index and the
    box's (nx, ny, nz)."""
    b = row_box[rr]
    d = dims[b]
    r = rr - row_start[b]
    return b, r % d[1], r // d[1], d[0] * r, d[0], d[1], d[2]


@tack.kernel
def _edge_codes(field, point_start, row_box, row_start, dims, levels, flags):
    # Each point's +x, +y, +z edges crossed (_BETWEEN) or not, a row a thread.
    level = levels[0]
    for rr in range(row_box.shape[0]):
        b, j, k, first, nx, ny, nz = _row_of(row_box, row_start, dims, rr)
        base = point_start[b]
        for i in range(nx):
            local = first + i
            p = base + local
            v0 = field.at(b, local)
            vx = field.at(b, local + 1) if i + 1 < nx else v0
            vy = field.at(b, local + nx) if j + 1 < ny else v0
            vz = field.at(b, local + nx * ny) if k + 1 < nz else v0
            f = 0
            if _crosses(v0, vx, level):
                f = 1
            if _crosses(v0, vy, level):
                f |= 4
            if _crosses(v0, vz, level):
                f |= 16
            if v0 != v0:
                f |= _MISSING
            flags[p] = tack.u8(f)


@tack.kernel
def _landings(field, point_start, row_box, row_start, dims, origins, spacings, levels,
              flags):
    # Crossed edges whose crossing lands on an end, recoded _AT_START / _AT_END; the
    # ids given _end_hit are the box's own, as the per-box path gives them.
    level = levels[0]
    for rr in range(row_box.shape[0]):
        b, j, k, first, nx, ny, nz = _row_of(row_box, row_start, dims, rr)
        base = point_start[b]
        for i in range(nx):
            p = first + i
            f = tack.i32(flags[base + p])
            if f != 0:
                for axis in range(3):
                    if ((f >> (2 * axis)) & 3) == _BETWEEN:
                        ii = i + (1 if axis == 0 else 0)
                        jj = j + (1 if axis == 1 else 0)
                        kk = k + (1 if axis == 2 else 0)
                        q = ii + nx * (jj + ny * kk)
                        hit = _end_hit(level, p, q, field.at(b, p), field.at(b, q),
                                       _position(origins, spacings, b, i, j, k),
                                       _position(origins, spacings, b, ii, jj, kk))
                        if hit == p:
                            f += (_AT_START - _BETWEEN) << (2 * axis)
                        elif hit == q:
                            f += (_AT_END - _BETWEEN) << (2 * axis)
                flags[base + p] = tack.u8(f)


@tack.func
def _flag_at(flags, base, nx, ny, i, j, k):
    return tack.i32(flags[base + i + nx * (j + ny * k)])


@tack.func
def _is_hit(flags, base, nx, ny, i, j, k):
    f = _flag_at(flags, base, nx, ny, i, j, k)
    hit = 0
    for axis in range(3):
        if _code(f, axis) == _AT_START:
            hit = 1
    if i > 0 and _code(_flag_at(flags, base, nx, ny, i - 1, j, k), 0) == _AT_END:
        hit = 1
    if j > 0 and _code(_flag_at(flags, base, nx, ny, i, j - 1, k), 1) == _AT_END:
        hit = 1
    if k > 0 and _code(_flag_at(flags, base, nx, ny, i, j, k - 1), 2) == _AT_END:
        hit = 1
    return hit


@tack.kernel
def _point_counts(flags, point_start, row_box, row_start, dims, counts, nans):
    for rr in range(row_box.shape[0]):
        b, j, k, first, nx, ny, nz = _row_of(row_box, row_start, dims, rr)
        base = point_start[b]
        total = 0
        missing = 0
        for i in range(nx):
            p = base + first + i
            if _is_hit(flags, base, nx, ny, i, j, k) == 1:
                flags[p] = flags[p] | tack.u8(_LANDED)
            f = tack.i32(flags[p])
            total += _emitted(f)
            if (f & _MISSING) != 0:
                missing += 1
        counts[rr] = total
        nans[rr] = missing


@tack.func
def _landing(flags, base, hexa, e, ci, cj, ck, nx, ny):
    axis, i, j, k = _edge_of(hexa, e, ci, cj, ck)
    p = i + nx * (j + ny * k)
    code = _code(_flag_at(flags, base, nx, ny, i, j, k), axis)
    out = tack.i64(-1)
    if code == _AT_START:
        out = p
    elif code == _AT_END:
        out = p + (1 if axis == 0 else (nx if axis == 1 else nx * ny))
    return out


@tack.func
def _kept(flags, base, hexa, case, t, ci, cj, ck, nx, ny):
    a = _landing(flags, base, hexa, hexa.contour_edge(case, 3 * t), ci, cj, ck, nx, ny)
    b = _landing(flags, base, hexa, hexa.contour_edge(case, 3 * t + 1), ci, cj, ck, nx, ny)
    c = _landing(flags, base, hexa, hexa.contour_edge(case, 3 * t + 2), ci, cj, ck, nx, ny)
    return not ((a >= 0 and (a == b or a == c)) or (b >= 0 and b == c))


@tack.func
def _cell_case(field, b, nx, ny, ci, cj, ck, level):
    # The hexahedron's corners (VTK's order: corner c at (a, b, d) with
    # c = 4d + 2b + (a ^ b)) at or above the level, as bits; none with a NaN.
    case = 0
    nan = 0
    for c in range(8):
        d = c >> 2
        bb = (c >> 1) & 1
        a = (c & 1) ^ bb
        x = field.at(b, (ci + a) + nx * ((cj + bb) + ny * (ck + d)))
        if x >= level:
            case |= 1 << c
        if x != x:
            nan = 1
    return 0 if nan == 1 else case


@tack.func
def _cell_row_of(crow_box, crow_start, row_start, dims, cc):
    """Cell row ``cc``'s box, its (cj, ck), its first point row (global) and the box's
    (nx, ny, nz)."""
    b = crow_box[cc]
    d = dims[b]
    r = cc - crow_start[b]
    cj = r % (d[1] - 1)
    ck = r // (d[1] - 1)
    return b, cj, ck, row_start[b] + cj + d[1] * ck, d[0], d[1], d[2]


@tack.kernel
def _triangle_counts(field, point_start, cell_start, crow_box, crow_start, row_start, dims,
                     hexa, flags, point_counts, marks, hidden, levels, counts):
    level = levels[0]
    for cc in range(crow_box.shape[0]):
        b, cj, ck, r0, nx, ny, nz = _cell_row_of(crow_box, crow_start, row_start, dims, cc)
        base = point_start[b]
        cx = nx - 1
        cy = ny - 1
        total = 0
        if (point_counts[r0] + point_counts[r0 + 1] + point_counts[r0 + ny]
                + point_counts[r0 + ny + 1]) > 0:
            for ci in range(cx):
                case = _cell_case(field, b, nx, ny, ci, cj, ck, level)
                if hidden == 1 and (tack.i32(marks.at(b, ci + cx * (cj + cy * ck)))
                                    & _MASKED) != 0:
                    case = 0
                for t in range(hexa.contour_count(case)):
                    if _kept(flags, base, hexa, case, t, ci, cj, ck, nx, ny):
                        total += 1
        counts[cc] = total


@tack.kernel
def _points(field, point_start, row_box, row_start, dims, origins, spacings, levels, flags,
            starts, total, positions):
    level = levels[0]
    for rr in range(row_box.shape[0]):
        b, j, k, first, nx, ny, nz = _row_of(row_box, row_start, dims, rr)
        base = point_start[b]
        at = starts[rr]
        end = starts[rr + 1] if rr + 1 < row_box.shape[0] else total
        for i in range(nx if end > at else 0):
            p = first + i
            f = tack.i32(flags[base + p])
            x0 = _position(origins, spacings, b, i, j, k)
            v0 = field.at(b, p)
            if _landed(f) == 1:
                positions[at] = x0
                at += 1
            for axis in range(3):
                if _code(f, axis) == _BETWEEN:
                    ii = i + (1 if axis == 0 else 0)
                    jj = j + (1 if axis == 1 else 0)
                    kk = k + (1 if axis == 2 else 0)
                    q = ii + nx * (jj + ny * kk)
                    x1 = _position(origins, spacings, b, ii, jj, kk)
                    v1 = field.at(b, q)
                    xa = x0
                    xb = x1
                    va = v0
                    vb = v1
                    if not _first_end(x0, x1, p, q):
                        xa = x1
                        xb = x0
                        va = v1
                        vb = v0
                    w = _crossing(level, va, vb)
                    positions[at] = xa + w * (xb - xa)
                    at += 1


@tack.kernel
def _triangles(field, point_start, cell_start, crow_box, crow_start, row_start, dims, hexa,
               flags, point_starts, triangle_starts, triangles, marks, hidden, levels, rows,
               boxes):
    level = levels[0]
    for cc in range(crow_box.shape[0]):
        b, cj, ck, r0, nx, ny, nz = _cell_row_of(crow_box, crow_start, row_start, dims, cc)
        base = point_start[b]
        cx = nx - 1
        cy = ny - 1
        at = triangle_starts[cc]
        end = triangle_starts[cc + 1] if cc + 1 < crow_box.shape[0] else triangles
        n0 = point_starts[r0]
        n1 = point_starts[r0 + 1]
        n2 = point_starts[r0 + ny]
        n3 = point_starts[r0 + ny + 1]
        for ci in range(cx if end > at else 0):
            m0 = n0 + _emitted(_flag_at(flags, base, nx, ny, ci, cj, ck))
            m1 = n1 + _emitted(_flag_at(flags, base, nx, ny, ci, cj + 1, ck))
            m2 = n2 + _emitted(_flag_at(flags, base, nx, ny, ci, cj, ck + 1))
            m3 = n3 + _emitted(_flag_at(flags, base, nx, ny, ci, cj + 1, ck + 1))
            case = _cell_case(field, b, nx, ny, ci, cj, ck, level)
            if hidden == 1 and (tack.i32(marks.at(b, ci + cx * (cj + cy * ck)))
                                & _MASKED) != 0:
                case = 0
            for t in range(hexa.contour_count(case)):
                if _kept(flags, base, hexa, case, t, ci, cj, ck, nx, ny):
                    for corner in range(3):
                        e = hexa.contour_edge(case, 3 * t + corner)
                        axis, i, j, k = _edge_of(hexa, e, ci, cj, ck)
                        s = (j - cj) + 2 * (k - ck)
                        start = _pick(s, n0, n1, n2, n3) if i == ci else _pick(s, m0, m1, m2,
                                                                                 m3)
                        f = _flag_at(flags, base, nx, ny, i, j, k)
                        code = _code(f, axis)
                        far = _pick(s + (2 if axis == 2 else 1), n0, n1, n2, n3)
                        if axis == 0:
                            far = _pick(s, m0, m1, m2, m3)
                        elif i != ci:
                            far = _pick(s + (2 if axis == 2 else 1), m0, m1, m2, m3)
                        rows[3 * at + corner] = _slot(f, axis, code, _landed(f), start, far)
                    boxes[at] = b
                    at += 1
            n0 = m0
            n1 = m1
            n2 = m2
            n3 = m3


@tack.kernel
def _taken_positions(kept, positions, out):
    for n in range(kept.shape[0]):
        out[n] = positions[kept[n]]


def _one_array(parts, what):
    whole = arrays._joined_view(parts)
    if whole is None:
        raise ValueError(f"{what}: the boxes' arrays are not views of one array, end to end")
    return whole


def tables(amr):
    """The hierarchy's box and row tables, on the host (NumPy): built once, placed on a
    device by ``prepare``."""
    boxes = [box for _, box in amr]
    dims = np.array([box.topology.point_dims for box in boxes], np.int64)
    if dims.shape[1] != 3:
        raise ValueError("3D boxes only")
    origins, spacings = [], []
    for box in boxes:
        g = box.geometry.values
        if not isinstance(g, arrays.UniformCoordinates) or getattr(g, "ORIENTED", 0):
            raise ValueError("image boxes (axis-aligned UniformCoordinates) only")
        origins.append((g.ox, g.oy, g.oz))
        spacings.append((g.sx, g.sy, g.sz))
    nrows = dims[:, 1] * dims[:, 2]
    ncrows = (dims[:, 1] - 1) * (dims[:, 2] - 1)
    cells = (dims - 1).prod(axis=1)
    starts = lambda n: np.concatenate([[0], np.cumsum(n)])[:-1].astype(np.int64)  # noqa: E731
    return {
        "boxes": boxes,
        "dims": dims,
        "point_start": starts(dims.prod(axis=1)),
        "cell_start": starts(cells),
        "row_start": starts(nrows),
        "crow_start": starts(ncrows),
        "row_box": np.repeat(np.arange(len(boxes), dtype=np.int32), nrows),
        "crow_box": np.repeat(np.arange(len(boxes), dtype=np.int32), ncrows),
        "origins": np.array(origins),
        "spacings": np.array(spacings),
        "npoints": int(dims.prod(axis=1).sum()),
        "gdt": boxes[0].geometry.values.dtype,
    }


def prepare(amr, field, host=None, device=None, float_dtype=None, layout="joined"):
    """Everything the passes read, on ``device`` (the current one by default): the
    tables, the boxes' point field ``field`` and ghost marks as one array each. With
    ``float_dtype`` (``tack.f32`` for Metal) the field, origins and spacings are
    converted to it (a copy); else the field is shared as it is and positions take
    the boxes' coordinate type. ``layout="table"`` (CPU) reads the boxes' own arrays
    through their addresses instead: they need not be views of one array."""
    from amr_fields import Joined, Table

    host = host or tables(amr)
    boxes = host["boxes"]
    parts = [b.fields[field].values for b in boxes]
    ghosts = [b.ghosts.cells if b.ghosts is not None else None for b in boxes]
    if layout == "table":
        if float_dtype is not None or (device is not None and device != tack.device("cpu")):
            raise ValueError("layout='table': CPU, the field's own dtype")
        fdt = host["gdt"]
        with (device or tack.get_default_device()):
            prepared = _tables_on_device(host, fdt)
            prepared["values"] = Table(parts)
            hidden = not any(g is None for g in ghosts)
            prepared["marks"] = Table(ghosts if hidden else [np.zeros(1, np.uint8)])
        prepared.update(hidden=int(hidden), npoints=host["npoints"], fdt=fdt,
                        vdt=parts[0].dtype, device=device or tack.get_default_device())
        return prepared
    values = _one_array(parts, field)
    marks = None if any(g is None for g in ghosts) else _one_array(ghosts, "ghost marks")
    fdt = float_dtype or host["gdt"]
    vdt = float_dtype or values.dtype
    convert = float_dtype is not None or (device is not None and device != values.device)
    with (device or tack.get_default_device()):
        if convert:
            values = tack.array(values.numpy().astype(vdt.numpy_dtype))
            if marks is not None:
                marks = tack.array(marks.numpy())
        prepared = _tables_on_device(host, fdt)
        nboxes = len(boxes)
        prepared["values"] = Joined(values, prepared["point_start"])
        prepared["marks"] = (Joined(marks, prepared["cell_start"]) if marks is not None
                             else Joined(tack.zeros((1,), tack.u8),
                                         tack.zeros((nboxes,), tack.i64)))
    prepared["hidden"] = 0 if marks is None else 1
    prepared["vdt"] = arrays.dtype_of(values)
    prepared["npoints"] = host["npoints"]
    prepared["fdt"] = fdt
    prepared["device"] = device or tack.get_default_device()
    return prepared


def _tables_on_device(host, fdt):
    prepared = {k: tack.array(host[k]) for k in ("point_start", "cell_start", "row_start",
                                                 "crow_start", "row_box", "crow_box")}
    prepared["dims"] = tack.array(host["dims"], tack.vec(3, tack.i64))
    prepared["origins"] = tack.array(host["origins"].astype(fdt.numpy_dtype), tack.vec(3, fdt))
    prepared["spacings"] = tack.array(host["spacings"].astype(fdt.numpy_dtype),
                                      tack.vec(3, fdt))
    return prepared


def hierarchy_surface(t, level):
    """``(positions, triangles, boxes, count, ntriangles)``: the contour at ``level`` of
    a hierarchy ``prepare``d, as one surface -- ``triangles`` three point ids a
    triangle, ``boxes`` each triangle's box (an index into the hierarchy's boxes in
    order)."""
    values, marks, hidden = t["values"], t["marks"], t["hidden"]
    with t["device"]:
        levels = tack.array(np.asarray([level], t["vdt"].numpy_dtype))
        hexa = shapes.Hexahedron()
        nrows = t["row_box"].shape[0]
        ncrows = t["crow_box"].shape[0]
        flags = tack.empty((t["npoints"],), tack.u8)
        common = (t["point_start"], t["row_box"], t["row_start"], t["dims"])
        _edge_codes(values, *common, levels, flags)
        _landings(values, *common, t["origins"], t["spacings"], levels, flags)
        point_counts = tack.empty((nrows,), tack.i64)
        nans = tack.empty((nrows,), tack.i64)
        _point_counts(flags, *common, point_counts, nans)
        triangle_counts = tack.empty((ncrows,), tack.i64)
        cell_rows = (values, t["point_start"], t["cell_start"], t["crow_box"], t["crow_start"],
                     t["row_start"], t["dims"], hexa, flags)
        _triangle_counts(*cell_rows, point_counts, marks, hidden, levels, triangle_counts)
        count = exclusive_scan(point_counts, point_counts, nrows)
        ntriangles = exclusive_scan(triangle_counts, triangle_counts, ncrows)
        positions = tack.empty((count,), tack.vec(3, t["fdt"]))
        if count:
            _points(values, *common, t["origins"], t["spacings"], levels, flags, point_counts,
                    count, positions)
        rows = tack.empty((3 * ntriangles,), tack.i64)
        tri_boxes = tack.empty((ntriangles,), tack.i32)
        if ntriangles:
            _triangles(*cell_rows, point_counts, triangle_counts, ntriangles, marks, hidden,
                       levels, rows, tri_boxes)
        del flags, point_counts, triangle_counts
        if count and (hidden or int(nans.numpy().sum())):
            used = tack.zeros((count,), tack.i32)
            if ntriangles:
                _mark_used(rows, used)
            new_ids, kept, kept_count = _compact(used, tack.i64)
            if kept_count < count:
                if ntriangles:
                    _renumbered(rows, new_ids)
                out = tack.empty((kept_count,), positions.dtype)
                if kept_count:
                    _taken_positions(kept, positions, out)
                positions, count = out, kept_count
    return positions, rows, tri_boxes, count, ntriangles
