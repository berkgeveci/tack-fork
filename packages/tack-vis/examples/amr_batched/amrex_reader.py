"""Read an AMReX plotfile into a ``vtkOverlappingAMR``, with NumPy, threads and MPI.

Each component read is one NumPy array holding every block's cells -- the levels
in order, each level's boxes in the order the plotfile lists them -- and each
block's VTK array is a view of it. VTK shares the view without a copy:
``numpy_to_vtk`` pins it on the VTK array's buffer, and the view holds the whole
array, which is freed when the last block's array goes. Components named
``<base>x``, ``<base>y`` and ``<base>z`` become one vector named ``<base>``: the
components are read as any others, then interleaved once into one array of
3-tuples (VTK's filters, and Tack's, work on interleaved vectors, and a read
cannot place values at every third element), each block's vector a view of it.
The blanking marks (``vtkGhostType``) are one array the same way.

Reading is planned, then done (``plan_reads``). A FAB is a header line and then
its components one after another, and AMReX writes a file's FABs one after
another, so where everything lies follows from ``Cell_H``'s offsets and the
boxes' sizes, with no header read first. A read is one ``preadv`` of a stretch
of a file into a list of places: each wanted run of values straight into its
view, header lines into small buffers (checked against their boxes), anything
else in between into scratch. ``gap`` is how much unwanted data a read may span
rather than end and start another: 0 (the default) joins only runs that touch
(a vector's components, say), and ``None`` reads whole FABs, a file's FABs in
reads of up to ``most`` bytes. Reads run on a pool of ``threads``: ``preadv``
releases the GIL, and an SSD serves many requests at once far faster than one at
a time (AMR-Wind's temperature, 1,463 MB in 15,598 boxes, uncached on a USB SSD:
plain reads take 12.7 s on one thread and 3.3 s on sixteen; this reader 3.9 s,
VTK's 24.6 s), while one large read reaches the drive's rate alone.

With ``comm`` (an mpi4py communicator, or anything with ``Get_rank``,
``Get_size`` and ``bcast``) rank 0 reads the headers and sends them to the
others; the FABs, ordered by file and offset, are cut into one contiguous stretch
per rank, balanced by cells, so each rank reads one region of the files (at most
the ends of its stretch share a file with another rank). Every rank's hierarchy
has every block's box and its own blocks' data, VTK's convention for a
distributed ``vtkOverlappingAMR``.

The boxes go to VTK in one call (``vtkOverlappingAMR.SetAMRBoxes``) where VTK has
it, one at a time otherwise. Like VTK's ``vtkAMReXGridReader``, the result has
the plotfile's time (``DATA_TIME_STEP``), blocks as ``vtkUniformGrid``s, and its
refined cells marked ``REFINEDCELL`` (``blank=False`` to skip): by VTK's
``vtkAMRUtilities.BlankCells`` where one process holds every block, and under MPI
from the boxes alone, so a rank marks its blocks under any rank's finer boxes
(VTK's serial blanking skips a finer box whose data is elsewhere).

Plotfiles of cell-centred data without ghost cells, whose FABs have header lines
(``Cell_H`` version 1), are read: the common case. Others are refused, naming
the reason.
"""

import concurrent.futures
import os
import re
import threading
from dataclasses import dataclass, field

import numpy as np

__all__ = ["AMReXPlotfileReader", "Level", "Plotfile", "Read", "plan_reads", "read_header",
           "read_plotfile", "vector_groups"]

_BOX = re.compile(r"\(\(([-\d,\s]+)\)\s*\(([-\d,\s]+)\)\s*\(([-\d,\s]+)\)\)")
_FAB = re.compile(rb"FAB\s*\(\((\d+),\s*\(([\d\s]+)\)\),\s*\((\d+),\s*\(([\d\s]+)\)\)\)"
                  rb"\(\(([-\d,\s]+)\)\s*\(([-\d,\s]+)\)\s*\(([-\d,\s]+)\)\)\s*(\d+)")
_LONGEST_HEADER = 4096          # a FAB's header line is a few hundred bytes at most
_REFINEDCELL = 8                # vtkDataSetAttributes::REFINEDCELL


def _ints(text):
    if isinstance(text, bytes):
        text = text.decode()
    return tuple(int(v) for v in text.replace(",", " ").split())


@dataclass
class Level:
    """A level of a plotfile: ``boxes`` (``(n, 2, 3)`` ints, each box's low and high
    cell, inclusive, as AMReX writes them; axes past the plotfile's dimension 0),
    each box's FAB as ``(file, offset)`` (``fabs``), its cell size (``spacing``), and
    its numbers of components and ghost cells."""

    boxes: np.ndarray
    fabs: list
    spacing: tuple
    components: int
    ghosts: int

    @property
    def cells(self):
        """Each box's number of cells."""
        return np.prod(self.boxes[:, 1] - self.boxes[:, 0] + 1, axis=1)


@dataclass
class Plotfile:
    """A plotfile's header: its components' ``names``, ``dimension`` (2 or 3),
    ``time``, ``origin`` (the problem domain's low corner, three values),
    ``ratios`` (each level's refinement ratio to the next), ``levels`` (up to the
    level asked for) and each FAB file's size (``sizes``)."""

    path: str
    names: list
    dimension: int
    time: float
    origin: tuple
    ratios: list
    levels: list
    sizes: dict = field(default_factory=dict)


def read_header(path, max_level=None):
    """The ``Plotfile`` at ``path``: its ``Header``, the ``Cell_H`` of each level up
    to ``max_level`` (every level when None) and its FAB files' sizes. No values
    are read."""
    with open(os.path.join(path, "Header")) as f:
        lines = iter(f.read().splitlines())
    version = next(lines)
    if not version.startswith("HyperCLaw"):
        raise ValueError(f"{path} is not an AMReX plotfile: its Header starts {version!r}")
    names = [next(lines).strip() for _ in range(int(next(lines)))]
    dimension = int(next(lines))
    if dimension not in (2, 3):
        raise NotImplementedError(f"{path} is a {dimension}D plotfile; 2D and 3D are read")
    time = float(next(lines))
    finest = int(next(lines))
    origin = tuple(float(v) for v in next(lines).split()) + (0.0,) * (3 - dimension)
    next(lines)                                         # the domain's high corner
    given = next(lines)                                 # ratios: an empty line for one level
    if "(" in given:                                    # an IntVect a level: (2,2,2)
        vectors = [_ints(v) for v in re.findall(r"\(([^()]*)\)", given)]
    else:
        vectors = [(int(v),) * dimension for v in given.split()]
    ratios = []
    for vector in vectors[:finest]:
        if len(set(vector[:dimension])) != 1:
            raise NotImplementedError(f"{path} refines levels by {vector}: VTK's AMR takes "
                                      "one ratio along every axis")
        ratios.append(vector[0])
    next(lines)                                         # each level's domain
    next(lines)                                         # each level's steps
    spacings = [tuple(float(v) for v in next(lines).split()) for _ in range(finest + 1)]
    next(lines)                                         # coordinate system
    next(lines)                                         # boundary width
    top = finest if max_level is None else min(finest, int(max_level))
    prefixes = []
    for _ in range(top + 1):
        grids = int(next(lines).split()[1])             # level, grids, time
        next(lines)                                     # steps
        for _ in range(grids * dimension):              # each grid's extent in space
            next(lines)
        prefixes.append(next(lines).strip())            # Level_<l>/Cell
    levels = [_read_level(path, prefix, spacings[level], dimension)
              for level, prefix in enumerate(prefixes)]
    sizes = {file: os.stat(file).st_size for lv in levels for file in {f for f, _ in lv.fabs}}
    return Plotfile(path, names, dimension, time, origin, ratios[:top], levels, sizes)


def _read_level(path, prefix, spacing, dimension):
    """A level's ``Cell_H``: its boxes and where each one's FAB is."""
    with open(os.path.join(path, prefix + "_H")) as f:
        text = f.read()
    version, _how, components, ghosts, rest = text.split(None, 4)
    if int(version) != 1:
        raise NotImplementedError(f"{prefix}_H is version {version}, whose FABs have no "
                                  "header lines; version 1 is read")
    count = int(re.match(r"\((\d+)", rest).group(1))
    boxes = np.zeros((count, 2, 3), np.int64)
    for k, (lo, hi, kind) in enumerate(_BOX.findall(rest)[:count]):
        if any(_ints(kind)):
            raise NotImplementedError(f"{prefix} holds nodal data; cell-centred data is read")
        boxes[k, 0, :dimension] = _ints(lo)[:dimension]
        boxes[k, 1, :dimension] = _ints(hi)[:dimension]
    directory = os.path.join(path, os.path.dirname(prefix))
    fabs = [(os.path.join(directory, name), int(offset))
            for name, offset in re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", rest)[:count]]
    if len(fabs) != count:
        raise ValueError(f"{prefix}_H lists {count} boxes and {len(fabs)} FABs")
    return Level(boxes, fabs, spacing, int(components), int(ghosts))


def vector_groups(names):
    """``{base: [<base>x, <base>y, <base>z]}`` for each base whose three components
    are all among ``names``, in the order the first of them comes."""
    present = set(names)
    groups = {}
    for name in names:
        base = name[:-1]
        if base and name[-1] == "x" and all(base + c in present for c in "yz"):
            groups[base] = [base + c for c in "xyz"]
    return groups


def _fab_header(head, where):
    """A FAB header line's values' NumPy type (byte order included), its box's low
    and high cells and its number of components."""
    match = _FAB.match(head)
    if match is None:
        raise ValueError(f"no FAB header at {where}: {bytes(head[:80])!r}")
    size, form, order_size, order = (int(match[1]), _ints(match[2]), int(match[3]),
                                     _ints(match[4]))
    kinds = {(8, 64): "f8", (4, 32): "f4"}
    if (size, form[0]) not in kinds or order_size != size:
        raise NotImplementedError(f"{where}: values of {size} bytes in the form {form}; IEEE "
                                  "doubles and floats are read")
    if order == tuple(range(size, 0, -1)):
        endian = "<"
    elif order == tuple(range(1, size + 1)):
        endian = ">"
    else:
        raise NotImplementedError(f"{where}: byte order {order}")
    return (np.dtype(endian + kinds[size, form[0]]), _ints(match[5]), _ints(match[6]),
            int(match[8]))


def _first_fab(plot):
    """The values' type and the length of the descriptor every FAB's header line
    starts with (up to its box), from the first FAB: every header read is checked
    against them, and every header's length is known from its box."""
    file, offset = next(fab for lv in plot.levels for fab in lv.fabs)
    with open(file, "rb") as f:
        f.seek(offset)
        head = f.read(_LONGEST_HEADER)
    dtype = _fab_header(head, f"{file} at {offset}")[0]
    return dtype, head.index(b")((") + 1


@dataclass
class Read:
    """One ``preadv``: ``length`` bytes of ``file`` from ``offset``, into ``places``,
    in order: ``(bytes, what)`` with ``what`` ``("values", name, box)`` (a run of a
    component of a box, at the box's place in this rank's arrays), ``("header",
    box)`` (a FAB's header line, checked) or None (read past)."""

    file: str
    offset: int
    places: list

    @property
    def length(self):
        return sum(n for n, _ in self.places)


def _header_length(prefix, box, components, dimension):
    """The length of a FAB's header line as AMReX writes it: the descriptor
    (``prefix`` bytes), its box (cell-centred), its number of components."""
    def corner(v):
        return ",".join(str(int(x)) for x in v[:dimension])

    text = f"(({corner(box[0])}) ({corner(box[1])}) ({','.join('0' * dimension)})) " \
        f"{components}\n"
    return prefix + len(text)


def _stretches(plot, itemsize, prefix, columns, mine):
    """Each file's FABs among ``mine`` (global box indices), in offset order, as
    ``(offset, bytes, what)`` stretches: header line, then each component's run,
    wanted (``("values", name, box)``) or not (None). A header's length is where
    the next FAB starts less this one's data (the file's end for its last), and
    must be the length of the line AMReX writes for its box: a file laid out
    otherwise (padding, say) is refused rather than read from the wrong places."""
    by_file = {}
    k = 0
    for lv in plot.levels:
        cells = lv.cells
        for b, (file, offset) in enumerate(lv.fabs):
            by_file.setdefault(file, []).append((offset, k, int(cells[b]), lv.components,
                                                 lv.boxes[b]))
            k += 1
    out = {}
    names = {c: name for name, c in columns.items()}
    for file, fabs in by_file.items():
        fabs.sort()
        stretches = []
        for i, (offset, k, cells, components, box) in enumerate(fabs):
            run = cells * itemsize
            end = fabs[i + 1][0] if i + 1 < len(fabs) else plot.sizes[file]
            header = end - offset - components * run
            expected = _header_length(prefix, box, components, plot.dimension)
            if header != expected:
                raise ValueError(f"the FABs of {file} are not laid out as AMReX writes them: "
                                 f"box {k}'s header line would be {header} bytes by Cell_H's "
                                 f"offsets, {expected} by its text")
            if k not in mine:
                continue
            stretches.append((offset, header, ("header", k)))
            at = offset + header
            for c in range(components):
                stretches.append((at, run, ("values", names[c], k) if c in names else None))
                at += run
        if stretches:
            out[file] = stretches
    return out


def plan_reads(plot, columns, itemsize, prefix, rank=0, size=1, gap=0, most=64 << 20,
               iov_max=None, per_block=32768):
    """This rank's boxes and reads: ``(boxes, reads)``, the boxes (global indices,
    levels in order) whose values it reads, and its ``Read``s.

    The FABs, ordered by file and offset, are cut into ``size`` contiguous stretches
    balanced by cells plus ``per_block`` a block (a block's reads, VTK grid and
    marks cost about as much as 1,500 cells' values uncached and tens of thousands
    from the file cache: AMR-Wind's coarse files hold many small blocks, each under
    many finer ones), and rank ``rank`` reads stretch ``rank``. ``columns`` maps
    each component read to its place in the FABs. A read starts at a wanted run
    and takes in what follows while the unwanted bytes since its last wanted run
    are at most ``gap`` (None: no limit, whole FABs), it is at most ``most`` bytes,
    and has at most ``iov_max`` places (the system's limit when None). ``prefix`` is
    the length of the descriptor FAB header lines start with (``_first_fab``)."""
    if iov_max is None:
        iov_max = min(os.sysconf("SC_IOV_MAX"), 1024) if hasattr(os, "sysconf") else 1024
    cells = np.concatenate([lv.cells for lv in plot.levels]).tolist()
    fabs = [fab for lv in plot.levels for fab in lv.fabs]
    order = sorted((file, offset, k, cells[k]) for k, (file, offset) in enumerate(fabs))
    weights = np.array([cells for *_, cells in order], np.float64) + per_block
    middle = np.cumsum(weights) - weights / 2
    owner = np.minimum((middle * size / max(weights.sum(), 1.0)).astype(np.int64), size - 1)
    mine = {k for (_, _, k, _), r in zip(order, owner) if r == rank}
    reads = []
    for file, stretches in _stretches(plot, itemsize, prefix, columns, mine).items():
        current, pending = None, []
        for offset, length, what in stretches:
            if what is None or what[0] == "header":
                if current is not None:
                    pending.append((length, what))
                continue
            skipped = sum(n for n, _ in pending)
            fits = (current is not None
                    and (gap is None or skipped <= gap)
                    and current.length + skipped + length <= most
                    and len(current.places) + len(pending) + 1 <= iov_max)
            if fits:
                current.places += pending + [(length, what)]
            else:
                if current is not None:
                    reads.append(current)
                current = Read(file, offset, [(length, what)])
            pending = []
        if current is not None:
            reads.append(current)
    for read in reads:                  # neighbouring places read past are one
        merged = []
        for n, what in read.places:
            if what is None and merged and merged[-1][1] is None:
                merged[-1] = (merged[-1][0] + n, None)
            else:
                merged.append((n, what))
        read.places = merged
    return sorted(mine), reads


_scratch = threading.local()


def _skip_buffer(n):
    buffer = getattr(_scratch, "buffer", None)
    if buffer is None or len(buffer) < n:
        buffer = _scratch.buffer = memoryview(bytearray(max(n, 1 << 20)))
    return buffer[:n]


def _do_read(read, fd, arrays, starts, place, boxes, dtype, dimension):
    """One planned read: its places filled, its header lines checked, its values put
    in the machine's byte order."""
    swap = dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">")
    if len(read.places) == 1:           # one run of values, the usual read: no lists
        n, (_, name, k) = read.places[0]
        a = starts[place[k]]
        view = arrays[name][a:a + n // dtype.itemsize]
        buffer = memoryview(view).cast("B")
        done = 0
        while done < n:
            got = os.preadv(fd, [buffer[done:]], read.offset + done)
            if got <= 0:
                raise EOFError(f"{read.file} ends {n - done} bytes short of a FAB")
            done += got
        if swap:
            view.byteswap(inplace=True)
        return
    views, headers, buffers = [], [], []
    for n, what in read.places:
        if what is None:
            buffers.append(_skip_buffer(n))
        elif what[0] == "header":
            head = bytearray(n)
            headers.append((what[1], head))
            buffers.append(memoryview(head))
        else:
            _, name, k = what
            a = starts[place[k]]
            view = arrays[name][a:a + n // dtype.itemsize]
            views.append(view)
            buffers.append(memoryview(view).cast("B"))
    offset, total = read.offset, read.length
    while total:
        got = os.preadv(fd, buffers, offset)
        if got <= 0:
            raise EOFError(f"{read.file} ends {total} bytes short of a FAB")
        offset += got
        total -= got
        while buffers and got >= len(buffers[0]):
            got -= len(buffers[0])
            buffers.pop(0)
        if got:
            buffers[0] = buffers[0][got:]
    for k, head in headers:
        found, lo, hi, _ = _fab_header(head, f"{read.file} (box {k})")
        if found != dtype:
            raise ValueError(f"{read.file}: box {k} holds {found}, the first FAB {dtype}")
        if lo[:dimension] != tuple(boxes[k, 0, :dimension]) or \
                hi[:dimension] != tuple(boxes[k, 1, :dimension]):
            raise ValueError(f"{read.file}: box {k} is {lo}-{hi} on disk, not "
                             f"{tuple(boxes[k, 0])}-{tuple(boxes[k, 1])}")
    if swap:
        for view in views:
            view.byteswap(inplace=True)


def _refined_marks(boxes, level_of, ratios, mine, starts, place, dimension):
    """``vtkGhostType`` for this rank's blocks, one u8 array for all of them:
    ``REFINEDCELL`` on each cell a finer box, coarsened, covers (any rank's box:
    every rank has them all). A level at a time, with no loop over its boxes: the
    pairs of a block and a finer box that meet (by a sweep along x), the rows
    (along x) of each pair's overlap, then the cells of the rows, all as arrays (a
    loop over the pairs was 0.9 s for AMR-Wind's blocks; VTK's BlankCells 0.06 s)."""
    marks = np.zeros(int(starts[-1]), np.uint8)
    levels = int(level_of.max()) + 1 if len(level_of) else 0
    mine = np.asarray(mine, np.int64)
    for level in range(levels - 1):
        children = boxes[level_of == level + 1]
        here = mine[level_of[mine] == level]
        if not len(children) or not len(here):
            continue
        r = np.array([ratios[level]] * dimension + [1] * (3 - dimension))
        clo, chi = children[:, 0] // r, children[:, 1] // r
        lo, hi = boxes[here, 0], boxes[here, 1]
        # The pairs that meet, by a sweep along x: the children sorted by their low
        # x, each block's candidates the run whose low x can reach it (no child is
        # wider than the widest), then the whole test on the candidates alone (every
        # pair of blocks and children was 0.5 s of AMR-Wind's 0.8).
        order = np.argsort(clo[:, 0], kind="stable")
        sorted_lo = clo[order, 0]
        widest = int((chi[:, 0] - clo[:, 0]).max())
        begin = np.searchsorted(sorted_lo, lo[:, 0] - widest, side="left")
        end = np.searchsorted(sorted_lo, hi[:, 0], side="right")
        counts = end - begin
        i = np.repeat(np.arange(len(here)), counts)
        j = order[np.repeat(begin - (np.cumsum(counts) - counts), counts) +
                  np.arange(int(counts.sum()))]
        meet = np.all((clo[j] <= hi[i]) & (chi[j] >= lo[i]), axis=1)
        i, j = i[meet], j[meet]
        if not len(i):
            continue
        # Each pair's overlap in its block's cells (inclusive), and the block's extent.
        a = np.maximum(clo[j], lo[i]) - lo[i]
        b = np.minimum(chi[j], hi[i]) - lo[i]
        extent = hi[i] - lo[i] + 1
        first = starts[[place[int(k)] for k in here[i]]]
        # The rows: one along x for each (y, z) of the overlap.
        ny, nz = b[:, 1] - a[:, 1] + 1, b[:, 2] - a[:, 2] + 1
        rows = ny * nz
        pair = np.repeat(np.arange(len(i)), rows)
        r_in = np.arange(int(rows.sum())) - np.repeat(np.cumsum(rows) - rows, rows)
        y = a[pair, 1] + r_in % ny[pair]
        z = a[pair, 2] + r_in // ny[pair]
        row_start = first[pair] + (z * extent[pair, 1] + y) * extent[pair, 0] + a[pair, 0]
        length = b[pair, 0] - a[pair, 0] + 1
        # The cells of the rows.
        cells = np.repeat(row_start - (np.cumsum(length) - length), length) + \
            np.arange(int(length.sum()))
        marks[cells] |= _REFINEDCELL
    return marks


def read_plotfile(path, fields=None, vectors=None, max_level=None, threads=16, blank=True,
                  gap=0, most=64 << 20, comm=None):
    """The AMReX plotfile at ``path`` as a ``vtkOverlappingAMR`` of its levels up to
    ``max_level`` (all when None).

    ``fields`` names what to read -- components, or vectors' names -- in the order
    the blocks' arrays take (all, when None). ``vectors`` maps a vector's name to its
    components' names; by default, each base with ``<base>x``, ``<base>y`` and
    ``<base>z`` components (``vector_groups``), and ``{}`` keeps every component a
    scalar, as VTK's reader does. A component a vector takes is not also a scalar
    unless ``fields`` names it. ``threads`` reads run at once; ``gap`` and ``most``
    shape them (``plan_reads``; ``gap=None`` reads whole FABs); ``blank`` marks
    refined cells, as VTK's reader does; ``comm`` reads across MPI ranks, each its
    own blocks."""
    from vtkmodules.vtkCommonCore import vtkIntArray
    from vtkmodules.vtkCommonDataModel import (
        vtkAMRBox,
        vtkAMRUtilities,
        vtkDataObject,
        vtkOverlappingAMR,
        vtkStructuredData,
        vtkUniformGrid,
    )

    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    shared = None
    if rank == 0:
        try:
            plot = read_header(path, max_level)
            for level, lv in enumerate(plot.levels):
                if lv.ghosts:
                    raise NotImplementedError(f"level {level} has {lv.ghosts} ghost cells a "
                                              "box; plotfiles without them are read")
            shared = (plot, *(_first_fab(plot) if any(lv.fabs for lv in plot.levels)
                              else (np.dtype("<f8"), 0)))
        except Exception as error:      # every rank raises it, none waits
            shared = error
    if comm is not None:
        shared = comm.bcast(shared, root=0)
    if isinstance(shared, Exception):
        raise shared
    plot, dtype, prefix = shared

    names, d = plot.names, plot.dimension
    groups = vector_groups(names) if vectors is None else {k: list(v) for k, v in
                                                            vectors.items()}
    for base, parts in groups.items():
        missing = [p for p in parts if p not in names]
        if missing:
            raise KeyError(f"vector {base!r} takes {missing}, which the plotfile lacks")
    if fields is None:
        taken = {p for parts in groups.values() for p in parts}
        scalars = [n for n in names if n not in taken]
    else:
        unknown = [f for f in fields if f not in names and f not in groups]
        if unknown:
            raise KeyError(f"no field {unknown} in {path}: it has {names}, and vectors "
                           f"{list(groups)}")
        scalars = [f for f in fields if f in names]
        groups = {base: groups[base] for base in fields if base in groups}
    read = list(dict.fromkeys(scalars + [p for parts in groups.values() for p in parts]))
    columns = {name: names.index(name) for name in read}

    levels = plot.levels
    boxes = np.concatenate([lv.boxes for lv in levels])
    cells = np.concatenate([lv.cells for lv in levels])
    level_of = np.repeat(np.arange(len(levels)), [len(lv.boxes) for lv in levels])
    mine, reads = plan_reads(plot, columns, dtype.itemsize, prefix, rank, size, gap, most)
    if not read:                        # no values: every block of this rank's stretch
        mine = plan_reads(plot, {names[0]: 0}, dtype.itemsize, prefix, rank, size, gap,
                          most)[0] if names else []
        reads = []
    place = {k: i for i, k in enumerate(mine)}
    starts = np.concatenate([[0], np.cumsum(cells[mine])]).astype(np.int64)
    # NumPy's own float64 (or float32), not dtype.newbyteorder("="): equal, but a
    # library that checks the type object (openPMD-api's store_chunk) refuses that.
    native = np.dtype(dtype.kind + str(dtype.itemsize))
    arrays = {name: np.empty(int(starts[-1]), native) for name in read}
    if reads:
        files = {r.file for r in reads}
        descriptors = {file: os.open(file, os.O_RDONLY) for file in files}
        # A task a file: its reads one after another, so a thread keeps to one
        # region and the pool does not hand out each small read on its own.
        by_file = {}
        for r in reads:
            by_file.setdefault(r.file, []).append(r)

        def task(file_reads):
            for r in file_reads:
                _do_read(r, descriptors[r.file], arrays, starts, place, boxes, dtype, d)

        try:
            with concurrent.futures.ThreadPoolExecutor(max(1, int(threads))) as pool:
                for future in [pool.submit(task, rs) for rs in by_file.values()]:
                    future.result()
        finally:
            for fd in descriptors.values():
                os.close(fd)
    # Under MPI the marks come from the boxes alone (VTK's serial BlankCells skips a
    # finer box whose data another rank holds); with every block here, VTK's
    # (60 ms for AMR-Wind, against 0.9 s for the same in NumPy).
    marks = _refined_marks(boxes, level_of, plot.ratios, mine, starts, place, d) \
        if blank and size > 1 else None

    amr = vtkOverlappingAMR()
    amr.Initialize([len(lv.boxes) for lv in levels])
    amr.SetOrigin(list(plot.origin))
    amr.SetGridDescription(vtkStructuredData.VTK_STRUCTURED_XYZ_GRID if d == 3
                           else vtkStructuredData.VTK_STRUCTURED_XY_PLANE)
    spacings = []
    for level, lv in enumerate(levels):
        # VTK's reader gives a 2D hierarchy's third axis the second's spacing.
        spacing = tuple(lv.spacing[:d]) + ((lv.spacing[1],) if d == 2 else ())
        spacings.append(np.asarray(spacing))
        amr.SetSpacing(level, list(spacing))
        amr.SetRefinementRatio(level, plot.ratios[level] if level < len(levels) - 1 else 1)
    corners = boxes.reshape(-1, 6).astype(np.int32)     # lo, then hi, inclusive
    if d == 2:
        corners[:, 5] = -1                              # an empty k axis, as VTK writes 2D
    if hasattr(amr, "SetAMRBoxes"):
        amr.SetAMRBoxes(vtkIntArray(corners))
    else:
        k = 0
        for level, lv in enumerate(levels):
            for i in range(len(lv.boxes)):
                amr.SetAMRBox(level, i, vtkAMRBox(corners[k, :3].tolist(),
                                                  corners[k, 3:].tolist()))
                k += 1

    # Each vector's components interleaved once; a component read only for a vector
    # goes once it is in.
    vectors = {}
    for base, parts in groups.items():
        vector = np.empty((int(starts[-1]), len(parts)), native)
        for c, part in enumerate(parts):
            vector[:, c] = arrays[part]
        vectors[base] = vector
    for part in {p for parts in groups.values() for p in parts} - set(scalars):
        del arrays[part]

    origin = np.asarray(plot.origin)
    firsts = np.concatenate([[0], np.cumsum([len(lv.boxes) for lv in levels])])
    for k in mine:
        level = int(level_of[k])
        spacing = spacings[level]
        lo, hi = boxes[k, 0], boxes[k, 1]
        dims = [int(hi[a] - lo[a]) + 2 if a < d else 1 for a in range(3)]
        corner = origin + np.where(np.arange(3) < d, lo, 0) * spacing
        grid = vtkUniformGrid(dimensions=dims, origin=corner.tolist(),
                              spacing=spacing.tolist())
        a, b = int(starts[place[k]]), int(starts[place[k] + 1])
        for name in scalars:
            grid.cell_data[name] = arrays[name][a:b]
        for base, vector in vectors.items():
            grid.cell_data[base] = vector[a:b]
        if marks is not None:
            grid.cell_data["vtkGhostType"] = marks[a:b]
        amr.SetDataSet(level, int(k - firsts[level]), grid)
    amr.GetInformation().Set(vtkDataObject.DATA_TIME_STEP(), plot.time)
    if blank and size == 1:
        vtkAMRUtilities.BlankCells(amr)
    elif blank:
        # As VTK's blanking leaves it: the parents and children of each block.
        amr.GetOverlappingAMRMetaData().GenerateParentChildInformation()
    return amr


def _algorithm_base():
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

    return VTKPythonAlgorithmBase


class AMReXPlotfileReader(_algorithm_base()):
    """``read_plotfile`` as a VTK source, for pipelines: ``FileName``, ``Fields``,
    ``Vectors``, ``MaxLevel``, ``Threads``, ``Blank``, ``Gap`` and ``Comm`` are its
    arguments."""

    def __init__(self, file_name=None, fields=None, vectors=None, max_level=None, threads=16,
                 blank=True, gap=0, comm=None):
        super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkOverlappingAMR")
        self._options = {"file_name": file_name, "fields": fields, "vectors": vectors,
                         "max_level": max_level, "threads": threads, "blank": blank,
                         "gap": gap, "comm": comm}

    def _set(self, key, value):
        if self._options[key] != value:
            self._options[key] = value
            self.Modified()

    def SetFileName(self, name):
        self._set("file_name", name)

    def GetFileName(self):
        return self._options["file_name"]

    def SetFields(self, fields):
        self._set("fields", None if fields is None else list(fields))

    def SetVectors(self, vectors):
        self._set("vectors", None if vectors is None else dict(vectors))

    def SetMaxLevel(self, level):
        self._set("max_level", level)

    def SetThreads(self, threads):
        self._set("threads", int(threads))

    def SetBlank(self, blank):
        self._set("blank", bool(blank))

    def SetGap(self, gap):
        self._set("gap", gap)

    def SetComm(self, comm):
        self._set("comm", comm)

    def RequestInformation(self, request, inInfo, outInfo):
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline

        name = self._options["file_name"]
        if name:
            comm = self._options["comm"]
            time = read_header(name, max_level=0).time if comm is None or \
                comm.Get_rank() == 0 else None
            if comm is not None:
                time = comm.bcast(time, root=0)
            info = outInfo.GetInformationObject(0)
            info.Set(vtkStreamingDemandDrivenPipeline.TIME_STEPS(), [time], 1)
            info.Set(vtkStreamingDemandDrivenPipeline.TIME_RANGE(), [time, time], 2)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        from vtkmodules.vtkCommonDataModel import vtkOverlappingAMR

        options = dict(self._options)
        name = options.pop("file_name")
        if not name:
            raise ValueError("AMReXPlotfileReader needs a FileName")
        output = vtkOverlappingAMR.GetData(outInfo)
        output.ShallowCopy(read_plotfile(name, **options))
        return 1
