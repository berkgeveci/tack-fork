"""Cell values to point values over a whole AMR hierarchy in one launch.

``project_to_points(amr, field, name)``: each box's point ``p`` takes the mean of its
box's cells around it -- up to 8, summed in increasing cell id in the field's type,
then divided by their count -- as ``td.project(box, field, H1(box))`` gives it box by
box (``_onto_points``: the point links in increasing order, ``segment_reduce``'s
mean). One kernel over every box, each box's points in order, its cells read from
the field's one array (the boxes' values must be views of one buffer, end to end,
as ``vtk_to_dataset(copy=False)`` shares the AMReX reader's). The points' values are one array too, each box's field
``name`` a view of it.

``ProjectToPointsAMR``: the same as a VTK filter, a ``vtkOverlappingAMR`` in and out,
in place of ``Project(to="points")``'s box-by-box run.
"""
import itertools

import numpy as np

import tack
from tack import arrays
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1
from tack.interop.vtk.algorithms import Parameter, TackAlgorithm


@tack.kernel
def _cells_to_points(cells, point_start, dims, out):
    # One iteration a box, its points in order: the box is the loop index and
    # (i, j, k) loop counters, so no search and no division a point -- 70 ms for
    # AMR-Wind's 206M points on 10 CPU cores, against 1.15 s a point an iteration
    # (a binary search over the boxes' offsets, then div/mod). On a GPU, rows
    # (box, k, j) would make more, evener iterations.
    for b in range(dims.shape[0] // 3):
        nx = dims[3 * b]
        ny = dims[3 * b + 1]
        nz = dims[3 * b + 2]
        p = point_start[b]
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    total = cells.at(b, 0) * 0
                    count = 0
                    for dk in range(2):
                        kk = k - 1 + dk
                        if kk >= 0 and kk < nz:
                            for dj in range(2):
                                jj = j - 1 + dj
                                if jj >= 0 and jj < ny:
                                    for di in range(2):
                                        ii = i - 1 + di
                                        if ii >= 0 and ii < nx:
                                            total = total + cells.at(
                                                b, ii + nx * (jj + ny * kk))
                                            count = count + 1
                    out[p] = total / count
                    p = p + 1


def project_to_points(amr, field, name, layout="joined"):
    """``amr`` (an ``AMRDataSet``) with each box's cell field ``field`` projected to its
    points as point field ``name``, in place; returned for chaining. ``layout``: how the
    kernel reads the boxes' cells (``amr_fields``): "joined" (views of one array) or
    "table" (any CPU arrays, through their addresses)."""
    from amr_fields import field_of

    boxes = [box for _, box in amr]
    fields = [box.fields[field] for box in boxes]
    parts = [f.values for f in fields]
    cells = field_of(parts, layout, repr(field))
    first = parts[0]
    point_dims = np.array([box.topology.point_dims for box in boxes], np.int64)
    if point_dims.shape[1] != 3:
        raise ValueError("3D boxes only")
    cell_dims = point_dims - 1
    point_start = np.concatenate([[0], np.cumsum(point_dims.prod(axis=1))])
    with first.device:
        out = tack.empty((int(point_start[-1]),), first.dtype)
        _cells_to_points(cells, tack.array(point_start),
                         tack.array(cell_dims.reshape(-1).astype(np.int64)), out)
    views = out._row_views(point_start.tolist())
    unit = fields[0].unit
    for box, view in zip(boxes, views):
        box.fields[name] = Field(H1(box), view, unit)
    return amr


class ProjectToPointsAMR(TackAlgorithm):
    """``project_to_points``: an AMR hierarchy's cell ``field`` at its boxes' points, as
    point array ``name``, every box in one launch. ``Project(to="points")`` on a
    ``vtkOverlappingAMR``, which runs box by box."""

    OUTPUT = None
    AMR_PORTS = (0,)

    field = Parameter(None, "the cell field's name")
    name = Parameter(None, "the point array's name; by default the field's")

    def compute(self, data):
        from tack.dataset import AMRDataSet

        if not isinstance(data, AMRDataSet):
            raise TypeError(f"ProjectToPointsAMR takes an AMR hierarchy, not a "
                            f"{type(data).__name__}")
        return project_to_points(data, self.field, self.name or self.field)
