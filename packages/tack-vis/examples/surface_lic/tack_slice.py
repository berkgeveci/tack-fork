# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``TackSlice``: a VTK algorithm (Python) that cuts a volume with Tack and outputs a
``vtkPolyData`` carrying what its field *is* on the GPU, for a mapper to draw:

- the slice's triangles, and each triangle's source cell (cell data ``TackCell``);
- Tack's packed buffer of the cut cells (field data ``TackPool``, 32-bit words);
- Tack's evaluation as Slang over that buffer (field data ``TackEvaluation``, one
  string: ``tack_cell_geometry``, ``tack_vector``, ``tack_scalar``);
- the plane's unit normal (field data ``TackPlaneNormal``).

The output is ordinary VTK data: it serializes as any other (a browser gets the arrays
as blobs and needs neither Tack nor Slang). How it is drawn is a mapper's decision
(``slice_mapper.LICMapperDelegate``).
"""

import numpy as np

from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import (vtkFloatArray, vtkPoints, vtkStringArray,
                                      vtkUnsignedIntArray)
from vtkmodules.vtkCommonDataModel import (vtkCellArray, vtkCompositeDataSet, vtkDataObject,
                                           vtkPolyData)

import exact
import scene
import tack.dataset as td


def _words(values):
    out = vtkUnsignedIntArray()
    out.SetNumberOfValues(len(values))
    np.asarray(out)[:] = np.asarray(values).view(np.uint32)
    return out


class TackSlice(VTKPythonAlgorithmBase):
    """The cut of the input's cells by a plane, Tack's way. Input: a dataset or a
    composite dataset (its first leaf is cut). ``SetPlane(origin, normal)``; ``refine``
    cuts every cell into refine^3 first (``td.subdivide``); ``vector`` and ``scalar``
    name the point fields to evaluate."""

    def __init__(self, vector="V", scalar="Temp", refine=1):
        super().__init__(nInputPorts=1, inputType="vtkDataObject", nOutputPorts=1,
                         outputType="vtkPolyData")
        self.vector, self.scalar, self.refine = vector, scalar, refine
        self.origin, self.normal = None, (1.0, 0.0, 0.0)
        self._key = self._data = self._locator = None
        self.timings = {}

    def SetPlane(self, origin, normal):
        self.origin = tuple(float(v) for v in origin)
        self.normal = tuple(float(v) for v in normal)
        self.Modified()

    def _tack_data(self, grid):
        """The input as Tack data (refined), made again only when the input changes."""
        key = (grid.GetMTime(), self.refine)
        if key != self._key:
            from tack.interop.vtk import vtk_to_dataset

            data = vtk_to_dataset(grid)
            if self.refine > 1:
                data = td.subdivide(data, level=self.refine, fields=[self.vector, self.scalar])
            self._data, self._locator, self._key = data, td.CellLocator(data), key
            self.bounds = grid.GetBounds()
        return self._data

    def RequestData(self, request, inInfo, outInfo):
        import time

        t0 = time.perf_counter()
        obj = vtkDataObject.GetData(inInfo[0])
        if isinstance(obj, vtkCompositeDataSet):
            it = obj.NewIterator()
            it.InitTraversal()
            obj = it.GetCurrentDataObject()
        data = self._tack_data(obj)
        b = self.bounds
        origin = self.origin or (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))
        points, triangles, _, unit, _ = scene.slice_triangles(
            data, origin, self.normal, vector=self.vector, scalar=self.scalar)
        out = vtkPolyData.GetData(outInfo)
        out.Initialize()
        if len(triangles):
            cells, _, _, _ = exact.slice_cells(data, points, triangles, locator=self._locator)
            cut, cells = exact.cut_cells(data, cells)
            evaluation, pool = exact.volume_functions(cut, self.vector, self.scalar)
            out.points = vtkPoints(data=vtkFloatArray(points.astype(np.float32)))
            conn = triangles.reshape(-1).astype(np.int64)
            out.polys = vtkCellArray(offsets=np.arange(0, len(conn) + 1, 3), connectivity=conn)
            cell_ids = _words(cells.astype(np.uint32))
            cell_ids.SetName("TackCell")
            out.GetCellData().AddArray(cell_ids)
            pool_words = _words(pool)
            pool_words.SetName("TackPool")
            out.GetFieldData().AddArray(pool_words)
            text = vtkStringArray()
            text.SetName("TackEvaluation")
            text.InsertNextValue(evaluation)
            out.GetFieldData().AddArray(text)
        n = vtkFloatArray()
        n.SetName("TackPlaneNormal")
        n.SetNumberOfComponents(3)
        n.InsertNextTuple(tuple(float(v) for v in unit))
        out.GetFieldData().AddArray(n)
        self.timings = {"tack_ms": 1000 * (time.perf_counter() - t0),
                        "triangles": len(triangles)}
        return 1
