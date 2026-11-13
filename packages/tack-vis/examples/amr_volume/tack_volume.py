# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``TackVolume``: a VTK algorithm (Python) that turns a volume into a ``vtkPolyData``
carrying what its field *is* on the GPU, for a mapper to volume render
(``volume_mapper.VolumeMapper``):

- the domain's bounding box as triangles (what the mapper rasterizes);
- Tack's packed buffer (field data ``TackPool``, 32-bit words);
- ``tack_sample(pool, p)`` as Slang over that buffer (field data ``TackEvaluation``);
- the bounds (``TackBounds``), the value range (``TackRange``) and the smallest cell
  size (``TackCellSize``).

Step 1: the input is a ``vtkImageData`` with a cell array (``array``), sampled by
``volume_tack.CellBox``. The Tack data and the Slang are made again only when the input
changes.
"""

import numpy as np
import volume_tack
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkStringArray, vtkUnsignedIntArray
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource

import tack


def _words(values):
    out = vtkUnsignedIntArray()
    out.SetNumberOfValues(len(values))
    np.asarray(out)[:] = np.asarray(values).view(np.uint32)
    return out


def _floats(name, values):
    out = vtkFloatArray()
    out.SetName(name)
    out.SetNumberOfComponents(len(values))
    out.InsertNextTuple([float(v) for v in values])
    return out


class TackVolume(VTKPythonAlgorithmBase):
    """The input's cell array ``array`` as a volume to render (see the module)."""

    def __init__(self, array="values"):
        super().__init__(nInputPorts=1, inputType="vtkImageData", nOutputPorts=1,
                         outputType="vtkPolyData")
        self.array = array
        self._key = None
        self.timings = {}

    def RequestData(self, request, inInfo, outInfo):
        import time

        t0 = time.perf_counter()
        image = vtkImageData.GetData(inInfo[0])
        out = vtkPolyData.GetData(outInfo)
        key = (image.GetMTime(), self.array)
        if key != self._key:
            values = np.asarray(image.GetCellData().GetArray(self.array)).astype(np.float32)
            dims = [d - 1 for d in image.GetDimensions()]
            spacing = image.GetSpacing()
            box = volume_tack.CellBox(tack.array(values.reshape(-1)), dims, image.GetOrigin(),
                                      spacing)
            evaluation, pool = volume_tack.sample_functions(box)
            self.box = box
            cube = vtkCubeSource(bounds=image.GetBounds())
            triangles = vtkTriangleFilter()
            triangles.SetInputConnection(cube.GetOutputPort())
            triangles.Update()
            self._surface = vtkPolyData()
            self._surface.ShallowCopy(triangles.GetOutput())
            self._surface.GetPointData().Initialize()
            fields = self._surface.GetFieldData()
            pool_words = _words(pool)
            pool_words.SetName("TackPool")
            fields.AddArray(pool_words)
            text = vtkStringArray()
            text.SetName("TackEvaluation")
            text.InsertNextValue(evaluation)
            fields.AddArray(text)
            fields.AddArray(_floats("TackBounds", image.GetBounds()))
            fields.AddArray(_floats("TackRange", (box.values.min(), box.values.max())))
            fields.AddArray(_floats("TackCellSize", (min(spacing),)))
            self._key = key
        out.ShallowCopy(self._surface)
        self.timings = {"tack_ms": 1000 * (time.perf_counter() - t0)}
        return 1
