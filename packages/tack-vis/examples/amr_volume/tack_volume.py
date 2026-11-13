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

The input holds cell-centred values (the cell array ``array``):

- a ``vtkImageData``, sampled by ``volume_tack.CellBox`` (trilinear over the cell
  centres);
- a ``vtkOverlappingAMR``, converted to Tack's ``AMRDataSet`` (which marks the
  refined cells) and sampled by ``volume_tack.AMRVolume``: tent basis functions of
  every active cell (``blend=1``), or per-level trilinear in the finest level holding
  the point (``blend=0``, the naive scheme, for comparison).

The Tack data and the Slang are made again only when the input or ``blend`` changes.
"""

import numpy as np
import volume_tack
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkStringArray, vtkUnsignedIntArray
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkImageData,
    vtkOverlappingAMR,
    vtkPolyData,
)
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

    def __init__(self, array="values", blend=1):
        super().__init__(nInputPorts=1, inputType="vtkDataObject", nOutputPorts=1,
                         outputType="vtkPolyData")
        self.array = array
        self.blend = blend
        self._key = None
        self.timings = {}

    def SetBlend(self, blend):
        self.blend = int(blend)
        self.Modified()

    def _sampler(self, data):
        """The input as a Tack sampler, and its bounds, value range and smallest cell."""
        if isinstance(data, vtkImageData):
            values = np.asarray(data.GetCellData().GetArray(self.array)).astype(np.float32)
            dims = [d - 1 for d in data.GetDimensions()]
            spacing = data.GetSpacing()
            box = volume_tack.CellBox(tack.array(values.reshape(-1)), dims, data.GetOrigin(),
                                      spacing)
            return box, data.GetBounds(), box.values, min(spacing)
        if isinstance(data, vtkOverlappingAMR):
            from tack.interop.vtk import vtk_to_dataset

            amr = vtk_to_dataset(data)
            spacings = []
            for level in range(data.GetNumberOfLevels()):
                h = [0.0, 0.0, 0.0]
                data.GetSpacing(level, h)
                spacings.append(h)
            # vtkOverlappingAMR's GetOrigin() and GetBounds() return pointers in Python
            # (VTK 9.7 and master); the filling form works. Level 0 covers the domain.
            bounds = [0.0] * 6
            data.GetBounds(bounds)
            volume = volume_tack.amr_volume(amr, spacings, bounds[::2], self.array, self.blend)
            self.amr = amr
            return volume, tuple(bounds), volume.values, min(spacings[-1])
        raise TypeError(f"TackVolume takes an image or an overlapping AMR, not "
                        f"{data.GetClassName()}")

    def RequestData(self, request, inInfo, outInfo):
        import time

        t0 = time.perf_counter()
        data = vtkDataObject.GetData(inInfo[0])
        out = vtkPolyData.GetData(outInfo)
        key = (data.GetMTime(), self.array, self.blend)
        if key != self._key:
            box, bounds, values, cell = self._sampler(data)
            t1 = time.perf_counter()
            evaluation, pool = volume_tack.sample_functions(box)
            self.box = box
            cube = vtkCubeSource(bounds=bounds)
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
            fields.AddArray(_floats("TackBounds", bounds))
            fields.AddArray(_floats("TackRange", (values.min(), values.max())))
            fields.AddArray(_floats("TackCellSize", (cell,)))
            self._key = key
            self.timings = {"convert_ms": 1000 * (t1 - t0),
                            "export_ms": 1000 * (time.perf_counter() - t1),
                            "pool_bytes": 4 * len(pool)}
        out.ShallowCopy(self._surface)
        self.timings["tack_ms"] = 1000 * (time.perf_counter() - t0)
        return 1
