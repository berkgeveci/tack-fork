# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``TackVolume``: a VTK algorithm (Python) that turns a volume into a ``vtkPolyData``
carrying what its field *is* on the GPU, for a mapper to volume render
(``volume_mapper.VolumeMapper``):

- the domain's bounding box as triangles (what the mapper rasterizes);
- Tack's packed buffer (field data ``TackPool``, 32-bit words);
- ``tack_sample(pool, p)`` as Slang over that buffer (field data ``TackEvaluation``);
- the bounds (``TackBounds``), the value range (``TackRange``) and the smallest cell
  size (``TackCellSize``);
- the macrocells for empty-space skipping: their grid (``TackMacroGrid``: origin,
  size, dimensions) and the values each can take (``TackMacroLow``,
  ``TackMacroHigh``), made by a Tack kernel over the cells
  (``volume_tack.macro_grid``).

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
    """The input's cell array ``array`` as a volume to render (see the module).

    For an overlapping AMR, what the client is given can be less than the input:

    - ``SetKeep(keep)``: the boxes held, per level (``volume_tack.select_boxes``), all
      by default; coarse cells stand in for the finer boxes left out;
    - ``SetAverageDown(True)``: the stand-ins hold their children's means (Tack's
      ``average_down``, once per input) rather than the input's coarse values;
    - ``SetQuantize(bits, scale)``: the values as 16- or 8-bit codes, on a linear or
      logarithmic scale.

    The value range (``TackRange``) is the whole input's, whatever is held, so that a
    transfer function's table does not move between levels of detail."""

    def __init__(self, array="values", blend=1, macrocells=64):
        super().__init__(nInputPorts=1, inputType="vtkDataObject", nOutputPorts=1,
                         outputType="vtkPolyData")
        self.array = array
        self.blend = blend
        self.macrocells = macrocells          # at most, along the longest axis
        self.keep = None
        self.average = False
        self.quantize = (None, "linear")
        self.histograms = False
        self._key = None
        self._converted = {}
        self.timings = {}

    def SetBlend(self, blend):
        self.blend = int(blend)
        self.Modified()

    def SetKeep(self, keep):
        self.keep = None if keep is None else [np.asarray(k, int) for k in keep]
        self.Modified()

    def SetAverageDown(self, average):
        self.average = bool(average)
        self.Modified()

    def SetHistograms(self, histograms):
        """Stand-in cells carry histograms of the finer data they stand in for
        (``volume_tack.HistogramAMRVolume``); needs ``SetKeep``."""
        self.histograms = bool(histograms)
        self.Modified()

    def SetQuantize(self, bits, scale="linear"):
        self.quantize = (bits, scale)
        self.Modified()

    def _hierarchy(self, data):
        """The input as an ``AMRDataSet`` (averaged down if asked) and its value range,
        made once per input."""
        from tack.interop.vtk import vtk_to_dataset

        key = (data.GetMTime(), self.array, self.average)
        if self._converted.get("key") != key:
            amr = vtk_to_dataset(data)
            if self.average:
                from tack.dataset.algorithms.amr_ghosts import average_down

                average_down(amr, self.array)
            lows, highs = [], []
            for _, piece in amr:
                values = piece.fields[self.array].values.numpy()
                lows.append(values.min())
                highs.append(values.max())
            self._converted = {"key": key, "amr": amr,
                               "range": (float(min(lows)), float(max(highs)))}
        return self._converted["amr"], self._converted["range"]

    def _full_ranges(self, data, bounds, coarse):
        """The macrocells' value ranges over every box of the input, made once."""
        key = (self._converted["key"], self.macrocells)
        if getattr(self, "_full", (None,))[0] != key:
            amr, _ = self._hierarchy(data)
            full = volume_tack.amr_volume(amr, self.spacings, bounds[::2], self.array)
            _, lows, highs = volume_tack.macro_grid(full, full.host["cells"], bounds, coarse,
                                                    self.macrocells)
            self._full = (key, lows.numpy(), highs.numpy())
        return self._full[1], self._full[2]

    def _sampler(self, data):
        """The input as a Tack sampler, and its bounds, cells held, value range,
        smallest cell and level 0's cell."""
        if isinstance(data, vtkImageData):
            values = np.asarray(data.GetCellData().GetArray(self.array)).astype(np.float32)
            dims = [d - 1 for d in data.GetDimensions()]
            spacing = data.GetSpacing()
            box = volume_tack.CellBox(tack.array(values.reshape(-1)), dims, data.GetOrigin(),
                                      spacing)
            return (box, data.GetBounds(), len(values.reshape(-1)),
                    (float(values.min()), float(values.max())), min(spacing), spacing)
        if isinstance(data, vtkOverlappingAMR):
            amr, value_range = self._hierarchy(data)
            spacings = []
            for level in range(data.GetNumberOfLevels()):
                h = [0.0, 0.0, 0.0]
                data.GetSpacing(level, h)
                spacings.append(h)
            # vtkOverlappingAMR's GetOrigin() and GetBounds() return pointers in Python
            # (VTK 9.7 and master); the filling form works. Level 0 covers the domain.
            bounds = [0.0] * 6
            data.GetBounds(bounds)
            bits, scale = self.quantize
            if self.histograms and self.keep is not None:
                volume = volume_tack.histogram_volume(amr, spacings, bounds[::2], self.array,
                                                      self.keep, self.blend)
            else:
                volume = volume_tack.amr_volume(amr, spacings, bounds[::2], self.array,
                                                self.blend, self.keep, bits, scale)
            self.amr, self.spacings = amr, spacings
            held = volume.host
            return (volume, tuple(bounds), held["cells"], value_range,
                    min(spacings[held["levels"] - 1]), spacings[0])
        raise TypeError(f"TackVolume takes an image or an overlapping AMR, not "
                        f"{data.GetClassName()}")

    def RequestData(self, request, inInfo, outInfo):
        import time

        t0 = time.perf_counter()
        data = vtkDataObject.GetData(inInfo[0])
        out = vtkPolyData.GetData(outInfo)
        keep = None if self.keep is None else tuple(k.tobytes() for k in self.keep)
        key = (data.GetMTime(), self.array, self.blend, keep, self.average, self.quantize,
               self.histograms)
        if key != self._key:
            box, bounds, cells, value_range, cell, coarse = self._sampler(data)
            t1 = time.perf_counter()
            grid, lows, highs = volume_tack.macro_grid(box, cells, bounds, coarse,
                                                       self.macrocells)
            if self.keep is not None:
                # What the macrocells can take is the full data's: the boxes held
                # alone would miss the peaks of the finer ones they stand in for, and
                # skip where those show. Joined with the held boxes' (a stand-in's tent
                # reaches further than its children's).
                full_lows, full_highs = self._full_ranges(data, bounds, coarse)
                lows = tack.array(np.minimum(lows.numpy(), full_lows))
                highs = tack.array(np.maximum(highs.numpy(), full_highs))
            t2 = time.perf_counter()
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
            kind = vtkStringArray()
            kind.SetName("TackKind")
            kind.InsertNextValue("mixture" if isinstance(box, volume_tack.HistogramAMRVolume)
                                 else "scalar")
            fields.AddArray(kind)
            if isinstance(box, volume_tack.HistogramAMRVolume):
                fields.AddArray(_floats("TackHistogram", (box.hlo, box.hstep)))
            fields.AddArray(_floats("TackBounds", bounds))
            fields.AddArray(_floats("TackRange", value_range))
            fields.AddArray(_floats("TackCellSize", (cell,)))
            fields.AddArray(_floats("TackMacroGrid", (*grid["origin"], *grid["size"],
                                                      *grid["dims"])))
            for name, array in (("TackMacroLow", lows), ("TackMacroHigh", highs)):
                out_array = vtkFloatArray(array.numpy())
                out_array.SetName(name)
                fields.AddArray(out_array)
            self._key = key
            self.timings = {"convert_ms": 1000 * (t1 - t0), "macro_ms": 1000 * (t2 - t1),
                            "export_ms": 1000 * (time.perf_counter() - t2),
                            "pool_bytes": 4 * len(pool), "macrocells": grid["dims"],
                            "cells": cells}
        out.ShallowCopy(self._surface)
        self.timings["tack_ms"] = 1000 * (time.perf_counter() - t0)
        return 1
