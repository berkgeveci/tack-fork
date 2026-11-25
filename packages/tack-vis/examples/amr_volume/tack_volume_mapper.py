# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A VTK volume mapper whose picture Tack renders.

``TackVolumeMapper`` is a ``vtkPythonWebGPUVolumeMapper`` (VTK's WebGPU renderer): once
the frame's opaque geometry is drawn VTK hands it the depth, ``tack_composite``'s
``TackRenderer`` marches the rays to it on Tack's device, and VTK blends the picture
over the frame. Nothing here is a shader, and Tack's device need not be the one VTK
draws with.

    mapper = TackVolumeMapper(amr, "mag_vorticity")
    volume = vtkVolume(mapper=mapper, property=...)   # colour, opacity, unit distance
    renderer.AddVolume(volume)

While the camera moves (the window's desired update rate at least
``interactive_rate``) the picture is made smaller (``interactive_scale``) with longer
steps, and VTK stretches it over the viewport.
"""

import time

import numpy as np
from tack_composite import TackRenderer
from vtkmodules.util.vtkWebGPUVolumeMapper import VTKPythonWebGPUVolumeMapperBase
from vtkmodules.vtkCommonCore import vtkFloatArray


class TackVolumeMapper(VTKPythonWebGPUVolumeMapperBase):
    """The cell array ``array`` of ``data`` (a ``vtkOverlappingAMR`` or a
    ``vtkImageData``), rendered by Tack. The volume's property gives the colour and
    scalar opacity functions (component 0) and the opacity's unit distance."""

    def __init__(self, data, array="values", macrocells=128, sample_distance=0.5):
        VTKPythonWebGPUVolumeMapperBase.__init__(self)
        self.tack = TackRenderer(data, array, macrocells=macrocells)
        self.SetInputDataObject(0, data)        # vtkVolume renders no mapper without one
        self.SetBounds(*self.tack.bounds)
        self.sample_distance = sample_distance
        # While the camera moves: steps of this many local cells and no shorter than
        # the bounds' diagonal over ``interactive_steps``, on a picture this much
        # smaller along each axis.
        self.interactive_rate = 1.0
        self.interactive_sample_distance = 1.0
        self.interactive_steps = 400
        self.interactive_scale = 0.5
        self.table_size = 256
        self._table = (None, None, None, None)
        self._image = None
        #: The last frame's parts in ms: Tack's march, and all of ``RenderImage``.
        self.timings = {}

    def _functions(self, volume):
        """The property's transfer functions as a table over their own range."""
        prop = volume.GetProperty()
        color, opacity = prop.GetRGBTransferFunction(0), prop.GetScalarOpacity(0)
        key = (color.GetMTime(), opacity.GetMTime(), self.table_size)
        if self._table[0] != key:
            lo = min(color.GetRange()[0], opacity.GetRange()[0])
            hi = max(color.GetRange()[1], opacity.GetRange()[1])
            self._table = (key, self.tack.table(color, opacity, lo, hi, self.table_size), lo, hi)
        return self._table[1:]

    def RenderImage(self, renderer, volume):
        t0 = time.perf_counter()
        w, h = self.GetDepthSize()
        depth = None
        if w > 0:
            depth = np.asarray(self.GetDepth()).reshape(h, w)[::-1]       # top row first
        else:
            w, h = renderer.GetSize()
        table, lo, hi = self._functions(volume)
        unit = volume.GetProperty().GetScalarOpacityUnitDistance(0) / self.tack.cell
        moving = renderer.GetRenderWindow().GetDesiredUpdateRate() >= self.interactive_rate
        options = {"sample_distance": self.sample_distance}
        size = (w, h)
        if moving:
            b = self.tack.bounds
            diagonal = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
            options = {"sample_distance": self.interactive_sample_distance,
                       "min_step": diagonal / self.interactive_steps, "jitter": 1.0}
            step = max(int(round(1.0 / self.interactive_scale)), 1)
            if step > 1 and depth is not None:
                depth = depth[::step, ::step]
                size = (depth.shape[1], depth.shape[0])
            elif step > 1:
                size = (max(w // step, 1), max(h // step, 1))
        picture = self.tack.render(renderer.GetActiveCamera(), size, table, lo, hi, depth,
                                   unit_distance=unit, aspect=w / h, **options)
        # VTK's rows start at the bottom. The array is kept until the next frame.
        self._image = np.ascontiguousarray(picture[::-1]).reshape(-1, 4)
        self.SetImage(vtkFloatArray(self._image), size[0], size[1])
        self.timings = {"march_ms": self.tack.timings["march_ms"],
                        "render_image_ms": 1000 * (time.perf_counter() - t0),
                        "size": size, "moving": moving}
