# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``LICSliceMapper``: a WebGPU mapper, written in Python, that draws a ``TackSlice``
output as an exact surface LIC. Users make one as any mapper::

    mapper = LICSliceMapper()
    mapper.SetInputConnection(tack_slice.GetOutputPort())
    mapper.SetScalarRange(lo, hi)            # the colour map's range
    renderer.SetSurfaceLIC(vtkWebGPUSurfaceLIC())

It subclasses ``vtkmodules.util.vtkMapper.VTKPythonMapperBase`` (``vtkPythonWebGPUMapper``):
before each frame binds its Slang resources the mapper brings its input up to date and
calls ``RequestSlang``, which, when the input or the colour range changed, sets up the
mapper's own Slang customization from the input:

- the module: the input's ``TackEvaluation`` (what the field is) and the LIC hook
  (``vtk_lic.HOOK``: how it is drawn: Newton in the pixel's cell, ``V`` and ``Temp``,
  the colour map, the LIC's inputs into the ids target);
- the input's ``TackPool`` and ``TackCell`` as the hook's buffers;
- the colour range from ``ScalarRange``, the plane's normal from the input.

Where there is no Python (a browser's VTK) the mapper is sent as a plain
``vtkWebGPUPolyDataMapper`` with that customization
(``vtkPythonWebGPUMapper.RegisterSerializationHandlers``), which draws it.
"""

import hashlib

from vtkmodules.util.vtkMapper import VTKPythonMapperBase
from vtkmodules.vtkRenderingWebGPU import vtkWebGPUShaderProperty

import vtk_lic


class LICSliceMapper(VTKPythonMapperBase):
    def __init__(self, noise_pixels=1.5):
        VTKPythonMapperBase.__init__(self)
        self.ScalarVisibilityOff()
        self.slang = vtkWebGPUShaderProperty()
        self.SetSlangShaderProperty(self.slang)
        self.uniforms = self.slang.GetFragmentCustomUniforms()
        self.uniforms.SetUniformf("hooks.noise_pixels", noise_pixels)
        self.body = None
        self.seen = None
        self.preparations = 0

    def RequestSlang(self, renderer, actor):
        poly = self.GetInput()
        if poly is None:
            return
        key = (poly.GetMTime(), tuple(self.GetScalarRange()))
        if key == self.seen:
            return
        self.seen = key
        fields = poly.GetFieldData()
        evaluation = fields.GetAbstractArray("TackEvaluation")
        pool = fields.GetArray("TackPool")
        cells = poly.GetCellData().GetArray("TackCell")
        if evaluation is None or pool is None or cells is None:
            return                      # an empty slice: nothing to draw
        body = evaluation.GetValue(0) + vtk_lic.HOOK
        if body != self.body:
            tag = hashlib.sha1(body.encode()).hexdigest()[:12]
            self.slang.SetSlangModule(f"module tack_lic_{tag};\nimport slangkit_hooks;\n\n"
                                      + body.replace("TackLIC", f"TackLIC_{tag}"))
            self.slang.SetSlangSpecialization("IShaderHooks", f"TackLIC_{tag}")
            self.body = body
        self.slang.SetSlangBuffer("hooks.pool", pool)
        self.slang.SetSlangBuffer("hooks.cells", cells)
        lo, hi = self.GetScalarRange()
        self.uniforms.SetUniformf("hooks.lo", lo)
        self.uniforms.SetUniformf("hooks.hi", hi)
        normal = fields.GetArray("TackPlaneNormal").GetTuple3(0)
        for name, value in zip("xyz", normal, strict=True):
            self.uniforms.SetUniformf(f"hooks.n{name}", value)
        self.preparations += 1
