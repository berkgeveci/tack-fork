# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VTK alone: a hook colouring each cell from a StructuredBuffer<float> field of the
hook type, bound with vtkWebGPUShaderProperty.SetSlangBuffer. Then the array is
modified and drawn again (the per-step path: an upload, no new bind group)."""
import numpy as np
import vtkmodules.vtkCommonDataModel  # noqa: F401
import vtkmodules.vtkRenderingCore as rr
import vtkmodules.vtkRenderingWebGPU  # noqa: F401
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.util.numpy_support import vtk_to_numpy

MODULE = """
module cell_colors;
import slangkit_hooks;
public struct CellColors : IShaderHooks {
    StructuredBuffer<float> rgb;     // three per cell
    public override void color(inout FragmentContext f) {
        uint c = f.cellId;
        f.diffuseColor = float3(rgb[3 * c], rgb[3 * c + 1], rgb[3 * c + 2]);
    }
    public override void lit(inout FragmentContext f) {
        f.color = float4(f.diffuseColor, 1.f);   // unlit, so pixels are the buffer's
    }
};
"""

sphere = vtkSphereSource(theta_resolution=8, phi_resolution=8)()
n = sphere.number_of_cells
rng = np.random.default_rng(1)
colors = rng.integers(0, 256, (n, 3)).astype(np.float32) / 255
values = vtkFloatArray(colors.reshape(-1))
mapper = rr.vtkPolyDataMapper(input_data=sphere)
actor = rr.vtkActor(mapper=mapper)
sp = actor.GetShaderProperty()
print(type(sp).__name__)
sp.SetSlangModule(MODULE)
sp.SetSlangSpecialization("IShaderHooks", "CellColors")
sp.SetSlangBuffer("hooks.rgb", values)
ren = rr.vtkRenderer(background=(0, 0, 0))
ren.AddActor(actor)
win = rr.vtkRenderWindow(size=(300, 300), off_screen_rendering=True)
win.AddRenderer(ren)
ren.ResetCamera()


def grab():
    win.Render()
    g = rr.vtkWindowToImageFilter(input=win, input_buffer_type=3)  # RGB
    g.Update()
    img = g.GetOutput()
    return vtk_to_numpy(img.GetPointData().GetScalars()).reshape(300, 300, -1)[..., :3]


def check(img, table):
    px = img.reshape(-1, 3).astype(int)
    px = px[px.sum(1) > 0]
    want = np.round(table * 255).astype(int)
    d = np.abs(px[:, None, :] - want[None]).max(2).min(1)
    return len(px), int(d.max()), len({tuple(p) for p in px})

for i in range(actor.GetNumberOfSlangParameters()):
    print("parameter", actor.GetSlangParameterName(i), actor.GetSlangParameterType(i))
a = grab()
print("first: pixels, worst distance to a buffer colour, distinct colours", check(a, colors))
colors2 = colors[::-1].copy()
values_np = vtk_to_numpy(values)
values_np[:] = colors2.reshape(-1)
values.Modified()
b = grab()
print("after Modified: ", check(b, colors2), "changed pixels", int((a != b).any(2).sum()))
