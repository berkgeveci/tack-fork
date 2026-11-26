# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Re 100 cavity on the refined quadtree, rendered with VTK: speed per cell
with the cells' edges; and the centreline against Ghia et al. (a chart)."""
import sys

import numpy as np
import vtkmodules.vtkCommonDataModel
import vtkmodules.vtkRenderingCore as rr
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

import tack
from tack.dataset import octree_lbm as lb

tack.set_default_device("cpu")
out = sys.argv[1]
N, U, Re = 32, 0.1, 100.0
tau = 3 * U * N / Re + 0.5
band = lambda x, y, h: (y > 24) if h > 0.75 else (y > 29)
s = lb.LBM2D(lb.lbm_mesh((N, N), band, 2, periodic=(False, False)), tau, lid=U)
s.initialize(1.0, 0.0, 0.0); s.step(8000)
rho, ux, uy = s.moments(); c, side = s.centres(); keep = np.flatnonzero(s.computational())
corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.5
pts = (c[keep, None, :] + side[keep, None, None] * corners[None]).reshape(-1, 2) / N
xyz = np.zeros((len(pts), 3), np.float32); xyz[:, :2] = pts
poly = vtkPolyData()
poly.points = vtkPoints(data=vtkFloatArray(xyz))
poly.polys = vtkCellArray(offsets=np.arange(0, 4 * len(keep) + 1, 4, dtype=np.int64), connectivity=np.arange(4 * len(keep), dtype=np.int64))
speed = vtkFloatArray(np.sqrt(ux[keep] ** 2 + uy[keep] ** 2).astype(np.float32) / U)
speed.SetName("speed")
poly.cell_data.AddArray(speed); poly.cell_data.SetActiveScalars("speed")
lut = vtkLookupTable(); lut.SetHueRange(0.667, 0.0); lut.SetTableRange(0, 1); lut.Build()
mapper = rr.vtkPolyDataMapper(); mapper.SetInputData(poly); mapper.SetScalarModeToUseCellData()
mapper.SetLookupTable(lut); mapper.SetScalarRange(0, 1)
cells = rr.vtkActor(mapper=mapper)
edges_mapper = rr.vtkPolyDataMapper(); edges_mapper.SetInputData(poly); edges_mapper.ScalarVisibilityOff()
edges = rr.vtkActor(mapper=edges_mapper)
edges.GetProperty().SetRepresentationToWireframe(); edges.GetProperty().SetColor(0.1, 0.1, 0.1)
edges.GetProperty().SetLineWidth(1.0); edges.GetProperty().SetOpacity(0.5)
bar = vtkScalarBarActor(); bar.SetLookupTable(lut); bar.SetTitle("speed / lid speed"); bar.SetNumberOfLabels(3)
bar.SetPosition(0.89, 0.1); bar.SetWidth(0.09); bar.SetHeight(0.8)
ren = rr.vtkRenderer(background=(1, 1, 1)); ren.AddActor(cells); ren.AddActor(edges); ren.AddViewProp(bar)
bar.GetTitleTextProperty().SetColor(0, 0, 0); bar.GetLabelTextProperty().SetColor(0, 0, 0)
win = rr.vtkRenderWindow(size=(1000, 800), off_screen_rendering=True); win.AddRenderer(ren)
cam = ren.GetActiveCamera(); cam.SetFocalPoint(0.62, 0.5, 0); cam.SetPosition(0.62, 0.5, 3); cam.SetParallelProjection(True); cam.SetParallelScale(0.53)
win.Render()
grab = vtkWindowToImageFilter(); grab.SetInput(win); grab.Update()
w = vtkPNGWriter(); w.SetFileName(out + "/lbm_cavity_speed.png"); w.SetInputConnection(grab.GetOutputPort()); w.Write()
print("cells", len(keep), "picture written")
