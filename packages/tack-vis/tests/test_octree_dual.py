# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An octree's dual grid (study): VTK's dual grid's cells, and contours on it
VTK's HyperTreeGrid contours, closed where the leaves' voxels crack."""

import numpy as np
import pytest
from test_octrees import needs_vtk

import tack
import tack.dataset as td
from tack.dataset.octree_dual import dual_grid

try:
    from vtkmodules.vtkFiltersCore import (
        vtkCleanPolyData,
        vtkFeatureEdges,
        vtkMassProperties,
        vtkTriangleFilter,
    )
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from vtkmodules.vtkFiltersHyperTree import vtkHyperTreeGridContour, vtkHyperTreeGridToDualGrid
    from vtkmodules.vtkFiltersSources import vtkRandomHyperTreeGridSource

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    pass


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _octree(dims, depth=4, seed=7):
    source = vtkRandomHyperTreeGridSource(dimensions=dims, max_depth=depth, seed=seed,
                                          split_fraction=0.45)
    source.Update()
    octree = vtk_to_dataset(source.GetOutput(), dtype=tack.f64)
    c = octree.centers() + np.asarray(octree.origin)
    octree.fields = {"phi": tack.array(np.linalg.norm(c - 0.13, axis=1), tack.f64)}
    return octree


def _surface(polydata):
    """Polygons, area, and open and non-manifold edges of a surface, merged."""
    if polydata.GetClassName() != "vtkPolyData":
        g = vtkGeometryFilter()
        g.SetInputData(polydata)
        g.Update()
        polydata = g.GetOutput()
    clean = vtkCleanPolyData()
    clean.SetInputData(polydata)
    clean.Update()
    counts = []
    for kind in ("boundary_edges", "non_manifold_edges"):
        edges = vtkFeatureEdges(boundary_edges=False, feature_edges=False,
                                non_manifold_edges=False, manifold_edges=False)
        setattr(edges, kind, True)
        edges.SetInputData(clean.GetOutput())
        edges.Update()
        counts.append(edges.GetOutput().GetNumberOfCells())
    triangles = vtkTriangleFilter()
    triangles.SetInputData(clean.GetOutput())
    triangles.Update()
    mass = vtkMassProperties()
    mass.SetInputData(triangles.GetOutput())
    mass.Update()
    return clean.GetOutput().GetNumberOfCells(), mass.GetSurfaceArea(), *counts


@needs_vtk
@pytest.mark.parametrize("dims", [(5, 4, 4), (5, 4, 1)])
def test_the_dual_grid_has_vtks_cells(dims):
    octree = _octree(dims)
    dual = dual_grid(octree)
    reference = vtkHyperTreeGridToDualGrid()
    reference.SetInputData(dataset_to_vtk(octree))
    reference.Update()
    assert dual.num_cells == reference.GetOutput().GetNumberOfCells()
    assert dual.num_points == octree.is_leaf.sum()


@needs_vtk
@pytest.mark.parametrize("depth", [4, 5])
def test_a_contour_of_the_dual_grid_is_vtks_and_closed(depth):
    octree = _octree((5, 4, 4), depth)
    mine = _surface(dataset_to_vtk(td.contour(dual_grid(octree), "phi", 6.0)))
    contour = vtkHyperTreeGridContour()
    contour.SetInputData(dataset_to_vtk(octree))
    contour.SetInputArrayToProcess(0, 0, 0, 1, "phi")
    contour.SetNumberOfContours(1)
    contour.SetValue(0, 6.0)
    contour.Update()
    theirs = _surface(contour.GetOutput())
    assert mine[0] == theirs[0]
    assert np.isclose(mine[1], theirs[1], rtol=1e-7)
    assert mine[2] == 0                                   # no open edge: no crack
    assert mine[3] == theirs[3]
