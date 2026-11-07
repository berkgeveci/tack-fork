# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""tack-vis's deprecations (``tack._deprecation``), declared here so the registry is
complete without importing what they deprecate."""

from tack._deprecation import deprecated

#: The split reading of a grid with polyhedra beside shaped cells (#307): two pieces
#: whose surfaces differ where a hexahedron and a polyhedron share a face that is
#: not planar, leaving slivers no piece holds (#337). One topology holds both now.
SPLIT = deprecated(
    'vtk_to_dataset(..., polyhedral="split")', since="0.8", removed_in="0.9",
    use="the default reading, the polyhedra beside the shaped cells in one "
        "UnstructuredTopology (polyhedral=True reads every cell as a polyhedron)",
    package="tack-vis")
