# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tack's filters as ParaView proxies, for a Python plugin to load at one go.

Every wrapper of ``tack.interop.vtk.algorithms`` with a panel (``_panels.PANELS``)
is a proxy here, ``Tack<Name>``: the wrapper with a setter for each property and
its ServerManager XML, made at import from the wrapper's parameters and the panel.
A plugin that imports them all registers them all::

    # /// script
    # dependencies = ["tack-vis"]
    # ///
    from tack.interop.paraview import *  # noqa: F403

    paraview_plugin_name = "Tack"

ParaView's loader collects the proxies from the plugin module's namespace, and
each process running a filter imports its class from this module. The filters
are in the Filters menu's Tack category; each has a Device property (where the
input's memory is, or a device found on this machine). Nothing here imports
ParaView.
"""

from tack.interop.vtk import algorithms as _algorithms

from ._generate import proxy
from ._panels import PANELS

__all__ = []

for _name, _panel in PANELS.items():
    _proxy = proxy(f"Tack{_name}", getattr(_algorithms, _name), _panel)
    globals()[_proxy.__name__] = _proxy
    __all__.append(_proxy.__name__)

del _name, _panel, _proxy

__all__ = tuple(__all__)
