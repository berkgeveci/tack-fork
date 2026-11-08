# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tack-core[cpu]>=0.8,<0.9",
#   "tack-vis>=0.8,<0.9",
# ]
# [tool.paraview.optional]
# cuda = ["tack-core[cuda]>=0.8,<0.9"]
# hip = ["tack-core[hip]>=0.8,<0.9"]
# metal = ["tack-core[metal]>=0.8,<0.9"]
# ///
# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tack's filters in ParaView, every one at one go.

Load this file as a plugin (Tools > Manage Plugins > Load New, or
``PV_PLUGIN_PATH``). Its packages come from the block above: on first load
ParaView makes an environment for them (after asking, with the GPU backends'
packages offered where their runtimes are found) and finds it again
afterwards. The filters themselves are made by ``tack.interop.paraview`` from
the installed Tack, so a newer Tack brings its filters without a new copy of
this file. They are in the Filters menu's Tack category.
"""

from tack.interop.paraview import *  # noqa: F403

paraview_plugin_name = "Tack"
paraview_plugin_version = "0.8"
