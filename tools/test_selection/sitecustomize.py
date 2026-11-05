# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Subprocesses of a depmap run record into the parent test file's map too.
import os

if os.environ.get("DEPMAP_TEST") and os.environ.get("DEPMAP_OUT"):
    import atexit
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import depmap_core
    depmap_core.start()
    atexit.register(depmap_core.dump, "sub")

# Then the interpreter's own sitecustomize (Homebrew's), which this one shadows.
import runpy as _runpy
import sysconfig as _sysconfig

_own = os.path.join(_sysconfig.get_paths()["stdlib"], "sitecustomize.py")
if os.path.exists(_own):
    _runpy.run_path(_own, run_name="sitecustomize")
