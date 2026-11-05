# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Record which repository files run, per test file (shared by the pytest plugin
and the subprocess sitecustomize)."""
import inspect
import json
import os
import sys

ROOT = os.environ.get("DEPMAP_ROOT", "")
OUT = os.environ.get("DEPMAP_OUT", "")
TOOL = 4
CO_OPTIMIZED = inspect.CO_OPTIMIZED
DISABLE = sys.monitoring.DISABLE

deps = {}          # test file -> set of repo-relative files
state = {"test": os.environ.get("DEPMAP_TEST")}


def _rel(path):
    if path and path.startswith(ROOT):
        return path[len(ROOT):].lstrip("/")
    return None


def note(path):
    rel = _rel(path)
    test = state["test"]
    if rel is not None and test is not None:
        deps.setdefault(test, set()).add(rel)


def _on_start(code, offset):
    # Functions only: module and class bodies run once, at import, and would
    # credit whichever test happened to import first.
    if code.co_flags & CO_OPTIMIZED:
        note(code.co_filename)
    return DISABLE


def start():
    sys.monitoring.use_tool_id(TOOL, "depmap")
    sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, _on_start)
    sys.monitoring.set_events(TOOL, sys.monitoring.events.PY_START)


def switch(test):
    if test != state["test"]:
        state["test"] = test
        os.environ["DEPMAP_TEST"] = test
        sys.monitoring.restart_events()


def dump(name):
    if not OUT:
        return
    data = {t: sorted(s) for t, s in deps.items()}
    with open(os.path.join(OUT, f"{name}-{os.getpid()}.json"), "w") as f:
        json.dump(data, f)


def _file_of(value):
    for attr in ("func", "python_func", "__wrapped__", "__func__"):
        inner = getattr(value, attr, None)
        if inner is not None and inner is not value:
            got = _file_of(inner)
            if got:
                return got
    code = getattr(value, "__code__", None)
    if code is not None:
        return code.co_filename
    if isinstance(value, type):
        try:
            return inspect.getfile(value)
        except TypeError:
            return None
    return None


def patch_tack():
    """Credit a kernel call with the kernel's file and every file lowering resolved
    into it (inlined @tack.func, template methods), whichever test lowered it."""
    import threading

    from tack.lang import call_bindings, kernel

    lowering = threading.local()
    resolve = call_bindings.CallBindings.resolve

    def resolve_noting(self, node):
        value = resolve(self, node)
        files = getattr(lowering, "files", None)
        if files is not None:
            for v in (self.owner, value):
                f = _file_of(v) if v is not None else None
                if f:
                    files.add(f)
        return value

    lower = kernel.Kernel._lower

    def lower_noting(self, vector_arrays, texture_arrays, template_args, debug=False):
        outer = getattr(lowering, "files", None)
        lowering.files = set()
        try:
            return lower(self, vector_arrays, texture_arrays, template_args, debug)
        finally:
            found = lowering.files
            lowering.files = outer
            if outer is not None:
                outer |= found
            self.__dict__.setdefault("_depmap_files", set()).update(found)
            for t in (template_args or {}).values() if isinstance(template_args, dict) else ():
                f = _file_of(type(t))
                if f:
                    self._depmap_files.add(f)

    call = kernel.Kernel.__call__

    def call_noting(self, *args, **kwargs):
        f = _file_of(self.func)
        if f:
            note(f)
        for a in args:
            f = _file_of(type(a))
            if f:
                note(f)
        try:
            return call(self, *args, **kwargs)
        finally:
            for f in self.__dict__.get("_depmap_files", ()):
                note(f)

    call_bindings.CallBindings.resolve = resolve_noting
    kernel.Kernel._lower = lower_noting
    kernel.Kernel.__call__ = call_noting
