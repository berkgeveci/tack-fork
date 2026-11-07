# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecations (``tack._deprecation``): none is past its date, and each warns at
the line that uses it, once."""

import warnings

import pytest

import tack
from tack._deprecation import Deprecation, deprecated, due, registry

pytestmark = pytest.mark.quick


def test_no_deprecation_is_past_its_release():
    """A release whose version reaches a deprecation's ``removed_in`` must remove it
    first (and its entry in the docs goes with it)."""
    late = due()
    assert not late, "remove before this release: " + "; ".join(
        f"{d.what} ({d.package} {d.removed_in})" for d in late)


def test_the_registry_holds_each_packages_declarations():
    found = registry()
    assert all(isinstance(d, Deprecation) for d in found)
    try:
        import tack.dataset  # noqa: F401
    except ImportError:
        return
    assert any(d.what == 'vtk_to_dataset(..., polyhedral="split")' for d in found)
    # Once a package reaches removed_in, due names it.
    assert any(d.package == "tack-vis" for d in due({"tack-vis": "0.9"}))


_SAMPLE = Deprecation("tack.something_old()", "0.8", "0.9", "tack.something_new()",
                      "tack-core")


def _inside_tack():
    """A function whose code is filed in Tack's own directory, as a deprecated
    function's is, two calls deep: ``warn`` passes over both."""
    import os

    namespace = {"_SAMPLE": _SAMPLE}
    source = "def inner():\n    _SAMPLE.warn()\n\ndef outer():\n    inner()\n"
    exec(compile(source, os.path.join(tack.__path__[0], "_a_deprecated_caller.py"), "exec"),
         namespace)
    return namespace["outer"]


def test_a_deprecation_warns_at_the_callers_line_once():
    import inspect

    uses_it = _inside_tack()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            line = inspect.currentframe().f_lineno + 1
            uses_it()                      # outside Tack: this line
    assert len(caught) == 1
    warning = caught[0]
    assert warning.lineno == line
    assert issubclass(warning.category, tack.TackDeprecationWarning)
    assert issubclass(warning.category, DeprecationWarning)
    assert str(warning.message) == ("tack.something_old() is deprecated since tack-core 0.8 "
                                    "and will be removed in 0.9: use tack.something_new()")
    assert warning.filename == __file__


def test_a_deprecation_is_removed_after_it_is_made():
    with pytest.raises(ValueError, match="must come after"):
        deprecated("x", since="0.9", removed_in="0.9", use="y")
