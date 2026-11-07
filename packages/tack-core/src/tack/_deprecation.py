# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecations: what is going away, since when, until when, and what to use.

A package declares each deprecation once, at import, in its declaration module
(``DECLARED_IN``: tack-core's here, tack-vis's in ``tack.dataset._deprecations``),
and the code it deprecates warns through it::

    SPLIT = deprecated('vtk_to_dataset(..., polyhedral="split")', since="0.8",
                       removed_in="0.9", use="the default reading",
                       package="tack-vis")
    ...
    SPLIT.warn()

``warn`` raises a ``TackDeprecationWarning``, a ``DeprecationWarning`` as NumPy's
are, at the line outside Tack that called in, so Python shows it once per such
line (its ``default`` action) and, by its default filters, in a script's own
code (``__main__``) and under pytest.

The declarations are the registry (``registry``): ``due`` lists those whose
``removed_in`` their package's version has reached, which a test refuses, so a
release cannot ship a deprecation past its date; the docs list the current
ones from it (``docs/_ext/deprecations.py``). Deprecate in one minor release and
remove in the next or later; before 1.0 a breaking change may still go without
one where keeping the old behaviour costs more than it saves, which the
CHANGELOG says. A deprecation's entry goes under ``### Deprecated``.
"""

import importlib
import os
import re
import sys
import warnings
from dataclasses import dataclass

__all__ = ["Deprecation", "TackDeprecationWarning", "deprecated", "due", "registry"]

#: The modules that declare each package's deprecations: importing them makes
#: the registry complete without importing what they deprecate (VTK's interop).
DECLARED_IN = ("tack._deprecation", "tack.dataset._deprecations")


class TackDeprecationWarning(DeprecationWarning):
    """Something Tack deprecated: it works as before until ``removed_in``, and the
    message says what to use instead."""


@dataclass(frozen=True)
class Deprecation:
    """One deprecation: ``what`` (as the caller writes it), the release that
    deprecated it (``since``), the one it may be removed in (``removed_in``),
    what to ``use`` instead, and the distribution whose versions those are."""

    what: str
    since: str
    removed_in: str
    use: str
    package: str

    @property
    def message(self):
        return (f"{self.what} is deprecated since {self.package} {self.since} and will be "
                f"removed in {self.removed_in}: use {self.use}")

    def warn(self):
        """A ``TackDeprecationWarning`` at the first line outside Tack on the stack."""
        warnings.warn(self.message, TackDeprecationWarning, stacklevel=_outside_tack())


_REGISTRY = []


def deprecated(what, *, since, removed_in, use, package="tack-core"):
    """Declare a deprecation (in its package's ``DECLARED_IN`` module, at import) and
    return it: the code it deprecates calls its ``warn()``."""
    if _version(removed_in) <= _version(since):
        raise ValueError(f"{what} is removed in {removed_in}, which must come after {since}")
    found = Deprecation(what, since, removed_in, use, package)
    _REGISTRY.append(found)
    return found


def registry():
    """Every declared deprecation, in the order declared: each package's declaration
    module imported first (one not installed is skipped)."""
    for name in DECLARED_IN:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    return tuple(_REGISTRY)


def due(versions=None):
    """The deprecations whose ``removed_in`` their package's version (``versions[package]``,
    by default the installed one) has reached: to be removed before that release."""
    from importlib.metadata import PackageNotFoundError, version

    found = []
    for d in registry():
        try:
            current = (versions or {}).get(d.package) or version(d.package)
        except PackageNotFoundError:
            continue
        if _version(current) >= _version(d.removed_in):
            found.append(d)
    return found


def _version(text):
    """A version's release numbers, ``"0.8"`` and ``"0.8.0"`` alike: ``(0, 8, 0)``."""
    numbers = [int(n) for n in re.match(r"\d+(?:\.\d+)*", str(text)).group(0).split(".")]
    return tuple(numbers + [0] * (3 - len(numbers)))


def _outside_tack():
    """The ``stacklevel`` of the first frame outside Tack's packages, from the caller of
    ``Deprecation.warn``: the line the warning is about, however many of Tack's own
    calls (decorators, helpers) lie between."""
    # The tack directories themselves (each package's portion), not what holds them:
    # installed, that is site-packages, where the caller's libraries are too.
    roots = tuple(p.rstrip(os.sep) + os.sep for p in
                  getattr(sys.modules.get("tack"), "__path__", ()))
    frame = sys._getframe(2)              # warn's caller
    level = 2
    while frame is not None and roots and frame.f_code.co_filename.startswith(roots):
        frame = frame.f_back
        level += 1
    return level
