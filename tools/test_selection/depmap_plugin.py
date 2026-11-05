# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""pytest -p depmap_plugin: which repository files each test file uses."""
import os

import depmap_core
import pytest


def pytest_configure(config):
    depmap_core.start()
    depmap_core.state["test"] = None
    depmap_core.patch_tack()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    depmap_core.switch(item.nodeid.split("::")[0])
    yield


def pytest_sessionfinish(session):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    depmap_core.dump(f"pytest-{worker}")
