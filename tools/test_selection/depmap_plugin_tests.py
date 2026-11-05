# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""pytest -p depmap_plugin_tests: which repository files each test uses.

The call phase is recorded per test (its node id); setup and teardown under
"setup::<file>", credited to every test of the file in the replay, since a
module-scoped fixture is built during whichever test needs it first."""
import json
import os

import depmap_core
import pytest

durations = {}


def pytest_configure(config):
    depmap_core.start()
    depmap_core.state["test"] = None
    depmap_core.patch_tack()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    depmap_core.switch("setup::" + item.nodeid.split("::")[0])
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    depmap_core.switch(item.nodeid)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    depmap_core.switch("setup::" + item.nodeid.split("::")[0])
    yield


def pytest_runtest_logreport(report):
    durations[report.nodeid] = durations.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    depmap_core.dump(f"pytest-{worker}")
    with open(os.path.join(depmap_core.OUT, f"durations-{worker}-{os.getpid()}.json"), "w") as f:
        json.dump(durations, f)
