import json
import os
import pathlib

import coverage
import pytest

durations: dict[str, float] = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    measuring = coverage.Coverage.current()
    measuring.switch_context(item.nodeid)
    yield
    measuring.switch_context("")


def pytest_runtest_logreport(report):
    durations[report.nodeid] = durations.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session):
    pathlib.Path(os.environ["TEST_MAP_DURATIONS"]).write_text(json.dumps(durations))
