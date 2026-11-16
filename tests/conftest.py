"""Shared test configuration and fixtures."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "r_parity: requires R + rpy2")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-r-parity"):
        return
    skip_r = pytest.mark.skip(reason="need --run-r-parity option to run")
    for item in items:
        if "r_parity" in item.keywords:
            item.add_marker(skip_r)


@pytest.fixture(autouse=True)
def _fresh_extrapolation_summaries():
    """Start each test with no extrapolation summary marked as shown.

    ``show_summary_once`` keeps a process-wide record, so a summary issued in
    one test would otherwise suppress the same wording in a later test."""
    from mboostrs._extrapolation import reset_shown_summaries

    reset_shown_summaries()
    yield


def pytest_addoption(parser):
    parser.addoption(
        "--run-r-parity",
        action="store_true",
        default=False,
        help="Run R parity tests (requires R + rpy2)",
    )


import importlib.util

HAS_RPY2 = importlib.util.find_spec("rpy2") is not None

try:
    import subprocess

    subprocess.run(["R", "--version"], capture_output=True, check=True)
    HAS_R = True
except (FileNotFoundError, subprocess.CalledProcessError):
    HAS_R = False


def r_call(fn, *args, packages=("mboost",)):
    """Call the R function ``fn`` with ``args`` and return its result.

    The test skips only when rpy2 cannot load R or when one of ``packages``
    is not installed in R. An exception raised by the R call itself is not
    caught, so an R error fails the test.
    """
    try:
        from rpy2.robjects.packages import isinstalled
    except Exception as exc:  # rpy2 missing, or R not found when rpy2 loads
        pytest.skip(f"rpy2 or R unavailable: {exc}")
    for package in packages:
        if not isinstalled(package):
            pytest.skip(f"R package {package!r} is not installed")
    return fn(*args)
