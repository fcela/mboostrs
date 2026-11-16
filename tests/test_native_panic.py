"""A Rust panic in a native call surfaces as a catchable MboostrsRuntimeError (#319)."""

import urllib.request

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs._errors import MboostrsRuntimeError
from mboostrs._workflow import Workflow, call_fit


class PanicException(BaseException):
    """Stand-in for pyo3's PanicException, which no module exports."""


def _panics(*_args, **_kwargs):
    raise PanicException("digamma requires x > 0, got 0")


def test_a_native_panic_becomes_a_runtime_error():
    with pytest.raises(MboostrsRuntimeError) as info:
        call_fit(_panics, 1, 2)
    assert info.value.code == "native_panic"
    assert info.value.subject == "_panics"
    assert isinstance(info.value.__cause__, PanicException)
    # Callers that catch RuntimeError (the autospec family probes) recover.
    assert isinstance(info.value, RuntimeError)


def test_a_native_panic_inside_a_workflow_becomes_a_runtime_error():
    from mboostrs.search_session import CostBudget

    with Workflow(CostBudget()), pytest.raises(MboostrsRuntimeError) as info:
        call_fit(_panics)
    assert info.value.code == "native_panic"


def test_other_base_exceptions_pass_through():
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        call_fit(interrupted)


MEUSE_URL = "https://raw.githubusercontent.com/filipkral/meuse/master/meuse.txt"


@pytest.fixture(scope="module")
def meuse():
    try:
        with urllib.request.urlopen(MEUSE_URL, timeout=20) as response:
            raw = response.read()
    except OSError as exc:  # offline CI
        pytest.skip(f"Meuse data unavailable: {exc}")
    frame = pl.read_csv(raw, null_values=["NA"])
    return frame.select("x", "y", "zinc", "elev", "dist", "ffreq", "soil", "lime").with_columns(
        pl.col("zinc").log().alias("log_zinc"),
        *(pl.col(c).cast(pl.Utf8).alias(c) for c in ("ffreq", "soil")),
    )


def test_autospec_on_meuse_log_zinc_completes(meuse):
    """Round-2 benchmark split 1 aborted the process from the GammaLSS probe."""
    order = np.random.default_rng(1).permutation(meuse.height)
    train = meuse[order[: int(round(meuse.height * 0.8))]].select(["log_zinc", "x", "y", "dist", "elev", "ffreq", "soil", "lime"])
    rec = mb.autospec(train, "log_zinc", budget=200, seed=0, coordinates=("x", "y"))
    assert "log_zinc ~" in rec.formula
