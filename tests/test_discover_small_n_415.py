"""Contrasts of dropped terms in ``discover`` (issue #415, small-sample coverage).

Backward elimination in the mean stage can drop a predictor's term, and before
this change ``result.contrast`` then returned an exact 0 with a zero-width
interval. On the claims benchmark at 1,500 rows, the spatial query was missed
because both coordinates were dropped on several draws. A dropped term's
contrast is now read from the REML refit of the model before elimination, and
its interval is widened to include 0, with the note "not selected".
"""

import importlib
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

discover_module = importlib.import_module("mboostrs.discover")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def frame_with_noise_column():
    """A smooth effect of ``x`` and a shift of ``z``; ``junk`` has no effect."""
    rng = np.random.default_rng(11)
    n = 400
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-1, 1, n)
    junk = rng.normal(size=n)
    y = np.sin(2 * x) + 0.8 * z + rng.normal(0, 0.4, n)
    return pl.DataFrame({"x": x, "z": z, "junk": junk, "y": y})


@pytest.fixture(scope="module")
def result(frame_with_noise_column):
    return _quiet(mb.discover, frame_with_noise_column, "y", seed=0, spread=False,
                  interactions=False)


def test_a_dropped_term_is_not_in_the_final_model(result):
    # The fixture is built so that elimination drops the noise column. The
    # assertions below need that to hold, so the test says so if it does not.
    assert "junk" not in result.formula, result.formula
    dropped = [e for e in result.log if e["stage"] == "mean effects"
               and e["candidate"] and "junk" in e["candidate"] and not e["adopted"]]
    assert dropped, "elimination should log the dropped noise column"


def test_a_dropped_term_contrast_is_marked_not_selected_and_is_not_exact_zero(result):
    contrast = result.contrast("junk", -1.0, 1.0)
    assert "not selected" in contrast.note
    assert contrast.estimate != 0.0
    assert contrast.lower is not None and contrast.upper is not None
    assert contrast.lower < contrast.upper


def test_a_dropped_term_interval_includes_zero(result):
    contrast = result.contrast("junk", -1.0, 1.0)
    assert contrast.lower <= 0.0 <= contrast.upper


def test_a_dropped_term_estimate_is_the_refit_contrast_of_the_full_model(result):
    frame = result._frame
    ref = discover_module._set_value(frame, "junk", -1.0)
    alt = discover_module._set_value(frame, "junk", 1.0)
    expected = discover_module._contrast_gam(result._dropped_refit, ref, alt, "junk",
                                             -1.0, 1.0, result.level)
    contrast = result.contrast("junk", -1.0, 1.0)
    assert contrast.estimate == pytest.approx(expected.estimate)


def test_a_selected_term_contrast_is_unchanged_by_the_dropped_refit(result):
    contrast = result.contrast("x", -1.0, 1.0)
    assert "not selected" not in contrast.note
    assert contrast.lower <= contrast.estimate <= contrast.upper


def test_a_variable_the_draft_never_had_returns_exact_zero(frame_with_noise_column):
    # `candidates` keeps junk out of the draft, so no term for it is ever fitted.
    fit = _quiet(mb.discover, frame_with_noise_column, "y", seed=0, spread=False,
                 interactions=False, candidates={"mu": ["x", "z"]})
    contrast = fit.contrast("junk", -1.0, 1.0)
    assert contrast.estimate == 0.0
    assert (contrast.lower, contrast.upper) == (0.0, 0.0)
    assert "not selected" in contrast.note
