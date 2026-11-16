"""mb.discover warns when elimination leaves an intercept-only mean model (#586).

On data with no mean structure, backward elimination drops every term and the
mean model becomes ``y ~ 1``. That is a valid result, but it predicts the
response mean for every row, so it must appear in ``result.warnings`` rather
than only in the log.
"""

import numpy as np
import polars as pl

import mboostrs as mb


def _no_mean_structure(seed=7, n=600):
    """x as noise, h (right-skewed) driving only the spread, so no mean term holds."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    h = np.exp(rng.normal(5.0, 1.0, n))
    sd = 0.15 + 0.6 * np.sqrt(np.exp(5.0) / h)
    y = sd * rng.normal(size=n)
    return pl.DataFrame({"x": x, "h": h, "y": y})


def test_intercept_only_mean_model_is_recorded_in_warnings():
    result = mb.discover(_no_mean_structure(), "y", seed=0)

    assert result.formula == "1"
    empty = [w for w in result.warnings if "no mean term survived" in w]
    assert len(empty) == 1
    assert "y ~ 1" in empty[0]
    assert "family Gaussian" in empty[0]


def test_the_log_note_about_the_empty_mean_model_is_kept():
    result = mb.discover(_no_mean_structure(), "y", seed=0)

    notes = [e for e in result.log if e["stage"] == "mean effects" and e["candidate"] == "1"]
    assert any("no mean term survived" in e["reason"] for e in notes)


def test_no_empty_model_warning_when_a_mean_term_survives():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(-2, 2, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    frame = pl.DataFrame({"x": x, "junk": rng.normal(size=n), "y": y})

    result = mb.discover(frame, "y", seed=0, interactions=False, spread=False)

    assert result.formula != "1"
    assert not any("no mean term survived" in w for w in result.warnings)
