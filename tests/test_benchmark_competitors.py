"""The competitor benchmark must stay honest when competitors are missing."""

import numpy as np
import pytest

from benches import bench_competitors as bc


def test_missing_competitors_are_reported_not_dropped():
    """A library that is not installed must appear as a labelled row."""
    rows = bc.run_scenario(
        "probe",
        [("NotReal", "NotRealModel", lambda *a: (_ for _ in ()).throw(ImportError()))],
        bc._regression_data(n=200),
        bc._r2,
        bc.R2_FLOOR,
    )
    assert len(rows) == 1
    assert rows[0].status == "not-installed"
    assert rows[0].fit_ms is None


def test_a_fast_but_wrong_model_is_flagged_below_the_floor():
    """Timing without accuracy is not a benchmark result."""

    def useless(Xtr, ytr, Xte):
        return 0.01, 0.01, np.zeros(Xte.shape[0]), []

    rows = bc.run_scenario(
        "probe", [("Lazy", "ZeroModel", useless)],
        bc._regression_data(n=200), bc._r2, bc.R2_FLOOR,
    )
    assert rows[0].status == "below-accuracy-floor"
    assert "floor" in rows[0].note


def test_errors_are_recorded_rather_than_crashing_the_run():
    def broken(Xtr, ytr, Xte):
        raise RuntimeError("kaboom")

    rows = bc.run_scenario(
        "probe", [("Broken", "M", broken)],
        bc._regression_data(n=200), bc._r2, bc.R2_FLOOR,
    )
    assert rows[0].status == "error"
    assert "kaboom" in rows[0].note


def test_mboostrs_clears_the_floor_on_the_additive_scenario():
    Xtr, ytr, Xte, yte = bc._regression_data(n=800)
    fit_ms, pred_ms, pred, caps = bc._mboostrs_regression(Xtr, ytr, Xte)
    assert bc._r2(yte, pred) > bc.R2_FLOOR
    assert fit_ms > 0 and pred_ms > 0
    assert "per-term effects" in caps


def test_render_carries_the_comparability_caveats():
    """A number from this file must not travel without its caveats."""
    rows = bc.run_scenario(
        "probe", [("NotReal", "M", lambda *a: (_ for _ in ()).throw(ImportError()))],
        bc._regression_data(n=200), bc._r2, bc.R2_FLOOR,
    )
    text = bc.render(rows)
    assert "not comparable across libraries" in text
    assert "shrinks sharply with n" in text
    assert "not installed" in text


def test_size_sweep_is_declared_so_one_row_cannot_stand_alone():
    assert len(bc.REGRESSION_SIZES) >= 3
    assert min(bc.REGRESSION_SIZES) < max(bc.REGRESSION_SIZES) / 10


def test_statsmodels_clears_the_floor_on_the_additive_scenario():
    """statsmodels' GLMGam is the harness's fourth REGRESSION competitor;
    it must actually run and clear the same accuracy floor as the others.

    statsmodels is an optional `bench-compare` extra, not a runtime dependency,
    so this skips where it is absent — the same treatment the harness gives every
    uninstalled competitor. Asserting instead would fail the suite in any
    environment without the extra, including CI.
    """
    pytest.importorskip("statsmodels")
    Xtr, ytr, Xte, yte = bc._regression_data(n=800)
    fit_ms, pred_ms, pred, caps = bc._statsmodels_regression(Xtr, ytr, Xte)
    assert bc._r2(yte, pred) > bc.R2_FLOOR
    assert fit_ms > 0 and pred_ms > 0
    assert "per-term effects" in caps
