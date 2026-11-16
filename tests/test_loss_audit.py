"""Resampling boundaries of the development-only loss audit."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benches"))
import audit_realdata_losses as audit
from audit_refit_conditioning import duplicate_predictors


@pytest.mark.parametrize("n", [100, 8192, 11934])
def test_outer_test_rows_never_enter_audit(n):
    order = np.random.default_rng(0).permutation(n)
    indices = audit.training_indices(n, 0)
    np.testing.assert_array_equal(indices, order[:int(0.8 * n)][:5000])
    outer_test = order[int(0.8 * n):]
    assert not np.intersect1d(indices, outer_test).size
    parts = list(audit.audit_partitions(len(indices), 3, 0))
    np.testing.assert_array_equal(np.sort(np.concatenate([h for _, h in parts])),
                                  np.arange(len(indices)))
    for fitting, held in parts:
        assert not np.intersect1d(fitting, held).size
        assert len(fitting) + len(held) == len(indices)


def test_discovery_and_prediction_never_receive_audit_outcomes(monkeypatch):
    import mboostrs as mb

    fitting = np.arange(6)
    held = np.arange(6, 9)
    frame = pl.DataFrame({"x": np.arange(9.), "y": np.arange(9.) + 100})
    discovery_rows, fit_rows, predicted_columns = [], [], []

    def discover(data, response, **kwargs):
        discovery_rows.append(data["y"].to_list())
        return SimpleNamespace(formula="y ~ bols(x)", family=mb.Gaussian(),
                               control=mb.boost_control(100), mstop=100,
                               folds_description="test", fits=0, report=lambda: "test")

    class Fit:
        def predict_over_mstop(self, grid, *, newdata, **kwargs):
            predicted_columns.append(newdata.columns)
            return np.zeros((len(grid), len(newdata)))

        def predict(self, newdata):
            predicted_columns.append(newdata.columns)
            return np.zeros(len(newdata))

    def fit(formula, *, data, **kwargs):
        fit_rows.append(data["y"].to_list())
        return Fit()

    monkeypatch.setattr(mb, "autospec", discover)
    monkeypatch.setattr(mb, "gamboost", fit)
    monkeypatch.setattr(mb, "gam_refit", lambda model, **kwargs: model)
    result = audit.run_fold(frame, "y", fitting, held, 0, 0, 200)
    assert discovery_rows == [list(np.arange(6.) + 100)]
    assert fit_rows == discovery_rows * 4
    assert all(columns == ["x"] for columns in predicted_columns)
    assert len(result["variants"]) == 4
    assert all(v["status"] == "success" for v in result["variants"])


def test_invalid_predictions_are_not_successes():
    with pytest.raises(ValueError, match="finite"):
        audit.score_predictions(np.array([1., 2.]), np.array([1., np.nan]))


def test_spline_ablation_preserves_other_arguments():
    formula = "y ~ bbs(x, df=4) + bbs(z, knots=10, df=3) + bols(a)"
    assert audit.with_knots(formula, 40) == (
        "y ~ bbs(x, df=4, knots=40) + bbs(z, df=3, knots=40) + bols(a)")


def test_deduplication_uses_predictors_only_and_requires_exact_equality():
    frame = pl.DataFrame({"a": [1., 2., 3.], "b": [1., 2., 3.],
                          "c": [1., 2., 3.00000001], "y": [1., 2., 3.]})
    assert duplicate_predictors(frame, "y") == {"b": "a"}
