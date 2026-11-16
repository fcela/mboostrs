"""Conformal results state the method-specific bound and assumptions."""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.conformal import ConformalInterval, _cv_plus_guaranteed_coverage


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(12)
    x = rng.uniform(-2, 2, 200)
    y = np.sin(x) + rng.normal(scale=.2 + .2 * (x + 2), size=200)
    return mb.gamboost("y ~ bbs(x)", {"x": x, "y": y}, control=mb.boost_control(50))


@pytest.mark.parametrize("method", ["split", "cqr"])
def test_split_methods_report_the_nominal_finite_sample_bound(fit, method):
    result = fit.conformal_interval(method=method, level=.9, seed=1)
    assert result.guaranteed_coverage == .9
    assert "finite-sample marginal" in result.guarantee
    assert any("fixed without calibration responses" in item for item in result.assumptions)
    assert result.n_training == 100


def test_cv_plus_reports_its_lower_bound_and_variable_width(fit):
    query = {"x": np.linspace(-1.9, 1.9, 40)}
    result = fit.conformal_interval(newdata=query, method="cv+", level=.9,
                                   n_folds=5, seed=1)
    expected = 1 - .2 - min(2 * (1 - 1 / 5) / (40 + 1), (1 - 5 / 200) / 6)
    assert result.guaranteed_coverage == pytest.approx(expected)
    assert result.guaranteed_coverage < result.level
    assert result.n_folds == 5
    assert np.ptp(result.width) > 1e-6


def test_unbalanced_cv_plus_does_not_invent_a_balanced_fold_bound():
    assert _cv_plus_guaranteed_coverage(101, 5, .9) is None


def test_cv_plus_balances_its_folds_and_reports_the_bound_for_the_rows_kept():
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 2, 203)
    fit = mb.gamboost("y ~ bbs(x)", {"x": x, "y": np.sin(x) + rng.normal(size=203)},
                      control=mb.boost_control(30))
    result = fit.conformal_interval(method="cv+", level=.9, n_folds=10, seed=2)
    assert result.n_set_aside == 3
    assert result.n_calibration == 200
    assert result.guaranteed_coverage == pytest.approx(_cv_plus_guaranteed_coverage(200, 10, .9))
    assert "3 rows set aside" in result.calibration_source


def _draw(rng, n):
    x = rng.uniform(-2, 2, n)
    return {"x": x, "y": np.sin(x) + 0.3 * rng.standard_t(3, size=n)}


def test_held_out_calibration_uses_the_fitted_model_and_the_order_statistic():
    rng = np.random.default_rng(5)
    train, calibration = _draw(rng, 150), _draw(rng, 39)
    fit = mb.gamboost("y ~ bbs(x)", train, control=mb.boost_control(60))
    result = fit.conformal_interval(calibration=calibration, level=.9)
    scores = np.sort(np.abs(calibration["y"] - fit.predict(newdata=calibration)))
    q = scores[int(np.ceil(40 * .9)) - 1]
    assert result.quantile == q
    np.testing.assert_allclose(result.lower, fit.predict() - q)
    np.testing.assert_allclose(result.upper, fit.predict() + q)
    assert result.n_calibration == 39 and result.guaranteed_coverage == .9
    assert "held out before the model was specified" in result.calibration_source


def test_held_out_calibration_covers_new_rows_at_the_nominal_level():
    coverage = []
    for replicate in range(100):
        rng = np.random.default_rng(100 + replicate)
        train, calibration, test = _draw(rng, 150), _draw(rng, 40), _draw(rng, 200)
        fit = mb.gamboost("y ~ bbs(x)", train, control=mb.boost_control(80))
        result = fit.conformal_interval(test, calibration=calibration, level=.9)
        coverage.append(result.empirical_coverage(test["y"]))
    # The guarantee is 0.9 <= coverage <= 0.9 + 1/41 in expectation; 100
    # replicates of 200 test rows give a Monte Carlo SE near 0.005.
    assert 0.885 <= np.mean(coverage) <= 0.94


def test_held_out_calibration_refuses_refitting_methods_and_tuning(fit):
    calibration = {"x": np.linspace(-1, 1, 30), "y": np.zeros(30)}
    with pytest.raises(NotImplementedError, match="calibration="):
        fit.conformal_interval(method="cv+", calibration=calibration)
    with pytest.raises(ValueError, match="mstop='tune'"):
        fit.conformal_interval(mstop="tune", calibration=calibration)


@pytest.mark.parametrize("field,value", [
    ("lower", np.array([[0.], [1.]])),
    ("upper", np.array([0.])),
    ("predicted", np.array([np.nan, 1.])),
])
def test_result_rejects_invalid_endpoint_shapes(field, value):
    arguments = {"lower": np.array([0., 1.]), "upper": np.array([1., 2.]),
                 "predicted": np.array([.5, 1.5]), "level": .9,
                 "method": "split", "n_calibration": 20}
    arguments[field] = value
    with pytest.raises(ValueError, match=field + "|same shape"):
        ConformalInterval(**arguments)


def test_result_arrays_are_owned_and_read_only():
    source = np.array([0., 1.])
    result = ConformalInterval(source, source + 1, source + .5, .9, "split", 20)
    source[:] = 99
    np.testing.assert_array_equal(result.lower, [0., 1.])
    with pytest.raises(ValueError, match="read-only"):
        result.lower[0] = 2


def test_weight_warning_states_the_unresolved_sampling_assumption():
    x = np.linspace(-1, 1, 100)
    fit = mb.glmboost("y ~ x", {"x": x, "y": x}, weights=np.linspace(1, 2, 100))
    with pytest.warns(UserWarning, match="target population.*sampling argument"):
        fit.conformal_interval(seed=1)
