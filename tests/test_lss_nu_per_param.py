"""Per-parameter LSS learning rates: upstream's ``boost_control(nu = c(...))``.

Upstream gamboostLSS resolves a ``nu`` request in ``mboostLSS_fit`` through
``check()``: a scalar is recycled, an unnamed length-``k`` vector is read in
family order, and a named vector must name every parameter and is reordered to
family order. mboostrs follows the same contract through the fit-level ``nu=``
argument on the LSS entry points: ``None`` keeps the control's scalar, a float
recycles, a list is family order, and a ``{param_name: float}`` dict names
every parameter. Range validation (``0 < nu <= 1``) happens in the engine.
"""

import pickle
import warnings

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = rng.normal(1 + 0.8 * x, np.exp(0.5 * x))
    return {"y": y, "x": x}


@pytest.fixture(scope="module")
def control():
    return mb.boost_control(40, 0.1)


def test_scalar_nu_recycles_to_the_control_path(data, control):
    default = mb.glmboostLSS("y ~ bols(x)", data, control=control)
    recycled = mb.glmboostLSS("y ~ bols(x)", data, control=control, nu=0.1)
    assert np.array_equal(default.risk, recycled.risk)
    for name in default.param_names:
        assert np.array_equal(default.fitted()[name], recycled.fitted()[name])


def test_dict_and_list_are_two_spellings_of_family_order(data, control):
    by_name = mb.glmboostLSS(
        "y ~ bols(x)", data, control=control, nu={"mu": 0.05, "sigma": 0.2})
    by_order = mb.glmboostLSS("y ~ bols(x)", data, control=control,
                              nu=[0.05, 0.2])
    assert np.array_equal(by_name.risk, by_order.risk)


def test_a_smaller_mu_rate_changes_the_trajectory(data, control):
    default = mb.glmboostLSS("y ~ bols(x)", data, control=control)
    slow_mu = mb.glmboostLSS("y ~ bols(x)", data, control=control,
                             nu={"mu": 0.05, "sigma": 0.2})
    assert default.risk[1] > slow_mu.risk[1]
    assert not np.allclose(default.risk, slow_mu.risk)


def test_missing_parameter_is_refused(data, control):
    with pytest.raises(ValueError, match="nu dict is missing"):
        mb.glmboostLSS("y ~ bols(x)", data, control=control, nu={"mu": 0.05})


def test_wrong_length_is_refused(data, control):
    with pytest.raises(ValueError, match="one entry per parameter"):
        mb.glmboostLSS("y ~ bols(x)", data, control=control, nu=[0.1])


def test_out_of_range_rate_is_refused_by_the_engine(data, control):
    with pytest.raises(ValueError, match=r"0 < nu <= 1"):
        mb.glmboostLSS("y ~ bols(x)", data, control=control, nu=[0.1, 1.5])


def test_setting_travels_through_cvrisk_and_stabsel(data, control):
    folds = mb.cv_weights(120, B=3, type="kfold", seed=1)
    grid = mb.diagonal_grid(40, 2, length_out=3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        risk_plain = np.asarray(mb.cvrisk_glmboostLSS(
            "y ~ bols(x)", data, control=control, folds=folds,
            grid=grid).risk_matrix)
        risk_slow = np.asarray(mb.cvrisk_glmboostLSS(
            "y ~ bols(x)", data, control=control, folds=folds, grid=grid,
            nu={"mu": 0.05, "sigma": 0.2}).risk_matrix)
    assert not np.allclose(risk_plain, risk_slow)
    hat_plain = mb.stabsel_glmboostLSS(
        "y ~ bols(x)", data, control=control, cutoff=0.8, q=1, PFER=1.5, B=4).phat
    hat_slow = mb.stabsel_glmboostLSS(
        "y ~ bols(x)", data, control=control, cutoff=0.8, q=1, PFER=1.5, B=4,
        nu={"mu": 0.05, "sigma": 0.2}).phat
    assert not np.allclose(hat_plain, hat_slow)


def test_gamboostlss_accepts_per_parameter_nu(data, control):
    default = mb.gamboostLSS("y ~ bbs(x, df=3)", data, control=control)
    slow_mu = mb.gamboostLSS("y ~ bbs(x, df=3)", data, control=control,
                             nu=[0.05, 0.2])
    assert default.risk[1] > slow_mu.risk[1]
    assert not np.allclose(default.risk, slow_mu.risk)

def test_per_parameter_nu_survives_slicing(data, control):
    """Slicing refits, so the per-parameter rates must reach that refit.

    `update()` carried the control's scalar `nu` but not the resolved
    per-parameter vector, so `model[:k]` silently fitted every parameter at the
    scalar rate.
    """
    tuned = mb.glmboostLSS("y ~ bols(x)", data, control=control,
                           nu={"mu": 0.1, "sigma": 0.02})
    truncated = mb.boost_control(20, control.nu)
    direct = mb.glmboostLSS("y ~ bols(x)", data, control=truncated,
                            nu={"mu": 0.1, "sigma": 0.02})
    scalar = mb.glmboostLSS("y ~ bols(x)", data, control=truncated)

    for model in (tuned[20], tuned.update(mstop=20)):
        for name in direct.param_names:
            assert np.allclose(model.fitted()[name], direct.fitted()[name]), name
        assert not np.allclose(model.fitted()["sigma"], scalar.fitted()["sigma"])


@pytest.mark.parametrize("entry, formula", [
    (mb.glmboostLSS, "y ~ bols(x)"),
    (mb.gamboostLSS, "y ~ bols(x) + bbs(x, df=3, center=True)"),
])
@pytest.mark.parametrize("method", ["cyclic", "noncyclic"])
@pytest.mark.parametrize("nu", [{"mu": 0.05, "sigma": 0.2}, 0.2])
def test_prediction_replay_preserves_parameter_rates(data, control, tmp_path,
                                                    entry, formula, method, nu):
    """Prediction on new rows must replay the path used to produce fitted values.

    The motorcycle case exposed nearly zero-width new-data centiles while its
    fitted centiles were sensible: stores replayed every parameter at the
    control's scalar rate. Check reordered rows and both serialization routes.
    """
    fit = entry(formula, data, control=control, method=method, nu=nu)
    indices = np.array([17, 3, 81, 0])
    newdata = {"x": data["x"][indices]}
    expected = fit.fitted(type="link")
    expected_centiles = fit.centiles(probs=[0.05, 0.5, 0.95])
    artifact = mb.save_model(fit, tmp_path / "parameter-rates.json")
    for model in (fit, pickle.loads(pickle.dumps(fit)), mb.load_model(artifact)):
        predictions = model.predict(newdata=newdata, type="link")
        for parameter in fit.param_names:
            np.testing.assert_allclose(
                predictions[parameter], expected[parameter][indices],
                rtol=1e-11, atol=1e-11,
            )
        centiles = model.centiles(newdata=newdata, probs=[0.05, 0.5, 0.95])
        for column in ("q5", "q50", "q95"):
            np.testing.assert_allclose(
                np.asarray(centiles[column]), np.asarray(expected_centiles[column])[indices],
                rtol=1e-11, atol=1e-11,
            )


@pytest.mark.parametrize("entry", [mb.glmboostLSS, mb.gamboostLSS])
def test_reported_coefficients_use_parameter_rates(data, control, entry):
    """A linear parameter's coefficients and offset reconstruct its predictor."""
    fit = entry("y ~ bols(x)", data, control=control, nu={"mu": 0.05, "sigma": 0.2})
    for parameter in fit.param_names:
        coefficients = fit.coef(parameter=parameter)
        if entry is mb.glmboostLSS:
            eta = (fit.offset_[parameter] + coefficients.get("(Intercept)", 0.0)
                   + coefficients.get("x", 0.0) * data["x"])
        else:
            intercept, slope = coefficients["x"]
            eta = coefficients["(offset)"] + intercept + slope * data["x"]
        np.testing.assert_allclose(eta, fit.fitted(parameter=parameter), atol=1e-11)


def test_partial_effects_and_path_norms_use_parameter_rates(data, control):
    rates = {"mu": 0.05, "sigma": 0.2}
    fit = mb.gamboostLSS("y ~ bols(x) + bbs(x, df=3, center=True)", data,
                        control=control, nu=rates)
    for parameter in fit.param_names:
        for newdata in (None, data):
            # The learners share a column name, so select each by public index.
            effects = [fit.partial_effects(parameter, which=index, newdata=newdata)[1]
                       for index in range(len(fit.base_learner_names(parameter)))]
            eta = fit.offset_[parameter] + sum(effects)
            np.testing.assert_allclose(eta, fit.fitted(parameter=parameter), atol=1e-11)
    for row in fit.path_frame(output="records"):
        assert row["scaled_coefficient_norm"] == pytest.approx(
            rates[row["parameter"]] * row["coefficient_norm"]
        )
