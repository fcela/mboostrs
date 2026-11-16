"""Transformed responses: `log(rent) ~ ...` through the fit (W06/C02).

Fitting a distribution to `log(rent)` by hand meant fabricating a `log_rent`
column, inverting quantiles with `exp()` by hand, remembering that `exp(mu)`
is a median rather than a mean, and adding the Jacobian `+ log(rent)` to the
negative log density on the euro scale. The formula may now carry the
transform on the response side, and the model carries the step: quantiles
invert exactly, densities carry the change-of-variables factor, and the
scores read on the scale the caller measured.

The response whitelist is the invertible single-column transforms — `log`,
`log1p`, `sqrt`, `exp` and the affine `scale`/`center` — because everything
downstream needs to come back. Clamps and `I()` are refused.
"""

import numpy as np
import pytest

import mboostrs as mb

scipy_stats = pytest.importorskip("scipy.stats")


def _lognormal_frame(seed=5, n=300):
    """Rent with a smooth log-mean and constant log-scale: lognormal rent."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    rent = np.exp(1.0 + 0.3 * x + 0.4 * rng.normal(size=n))
    return {"rent": rent, "x": x}, rent


def test_intervals_match_the_lognormal_reference_quantiles():
    """`prediction_interval` on the original scale equals `exp` of the
    log-scale quantiles, which equals scipy's lognormal percentiles."""
    data, rent = _lognormal_frame()
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"},
                         data, family=mb.GaussianLSS(),
                         control=mb.boost_control(200, 0.1))
    rows = fit.prediction_interval(level=0.9).to_dicts()
    parameters = fit.params(newdata=data, type="response")
    mu, sigma = np.asarray(parameters["mu"]), np.asarray(parameters["sigma"])
    for probability, key in ((0.05, "lower"), (0.5, "median"), (0.95, "upper")):
        reference = scipy_stats.lognorm.ppf(
            probability, s=sigma, scale=np.exp(mu))
        np.testing.assert_allclose([row[key] for row in rows], reference,
                                   rtol=1e-12)


def test_original_scale_scores_carry_the_jacobian():
    """`evaluate_distribution` on the original response equals the direct
    lognormal negative log density — the `+ log(rent)` term included."""
    data, rent = _lognormal_frame()
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"},
                         data, family=mb.GaussianLSS(),
                         control=mb.boost_control(200, 0.1))
    result = mb.evaluate_distribution(fit, data, data_role="training")
    parameters = fit.params(newdata=data, type="response")
    mu, sigma = np.asarray(parameters["mu"]), np.asarray(parameters["sigma"])
    direct = -scipy_stats.lognorm.logpdf(rent, s=sigma, scale=np.exp(mu))
    assert result.metrics["mean_log_score"] == pytest.approx(
        float(np.mean(direct)), abs=1e-10)
    assert any("log(rent)" in line and "original scale" in line
               for line in result.assumptions)


def test_the_score_identity_between_scales():
    """The declarative fit and the hand-derived one are the same model, and
    the original-scale NLL is the log-scale NLL plus mean(log(rent)) — the
    Jacobian term, verified as an identity between the two."""
    data, rent = _lognormal_frame()
    control = mb.boost_control(200, 0.1)
    declared = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                               "sigma": "log(rent) ~ 1"},
                              data, family=mb.GaussianLSS(), control=control)
    hand = {"log_rent": np.log(rent), "x": data["x"]}
    derived = mb.gamboostLSS({"mu": "log_rent ~ bbs(x, df=4)",
                              "sigma": "log_rent ~ 1"},
                             hand, family=mb.GaussianLSS(), control=control)
    a = declared.params(newdata=data, type="response")
    b = derived.params(newdata=hand, type="response")
    np.testing.assert_allclose(np.asarray(a["mu"]), np.asarray(b["mu"]),
                               rtol=1e-12)
    log_scale = mb.evaluate_distribution(derived, hand, response="log_rent",
                                          data_role="training")
    original = mb.evaluate_distribution(declared, data, data_role="training")
    assert original.metrics["mean_log_score"] == pytest.approx(
        log_scale.metrics["mean_log_score"] + float(np.mean(np.log(rent))),
        abs=1e-10)


def test_the_median_is_not_the_mean():
    """`prediction_interval`'s median is exp(mu); the lognormal *mean* is
    exp(mu + sigma^2/2). They differ, and the model must not conflate them."""
    data, _rent = _lognormal_frame()
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"},
                         data, family=mb.GaussianLSS(),
                         control=mb.boost_control(200, 0.1))
    parameters = fit.params(newdata=data, type="response")
    mu, sigma = np.asarray(parameters["mu"]), np.asarray(parameters["sigma"])
    rows = fit.prediction_interval(level=0.5).to_dicts()
    medians = np.asarray([row["median"] for row in rows])
    np.testing.assert_allclose(medians, np.exp(mu), rtol=1e-12)
    means = np.exp(mu + sigma ** 2 / 2)
    assert np.all(means > medians)
    assert np.mean(means / medians) == pytest.approx(
        float(np.mean(np.exp(sigma ** 2 / 2))), rel=1e-6)


def test_pit_is_calibrated_on_the_original_response():
    data, _rent = _lognormal_frame(n=400)
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1 + bbs(x, df=4)"},
                         data, family=mb.GaussianLSS(),
                         control=mb.boost_control(200, 0.1))
    pit = np.asarray(fit.pit(newdata=data))
    # A calibrated predictive distribution leaves PIT values uniform: their
    # mean is 1/2 and their spread is that of a uniform (1/12).
    assert abs(pit.mean() - 0.5) < 0.05
    assert abs(pit.var() - 1.0 / 12.0) < 0.01


def test_an_affine_response_round_trips_to_the_raw_fit():
    """`scale(rent) ~ ...` standardizes the response with its training mean
    and sd, and the interval comes back on the original scale. The two fits
    target the same conditional quantiles, but they optimize on different
    scales (different family offsets, gradient magnitudes), so at a finite
    `mstop` they agree to a few percent rather than exactly; what the test
    pins is the inverse transform (rent-scale values, not standardized ones)
    and the agreement of the two answers."""
    data, _rent = _lognormal_frame()
    raw = mb.gamboostLSS({"mu": "rent ~ bbs(x, df=4)", "sigma": "rent ~ 1"},
                         data, family=mb.GaussianLSS(),
                         control=mb.boost_control(100, 0.1))
    scaled = mb.gamboostLSS({"mu": "scale(rent) ~ bbs(x, df=4)",
                             "sigma": "scale(rent) ~ 1"},
                            data, family=mb.GaussianLSS(),
                            control=mb.boost_control(100, 0.1))
    raw_rows = raw.prediction_interval(level=0.9).to_dicts()
    scaled_rows = scaled.prediction_interval(level=0.9).to_dicts()
    raw_medians = np.asarray([row["median"] for row in raw_rows])
    scaled_medians = np.asarray([row["median"] for row in scaled_rows])
    # On the original scale, not the standardized one.
    assert scaled_medians.min() > 0.5 * raw_medians.min()
    assert np.corrcoef(scaled_medians, raw_medians)[0, 1] > 0.99
    np.testing.assert_allclose(scaled_medians, raw_medians, rtol=0.05)


def test_both_lss_frontends_and_the_standard_frontends_accept_the_transform():
    data, _rent = _lognormal_frame(n=150)
    f1 = mb.glmboostLSS({"mu": "log(rent) ~ x", "sigma": "log(rent) ~ 1"},
                        data, family=mb.GaussianLSS(),
                        control=mb.boost_control(100, 0.1))
    assert f1.prediction_interval(level=0.9) is not None
    f2 = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                         "sigma": "log(rent) ~ 1"},
                        data, family=mb.GaussianLSS(),
                        control=mb.boost_control(100, 0.1))
    assert f2.pit(newdata=data) is not None
    g = mb.gamboost("log(rent) ~ bbs(x, df=4)", data, family=mb.Gaussian(),
                    control=mb.boost_control(100, 0.1))
    # A standard model reports its fit on the transformed scale and claims
    # nothing about the original one: its mean is a mean of log rent.
    assert np.all(np.isfinite(np.asarray(g.predict(type="link"))))
    linear = mb.glmboost("log(rent) ~ x", data, family=mb.Gaussian(),
                         control=mb.boost_control(100, 0.1))
    assert np.all(np.isfinite(np.asarray(linear.predict(type="link"))))


def test_transformed_predictors_replay_on_new_rows():
    """A transformed predictor (`log(area)`) is materialized for the fit and
    replayed for new rows from the predictor recipe."""
    data, _rent = _lognormal_frame(n=150)
    data["area"] = np.exp(data["x"]) + 1.0
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(log(area), df=4)",
                          "sigma": "log(rent) ~ 1"},
                         data, family=mb.GaussianLSS(),
                         control=mb.boost_control(100, 0.1))
    fresh = {"rent": np.full(3, 100.0),
             "area": np.array([4.0, 2.5, 1.5])}
    prediction = fit.predict(newdata=fresh, type="response")
    assert np.all(np.isfinite(np.asarray(prediction["mu"])))
    medians = [row["median"] for row in
               fit.prediction_interval(newdata=fresh).to_dicts()]
    assert np.all(np.isfinite(np.asarray(medians)))


def test_non_invertible_response_transforms_are_refused():
    data, _rent = _lognormal_frame(n=80)
    with pytest.raises(ValueError, match="not invertible"):
        mb.gamboost("pmin(rent, 500) ~ bbs(x, df=4)", data,
                    family=mb.Gaussian())
    with pytest.raises(ValueError, match="not invertible"):
        mb.gamboostLSS({"mu": "I(rent / 100) ~ bbs(x, df=4)",
                        "sigma": "I(rent / 100) ~ 1"},
                       data, family=mb.GaussianLSS())


def test_the_artifact_round_trips_the_response_transform():
    import pathlib
    import tempfile

    data, _rent = _lognormal_frame(n=150)
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"},
                         data, family=mb.GaussianLSS(),
                         control=mb.boost_control(100, 0.1))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "transformed-response.json"
        mb.save_model(fit, path)
        loaded = mb.load_model(path)
        rows = loaded.prediction_interval(level=0.9).to_dicts()
        reference = fit.prediction_interval(level=0.9).to_dicts()
        np.testing.assert_allclose([r["median"] for r in rows],
                                   [r["median"] for r in reference],
                                   rtol=1e-12)
        np.testing.assert_allclose(np.asarray(loaded.pit(newdata=data)),
                                   np.asarray(fit.pit(newdata=data)),
                                   atol=1e-12)


def test_cross_validation_runs_on_a_transformed_response():
    data, _rent = _lognormal_frame(n=150)
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"},
                         data, family=mb.GaussianLSS(),
                         control=mb.boost_control(100, 0.1))
    risk = fit.cvrisk(B=3, seed=1)
    assert len(risk.optimal_mstop) == 2
