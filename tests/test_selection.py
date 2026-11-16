"""Deselection and probing: selecting terms after, or instead of, CV stopping.

On an additive simulation with five signal and five noise features, a
CV-tuned `gamboost` keeps most noise features (a known property of early
stopping tuned for prediction). Deselection and probing should recover the
signal set.
"""

import json

import numpy as np
import pandas as pd
import pytest

import mboostrs as mb


def _additive(seed=0, n=600):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 8))
    y = (0.6 * X[:, 0] + np.sin(1.5 * X[:, 1]) + 0.4 * X[:, 2] ** 2
         + (X[:, 3] > 0.5) + rng.normal(0, 0.5, n))
    data = {f"x{i + 1}": X[:, i] for i in range(8)}
    data["y"] = y
    return data


SIGNAL = {"x1", "x2", "x3", "x4"}


def _columns(terms):
    return {t.split("(")[1].split(",")[0].strip(")").strip() if "(" in t else t for t in terms}


def test_deselect_removes_noise_terms_a_cv_tuned_fit_kept():
    data = _additive()
    rhs = " + ".join(f"bbs(x{i}, df=4)" for i in range(1, 9))
    tuned = mb.tune_mstop(f"y ~ {rhs}", data).model
    result = mb.deselect(tuned)

    assert _columns(result.kept) == SIGNAL
    assert {t for t, _ in result.dropped} and all(share < 0.01 for _, share in result.dropped)
    assert set(result.model.selected) <= SIGNAL
    json.dumps(result.to_dict())


def test_deselect_on_glmboost_keeps_factor_terms_whole():
    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "cat": rng.choice(["a", "b", "c"], n)})
    frame["y"] = frame["x1"] + (frame["cat"] == "b") + rng.normal(0, 0.5, n)
    tuned = mb.tune_mstop("y ~ bbs(x1) + bbs(x2) + bols(cat)", frame).model
    result = mb.deselect(tuned, threshold=0.02)
    assert "bols(cat)" in result.kept


def test_probe_selects_the_signal_before_the_first_shadow():
    data = _additive(seed=2)
    rhs = " + ".join(f"bbs(x{i}, df=4)" for i in range(1, 9))
    result = mb.probe(f"y ~ {rhs}", data)

    assert SIGNAL <= _columns(result.selected)
    assert len(result.selected) <= 5
    assert result.first_shadow.startswith("shadow__")
    json.dumps(result.to_dict())


def test_selection_refuses_models_it_cannot_refit():
    data = _additive()
    lss = mb.gamboostLSS({"mu": "y ~ bbs(x1)", "sigma": "y ~ bbs(x1)"}, data,
                         families=mb.GaussianLSS())
    with pytest.raises(TypeError, match="gamboost models"):
        mb.gam_refit(lss)
    with pytest.raises(ValueError, match="threshold"):
        mb.deselect(mb.gamboost("y ~ bbs(x1)", data=data), threshold=0)


def test_a_fold_that_explodes_is_named_as_deciding_mstop():
    from types import SimpleNamespace

    from mboostrs.cvrisk import warn_if_one_fold_dominates

    steady = np.full((5, 50), 0.4)
    exploding = steady.copy()
    exploding[4, 10:] = np.linspace(0.4, 160.0, 40)
    with pytest.warns(RuntimeWarning, match="fold 5 of 5 .* follows that one fold"):
        warn_if_one_fold_dominates(SimpleNamespace(risk_matrix=exploding, optimal_mstop=11))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_one_fold_dominates(SimpleNamespace(risk_matrix=steady, optimal_mstop=40))


def test_autospec_reports_a_heavy_tailed_predictor_with_the_log_remedy():
    rng = np.random.default_rng(0)
    n = 400
    occupancy = np.exp(rng.normal(1.0, 0.8, n))
    occupancy[:3] = [300.0, 450.0, 600.0]          # a few extreme rows
    data = {"occupancy": occupancy, "x": rng.normal(size=n)}
    data["y"] = np.log(occupancy) + data["x"] + rng.normal(0, 0.3, n)
    rec = mb.autospec(data, "y")
    finding = next(f for f in rec.hygiene if f.column == "occupancy")
    assert finding.kind == "right-skewed predictor"
    assert "bbs(log(occupancy))" in str(finding)


def test_gam_refit_estimates_smoothness_and_noise_and_covers_the_truth():
    """REML picks about one degree of freedom for a straight line and more for
    a sine; sigma matches the noise; the 90% bands cover the true centred
    effects at close to the nominal rate."""
    rng = np.random.default_rng(3)
    n = 800
    x1, x2 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    data = {"x1": x1, "x2": x2, "y": 0.5 * x1 + np.sin(1.5 * x2) + rng.normal(0, 0.5, n)}
    tuned = mb.tune_mstop("y ~ bbs(x1, df=4) + bbs(x2, df=4)", data).model
    refit = mb.gam_refit(tuned)

    assert refit.edf["bbs(x1, df=4)"] < 2.0
    assert refit.edf["bbs(x2, df=4)"] > 3.0
    assert refit.sigma == pytest.approx(0.5, abs=0.05)
    grid = {"x1": np.linspace(-1.8, 1.8, 200), "x2": np.linspace(-1.8, 1.8, 200)}
    effects = refit.effects(grid, level=0.9)
    covered = []
    for term, truth in (("bbs(x1, df=4)", 0.5 * grid["x1"]),
                        ("bbs(x2, df=4)", np.sin(1.5 * grid["x2"]))):
        estimate, lower, upper = effects[term]
        # The fit centres each term over the training rows; compare shapes.
        shift = np.mean(estimate - truth)
        covered.append(np.mean((truth + shift >= lower) & (truth + shift <= upper)))
    assert np.mean(covered) > 0.75
    json.dumps(refit.to_dict())
    assert refit.conditional_on_selection


def test_gam_refit_refuses_what_it_cannot_refit():
    data = _additive()
    fit = mb.gamboost("y ~ bbs(x1)", data=data, family=mb.Laplace())
    with pytest.raises(NotImplementedError, match="Laplace"):
        mb.gam_refit(fit)
    tree = mb.gamboost("y ~ btree(x1) + bbs(x2)", data=data)
    with pytest.raises(NotImplementedError, match="btree"):
        mb.gam_refit(tree, which="all")


def test_gam_refit_handles_a_varying_coefficient():
    """`bbs(x, by=t)` refits as t times a centred smooth of x; the treatment
    effect at x is bols(t)'s slope plus that smooth."""
    rng = np.random.default_rng(5)
    n = 1500
    x = rng.uniform(-2, 2, n)
    t = rng.integers(0, 2, n).astype(float)
    y = np.sin(x) + (1 + 0.5 * x) * t + rng.normal(0, 0.5, n)
    data = {"x": x, "t": t, "y": y}
    tuned = mb.tune_mstop("y ~ bbs(x, df=4) + bols(t) + bbs(x, by=t, df=4)", data).model
    refit = mb.gam_refit(tuned, which="all")
    grid = np.linspace(-1.5, 1.5, 50)
    tau = (refit.predict({"x": grid, "t": np.ones(50)})
           - refit.predict({"x": grid, "t": np.zeros(50)}))
    assert np.sqrt(np.mean((tau - (1 + 0.5 * grid)) ** 2)) < 0.15


@pytest.mark.parametrize("family", ["Binomial", "Poisson", "GammaReg"])
def test_gam_refit_fits_glm_families_on_the_canonical_scale(family):
    """Penalized IRLS with REML: the refit recovers a sine on the link scale."""
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.uniform(-2, 2, n)
    eta = 0.3 + np.sin(x)
    if family == "Binomial":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
        fam = mb.Binomial()
    elif family == "Poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
        fam = mb.Poisson()
    else:
        y = rng.gamma(shape=5.0, scale=np.exp(eta) / 5.0)
        fam = mb.GammaReg()
    fit = mb.gamboost("y ~ bbs(x, df=4)", data={"x": x, "y": y}, family=fam,
                      control=mb.boost_control(300, 0.1))
    refit = mb.gam_refit(fit)
    grid = {"x": np.linspace(-1.8, 1.8, 60)}
    estimate, lower, upper = refit.effects(grid, level=0.9)["bbs(x, df=4)"]
    truth = np.sin(grid["x"])
    shift = np.mean(estimate - truth)
    assert np.sqrt(np.mean((estimate - shift - truth) ** 2)) < 0.15
    assert np.mean((truth + shift >= lower) & (truth + shift <= upper)) > 0.7
    response = refit.predict(grid, type="response")
    assert np.all(response > 0)


def _lss_data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (5, n))
    y = np.sin(2 * x[0]) + rng.normal(size=n) * np.exp(0.5 * x[1])
    return {"y": y, **{f"x{i}": x[i] for i in range(5)}}


def test_deselect_noncyclic_lss_drops_terms_per_parameter():
    data = _lss_data()
    fit = mb.glmboostLSS("y ~ x0 + x1 + x2 + x3 + x4", data=data,
                         families=mb.GaussianLSS(),
                         control=mb.boost_control(300, 0.1), method="noncyclic")
    result = mb.deselect(fit)
    assert ("mu", "x0") in result.kept
    assert ("sigma", "x1") in result.kept
    assert set(result.formula) == {"mu", "sigma"}
    assert all(share < 0.01 for _, _, share in result.dropped)
    assert result.model._method == "noncyclic"
    assert set(result.to_dict()["mstop"]) == {"mu", "sigma"}


def test_deselect_lss_leaves_an_emptied_parameter_intercept_only():
    data = _lss_data()
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x0) + bbs(x2)", "sigma": "y ~ bols(x3)"},
                         data=data, families=mb.GaussianLSS(),
                         control=mb.boost_control(200, 0.1), method="noncyclic")
    result = mb.deselect(fit, threshold=0.2)
    assert result.formula["sigma"] == "y ~ 1"
    assert result.formula["mu"] == "y ~ bbs(x0)"


def test_deselect_refuses_a_cyclic_lss_fit():
    fit = mb.glmboostLSS("y ~ x0 + x1", data=_lss_data(), families=mb.GaussianLSS())
    with pytest.raises(NotImplementedError, match="noncyclic"):
        mb.deselect(fit)


def test_gam_refit_reml_matches_mgcv_on_bodyfat():
    """mgcv: gam(DEXfat ~ s(waistcirc, bs="ps", k=24, m=c(2,2)) +
    s(hipcirc, ...) + age, method="REML") gives edf 1.00 and 1.75 and sigma
    3.999. The REML search used to score a singular corner as a
    log-determinant of minus infinity and stopped at edf 1.91 for hipcirc."""
    from mboostrs.datasets import bodyfat

    fit = mb.gamboost("DEXfat ~ bbs(waistcirc, df=4) + bbs(hipcirc, df=4) + bols(age)",
                      data=bodyfat(), control=mb.boost_control(200, 0.1))
    refit = mb.gam_refit(fit)
    assert refit.edf["bbs(waistcirc, df=4)"] == pytest.approx(1.00, abs=0.1)
    assert refit.edf["bbs(hipcirc, df=4)"] == pytest.approx(1.75, abs=0.1)
    assert refit.sigma == pytest.approx(3.999, abs=0.02)


def test_gam_refit_predicts_a_new_group_at_the_average_effect():
    rng = np.random.default_rng(3)
    b = rng.normal(size=30)
    g = np.repeat(np.arange(30), 20)
    x = rng.uniform(-1, 1, 600)
    data = {"x": x, "g": g, "y": x + b[g] + rng.normal(scale=0.3, size=600)}
    fit = mb.gamboost("y ~ bbs(x, df=4) + brandom(g)", data=data,
                      control=mb.boost_control(2000, 0.1))
    refit = mb.gam_refit(fit)
    new = {"x": np.zeros(2), "g": np.array([0, 999])}
    with pytest.raises(ValueError, match="unseen_levels='zero_effect'"):
        refit.predict(new)
    effect, lower, upper = refit.effects(new, unseen_levels="zero_effect")["brandom(g)"]
    assert effect[1] == 0.0 and lower[1] == 0.0 and upper[1] == 0.0
    assert effect[0] != 0.0


def test_gam_refit_fits_negative_binomial_counts_with_an_exposure_offset():
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, 1500)
    exposure = rng.uniform(0.2, 2.0, 1500)
    mu = exposure * np.exp(-0.5 + 0.8 * np.sin(2 * x))
    data = {"x": x, "y": rng.negative_binomial(2, 2 / (2 + mu)).astype(float)}
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.NBinomial(),
                      offset=np.log(exposure), control=mb.boost_control(1500, 0.1))
    refit = mb.gam_refit(fit)
    grid = np.linspace(-0.9, 0.9, 19)
    effect = refit.effects({"x": grid})["bbs(x, df=4)"][0]
    truth = 0.8 * np.sin(2 * grid)
    assert np.sqrt(np.mean((effect - effect.mean() - (truth - truth.mean())) ** 2)) < 0.15
    # Doubling the exposure adds log 2 to the linear predictor.
    once = refit.predict({"x": grid}, offset=np.zeros(19))
    twice = refit.predict({"x": grid}, offset=np.full(19, np.log(2)))
    assert np.allclose(twice - once, np.log(2))


def test_gam_refit_refits_a_cox_model_as_mgcv_cox_ph_does():
    """The Cox refit alternates a Breslow baseline with a Poisson step. On a
    simulated proportional-hazards sample, mgcv's `gam(time ~ s(x1, bs="ps",
    k=24, m=c(2,2)) + ..., family=cox.ph(), weights=status, method="REML")`
    started with x3 fully smoothed reaches REML score 3742.642 with edf 5.516
    1 1.002 1.782 1.001 1.731 1 1.001; its default start stops at a second
    local minimum with x3 at edf 2.60 and a higher score (3742.703). The
    refit finds the lower one."""
    from pathlib import Path

    import pandas as pd

    data = pd.read_csv(Path(__file__).parent / "fixtures" / "cox_refit.csv")
    formula = "Surv(time, status) ~ " + " + ".join(f"bbs(x{j}, df=4)" for j in range(1, 9))
    fit = mb.gamboost(formula, data=data, family=mb.CoxPH(),
                      control=mb.boost_control(300, 0.1))
    refit = mb.gam_refit(fit, which="all")
    expected = [5.516, 1, 1.002, 1.782, 1.001, 1.731, 1, 1.001]
    assert list(refit.edf.values()) == pytest.approx(expected, abs=0.02)
    assert refit.convergence["method"] == "outer"
    grid = np.linspace(-1.35, 1.35, 25)
    new = {f"x{j}": np.zeros(25) for j in range(1, 9)}
    new["x1"] = grid
    effect = refit.effects(new)["bbs(x1, df=4)"][0]
    truth = 0.8 * np.sin(1.5 * grid)
    assert np.sqrt(np.mean((effect - effect.mean() - (truth - truth.mean())) ** 2)) < 0.2


@pytest.mark.parametrize("surface_term", ["bspatial(lon, lat, knots=8)",
                                          "bbs(lon, knots=8) %X% bbs(lat, knots=8)"])
def test_gam_refit_fits_a_tensor_surface_with_one_smoothing_parameter_per_margin(surface_term):
    rng = np.random.default_rng(6)
    lon, lat = rng.uniform(0, 1, 800), rng.uniform(0, 1, 800)
    surface = np.exp(-((lon - 0.3) ** 2 + (lat - 0.7) ** 2) / 0.05)
    data = {"lon": lon, "lat": lat, "y": surface + rng.normal(scale=0.3, size=800)}
    fit = mb.gamboost(f"y ~ {surface_term}", data=data, control=mb.boost_control(500, 0.1))
    refit = mb.gam_refit(fit, which="all")
    (term,) = refit.terms
    assert len(refit.smoothing[term]) == 2
    grid = {"lon": np.array([0.3, 0.8]), "lat": np.array([0.7, 0.2])}
    estimate, lower, upper = refit.effects(grid)[term]
    assert estimate[0] - estimate[1] == pytest.approx(1.0, abs=0.25)
    assert np.all(lower < estimate) and np.all(estimate < upper)


def test_gam_refit_corrected_covariance_widens_the_intervals():
    """`corrected=True` adds the first-order uncertainty of the smoothing
    parameters: J V_rho J' is positive semi-definite, so no standard error
    shrinks. On bodyfat the uncorrected standard errors match mgcv's
    `predict(se.fit=TRUE)` to four decimals."""
    from mboostrs.datasets import bodyfat

    data = bodyfat()
    fit = mb.gamboost("DEXfat ~ bbs(waistcirc, df=4) + bbs(hipcirc, df=4) + bols(age)",
                      data=data, control=mb.boost_control(200, 0.1))
    refit = mb.gam_refit(fit)
    rows = {c: np.asarray(data[c])[:6] for c in ("waistcirc", "hipcirc", "age")}
    plain = refit.effects(rows)["bbs(hipcirc, df=4)"]
    wide = refit.effects(rows, corrected=True)["bbs(hipcirc, df=4)"]
    np.testing.assert_allclose(plain[0], wide[0])
    assert np.all(wide[2] - wide[1] >= plain[2] - plain[1] - 1e-12)
    assert np.any(wide[2] - wide[1] > plain[2] - plain[1] + 1e-6)
    np.testing.assert_allclose((plain[2] - plain[1])[:3] / (2 * 1.6448536269514722),
                               [0.7609, 1.0809, 0.5309], atol=1e-4)
    diff = refit.vcov("corrected") - refit.vcov("bayesian")
    assert np.min(np.linalg.eigvalsh((diff + diff.T) / 2)) > -1e-10


def _lss_refit_data():
    rng = np.random.default_rng(1)
    n = 600
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(6 * x1) + np.exp(-0.7 + x2) * rng.normal(size=n)
    return {"y": y, "x1": x1, "x2": x2}


def test_gamlss_refit_matches_mgcv_gaulss():
    """mgcv: gam(list(y ~ s(x1, bs="ps", k=24, m=c(2,2)) + s(x2, ...),
    ~ s(x1, ...) + s(x2, ...)), family=gaulss(), method="REML") gives edf
    6.635 1.789 3.963 1.001 and, on the grid below, means 0.24 0.934 0.102
    -0.841 -0.396 and standard deviations 0.565 0.599 0.897 1.048 1.454
    (gaulss models 1/sigma; compare the fitted distribution)."""
    data = _lss_refit_data()
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x1) + bbs(x2)", "sigma": "y ~ bbs(x1) + bbs(x2)"},
                         data=data, families=mb.GaussianLSS(),
                         control=mb.boost_control(500, 0.1))
    refit = mb.gamlss_refit(fit, which="all")
    assert refit.convergence["converged"]
    edf = [*refit.edf["mu"].values(), *refit.edf["sigma"].values()]
    assert edf == pytest.approx([6.635, 1.789, 3.963, 1.001], abs=0.03)
    grid = {"x1": np.linspace(0.05, 0.95, 5), "x2": np.linspace(0.05, 0.95, 5)}
    predicted = refit.predict(grid, type="response")
    np.testing.assert_allclose(predicted["mu"], [0.24, 0.934, 0.102, -0.841, -0.396], atol=2e-3)
    np.testing.assert_allclose(predicted["sigma"], [0.565, 0.599, 0.897, 1.048, 1.454],
                               atol=3e-3)
    assert not refit.conditional_on_selection
    json.dumps(refit.to_dict())


def test_gamlss_refit_effects_and_intervals():
    data = _lss_refit_data()
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x1)", "sigma": "y ~ bbs(x2)"}, data=data,
                         families=mb.GaussianLSS(), control=mb.boost_control(300, 0.1))
    refit = mb.gamlss_refit(fit)
    grid = {"x1": np.linspace(0.05, 0.95, 20), "x2": np.linspace(0.05, 0.95, 20)}
    effects = refit.effects(grid)
    assert set(effects) == {"mu", "sigma"}
    estimate, lower, upper = effects["sigma"]["bbs(x2)"]
    assert np.all(lower < estimate) and np.all(estimate < upper)
    # log sd rises by 0.9 over the grid (slope 1 in x2).
    assert estimate[-1] - estimate[0] == pytest.approx(0.9, abs=0.3)
    wide = refit.effects(grid, parameter="sigma", corrected=True)["bbs(x2)"]
    assert np.all(wide[2] - wide[1] >= upper - lower - 1e-12)
    with pytest.raises(ValueError, match="parameter must be one of"):
        refit.effects(grid, parameter="nu")


def test_refits_pickle_and_predict_the_same():
    import pickle

    data = _lss_refit_data()
    refit = mb.gam_refit(mb.gamboost("y ~ bbs(x1) + bbs(x2)", data=data))
    again = pickle.loads(pickle.dumps(refit))
    np.testing.assert_array_equal(again.predict(data), refit.predict(data))
    assert again.edf == refit.edf
    lss = mb.gamlss_refit(mb.gamboostLSS({"mu": "y ~ bbs(x1)", "sigma": "y ~ bbs(x2)"},
                                         data=data, families=mb.GaussianLSS()))
    lss_again = pickle.loads(pickle.dumps(lss))
    np.testing.assert_array_equal(lss_again.predict(data, parameter="sigma", type="response"),
                                  lss.predict(data, parameter="sigma", type="response"))


def test_gamlss_draws_give_centile_intervals():
    data = _lss_refit_data()
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x1)", "sigma": "y ~ bbs(x2)"}, data=data,
                         families=mb.GaussianLSS(), control=mb.boost_control(300, 0.1))
    refit = mb.gamlss_refit(fit)
    grid = {"x1": np.array([0.2, 0.5, 0.8]), "x2": np.array([0.2, 0.5, 0.8])}
    draws = refit.draws(grid, 3000, seed=3)
    assert draws["mu"].shape == (3000, 3) and np.all(draws["sigma"] > 0)
    centile = draws["mu"] + 1.2816 * draws["sigma"]
    point = (refit.predict(grid, parameter="mu", type="response")
             + 1.2816 * refit.predict(grid, parameter="sigma", type="response"))
    low, high = np.percentile(centile, [5, 95], axis=0)
    assert np.all((low < point) & (point < high))
    # The same seed gives the same draws.
    np.testing.assert_array_equal(refit.draws(grid, 10, seed=3)["mu"],
                                  refit.draws(grid, 10, seed=3)["mu"])


def test_gamlss_refit_matches_mgcv_gammals():
    """mgcv: gam(list(y ~ s(x1, bs="ps", k=24, m=c(2,2)) + s(x2, ...),
    ~ s(x1, ...) + s(x2, ...)), family=gammals(), method="REML") on this
    sample gives edf 6.019 5.156 1.001 1.000 and log-likelihood -1204.619.
    gammals parametrizes the scale differently, so the comparison is on the
    degrees of freedom and on the fitted distribution's log-likelihood."""
    from scipy.stats import gamma

    rng = np.random.default_rng(5)
    n = 800
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    mu, shape = np.exp(0.5 + np.sin(4 * x1)), np.exp(1.0 + x2)
    data = {"y": rng.gamma(shape, mu / shape), "x1": x1, "x2": x2}
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x1) + bbs(x2)", "sigma": "y ~ bbs(x1) + bbs(x2)"},
                         data=data, families=mb.GammaLSS(),
                         control=mb.boost_control(500, 0.1))
    refit = mb.gamlss_refit(fit, which="all")
    edf = [*refit.edf["mu"].values(), *refit.edf["sigma"].values()]
    assert edf == pytest.approx([6.019, 5.156, 1.001, 1.000], abs=0.01)
    fitted = refit.predict(data, type="response")
    loglik = gamma.logpdf(data["y"], a=fitted["sigma"], scale=fitted["mu"] / fitted["sigma"]).sum()
    assert loglik == pytest.approx(-1204.619, abs=0.05)


def test_gam_refit_loglik_matches_mgcv():
    """mgcv's logLik on the bodyfat REML fit is -196.6984 (Gaussian, maximum-
    likelihood variance)."""
    from mboostrs.datasets import bodyfat

    fit = mb.gamboost("DEXfat ~ bbs(waistcirc, df=4) + bbs(hipcirc, df=4) + bols(age)",
                      data=bodyfat(), control=mb.boost_control(200, 0.1))
    refit = mb.gam_refit(fit)
    assert refit.loglik == pytest.approx(-196.6984, abs=1e-3)
    assert refit.df == pytest.approx(sum(refit.edf.values()) + 2, abs=1e-8)
    assert refit.aic() == pytest.approx(-2 * refit.loglik + 2 * refit.df)


# ------------------------------------------------------ one vocabulary for "which columns"
def _two_driver_frame(seed=0, n=400, noise=5):
    rng = np.random.default_rng(seed)
    data = {f"x{i}": rng.uniform(-1, 1, n) for i in range(1, 3 + noise)}
    data["y"] = np.sin(np.pi * data["x1"]) + data["x2"] + rng.normal(0, 0.5, n)
    return data


def test_term_variables_names_the_columns_a_term_reads():
    from mboostrs.formula import term_variables

    assert term_variables("bbs(x1, df=4)") == ("x1",)
    assert term_variables("bols(x2, by=z)") == ("x2", "z")
    assert term_variables("bbs(log(x5))") == ("x5",)
    assert term_variables("bbs(a, df=3) %X% bbs(b)") == ("a", "b")
    assert term_variables("bspatial(lon, lat, df=6)") == ("lon", "lat")
    assert term_variables('bmono(x, constraint="increasing")') == ("x",)


def test_every_selection_result_reports_the_selected_columns():
    """stabsel, deselect and probe answer the same question; each result
    carries `variables`, the data columns of what it selected, whatever the
    formula's spelling of the terms."""
    data = _two_driver_frame()
    formula = "y ~ " + " + ".join(f"bbs(x{i}, df=4)" for i in range(1, 8))
    model = mb.tune_mstop(formula, data).model
    stab = mb.stabsel(model, PFER=1, cutoff=0.75, verbose=False)
    assert stab.variables == ("x1", "x2")
    assert stab.selected_names == ["x1", "x2"]
    assert mb.deselect(model).variables == ("x1", "x2")
    probing = mb.probe(model)
    assert set(probing.variables) >= {"x1", "x2"}
    assert probing.model.family.__class__.__name__ == "Gaussian"


def test_probe_takes_a_fitted_model_and_reuses_its_family():
    rng = np.random.default_rng(1)
    n = 500
    data = {f"x{i}": rng.uniform(-1, 1, n) for i in range(1, 6)}
    eta = 1.5 * np.sin(np.pi * data["x1"])
    data["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    model = mb.gamboost("y ~ .", data, family=mb.Binomial())
    probing = mb.probe(model)
    assert "x1" in probing.variables
    assert type(probing.model._family).__name__ == "Binomial"
    with pytest.raises(TypeError, match="reuses the model's data"):
        mb.probe(model, data)


def test_gam_refit_reads_bare_names_as_the_smooths_gamboost_fitted():
    """`y ~ .` fits each numeric column as a default bbs; the refit refits
    the same terms instead of refusing the bare names."""
    data = _two_driver_frame(noise=2)
    model = mb.tune_mstop("y ~ .", data).model
    refit = mb.gam_refit(mb.deselect(model).model)
    grid = {k: np.zeros(21) for k in data if k != "y"}
    grid["x1"] = np.linspace(-0.9, 0.9, 21)
    effect = np.asarray(refit.predict(grid))
    truth = np.sin(np.pi * grid["x1"])
    assert np.corrcoef(effect, truth)[0, 1] > 0.98


def test_gam_refit_holds_a_term_it_cannot_represent():
    """`unsupported="hold"` keeps a tree term at its boosted contribution
    as a known offset and refits the smooth around it; predictions include
    the held term."""
    rng = np.random.default_rng(3)
    n = 600
    data = {"x": rng.uniform(-1, 1, n), "s": rng.uniform(-1, 1, n)}
    data["y"] = np.sin(np.pi * data["x"]) + 1.0 * (data["s"] > 0.2) + rng.normal(0, 0.3, n)
    model = mb.tune_mstop("y ~ bbs(x) + btree(s)", data).model
    with pytest.raises(NotImplementedError, match="unsupported='hold'"):
        mb.gam_refit(model)
    refit = mb.gam_refit(model, unsupported="hold")
    assert refit.held == ("btree(s)",)
    assert set(refit.effects(data)) == {"bbs(x)"}
    truth = np.sin(np.pi * data["x"]) + 1.0 * (data["s"] > 0.2)
    residual = refit.predict(data) - truth
    assert np.sqrt(np.mean((residual - residual.mean()) ** 2)) < 0.15


def test_autospec_can_finish_with_a_reml_refit():
    rng = np.random.default_rng(4)
    n = 500
    data = {f"x{i}": rng.uniform(-1, 1, n) for i in range(1, 6)}
    data["y"] = np.sin(np.pi * data["x1"]) + 0.8 * data["x2"] + rng.normal(0, 0.4, n)
    rec = mb.autospec(data, "y", budget=100, seed=0)
    refit = rec.fit(data, refit="reml")
    assert isinstance(refit, mb.GamRefit)
    kept = {v for term in refit.terms for v in mb.formula.term_variables(term)}
    assert {"x1", "x2"} <= kept
    with pytest.raises(ValueError, match="refit must be"):
        rec.fit(data, refit="ml")


def _spread_moves_with_group(seed=1, n=800):
    """A linear mean with a spread 2.5 times wider in one group: autospec's
    scale probe recommends GaussianLSS with ``sigma ~ part_time``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    part_time = rng.choice([0.0, 1.0], size=n)
    sigma = np.where(part_time > 0.5, 2.5, 1.0)
    return {"y": 1.0 + 0.8 * x + sigma * rng.normal(size=n), "x": x,
            "part_time": part_time}


def test_autospec_reml_refit_of_a_distributional_recommendation():
    """A GaussianLSS recommendation refits with gamlss_refit. The route used
    to call gam_refit's GAMBoostModel path and failed with AttributeError:
    'LSSModel' object has no attribute '_learner_names'."""
    data = _spread_moves_with_group()
    rec = mb.autospec(data, "y", budget=60, mstop=60)
    assert isinstance(rec.family, mb.GaussianLSS)
    refit = rec.fit(data, refit="reml")
    assert isinstance(refit, mb.GamlssRefit)
    assert refit.convergence["converged"]
    assert any("part_time" in term for term in refit.terms["sigma"])
    sigma = refit.predict({"x": np.zeros(2), "part_time": np.array([0.0, 1.0])},
                          type="response")["sigma"]
    assert sigma[1] / sigma[0] == pytest.approx(2.5, rel=0.25)


def test_autospec_keeps_the_boosted_lss_model_when_gamlss_refit_cannot_represent_it(monkeypatch):
    """When gamlss_refit raises NotImplementedError (a btree or bmono term in
    a parameter's formula), fit(refit="reml") returns the boosted LSSModel
    with a RuntimeWarning. The raise is simulated here, not produced by a
    btree or bmono term."""
    import mboostrs.selection as selection

    data = _spread_moves_with_group()
    rec = mb.autospec(data, "y", budget=60, mstop=60)
    assert isinstance(rec.family, mb.GaussianLSS)

    def cannot_represent(model, **kwargs):
        raise NotImplementedError("btree term has no gamlss_refit counterpart")

    monkeypatch.setattr(selection, "gamlss_refit", cannot_represent)
    with pytest.warns(RuntimeWarning, match="cannot refit this distributional model"):
        result = rec.fit(data, refit="reml")
    assert not isinstance(result, mb.GamlssRefit)
    assert type(result).__name__ == "LSSModel"


def test_autospec_keeps_the_boosted_lss_model_when_the_gamlss_refit_does_not_converge(monkeypatch):
    """An unconverged gamlss_refit is not adopted: fit(refit="reml") returns
    the boosted LSSModel with a RuntimeWarning that carries the stop reason."""
    import dataclasses

    import mboostrs.selection as selection

    data = _spread_moves_with_group()
    rec = mb.autospec(data, "y", budget=60, mstop=60)
    real = selection.gamlss_refit

    def unconverged(model, **kwargs):
        refit = real(model, **kwargs)
        return dataclasses.replace(
            refit, convergence={**refit.convergence, "converged": False,
                                "stop_reason": "no_descent"})

    monkeypatch.setattr(selection, "gamlss_refit", unconverged)
    with pytest.warns(RuntimeWarning, match="no_descent"):
        result = rec.fit(data, refit="reml")
    assert not isinstance(result, mb.GamlssRefit)
    assert type(result).__name__ == "LSSModel"


def test_autospec_does_not_deselect_a_cyclic_lss_recommendation_before_refit(monkeypatch):
    """Deselection runs only for a noncyclic LSS fit. The autospec
    recommendation fits cyclically, so deselect is never called and the
    refit still completes."""
    import mboostrs.selection as selection

    data = _spread_moves_with_group()
    rec = mb.autospec(data, "y", budget=60, mstop=60)

    def must_not_run(*args, **kwargs):
        raise AssertionError("deselect was called on a cyclic LSS fit")

    monkeypatch.setattr(selection, "deselect", must_not_run)
    refit = rec.fit(data, refit="reml")
    assert isinstance(refit, mb.GamlssRefit)
    assert refit.convergence["converged"]


def test_autospec_keeps_the_boosted_model_when_the_refit_does_not_converge(monkeypatch):
    """A refit whose smoothing parameters are not at a REML optimum is not
    adopted (issue #70): the boosted model comes back with a warning naming
    the stop reason."""
    import mboostrs.selection as selection

    rng = np.random.default_rng(4)
    n = 300
    data = {f"x{i}": rng.uniform(-1, 1, n) for i in range(1, 4)}
    data["y"] = np.sin(np.pi * data["x1"]) + 0.8 * data["x2"] + rng.normal(0, 0.4, n)
    rec = mb.autospec(data, "y", budget=60, seed=0)
    real = selection.gam_refit

    def unconverged(model, **kwargs):
        import dataclasses

        refit = real(model, **kwargs)
        return dataclasses.replace(
            refit, convergence={**refit.convergence, "converged": False,
                                "stop_reason": "no_descent"})

    monkeypatch.setattr(selection, "gam_refit", unconverged)
    with pytest.warns(RuntimeWarning, match="no_descent"):
        result = rec.fit(data, refit="reml")
    assert isinstance(result, mb.GAMBoostModel)


def test_gam_refit_reports_term_tests_as_mgcv_summary_does():
    """edf, reference df, F and p-value per term. mgcv 1.9.3 on the same
    basis (bs="ps", k=24, m=c(2,2), REML) gives Ref.df 1.0003, 3.2897,
    1.0003, 5.8200; F 23.479, 10.045, 4.788, 4.067; p 9.3966e-6, 1.2716e-5,
    0.032484, 0.0019175. mgcv evaluates the tail with Davies' algorithm at an
    absolute tolerance of 2e-5, so small p-values agree to that tolerance."""
    from mboostrs.datasets import bodyfat

    fit = mb.gamboost("DEXfat ~ bbs(waistcirc) + bbs(hipcirc) + bbs(age) + bbs(kneebreadth)",
                      data=bodyfat())
    tests = mb.gam_refit(fit, which="all").tests
    expected = {"bbs(waistcirc)": (1.0003, 23.479, 9.396601e-06),
                "bbs(hipcirc)": (3.2897, 10.045, 1.271641e-05),
                "bbs(age)": (1.0003, 4.788, 0.03248401),
                "bbs(kneebreadth)": (5.8200, 4.067, 0.001917483)}
    for term, (ref_df, f_stat, p_value) in expected.items():
        assert tests[term]["ref_df"] == pytest.approx(ref_df, abs=2e-3)
        assert tests[term]["statistic"] == pytest.approx(f_stat, rel=2e-3)
        assert tests[term]["p_value"] == pytest.approx(p_value, abs=2e-5)
    summary = mb.gam_refit(fit, which="all").summary()
    assert "p_value" in str(summary) and "F" in str(summary)


def test_known_scale_term_tests_use_the_chi_square():
    rng = np.random.default_rng(8)
    n = 800
    data = {"x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n)}
    data["y"] = rng.poisson(np.exp(0.5 + np.sin(np.pi * data["x"])))
    fit = mb.gamboost("y ~ bbs(x) + bbs(z)", data, family=mb.Poisson())
    refit = mb.gam_refit(fit, which="all")
    tests = refit.tests
    assert tests["bbs(x)"]["p_value"] < 1e-10
    assert tests["bbs(z)"]["p_value"] > 0.01
    assert "chi_sq" in str(refit.summary())


def test_per_learner_prediction_of_a_string_group_column():
    """`predict(which=)` ran the recipe twice, so an encoded group label read
    as a new one and every per-learner prediction of a `brandom` term over a
    string column failed, on new rows and on the training rows."""
    import polars as pl

    rng = np.random.default_rng(0)
    n = 300
    d = pl.DataFrame({"g": rng.choice(["a", "b", "c", "d"], n), "x": rng.uniform(-1, 1, n)})
    d = d.with_columns(y=pl.Series(np.sin(3 * d["x"].to_numpy())
                                   + (d["g"] == "b").to_numpy() + rng.normal(0, 0.3, n)))
    model = mb.gamboost("y ~ brandom(g) + bbs(x)", d)
    new = np.asarray(model.predict(newdata=d.head(5), which="g")).ravel()
    fitted = np.asarray(model.predict(which="g")).ravel()[:5]
    np.testing.assert_allclose(new, fitted, atol=1e-12)


def test_distributional_deselection_reports_columns_per_parameter():
    rng = np.random.default_rng(0)
    n = 600
    d = {f"x{i}": rng.uniform(-1, 1, n) for i in range(1, 7)}
    d["y"] = np.sin(np.pi * d["x1"]) + np.exp(0.6 * d["x3"]) * rng.normal(size=n)
    rhs = " + ".join(f"bbs(x{i})" for i in range(1, 7))
    model = mb.gamboostLSS({"mu": f"y ~ {rhs}", "sigma": f"y ~ {rhs}"}, data=d,
                           families=mb.GaussianLSS(), control=mb.boost_control(1000, 0.1),
                           method="noncyclic")
    folds = mb.cv_weights(n, "kfold", 5, seed=0)
    kept = mb.deselect(model.with_mstop(model.cvrisk(folds=folds).optimal_mstop)).variables
    assert set(kept) == {"mu", "sigma"}
    assert "x1" in kept["mu"] and "x3" in kept["sigma"]


def test_centred_effects_are_the_contrast_over_the_rows():
    """`effects(centre=True)` subtracts the mean over the rows of `newdata`
    from the estimate and gives that contrast's interval."""
    import numpy as np

    import mboostrs as mb

    rng = np.random.default_rng(1)
    n = 400
    x, z = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    data = {"x": x, "z": z, "y": np.sin(3 * x) + z + rng.normal(scale=0.3, size=n)}
    refit = mb.gam_refit(mb.gamboost("y ~ bbs(x) + bbs(z)", data,
                                     control=mb.boost_control(300, 0.1)))
    new = {"x": np.linspace(-0.9, 0.9, 50), "z": np.zeros(50)}
    plain = refit.effects(new)["bbs(x)"]
    centred = refit.effects(new, centre=True)["bbs(x)"]
    np.testing.assert_allclose(centred[0], plain[0] - plain[0].mean(), atol=1e-12)
    assert abs(centred[0].mean()) < 1e-12
    assert np.all(centred[2] > centred[1])


def test_gam_refit_merges_a_linear_plus_centred_smooth_decomposition():
    """`bols(x) + bbs(x, center=TRUE)` refits as the one smooth `bbs(x)`."""
    rng = np.random.default_rng(2)
    n = 600
    x, z = rng.uniform(-2, 2, (2, n))
    data = {"x": x, "z": z, "y": np.sin(x) + 0.5 * x + 0.3 * z + rng.normal(0, .3, n)}
    split = mb.gamboost("y ~ bols(x) + bbs(x, center=TRUE, df=1) + bols(z)", data,
                        control=mb.boost_control(300))
    whole = mb.gamboost("y ~ bbs(x) + bols(z)", data, control=mb.boost_control(300))
    merged, plain = mb.gam_refit(split), mb.gam_refit(whole)
    assert merged.terms == plain.terms == ("bbs(x)", "bols(z)")
    np.testing.assert_allclose(merged.predict(data), plain.predict(data), atol=1e-10)


def test_gam_refit_keeps_a_one_column_varying_slope():
    """`bols(x, by=z, intercept=False)` is one column, x*z, identifiable
    without a sum-to-zero constraint; the refit keeps and estimates it."""
    rng = np.random.default_rng(1)
    n = 1200
    x, z = rng.uniform(-1, 1, (2, n))
    data = {"x": x, "z": z, "y": np.sin(2 * x) + z + 1.5 * x * z + rng.normal(0, .3, n)}
    model = mb.gamboost("y ~ bbs(x) + bbs(z) + bols(x, by=z, intercept=False)", data,
                        control=mb.boost_control(400))
    refit = mb.gam_refit(model)
    assert "bols(x, by=z, intercept=False)" in refit.terms
    grid = {"x": np.array([0.5, 0.5]), "z": np.array([-0.5, 0.5])}
    assert np.diff(refit.predict(grid))[0] > 1.0


def test_lss_deselect_and_refit_keep_tensor_pairs():
    """A distributional fit with a `%X%` pair, deselected and refitted with
    REML, still predicts new rows and refits the pair as a tensor."""
    rng = np.random.default_rng(1)
    n = 800
    x1, x2 = rng.uniform(-1, 1, (2, n))
    y = x1 * x2 + np.exp(-.5 + .5 * x1 * x2) * rng.normal(size=n)
    data = {"x1": x1, "x2": x2, "y": y}
    pair = "bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5)"
    model = mb.gamboostLSS(f"y ~ bbs(x1) + bbs(x2) + {pair}", data, family=mb.GaussianLSS(),
                           control=mb.boost_control(300, .1), method="noncyclic")
    reduced = mb.deselect(model, seed=0).model
    new = {"x1": x1[:5], "x2": x2[:5]}
    np.testing.assert_allclose(np.asarray(reduced.predict(newdata=new, type="response")["mu"]),
                               np.asarray(reduced.predict(type="response")["mu"])[:5])
    refit = mb.gamlss_refit(reduced)
    assert pair in refit.terms["mu"]
    assert np.isfinite(np.asarray(refit.predict(new, type="response")["mu"])).all()
