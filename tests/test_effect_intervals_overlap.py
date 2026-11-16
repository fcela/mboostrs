"""Per-row intervals of the effect model, and the overlap report."""

import warnings
from statistics import NormalDist

import numpy as np
import pytest

import mboostrs as mb


def spec(covariates=("x1", "x2", "x3")):
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=tuple(covariates),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="the covariates are the common causes",
    )


def heterogeneous(n, seed, shift=0.0):
    """Confounded treatment, outcome with a smooth nonlinear baseline, and a
    known additive effect tau(x) = 1 + 0.8 x1 + 0.5 sin(2 x2)."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.uniform(-1.5, 1.5, (3, n))
    x1 = x1 + shift
    e = 1 / (1 + np.exp(-(0.8 * x1 + 0.6 * x2)))
    a = rng.binomial(1, e).astype(float)
    tau = 1 + 0.8 * x1 + 0.5 * np.sin(2 * x2)
    y = np.sin(x1) + 0.7 * x2 + 0.3 * x3 + tau * a + rng.normal(0, 0.5, n)
    return {"a": a, "y": y, "x1": x1, "x2": x2, "x3": x3}, tau


def test_per_row_intervals_cover_the_known_effect_near_nominal():
    """Coverage of the 95% and 90% per-row intervals over fresh rows, pooled
    over four simulated data sets. The interval is the second stage's
    Bayesian interval given the cross-fitted nuisances, so the check is
    near nominal rather than exact."""
    hits95 = hits90 = total = 0
    for seed in range(4):
        data, _ = heterogeneous(1200, seed=2000 + seed)
        fresh, tau_fresh = heterogeneous(600, seed=3000 + seed)
        # learner="r" is the unbounded R-learner pseudo-outcome. The default
        # clips its 1% tails, which shrinks the effect where the pseudo-outcome
        # is extreme and lowers the coverage of this clean design to about 0.84.
        fx = mb.effect_model(data, spec(), seed=seed, terms="smooth", learner="r")
        assert fx.terms == "smooth"
        newdata = {k: fresh[k] for k in ("x1", "x2", "x3")}
        for level, counter in ((0.95, "95"), (0.90, "90")):
            frame = fx.predict(newdata, interval=True, level=level)
            inside = ((frame["lower"].to_numpy() <= tau_fresh)
                      & (tau_fresh <= frame["upper"].to_numpy()))
            if counter == "95":
                hits95 += int(inside.sum())
            else:
                hits90 += int(inside.sum())
        total += tau_fresh.size
    assert 0.90 <= hits95 / total <= 0.995
    assert 0.85 <= hits90 / total <= 0.98


def test_interval_standard_error_is_the_summed_prediction_se():
    """The standard error of the summed linear predictor. With one smooth
    term, the intercept and the centred term are uncorrelated, so the
    prediction variance is the term's effect variance plus the intercept's."""
    from mboostrs.families import Gaussian
    from mboostrs.selection import gam_refit

    data, _ = heterogeneous(800, seed=11)
    model = mb.gamboost("y ~ bbs(x1)", data=data, family=Gaussian(),
                        control=mb.boost_control(10, .1))
    refit = gam_refit(model, which="all")
    new = {"x1": np.array([-1.0, 0.0, 1.0])}
    frame = refit.predict(new, interval=True, level=0.9, output="records")
    fit = np.array([r["estimate"] for r in frame])
    se = np.array([r["se"] for r in frame])
    lower = np.array([r["lower"] for r in frame])
    upper = np.array([r["upper"] for r in frame])
    (effect, effect_lower, effect_upper), = refit.effects(new, level=0.9).values()
    effect_se = (effect_upper - effect_lower) / (2 * NormalDist().inv_cdf(0.95))
    intercept_var = refit.vcov()[0, 0]
    np.testing.assert_allclose(se ** 2, effect_se ** 2 + intercept_var, rtol=1e-8)
    np.testing.assert_allclose(fit, refit.predict(new))
    z = NormalDist().inv_cdf(0.95)
    np.testing.assert_allclose(upper - fit, z * se)
    np.testing.assert_allclose(fit - lower, z * se)


def test_boosted_candidate_has_no_per_row_interval():
    data, _ = heterogeneous(400, seed=5)
    fx = mb.effect_model(data, spec(), terms="stumps", seed=1)
    with pytest.raises(NotImplementedError, match="terms='smooth'"):
        fx.predict({"x1": [0.0], "x2": [0.0], "x3": [0.0]}, interval=True)


def test_overlap_report_locates_a_deterministic_no_treatment_region():
    """Propensity is near zero for x1 < -1 and ordinary elsewhere, so the
    rows outside the bounds are exactly those, and the region is the span of
    x1 where they lie."""
    rng = np.random.default_rng(8)
    n = 4000
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.normal(size=n)
    propensity = np.where(x1 < -1, 0.005, 1 / (1 + np.exp(-(0.5 * x2 + 0.3 * x1))))
    report = mb.overlap_report(propensity, data={"x1": x1, "x2": x2},
                               covariates=("x1", "x2"))
    expected_share = np.mean(x1 < -1)
    assert report.share_outside == pytest.approx(expected_share, abs=0.01)
    assert report.n_below == int(np.sum(x1 < -1))
    assert report.n_above == 0
    region = next(r for r in report.regions if r["covariate"] == "x1")
    low, high = region["region"]
    assert low == pytest.approx(-2, abs=0.1) and high == pytest.approx(-1, abs=0.1)
    assert region["share_outside_in_region"] > 0.9
    assert region["share_outside_elsewhere"] < 0.01
    x2_region = next(r for r in report.regions if r["covariate"] == "x2")
    assert x2_region["region"] is None
    assert report.estimate is None and report.estimate_inside is None
    assert "outside [0.02, 0.98]" in report.describe()


def test_overlap_report_lists_levels_of_a_categorical_covariate():
    n = 600
    rng = np.random.default_rng(9)
    site = np.where(rng.random(n) < 0.5, "north", "south")
    propensity = np.where(site == "north", 0.01, 0.5)
    report = mb.overlap_report(propensity, data={"site": site}, covariates=("site",))
    column = report.regions[0]
    assert column["kind"] == "categorical"
    assert column["flagged_levels"] == ("north",)


def test_overlap_report_refuses_bad_inputs():
    with pytest.raises(ValueError, match="bounds"):
        mb.overlap_report(np.array([0.3, 0.6]), bounds=(0.9, 0.1))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        mb.overlap_report(np.array([0.3, 1.2]))
    with pytest.raises(ValueError, match="covariates"):
        mb.overlap_report(np.array([0.3, 0.6]), data={"x": [1.0, 2.0]})


def test_aipw_restricted_estimate_is_the_mean_score_inside_the_bounds():
    """Rows with x1 < -1 are never treated, so their propensity is pushed
    toward zero and the report counts them as outside the bounds."""
    rng = np.random.default_rng(12)
    n = 900
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.normal(size=n)
    e = 1 / (1 + np.exp(-(1.2 * x1 + 0.5 * x2)))
    a = np.where(x1 < -1, 0.0, rng.binomial(1, e)).astype(float)
    tau = 2.0
    y = x1 + 0.5 * x2 + tau * a + rng.normal(0, 1, n)
    data = {"a": a, "y": y, "x1": x1, "x2": x2}
    item = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 are the common causes",
    )
    result = mb.aipw(data, item, folds=4, mstop=100, tune=False, seed=3)
    overlap = result.diagnostics["overlap"]
    assert overlap["n"] == n
    assert overlap["n_outside"] == overlap["n_below"] + overlap["n_above"]
    inside = (result.propensity >= 0.02) & (result.propensity <= 0.98)
    assert overlap["n_inside"] == int(inside.sum())
    assert overlap["estimate"] == pytest.approx(result.estimate)
    assert overlap["estimate_inside"] == pytest.approx(result.scores[inside].mean())
    assert np.isfinite(overlap["standard_error_inside"])
    regions = {r["covariate"]: r for r in overlap["regions"]}
    assert regions["x1"]["kind"] == "numeric"


def test_effect_model_records_overlap_and_flags_new_rows_once():
    rng = np.random.default_rng(21)
    n = 900
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    x3 = rng.uniform(-1, 1, n)
    e = 1 / (1 + np.exp(-4 * x1))
    a = rng.binomial(1, e).astype(float)
    y = x2 + (1 + 0.5 * x2) * a + rng.normal(0, .5, n)
    data = {"a": a, "y": y, "x1": x1, "x2": x2, "x3": x3}
    fx = mb.effect_model(data, spec(), seed=2, terms="smooth")
    assert fx.diagnostics["overlap"]["n"] == n
    assert fx.overlap_bounds == (0.02, 0.98)

    # Two rows far outside the training range of x1 have propensities beyond
    # the bounds; the two inside rows do not.
    new = {"x1": np.array([-6.0, 0.0, 6.0, 0.1]), "x2": np.zeros(4),
           "x3": np.zeros(4)}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        frame = fx.predict(new, interval=True)
    flags = frame["outside_overlap"].to_list()
    assert flags == [True, False, True, False]
    overlap_warnings = [w for w in record if "outside [0.02, 0.98]" in str(w.message)]
    assert len(overlap_warnings) == 1
    assert "2 of 4 new rows" in str(overlap_warnings[0].message)

    # Without interval the effects come back as an array, with the same warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        plain = fx.predict(new)
    assert isinstance(plain, np.ndarray) and plain.shape == (4,)
    assert sum("2 of 4 new rows" in str(w.message) for w in record) == 1


def test_predict_without_covariates_returns_effects_with_a_warning():
    rng = np.random.default_rng(22)
    n = 500
    x1, x2 = rng.uniform(-1, 1, (2, n))
    a = rng.binomial(1, 1 / (1 + np.exp(-x2))).astype(float)
    y = x1 + (1 + x1) * a + rng.normal(0, .5, n)
    item = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="x2 is the common cause",
    )
    fx = mb.effect_model({"a": a, "y": y, "x1": x1, "x2": x2}, item,
                         modifiers=("x1",), seed=1, terms="smooth")
    with pytest.warns(UserWarning, match="cannot check overlap"):
        frame = fx.predict({"x1": [0.0, 0.5]}, interval=True)
    assert frame["outside_overlap"].to_list() == [None, None]
    with pytest.warns(UserWarning, match="cannot check overlap"):
        plain = fx.predict({"x1": [0.0, 0.5]})
    assert frame["estimate"].to_list()[0] == pytest.approx(plain[0], abs=1e-8)
