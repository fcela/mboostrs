"""Variable importance, bootstrap intervals and LSS quantiles, computed in the core.

The core (`mboostrs_core::inference`, `mboostrs_core::lss_quantile`) owns the
arithmetic; these tests pin it to independent NumPy computations, check that
the Python front ends call it, and check the R reference where R mboost has
the same function.
"""

import json
import subprocess
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust
from mboostrs import gamlss as G
from mboostrs._attribution import attribute_by_name
from mboostrs._bootstrap import (
    AT_CAP,
    FAILED,
    UNTUNED,
    BootstrapTally,
    bootstrap_outcome,
    group_bootstrap_counts,
)


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(21)
    n = 120
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    x3 = rng.normal(size=n)
    y = 2 * x1 - x2 + np.sin(3 * x1) + rng.normal(scale=0.4, size=n)
    group = np.repeat(np.arange(30), 4).astype(float)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3, "g": group}


# ─── variable importance ────────────────────────────────────────────


def test_glm_varimp_equals_the_risk_path_attributed_by_hand(regression_data):
    model = mb.glmboost("y ~ x1 + x2 + x3", data=regression_data,
                        control=mb.boost_control(mstop=40, nu=0.1))
    risk = np.asarray(model._risk, dtype=float)
    n = len(model._fitted)
    changes = -np.diff(risk) / n
    names = model._learner_names
    expected = {name: 0.0 for name in names}
    for step, learner in enumerate(model._selected):
        expected[names[learner]] += changes[step]

    raw = model.varimp(percent=False)["reduction"]
    for name in names:
        assert raw[name] == pytest.approx(expected[name], rel=0, abs=1e-12)

    shares = model.varimp(percent=True)["reduction"]
    total = sum(abs(v) for v in expected.values())
    for name in names:
        assert shares[name] == pytest.approx(expected[name] / total, rel=0, abs=1e-12)
    assert sum(abs(v) for v in shares.values()) == pytest.approx(1.0, abs=1e-12)


def test_glm_selection_frequencies_count_each_update_once(regression_data):
    model = mb.glmboost("y ~ x1 + x2 + x3", data=regression_data,
                        control=mb.boost_control(mstop=40, nu=0.1))
    freq = model.varimp(percent=False)["selfreq"]
    assert sum(freq.values()) == pytest.approx(1.0, abs=1e-12)
    for name in model._learner_names:
        count = sum(1 for j in model._selected if model._learner_names[j] == name)
        assert freq[name] == pytest.approx(count / len(model._selected), abs=1e-12)


def test_gam_varimp_keys_every_base_learner_and_sums_to_one(regression_data):
    model = mb.gamboost("y ~ bbs(x1, df=4) + bols(x2) + bbs(x3, df=3)",
                        data=regression_data, control=mb.boost_control(mstop=30, nu=0.1))
    out = model.varimp(percent=True)
    assert set(out["reduction"]) == set(model.base_learner_names())
    assert sum(abs(v) for v in out["reduction"].values()) == pytest.approx(1.0, abs=1e-12)
    assert set(out["variable_names"]) == set(model.base_learner_names())


def test_attribute_by_name_reports_unselected_names_as_zero():
    reduction, selfreq = attribute_by_name(
        [1.0, 0.5, 0.0], ["a", "a", "c"], ["a", "b", "c", "a"], percent=False)
    assert list(reduction) == ["a", "b", "c"]
    assert reduction == {"a": 1.5, "b": 0.0, "c": 0.0}
    assert selfreq == pytest.approx({"a": 2 / 3, "b": 0.0, "c": 1 / 3})


def test_attribute_by_name_shares_are_relative_to_absolute_total():
    reduction, _ = attribute_by_name([2.0, -1.0], ["a", "b"], ["a", "b"], percent=True)
    assert reduction == pytest.approx({"a": 2 / 3, "b": -1 / 3})


def test_lss_varimp_sums_the_path_risk_changes_per_learner_and_parameter(regression_data):
    lss = mb.gamboostLSS({"mu": "y ~ bbs(x1, df=4) + bols(x2)", "sigma": "y ~ bols(x3)"},
                         data=regression_data, families=mb.GaussianLSS(),
                         control=mb.boost_control(mstop=40, nu=0.1), method="noncyclic")
    rows = lss.path_frame(output="records")
    expected_reduction, expected_count = {}, {}
    for row in rows:
        key = f"{row['learner']}.{row['parameter']}"
        change = row.get("risk_change")
        expected_reduction[key] = expected_reduction.get(key, 0.0) + (
            float(change) if change is not None else 0.0)
        expected_count[key] = expected_count.get(key, 0) + 1
    out = lss.varimp(percent=False)
    assert set(out["reduction"]) == set(expected_reduction)
    for key, value in expected_reduction.items():
        assert out["reduction"][key] == pytest.approx(value, rel=0, abs=1e-12)
        assert out["selfreq"][key] == pytest.approx(
            expected_count[key] / len(rows), rel=0, abs=1e-12)


def test_core_varimp_rejects_mismatched_updates():
    with pytest.raises(ValueError, match="pair up one to one"):
        _rust.varimp_attribution([1.0, 2.0], [0], 1, False)


# ─── bootstrap draws, tally and percentile intervals ────────────────


def test_group_draws_move_whole_groups_and_respect_eligibility():
    groups = np.repeat([0, 1, 2, 3], 3)
    eligible = np.ones(12, dtype=bool)
    eligible[5] = False
    counts = group_bootstrap_counts(groups, eligible, 40, 3)
    assert counts.shape == (12, 40)
    for column in counts.T:
        for g in range(4):
            rows = np.flatnonzero((groups == g) & eligible)
            if rows.size:
                assert len(set(column[rows])) == 1
        assert column[5] == 0.0


def test_group_draws_repeat_for_a_seed_and_change_with_it():
    groups = np.repeat(np.arange(10), 2)
    eligible = np.ones(20, dtype=bool)
    first = group_bootstrap_counts(groups, eligible, 25, 17)
    again = group_bootstrap_counts(groups, eligible, 25, 17)
    other = group_bootstrap_counts(groups, eligible, 25, 18)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_tally_bits_are_counted_per_replicate():
    tally = BootstrapTally(B=6)
    tally.mark(0, FAILED, ValueError("a"))
    tally.mark(1, UNTUNED, RuntimeError("b"))
    tally.mark(1, FAILED, ValueError("c"))
    tally.mark(3, AT_CAP)
    assert _rust.bootstrap_counts(tally.flags) == (2, 1, 1)
    with pytest.warns(RuntimeWarning, match="2 of 6 bootstrap replicates failed"):
        counts = bootstrap_outcome(tally, B=6, what="coefficient")
    assert counts["n_failed"] == 2
    assert counts["n_untuned"] == 1
    assert counts["n_at_mstop_cap"] == 1
    assert counts["n_used"] == 4


def test_percentile_interval_matches_numpy_nanquantile_per_column():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(60, 4))
    values[::7, 1] = np.nan
    values[:, 3] = np.nan
    level = 0.9
    lower, upper = _rust.percentile_interval(values, level, False)
    q_low, q_high = (1 - level) / 2, 1 - (1 - level) / 2
    np.testing.assert_allclose(lower[:3], np.nanquantile(values, q_low, axis=0)[:3],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(upper[:3], np.nanquantile(values, q_high, axis=0)[:3],
                               rtol=0, atol=1e-12)
    assert np.isnan(lower[3]) and np.isnan(upper[3])


def test_percentile_interval_centres_each_replicate_before_the_quantile():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(40, 25)) + rng.normal(size=(40, 1)) * 3.0
    centred = values - values.mean(axis=1, keepdims=True)
    lower, upper = _rust.percentile_interval(values, 0.8, True)
    q_low, q_high = 0.1, 0.9
    np.testing.assert_allclose(lower, np.quantile(centred, q_low, axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(upper, np.quantile(centred, q_high, axis=0), rtol=0, atol=1e-12)


def test_percentile_interval_refuses_a_level_outside_the_unit_interval():
    with pytest.raises(ValueError, match="level must lie in"):
        _rust.percentile_interval(np.ones((3, 2)), 1.5, False)


# ─── confint on fitted models ───────────────────────────────────────


def test_glm_confint_is_identical_for_a_repeated_seed(regression_data):
    model = mb.glmboost("y ~ x1 + x2 + x3", data=regression_data,
                        control=mb.boost_control(mstop=30, nu=0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = model.confint(B=10, B_mstop=0, seed=4)
        second = model.confint(B=10, B_mstop=0, seed=4)
    for name in first["lower"]:
        assert first["lower"][name] == second["lower"][name]
        assert first["upper"][name] == second["upper"][name]


def test_glm_group_confint_resamples_groups(regression_data):
    model = mb.glmboost("y ~ x1 + x2 + x3", data=regression_data,
                        control=mb.boost_control(mstop=30, nu=0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = model.confint(B=12, B_mstop=0, seed=2, groups="g")
    assert ci["resampled"] == "groups"
    assert ci["n_used"] == 12
    assert all(ci["lower"][k] <= ci["upper"][k] for k in ci["lower"])


def test_gam_confint_centred_bands_are_centred_per_replicate(regression_data):
    model = mb.gamboost("y ~ bbs(x1, df=4) + bols(x2)", data=regression_data,
                        control=mb.boost_control(mstop=25, nu=0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = model.confint(B=8, B_mstop=0, seed=6, centre=True)
    boot = np.asarray(ci["boot_pred"]["x1"], dtype=float)
    centred = boot - boot.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(ci["lower"]["x1"], np.quantile(centred, 0.025, axis=0),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(ci["upper"]["x1"], np.quantile(centred, 0.975, axis=0),
                               rtol=0, atol=1e-12)


# ─── LSS quantiles ──────────────────────────────────────────────────

CONTINUOUS = [
    (G.GaussianLSS(), dict(mu=[0.3, 1.5], sigma=[0.8, 2.0])),
    (G.GammaLSS(), dict(mu=[2.0, 5.0], sigma=[1.5, 3.0])),
    (G.StudentTLSS(), dict(mu=[0.0, 1.0], sigma=[1.0, 0.5], df=[4.0, 12.0])),
    (G.BetaLSS(), dict(mu=[0.3, 0.7], phi=[4.0, 9.0])),
    (G.InverseGaussianLSS(), dict(mu=[1.0, 2.5], sigma=[0.5, 0.8])),
    (G.LogisticLSS(), dict(mu=[0.0, 2.0], sigma=[1.0, 0.4])),
    (G.GumbelLSS(), dict(mu=[0.0, 1.0], sigma=[1.0, 2.0])),
    (G.Pareto2LSS(), dict(mu=[2.0, 4.0], sigma=[2.5, 3.5])),
    (G.LogNormalLSS(), dict(mu=[0.2, 1.0], sigma=[0.6, 0.9])),
    (G.LogLogLSS(), dict(mu=[0.2, 1.0], sigma=[0.6, 0.9])),
    (G.WeibullLSS(), dict(mu=[0.2, 1.0], sigma=[0.6, 1.4])),
]


@pytest.mark.parametrize("family, params", CONTINUOUS, ids=lambda v: type(v).__name__)
def test_continuous_quantiles_invert_the_family_cdf(family, params):
    probs = np.array([0.02, 0.25, 0.5, 0.8, 0.98])
    p = np.broadcast_to(probs[:, None], (probs.size, 2))
    par = {k: np.broadcast_to(np.asarray(v, dtype=float)[None, :], p.shape)
           for k, v in params.items()}
    quant = family.quantile(p, par)
    back = family.cdf(quant, par)
    np.testing.assert_allclose(back, p, rtol=0, atol=1e-7)


@pytest.mark.parametrize("family, params", [
    (G.PoissonLSS(), dict(mu=[0.7, 4.0])),
    (G.NBinomialLSS(), dict(mu=[2.0, 6.0], sigma=[1.5, 0.5])),
    (G.NegativeBinomialLSS(), dict(mu=[2.0, 6.0], sigma=[0.6, 0.2])),
    (G.ZIPoLSS(), dict(mu=[2.0, 5.0], sigma=[0.2, 0.4])),
    (G.ZINBLSS(), dict(mu=[2.0, 5.0], sigma=[0.8, 0.3], nu=[0.2, 0.1])),
], ids=lambda v: type(v).__name__)
def test_count_quantiles_are_the_smallest_count_reaching_the_probability(family, params):
    probs = np.array([0.05, 0.5, 0.9, 0.99])
    p = np.broadcast_to(probs[:, None], (4, 2))
    par = {k: np.broadcast_to(np.asarray(v, float)[None, :], p.shape) for k, v in params.items()}
    quant = family.quantile(p, par)
    assert np.all(family.cdf(quant, par) >= p - 1e-12)
    below = np.maximum(quant - 1.0, 0.0)
    strictly = family.cdf(below, par) < p
    assert np.all(strictly | (quant == 0))


def test_core_quantile_names_the_family_in_its_refusals():
    cols = [[0.5], [1.0], [0.0]]
    with pytest.raises(ValueError, match="GaussianLSS sigma must be positive"):
        _rust.lss_quantile("GaussianLSS", cols)
    with pytest.raises(ValueError, match="BetaLSS quantile probabilities must be finite"):
        _rust.lss_quantile("BetaLSS", [[1.0], [0.4], [2.0]])
    with pytest.raises(ValueError, match="no quantile function for family 'NoSuchLSS'"):
        _rust.lss_quantile("NoSuchLSS", [[0.5], [0.0]])


def test_quantile_method_and_core_return_the_same_values():
    family = G.GammaLSS()
    params = dict(mu=np.array([1.0, 2.0, 3.0]), sigma=np.array([2.0, 0.5, 4.0]))
    p = np.array([0.1, 0.5, 0.9])
    via_method = family.quantile(p, params)
    values, unconverged = _rust.lss_quantile(
        "GammaLSS", [p.tolist(), params["mu"].tolist(), params["sigma"].tolist()])
    assert unconverged == 0
    np.testing.assert_allclose(via_method, values, rtol=0, atol=1e-15)


def test_binomial_quantile_default_and_explicit_trials_agree():
    p = np.array([0.1, 0.6, 0.95])
    mu = np.array([0.3, 0.5, 0.8])
    one = G.BinomialLSS().quantile(p, dict(mu=mu))
    explicit = G.BinomialLSS().quantile(p, dict(mu=mu), size=1)
    np.testing.assert_array_equal(one, explicit)
    three = G.BinomialLSS().quantile(p, dict(mu=mu), size=3)
    assert np.all(three >= one)


@pytest.fixture(scope="module")
def gauss_model():
    rng = np.random.default_rng(4)
    x = rng.uniform(size=80)
    y = 1 + 2 * x + rng.normal(scale=0.5 + 0.8 * x)
    return mb.glmboostLSS("y ~ x", data={"y": y, "x": x}, family=G.GaussianLSS(),
                          control=mb.boost_control(mstop=60, nu=0.1))


def test_prediction_interval_takes_its_quantiles_from_the_core(gauss_model, monkeypatch):
    calls = []
    real = _rust.lss_quantile

    def counting(family, columns):
        calls.append(family)
        return real(family, columns)

    monkeypatch.setattr(_rust, "lss_quantile", counting)
    rows = gauss_model.prediction_interval(level=0.9).to_dicts()
    assert calls == ["GaussianLSS"] * 3  # lower, median, upper
    assert all(r["lower"] < r["median"] < r["upper"] for r in rows)


def test_prediction_interval_bounds_are_the_normal_quantiles(gauss_model):
    rows = gauss_model.prediction_interval(level=0.9).to_dicts()
    params = gauss_model.params(type="response")
    mu = np.asarray(params["mu"], dtype=float)
    sigma = np.asarray(params["sigma"], dtype=float)
    z = 1.6448536269514722  # qnorm(0.95)
    lower = np.array([r["lower"] for r in rows])
    upper = np.array([r["upper"] for r in rows])
    np.testing.assert_allclose(lower, mu - z * sigma, rtol=0, atol=1e-10)
    np.testing.assert_allclose(upper, mu + z * sigma, rtol=0, atol=1e-10)


def test_known_variance_family_prediction_interval_is_refused_with_a_reason(
        gauss_model, monkeypatch):
    monkeypatch.setattr(gauss_model, "_family", G.GaussianKnownVarLSS())
    with pytest.raises(NotImplementedError, match="known sampling variance"):
        gauss_model.prediction_interval(level=0.9)


def test_families_without_a_core_quantile_are_not_listed():
    assert "PIGLSS" not in G._CORE_QUANTILE_FAMILIES
    assert "DirichletLSS" not in G._CORE_QUANTILE_FAMILIES


# ─── R reference ────────────────────────────────────────────────────

R_VARIMP_SCRIPT = r"""
suppressMessages(library(mboost))
set.seed(42)
n <- 150
x1 <- runif(n)
x2 <- runif(n)
x3 <- runif(n)
y <- 2 * x1 - x2 + rnorm(n, sd = 0.4)
dat <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
mod <- glmboost(y ~ x1 + x2 + x3, data = dat,
                control = boost_control(mstop = 60, nu = 0.1))
vi <- varimp(mod, percent = FALSE)
result <- list(
  names = names(vi),
  reduction = as.numeric(vi),
  x1 = x1, x2 = x2, x3 = x3, y = y
)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA))
"""


@pytest.mark.r_parity
def test_glmboost_varimp_reductions_match_mboost():
    try:
        result = subprocess.run(["Rscript", "-e", R_VARIMP_SCRIPT],
                                capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        pytest.skip("Rscript not available")
    assert result.returncode == 0, result.stderr
    r_out = json.loads(result.stdout.strip())
    data = {k: np.asarray(r_out[k], dtype=float) for k in ("y", "x1", "x2", "x3")}
    model = mb.glmboost("y ~ x1 + x2 + x3", data=data,
                        control=mb.boost_control(mstop=60, nu=0.1))
    reduction = model.varimp(percent=False)["reduction"]
    for name, r_value in zip(r_out["names"], r_out["reduction"], strict=True):
        assert reduction[name] == pytest.approx(r_value, rel=0, abs=1e-8), name
