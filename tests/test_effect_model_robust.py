"""Bounded second stage of effect_model, categorical covariates, and the
propensity of new rows."""

import warnings

import numpy as np
import pytest

import mboostrs as mb
import mboostrs._rust as rust
import mboostrs.causal as causal


def spec(covariates=("x1", "x2", "region")):
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=tuple(covariates),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="the covariates are the common causes",
    )


def heavy_tailed(n, seed):
    """Confounded treatment with a few rows near the propensity bounds, a
    heavy-tailed outcome and a three-level categorical covariate."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    region = rng.choice(np.array(["north", "south", "west"], dtype=object), n)
    e = 1 / (1 + np.exp(-(2.5 * x1 + 0.5 * x2)))
    a = rng.binomial(1, e).astype(float)
    tau = 1.0 + 0.5 * x1
    y = np.exp(0.8 * x1 + 0.4 * x2 + rng.standard_t(2.5, n)) + tau * a
    return {"a": a, "y": y, "x1": x1, "x2": x2, "region": region}


def light_tailed(n, seed):
    """Mild confounding, normal noise and a constant effect of 1: the effect
    stays within a few outcome interquartile ranges."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    region = rng.choice(np.array(["north", "south", "west"], dtype=object), n)
    e = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    a = rng.binomial(1, e).astype(float)
    y = x1 + 0.5 * x2 + rng.normal(0, 1, n) + a
    return {"a": a, "y": y, "x1": x1, "x2": x2, "region": region}


def test_winsorized_pseudo_outcome_clips_at_type_seven_quantiles():
    pseudo = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 1e9])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    values, lower, upper, n_clipped = rust.winsorize_pseudo_outcome(pseudo, weights, 0.2)
    assert lower == pytest.approx(1.8)
    assert upper == pytest.approx(23.2)
    np.testing.assert_allclose(values, [1.8, 2.0, 3.0, 4.0, 23.2, 1e9])
    assert n_clipped == 2


def test_unknown_learner_is_refused_with_the_choices_named():
    data = heavy_tailed(120, seed=1)
    with pytest.raises(ValueError, match="'r_winsorized' or 'r'"):
        mb.effect_model(data, spec(), learner="dr")


def test_default_clips_the_pseudo_outcome_and_r_keeps_it():
    data = heavy_tailed(400, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bounded = mb.effect_model(data, spec(), terms="smooth")
        unbounded = mb.effect_model(data, spec(), terms="smooth", learner="r")
    assert bounded.diagnostics["learner"] == "r_winsorized"
    lower, upper = bounded.diagnostics["pseudo_bounds"]
    assert lower < upper
    assert bounded.diagnostics["n_winsorized"] > 0
    assert unbounded.diagnostics["learner"] == "r"
    assert unbounded.diagnostics["pseudo_bounds"] is None
    assert unbounded.diagnostics["n_winsorized"] == 0
    assert "Pseudo-outcome r_winsorized, clipped to [" in bounded.describe()
    assert "Pseudo-outcome r." in unbounded.describe()


def test_effects_beyond_ten_outcome_iqrs_are_warned():
    # Heavy-tailed outcome and propensities near the bounds: the fitted effect
    # reaches hundreds of times the outcome's interquartile range.
    data = heavy_tailed(300, seed=3)
    with pytest.warns(UserWarning, match="times the interquartile range of the outcome"):
        fx = mb.effect_model(data, spec(), terms="smooth")
    assert fx.diagnostics["effect_range"]["beyond_range"] is True


def test_the_warning_threshold_is_ten_times_the_outcome_iqr(monkeypatch):
    data = light_tailed(300, seed=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fx = mb.effect_model(data, spec(), terms="smooth")
    assert not any("interquartile range" in str(w.message) for w in caught)
    assert fx.diagnostics["effect_range"]["beyond_range"] is False
    assert fx.diagnostics["effect_range"]["multiple"] == 10.0
    # The multiple is lowered so that the same fit now exceeds it.
    monkeypatch.setattr(causal, "_EFFECT_RANGE_MULTIPLE", 1e-3)
    with pytest.warns(UserWarning, match="times the interquartile range of the outcome"):
        again = mb.effect_model(data, spec(), terms="smooth")
    assert again.diagnostics["effect_range"]["beyond_range"] is True


def test_categorical_covariate_is_accepted_and_coded_the_same_for_new_rows():
    data = heavy_tailed(300, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = mb.effect_model(data, spec(), terms="smooth")
    new = heavy_tailed(50, seed=6)
    tau = fx.predict(new)
    assert tau.shape == (50,)
    assert np.all(np.isfinite(tau))
    frame = fx.predict(new, interval=True)
    assert len(frame["estimate"]) == 50


def test_propensity_of_new_rows_is_a_probability_per_row():
    data = heavy_tailed(300, seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = mb.effect_model(data, spec(), terms="smooth")
    new = heavy_tailed(40, seed=8)
    e_new = fx.propensity(new)
    assert e_new.shape == (40,)
    assert np.all((e_new > 0) & (e_new < 1))
    # The array attribute is still the cross-fitted propensity of the fitting rows.
    assert np.asarray(fx.propensity).shape == (300,)


def test_propensity_names_a_missing_covariate():
    data = heavy_tailed(300, seed=9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = mb.effect_model(data, spec(), terms="smooth")
    new = heavy_tailed(20, seed=10)
    del new["x2"]
    with pytest.raises(ValueError, match="'x2'"):
        fx.propensity(new)


def test_rank_by_effect_takes_its_propensity_from_the_method():
    data = heavy_tailed(300, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = mb.effect_model(data, spec(), terms="smooth")
    new = heavy_tailed(100, seed=12)
    ranking = mb.rank_by_effect(fx, new, share=0.2)
    expected = fx.propensity(new)
    np.testing.assert_allclose([row["propensity"] for row in ranking.rows], expected)
    assert len(ranking.chosen) == 20
