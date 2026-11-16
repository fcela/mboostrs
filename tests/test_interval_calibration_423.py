"""Propensity learner choice (aipw, effect_model) and the second-stage row bootstrap.

The design has an eligibility cliff: the true propensity is 0.05 above a prior
earnings of 40,000 and about 0.55 below it. A smooth term spreads the step over
its basis, so a formula with a depth-one tree per numeric covariate should fit it
better; the tests check that the choice follows the out-of-fold deviance, that the
tree formula is used when it wins, and that the second-stage bootstrap gives
per-row intervals.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust

COVARIATES = ("prior", "age", "educ")


def head(data, k):
    return {key: np.asarray(value)[:k] for key, value in data.items()}


def spec():
    return mb.CausalSpec(
        treatment="enrolled", outcome="earnings", covariates=COVARIATES,
        estimand="ate", outcome_type="continuous",
        time_zero="enrolment decision", follow_up="one year",
        adjustment_rationale="pre-application characteristics",
    )


def cliff_design(n=1200, seed=5, cliff=True):
    """Enrolment with an eligibility cliff at prior earnings 40,000 (or none, when
    ``cliff=False``), and an effect that varies smoothly with age.

    Returns the data frame, the true propensity and the true effect at each row."""
    rng = np.random.default_rng(seed)
    prior = rng.uniform(10_000, 80_000, n)
    age = rng.uniform(20, 60, n)
    educ = rng.integers(8, 17, n).astype(float)
    if cliff:
        logit = np.where(prior > 40_000, -3.0, 0.2) + 0.1 * (age - 40) / 10
    else:
        logit = 0.3 * (prior - 45_000) / 20_000 + 0.1 * (age - 40) / 10
    e = 1 / (1 + np.exp(-logit))
    enrolled = rng.binomial(1, e).astype(float)
    tau = 2.0 + 1.5 * np.sin(age / 10)
    mu = 1.0 + 0.00005 * prior + 0.5 * educ + 0.1 * age
    earnings = mu + tau * enrolled + rng.normal(0, 2.0, n)
    frame = {"enrolled": enrolled, "earnings": earnings, "prior": prior,
             "age": age, "educ": educ}
    return frame, e, tau


@pytest.fixture(scope="module")
def cliff_data():
    return cliff_design()


def test_the_default_propensity_is_the_smooth_formula_alone(cliff_data):
    """The default keeps the base formula and fits no tree candidate."""
    frame, _, _ = cliff_data
    result = mb.aipw(frame, spec(), folds=3, seed=2, tune=False, mstop=80)
    record = result.diagnostics["propensity_learner"]
    assert record["requested"] == "smooth"
    assert record["candidates"] == ["smooth"]
    assert record["chosen"] == "smooth"
    assert "btree" not in result.nuisance["terms"]["propensity"]
    explicit = mb.aipw(frame, spec(), folds=3, seed=2, tune=False, mstop=80,
                       propensity_learner="smooth")
    assert result.estimate == pytest.approx(explicit.estimate, rel=1e-12)


def test_auto_records_both_deviances_and_keeps_the_lower_one(cliff_data):
    frame, _, _ = cliff_data
    result = mb.aipw(frame, spec(), folds=3, seed=2, tune=False, mstop=80,
                     propensity_learner="auto")
    record = result.diagnostics["propensity_learner"]
    assert record["requested"] == "auto"
    assert record["candidates"] == ["smooth", "smooth+trees"]
    deviance = record["out_of_fold_deviance"]
    assert set(deviance) == {"smooth", "smooth+trees"}
    assert record["chosen"] == min(deviance, key=deviance.get)
    assert result.nuisance["terms"]["propensity"] == record["terms"][record["chosen"]]


def test_auto_takes_the_tree_formula_at_a_cliff(cliff_data):
    frame, _, _ = cliff_data
    result = mb.aipw(frame, spec(), folds=3, seed=2, tune=False, mstop=80,
                     propensity_learner="auto")
    record = result.diagnostics["propensity_learner"]
    assert record["chosen"] == "smooth+trees"
    assert record["out_of_fold_deviance"]["smooth+trees"] < record["out_of_fold_deviance"]["smooth"]
    assert "btree(prior, maxdepth=1)" in result.nuisance["terms"]["propensity"]
    assert "Propensity: formula 'smooth+trees'" in result.describe()


def test_tree_formula_lowers_propensity_error_at_the_cliff(cliff_data):
    frame, e_true, _ = cliff_data
    errors = {}
    for learner in ("smooth", "smooth+trees"):
        result = mb.aipw(frame, spec(), folds=3, seed=2, tune=False, mstop=80,
                         propensity_learner=learner)
        errors[learner] = float(np.mean(np.abs(np.asarray(result.propensity) - e_true)))
    assert errors["smooth+trees"] < errors["smooth"]


def test_explicit_learner_fits_only_its_candidate(cliff_data):
    frame, _, _ = cliff_data
    for learner in ("smooth", "smooth+trees"):
        result = mb.aipw(frame, spec(), folds=3, seed=2, tune=False, mstop=80,
                         propensity_learner=learner)
        record = result.diagnostics["propensity_learner"]
        assert record["requested"] == learner
        assert record["candidates"] == [learner]
        assert record["chosen"] == learner


def test_learner_is_refused_with_propensity_terms_and_checked(cliff_data):
    frame, _, _ = cliff_data
    with pytest.raises(ValueError, match="propensity_learner"):
        mb.aipw(frame, spec(), folds=3, tune=False, mstop=20,
                propensity_learner="smooth+trees",
                propensity_terms="prior + age + educ")
    with pytest.raises(ValueError, match="propensity_learner"):
        mb.aipw(frame, spec(), folds=3, tune=False, mstop=20, propensity_learner="trees")


def test_custom_propensity_terms_are_recorded_as_custom(cliff_data):
    frame, _, _ = cliff_data
    result = mb.aipw(frame, spec(), folds=3, seed=2, tune=False, mstop=40,
                     propensity_terms="prior + age + educ")
    assert result.diagnostics["propensity_learner"]["chosen"] == "custom"
    assert "custom formula from propensity_terms" in result.describe()


def test_effect_model_uses_the_same_choice_at_a_cliff(cliff_data):
    """With propensity_learner="auto", effect_model keeps the tree formula at the cliff."""
    frame, _, _ = cliff_data
    fx = mb.effect_model(frame, spec(), terms="smooth", folds=3, seed=2,
                         propensity_learner="auto")
    record = fx.diagnostics["propensity_learner"]
    assert record["chosen"] == "smooth+trees"
    assert fx.propensity_model is not None
    assert fx.describe()


def test_binomial_deviance_binding_matches_the_formula():
    rng = np.random.default_rng(1)
    t = rng.binomial(1, 0.4, 50).astype(float)
    p = rng.uniform(0.05, 0.95, 50)
    w = rng.uniform(0.5, 2.0, 50)
    expected = -2 * np.sum(w * (t * np.log(p) + (1 - t) * np.log(1 - p))) / np.sum(w)
    np.testing.assert_allclose(_rust.binomial_deviance(t, p, w), expected, rtol=1e-12)


def test_second_stage_bootstrap_with_unit_counts_reproduces_the_refit(cliff_data):
    """Multiplicities of one leave the weights as they were, so the draw is the refit."""
    frame, _, _ = cliff_data
    fx = mb.effect_model(frame, spec(), terms="smooth", folds=3, seed=2,
                         interval_method="refit")
    model = fx.model
    data = model._training_data
    response = np.asarray(data["tau_pseudo_outcome"], dtype=float)
    boot = model.bootstrap(np.ones((2, response.size)), response)
    assert boot.n_draws == 2
    assert all(boot.converged())
    rows = {name: np.asarray(data[name])[:25] for name in fx.modifiers}
    base = np.asarray(model.predict(newdata=rows, extrapolation="clip"), dtype=float)
    draws = boot.predict(rows, extrapolation="clip")
    assert draws.shape == (2, 25)
    np.testing.assert_allclose(draws[0], base, atol=1e-6)


def test_second_stage_bootstrap_gives_per_row_intervals(cliff_data):
    frame, _, _ = cliff_data
    fx = mb.effect_model(frame, spec(), terms="smooth", folds=3, seed=2,
                         interval_method="second_stage_bootstrap", bootstrap_draws=40)
    assert fx.interval_method == "second_stage_bootstrap"
    assert fx.diagnostics["interval"]["second_stage_draws"] == 40
    assert fx.diagnostics["interval"]["second_stage_converged"] == 40
    rows = head(frame, 60)
    pred = fx.predict(rows, interval=True, level=0.9)
    lower = pred["lower"].to_numpy()
    upper = pred["upper"].to_numpy()
    estimate = pred["estimate"].to_numpy()
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(upper > lower)
    assert np.mean((lower <= estimate) & (estimate <= upper)) > 0.9
    # The point estimate is the full-sample fit, the same as the refit's.
    refit = mb.effect_model(frame, spec(), terms="smooth", folds=3, seed=2,
                            interval_method="refit")
    np.testing.assert_allclose(estimate, refit.predict(rows), rtol=1e-10, atol=1e-10)


def test_second_stage_bootstrap_takes_an_aipw_nuisance(cliff_data):
    frame, _, _ = cliff_data
    nuisance = mb.aipw(frame, spec(), folds=3, seed=2, tune=False, mstop=60)
    fx = mb.effect_model(frame, spec(), nuisance=nuisance, terms="smooth",
                         interval_method="second_stage_bootstrap", bootstrap_draws=10)
    assert fx.diagnostics["interval"]["second_stage_draws"] == 10


def test_auto_interval_is_the_second_stage_bootstrap_for_a_smooth_fit(cliff_data):
    frame, _, _ = cliff_data
    fx = mb.effect_model(frame, spec(), terms="smooth", folds=3, seed=2, bootstrap_draws=5)
    assert fx.interval_method == "second_stage_bootstrap"
    assert fx.diagnostics["interval"]["method"] == "second_stage_bootstrap"


def test_second_stage_bootstrap_refuses_boosted_terms_and_auto_falls_back(cliff_data):
    frame, _, _ = cliff_data
    with pytest.raises(ValueError, match="smooth second stage"):
        mb.effect_model(frame, spec(), terms="stumps", folds=3, seed=2,
                        interval_method="second_stage_bootstrap")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = mb.effect_model(frame, spec(), terms="stumps", folds=3, seed=2)
    assert fx.interval_method == "refit"
    with pytest.raises(NotImplementedError, match="interval=True"):
        fx.predict(head(frame, 5), interval=True)


def test_second_stage_bootstrap_needs_at_least_two_draws(cliff_data):
    frame, _, _ = cliff_data
    with pytest.raises(ValueError, match="bootstrap_draws"):
        mb.effect_model(frame, spec(), terms="smooth", folds=3, seed=2,
                        interval_method="second_stage_bootstrap", bootstrap_draws=1)
