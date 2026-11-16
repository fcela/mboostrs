"""The step rule: `nu` as a fraction of the loss-optimal step.

A fixed `nu·h` step is a different amount of progress for every family: for
Poisson with counts around 150 it is about `nu·mu` Newton steps and pushes the
linear predictor past `exp` overflow at iteration 2 (the same divergence R
reports as "Infinite residuals: please decrease step-size nu"); at a 2% event
rate it is about twelve times smaller than a Newton step and spends thousands
of iterations. `step="line_search"` finds the multiple `rho` of the winner's
direction that minimizes the training loss each boosting step, then takes
`nu·rho·h`, so the meaning of `nu` stops depending on the response scale.

The evidence lives in `crates/mboostrs-core/src/engine.rs` (the large-count and
rare-event tests, the squared-error bit-identity, the coefficient-replay
identity); these tests check the Python surface: the control record, the
binding wiring on every fit and refit path, and the artifacts.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._control import BoostControl


def large_counts(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    mu = 150 + 40 * x1 + 20 * x2
    y = np.maximum(np.rint(mu + rng.normal(scale=8, size=n)), 1.0)
    return {"y": y, "x1": x1, "x2": x2}


def rare_events(seed=1, n=500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    y = np.zeros(n)
    y[:10] = 1.0
    return {"y": y, "x": x}


# ─── the control record ────────────────────────────────────────────────


def test_the_default_step_is_the_classic_fixed_one():
    control = mb.boost_control(100, 0.1)
    assert control.step == "fixed"
    assert control.step_iterations == 4
    assert "step" not in control.describe()
    described = mb.boost_control(100, 0.1, step="line_search").describe()
    assert "step='line_search'" in described


def test_the_step_is_validated_where_it_is_written():
    with pytest.raises(ValueError, match="step must be 'fixed' or 'line_search'"):
        mb.boost_control(100, 0.1, step="bogus")
    with pytest.raises(ValueError, match="step_iterations must be at least 1"):
        mb.boost_control(100, 0.1, step="line_search", step_iterations=0)


def test_a_record_from_before_the_rule_exists_reads_as_fixed():
    """Older artifacts stored 4-field records; their fits stay fixed-step."""
    legacy = BoostControl(50, 0.1, "inbag", False)
    assert legacy.step == "fixed"
    assert legacy.step_iterations == 4
    from mboostrs._control import step_kwargs
    assert step_kwargs(legacy) == {"step": "fixed", "step_iterations": 4}
    # A bare tuple a caller may still pass reads the same way.
    assert step_kwargs((50, 0.1, "inbag", False)) == {"step": "fixed",
                                                     "step_iterations": 4}


def test_a_plain_control_tuple_is_still_accepted_by_the_fit():
    data = rare_events()
    fit = mb.glmboost("y ~ bols(x)", data=data, family=mb.Binomial(),
                      control=(50, 0.1, "inbag", False))
    assert fit.control.step == "fixed"


# ─── what the rule changes ──────────────────────────────────────────────


def test_a_large_count_poisson_fit_diverges_fixed_and_fits_searched():
    data = large_counts()
    with pytest.raises((ValueError, RuntimeError),
                       match="risk became non-finite"):
        mb.glmboost("y ~ bols(x1) + bols(x2)", data=data, family=mb.Poisson(),
                    control=mb.boost_control(50, 0.1))
    fit = mb.glmboost("y ~ bols(x1) + bols(x2)", data=data, family=mb.Poisson(),
                      control=mb.boost_control(50, 0.1, step="line_search"))
    assert np.isfinite(fit.risk_).all()
    assert fit.risk_[-1] < fit.risk_[0]


def test_a_squared_error_fit_is_untouched_by_the_rule():
    """The least-squares step is already loss-optimal (`rho = 1` exactly), so
    enabling the search must change nothing, on any engine route."""
    rng = np.random.default_rng(3)
    n = 400
    x = rng.uniform(-2, 2, n)
    data = {"y": 1.0 + 0.8 * x + rng.normal(scale=0.3, size=n), "x": x}
    formulas = ["y ~ bols(x)", "y ~ bbs(x, df=4)"]
    for formula in formulas:
        fixed = mb.gamboost(formula, data=data, control=mb.boost_control(120, 0.1))
        searched = mb.gamboost(
            formula, data=data, control=mb.boost_control(120, 0.1,
                                                          step="line_search"))
        assert fixed.selected == searched.selected, formula
        assert np.array_equal(fixed.risk_, searched.risk_), formula
        assert np.array_equal(np.asarray(fixed.fitted()),
                              np.asarray(searched.fitted())), formula


def test_a_rare_event_binomial_fit_needs_far_fewer_iterations():
    data = rare_events()
    def fit(control):
        return mb.glmboost("y ~ bols(x)", data=data, family=mb.Binomial(),
                           control=control)
    fixed = fit(mb.boost_control(300, 0.1))
    searched = fit(mb.boost_control(50, 0.1, step="line_search"))
    # At an equal budget the searched path is ahead, and 50 searched
    # iterations reach the fixed rule's 300-iteration risk to within a
    # thousandth of the reduction either achieves.
    short_fixed = fit(mb.boost_control(50, 0.1))
    assert searched.risk_[-1] < short_fixed.risk_[-1]
    reduction = fixed.risk_[0] - fixed.risk_[-1]
    assert searched.risk_[-1] <= fixed.risk_[-1] + 1e-3 * reduction


# ─── the rule travels with the model ────────────────────────────────────


def test_cvrisk_carries_the_rule_to_every_fold():
    data = large_counts(2, n=600)
    fit = mb.glmboost("y ~ bols(x1) + bols(x2)", data=data, family=mb.Poisson(),
                      control=mb.boost_control(40, 0.1, step="line_search"))
    result = fit.cvrisk(type="kfold", B=3, seed=5)
    assert np.isfinite(result.aggregate).all()


def test_update_and_continue_carry_the_rule():
    data = rare_events()
    fit = mb.glmboost("y ~ bols(x)", data=data, family=mb.Binomial(),
                      control=mb.boost_control(30, 0.1, step="line_search"))
    assert fit.update(mstop=20).control.step == "line_search"
    assert fit.continue_to(40).control.step == "line_search"


def test_an_artifact_round_trips_the_rule():
    data = rare_events()
    fit = mb.glmboost("y ~ bols(x)", data=data, family=mb.Binomial(),
                      control=mb.boost_control(25, 0.1, step="line_search"))
    mb.save_model(fit, "/tmp/step_rule_model.json")
    loaded = mb.load_model("/tmp/step_rule_model.json")
    assert loaded.control.step == "line_search"
    assert loaded.control.step_iterations == 4
    assert np.array_equal(np.asarray(loaded.fitted()), np.asarray(fit.fitted()))


def test_stabsel_runs_on_a_searched_fit():
    rng = np.random.default_rng(2)
    data = {"y": (rng.uniform(size=300) < 0.1).astype(float),
            "x": rng.uniform(size=300), "z": rng.uniform(size=300)}
    fit = mb.glmboost("y ~ bols(x) + bols(z)", data=data, family=mb.Binomial(),
                      control=mb.boost_control(20, 0.1, step="line_search"))
    result = fit.stabsel(cutoff=0.75, q=1, PFER=2.0, B=15)
    assert result.phat is not None


def test_the_cyclic_lss_fits_search_each_parameter():
    rng = np.random.default_rng(4)
    n = 400
    x1 = rng.uniform(-1, 1, n)
    data = {"y": 2.0 + x1 + rng.normal(scale=0.4, size=n), "x1": x1}
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x1, df=3)", "sigma": "y ~ 1"},
                         data=data, family=mb.GaussianLSS(),
                         control=mb.boost_control(30, 0.1, step="line_search"))
    assert np.isfinite(fit.risk_).all()
    assert fit.risk_[-1] < fit.risk_[0]
    cv = fit.cvrisk(folds=mb.cv_weights(len(data["y"]), type="kfold",
                                      B=3, seed=1))
    assert np.isfinite(cv.aggregate).all()


def test_the_noncyclic_method_refuses_the_rule_with_the_reason():
    rng = np.random.default_rng(0)
    data = {"y": rng.normal(size=120), "x": rng.normal(size=120)}
    with pytest.raises((ValueError, RuntimeError), match="non-cyclic"):
        mb.glmboostLSS({"mu": "y ~ bols(x)", "sigma": "y ~ 1"}, data=data,
                       family=mb.GaussianLSS(), method="noncyclic",
                       control=mb.boost_control(15, 0.1, step="line_search"))
