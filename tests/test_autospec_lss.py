"""The family probe's distributional candidates (#228).

The probe's mean-family candidates cannot express a precision or a shape
that moves with the covariates, so a budgeted search now offers BetaLSS
inside (0, 1) and GammaLSS for a spread that grows with the level, scores
each against the surviving mean family on held-out log score, and adopts
the distributional family only when it clearly wins. These tests pin the
three outcomes: adoption when the second parameter carries signal, refusal
when it does not, and a recipe whose fit() builds the distributional model.
"""

import numpy as np
import pytest

import mboostrs as mb

N = 400
BUDGET = 50


def _bell_in_interval(seed):
    """A Gaussian bell away from the interval's ends: beta gains nothing."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, N)
    y = np.clip(0.5 + 0.1 * x + rng.normal(0, 0.1, N), 0.02, 0.98)
    return {"y": y, "x": x}


def _beta_moving_precision(seed):
    """Beta data whose precision moves with x: the second predictor pays."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, N)
    mu = 1 / (1 + np.exp(-(0.3 + 0.6 * x)))
    phi = np.exp(2.2 + 0.8 * x)
    return {"y": rng.beta(mu * phi, (1 - mu) * phi), "x": x}


def _gamma_moving_shape(seed):
    """Gamma data whose coefficient of variation moves with x."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, N)
    mu = np.exp(0.4 + 0.5 * x)
    cv = np.exp(0.1 + 0.45 * x)
    shape = 1.0 / cv ** 2
    return {"y": rng.gamma(shape, mu / shape), "x": x}


def test_beta_lss_is_adopted_when_the_precision_moves():
    rec = mb.autospec(_beta_moving_precision(7), "y",
                      budget=BUDGET, mstop=100, seed=0)
    evidence = rec.evidence["family"]

    assert isinstance(rec.family, mb.BetaLSS)
    assert evidence.lss_score < evidence.mean_log_score
    assert evidence.score_difference_se > 0
    assert evidence.lss_mstop
    assert "nats/row" in evidence.sentence
    assert "BetaLSS carries both parameters" in evidence.sentence


def test_beta_lss_is_refused_for_a_bell_inside_the_interval():
    rec = mb.autospec(_bell_in_interval(14), "y",
                      budget=BUDGET, mstop=100, seed=0)
    evidence = rec.evidence["family"]

    assert isinstance(rec.family, mb.Gaussian)
    assert evidence.lss_score > evidence.mean_log_score
    assert "not clearly better, so Gaussian" in evidence.sentence


def test_gamma_lss_is_adopted_when_the_shape_moves():
    rec = mb.autospec(_gamma_moving_shape(11), "y",
                      budget=BUDGET, mstop=100, seed=0)
    evidence = rec.evidence["family"]

    assert isinstance(rec.family, mb.GammaLSS)
    assert evidence.lss_score < evidence.mean_log_score
    assert "GammaLSS carries both parameters" in evidence.sentence


def test_the_lss_recipe_fits_a_distributional_model():
    data = _beta_moving_precision(7)
    rec = mb.autospec(data, "y", budget=BUDGET, mstop=100, seed=0)
    model = rec.fit(data)

    assert type(model).__name__ == "LSSModel"
    assert list(model.param_names) == ["mu", "phi"]
    parameters = model.predict_parameters(newdata={"x": np.array([0.5, -1.0])})
    assert set(parameters) == {"mu", "phi"}
    assert all(np.isfinite(values).all() for values in parameters.values())
    # mu on the probability (response) scale: negative x, positive effect of
    # x on the mean (#326: predict_parameters() defaults to the response).
    assert parameters["mu"][0] > parameters["mu"][1]


def test_the_search_stops_at_the_family_and_says_so():
    rec = mb.autospec(_beta_moving_precision(7), "y",
                      budget=BUDGET, mstop=100, seed=0)

    assert "search_stopped_at_family" in rec.evidence
    assert "mu predictor" in rec.evidence["search_stopped_at_family"]
    report = rec.report()
    assert "the family probe adopted BetaLSS" in report
    # Deselection and the mean-model step count do not apply to the
    # distributional model; the shape probe runs on its mu predictor.
    assert "mstop" not in rec.measured
    assert "deselect" not in rec.evidence


def test_without_a_budget_the_proportion_keeps_its_mean_family():
    """The distributional check costs one fit per fold, so it only runs
    when the budget allows it; the no-budget route keeps the pointer to
    BetaLSS in its warning instead."""
    with pytest.warns(UserWarning, match=r"lies in \(0, 1\).*BetaLSS"):
        rec = mb.autospec(_beta_moving_precision(7), "y")

    assert isinstance(rec.family, mb.Gaussian)
    assert "family" not in rec.evidence
    assert "search_stopped_at_family" not in rec.evidence


def test_lss_family_reprs_read_like_the_mean_families():
    """A recommendation's recorded family name is rendered in reports and
    benchmark artifacts, so an LSS family prints as `BetaLSS()` the way
    `GammaReg()` does, not as a memory address; a set stabilization is
    printed and the default is not."""
    assert repr(mb.BetaLSS()) == "BetaLSS()"
    assert repr(mb.GammaLSS()) == "GammaLSS()"
    assert repr(mb.GaussianLSS(stabilization="MAD")) == \
        "GaussianLSS(stabilization='MAD')"


# ── stabilized candidate, extended grid and the mean check ─────────────────


def test_the_adopted_lss_family_is_stabilized_and_its_checks_are_recorded():
    """The candidate is fitted with L2 gradient stabilization, the final fit
    keeps it, and the evidence carries both checks with their numbers."""
    data = _gamma_moving_shape(11)
    rec = mb.autospec(data, "y", budget=BUDGET, mstop=100, seed=0)
    evidence = rec.evidence["family"]

    assert isinstance(rec.family, mb.GammaLSS)
    assert rec.family.stabilization == "L2"
    assert evidence.lss_stabilization == "L2"
    assert evidence.lss_mean_check == "passed"
    assert np.isfinite(evidence.lss_mse) and np.isfinite(evidence.mean_family_mse)
    assert evidence.lss_mse <= 1.02 * evidence.mean_family_mse or (
        evidence.lss_mse - evidence.mean_family_mse < 2.0 * evidence.lss_mse_se)
    assert "squared error of its mean" in evidence.sentence
    assert "L2 stabilization" in evidence.sentence
    model = rec.fit(data)
    assert model._stabilization == "L2"


def test_a_worse_mean_rejects_the_lss_candidate(monkeypatch):
    """The log score wins, but the candidate's mean is fitted worse than the
    mean family's by more than 2% and two paired standard errors: the mean
    family stands and the evidence says why."""
    from mboostrs import _autospec

    real = _autospec._lss_fold_mean_mse

    def worse_mean(*args, **kwargs):
        errors, fits = real(*args, **kwargs)
        return 1.5 * errors, fits

    monkeypatch.setattr(_autospec, "_lss_fold_mean_mse", worse_mean)
    rec = mb.autospec(_gamma_moving_shape(11), "y",
                      budget=BUDGET, mstop=100, seed=0)
    evidence = rec.evidence["family"]

    assert not hasattr(rec.family, "param_names")
    assert evidence.lss_score < evidence.mean_log_score
    assert evidence.lss_mean_check == "failed"
    assert evidence.lss_mse > 1.4 * evidence.mean_family_mse
    assert "a worse mean" in evidence.sentence
    assert "carries both parameters" not in evidence.sentence


def test_the_mean_rule_is_the_squared_error_rule():
    from mboostrs._autospec import _mse_worse

    assert _mse_worse(1.05, 1.0, 0.01)
    assert not _mse_worse(1.05, 1.0, 0.05)      # inside two standard errors
    assert not _mse_worse(1.01, 1.0, 0.001)     # inside the 2% margin
    assert _mse_worse(1.6, 1.0, 10.0)           # a runaway fit


def test_the_lss_step_grid_is_extended_while_the_optimum_is_at_its_top():
    """A grid whose top is 4 steps ends at its edge on this data; the grid is
    doubled until the optimum is inside it."""
    import warnings

    from mboostrs._autospec import _lss_candidate, _lss_fold_logscores

    data = _gamma_moving_shape(11)
    folds = mb.cv_weights(N, "kfold", 5, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _lss_fold_logscores(
            data, "y", "y ~ bbs(x)", _lss_candidate("GammaLSS"), folds,
            None, 4, 0.3, None)
    *_, at_boundary, largest = result

    assert largest > 4
    assert not at_boundary


# ── probes on the mu predictor ─────────────────────────────────────────────


def _gamma_step_in_mean(seed):
    """Gamma data whose mean has a step in x and whose shape moves with x."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, N)
    z = rng.normal(size=N)
    mu = np.exp(0.4 + 1.2 * (x > 0.3) + 0.2 * z)
    shape = np.exp(0.8 + 0.7 * x)
    return {"y": rng.gamma(shape, mu / shape), "x": x, "z": z}


def test_the_shape_probe_runs_on_the_mu_predictor_of_an_lss_family():
    data = _gamma_step_in_mean(3)
    # The distributional route gets half of the fits left after the family
    # check and the comparison's set-aside; 200 leaves it enough for the
    # shape probe.
    rec = mb.autospec(data, "y", budget=200, mstop=100, seed=0)

    assert isinstance(rec.family, mb.GammaLSS)
    assert "shapes" in rec.measured
    rows = {row.column: row for row in rec.evidence["shapes"]}
    assert set(rows) == {"x", "z"}
    assert type(rec.fit(data)).__name__ == "LSSModel"


def test_a_mu_change_that_worsens_the_mean_is_refused(monkeypatch):
    """A change that lowers the family's held-out loss is not adopted when
    the squared error of mu rises (the mean rule of the family choice)."""
    from mboostrs import _autospec

    real = _autospec._lss_fold_mean_mse
    calls = {"n": 0}

    def worse(*args, **kwargs):
        errors, fits = real(*args, **kwargs)
        calls["n"] += 1
        # the family check's own call is first; every later pair of calls
        # (candidate, incumbent) has the candidate's mean 1.6 times as bad
        return (1.6 * errors if calls["n"] > 1 and calls["n"] % 2 == 0 else errors), fits

    monkeypatch.setattr(_autospec, "_lss_fold_mean_mse", worse)
    rec = mb.autospec(_gamma_step_in_mean(3), "y", budget=200, mstop=100, seed=0)

    assert "shapes" in rec.measured      # the probe ran, so the refusal is real
    assert not rec.evidence.get("lss_mu_search", {}).get("shapes")
