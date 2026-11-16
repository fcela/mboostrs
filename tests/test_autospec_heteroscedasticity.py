"""autospec's scale probe: a Gaussian whose spread moves with a covariate (#322).

The family probe asks which distribution the response comes from, assuming one
spread for every row. For a Gaussian the second question is whether the spread
itself depends on a predictor -- which is GaussianLSS, with a scale equation.
These tests pin the outcomes: adoption when a covariate drives the spread
(including a two-level covariate the older `explore.heteroscedasticity` screen
skipped), refusal when the spread is constant, no proposal without a budget,
and a graceful decline when the candidate's fit panics (a `BaseException`, #319)
rather than aborting the whole search.
"""

import numpy as np

import mboostrs as mb

BUDGET = 60
MSTOP = 60


def _spread_moves_with_group(seed):
    """A homoscedastic mean with a group-dependent spread: part-time work with
    a wider log-wage spread than full-time work, the #322 shape."""
    rng = np.random.default_rng(seed)
    n = 800
    x = rng.normal(size=n)
    part_time = rng.choice([0.0, 1.0], size=n)
    sigma = np.where(part_time > 0.5, 2.5, 1.0)
    y = 1.0 + 0.8 * x + sigma * rng.normal(size=n)
    return {"y": y, "x": x, "part_time": part_time}


def _constant_spread(seed):
    """The same mean and covariates, with one spread for every row."""
    rng = np.random.default_rng(seed)
    n = 600
    x = rng.normal(size=n)
    part_time = rng.choice([0.0, 1.0], size=n)
    y = 1.0 + 0.8 * x + 0.5 * part_time + rng.normal(size=n)
    return {"y": y, "x": x, "part_time": part_time}


def test_gaussian_lss_is_adopted_when_the_spread_moves_with_a_covariate():
    """The residual screen regresses log(residual^2) on the covariates; a
    two-level covariate clears the known-variance |t| bar, so GaussianLSS is
    offered sigma ~ part_time and wins on held-out log score."""
    rec = mb.autospec(_spread_moves_with_group(1), "y",
                      budget=BUDGET, mstop=MSTOP)

    assert isinstance(rec.family, mb.GaussianLSS)
    assert "heteroscedasticity" in rec.measured
    assert isinstance(rec.formula, dict)
    assert set(rec.formula) == {"mu", "sigma"}
    assert "part_time" in rec.formula["sigma"]
    evidence = rec.evidence["family"]
    assert evidence.lss_score < evidence.mean_log_score
    assert evidence.score_difference_se > 0
    assert "GaussianLSS carries both parameters" in evidence.sentence


def test_the_heteroscedastic_recipe_fits_a_scale_model():
    """rec.fit() builds the distributional model the scale equation names."""
    data = _spread_moves_with_group(1)
    rec = mb.autospec(data, "y", budget=BUDGET, mstop=MSTOP)
    model = rec.fit(data)

    assert type(model).__name__ == "LSSModel"
    assert list(model.param_names) == ["mu", "sigma"]
    # The fitted scale is wider where the covariate marks the noisy group.
    parameters = model.predict_parameters(
        newdata={"x": np.array([0.0, 0.0]), "part_time": np.array([0.0, 1.0])})
    assert np.isfinite(parameters["sigma"]).all()
    assert parameters["sigma"][1] > parameters["sigma"][0]


def test_a_constant_spread_keeps_the_mean_only_gaussian():
    """The screen runs and reports, but no covariate drives the spread, so the
    mean-only Gaussian stands -- and the screen is recorded as measured."""
    rec = mb.autospec(_constant_spread(7), "y", budget=BUDGET, mstop=MSTOP)

    assert type(rec.family).__name__ == "Gaussian"
    assert "heteroscedasticity" in rec.measured
    evidence = rec.evidence["heteroscedasticity"]
    assert type(evidence.recommended).__name__ == "Gaussian"
    assert "no covariate's |t| clears" in evidence.sentence
    assert "spread:" in rec.report()


def test_without_a_budget_no_scale_equation_is_proposed():
    """The probe costs one fit per fold, so the default keeps the dtype
    Gaussian and spends nothing."""
    rec = mb.autospec(_spread_moves_with_group(1), "y")

    assert type(rec.family).__name__ == "Gaussian"
    assert "heteroscedasticity" not in rec.evidence
    assert rec.fits == 0


class _Panic(BaseException):
    """Stands in for the native `PanicException` the #319 finding hit."""


def test_a_panicking_lss_candidate_declines_instead_of_crashing(monkeypatch):
    """A candidate fit that raises a `BaseException` (not an `Exception`) must
    make the probe decline, not propagate and kill autospec (#319)."""
    import mboostrs.gamlss as gamlss

    def boom(*args, **kwargs):
        raise _Panic("digamma(0)")

    monkeypatch.setattr(gamlss, "cvrisk_gamboostLSS", boom)
    rec = mb.autospec(_spread_moves_with_group(1), "y",
                      budget=BUDGET, mstop=MSTOP)

    assert type(rec.family).__name__ == "Gaussian"
    assert "could not be fitted" in rec.evidence["heteroscedasticity"].sentence
