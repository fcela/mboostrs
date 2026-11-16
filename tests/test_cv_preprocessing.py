"""Fixed-design resampling and validation of a refitted preprocessing pipeline."""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._cv_preprocessing import check_preprocessing
from mboostrs._refit import RefitSpec


def _data(family):
    rng = np.random.default_rng(71)
    x = rng.normal(size=36)
    z = rng.normal(size=36)
    if isinstance(family, mb.NBinomial):
        y = rng.negative_binomial(2, 0.5, len(x)).astype(float)
    elif isinstance(family, mb.GammaReg):
        y = rng.gamma(2, 1, len(x))
    elif isinstance(family, mb.Poisson):
        y = rng.poisson(np.exp(0.3 * x)).astype(float)
    elif isinstance(family, mb.Binomial):
        y = rng.binomial(1, 0.5, len(x)).astype(float)
    else:
        y = 0.5 * x + z + rng.normal(size=len(x))
    return {"x": x, "z": z, "y": y}


@pytest.mark.parametrize("fit,formula", [
    (mb.glmboost, "y ~ scale(x) + z"),
    (mb.gamboost, "y ~ bbs(scale(x), knots=4) + bols(z)"),
])
@pytest.mark.parametrize("family", [mb.Gaussian(), mb.NBinomial(), mb.GammaReg(),
                                     mb.Huber(), mb.Binomial(), mb.Poisson(), mb.Quantile(tau=0.7)])
def test_refitted_cv_matches_training_only_fits(fit, formula, family):
    from mboostrs._rust import family_held_out_score
    from mboostrs.families import _family_params, _family_rust_name

    data = _data(family)
    weights = np.tile([1.0, 2.0, 3.0], 12)
    control = mb.boost_control(8, 0.02)
    model = fit(formula, data, family=family, weights=weights, weight_policy="raw", control=control)
    assignment = np.arange(36) % 3
    folds = (assignment[:, None] != np.arange(3)).astype(float)
    result = model.cvrisk(folds=folds, preprocessing="refit", _inner=True)
    for b in range(3):
        train, held = assignment != b, assignment == b
        train_data = {key: value[train] for key, value in data.items()}
        held_data = {key: value[held] for key, value in data.items()}
        reference = fit(formula, train_data, family=family, control=control,
                         weights=weights[train], weight_policy="raw")
        nuisance = (reference._nuisance_vec_path if fit is mb.glmboost
                    else reference._family_sigma_path)
        for k in range(9):
            eta = np.ascontiguousarray(reference.predict(held_data, mstop=k, type="link"))
            ns = nuisance[k] if nuisance else []
            expected = family_held_out_score(_family_rust_name(family), _family_params(family),
                data["y"][held], eta, weights[held], ns[0] if ns else None, None)
            np.testing.assert_allclose(result.risk_matrix[b, k], expected, rtol=1e-11, atol=1e-11)
    assert result.preprocessing == "refit"
    assert result.to_dict()["preprocessing"] == "refit"
    assert result.report().preprocessing == "refit"
    np.testing.assert_allclose(result.aggregate, result.risk_matrix.mean(axis=0))
    assert result.optimal_mstop > 0


def test_held_out_rows_do_not_set_scaling(monkeypatch):
    x = np.r_[np.arange(12, dtype=float), 1000.0, 2000.0]
    data = {"x": x, "y": np.sin(x)}
    observed = []
    build = RefitSpec.build

    def record(spec, **options):
        model = build(spec, **options)
        scale = next(step for step in model._transform_recipe.steps if step.kind == "scale")
        observed.append(scale.params[0])
        return model

    model = mb.glmboost("y ~ scale(x)", data, control=mb.boost_control(4))
    folds = np.r_[np.ones(12), 0, 0][:, None]
    monkeypatch.setattr(RefitSpec, "build", record)
    model.cvrisk(folds=folds, preprocessing="refit", at_mstop=3)
    assert observed == [5.5]
    assert model.cvrisk(folds=folds, at_mstop=3).preprocessing == "fixed"


@pytest.mark.parametrize("preprocessing", ["fixed", "refit"])
def test_structured_held_out_mask_preserves_unused_rows(preprocessing):
    data = {"x": np.arange(12, dtype=float), "y": np.r_[np.arange(8, dtype=float), np.full(4, 1e6)]}
    weights = np.r_[np.ones(6), np.zeros(6)][:, None]
    held = np.zeros_like(weights, dtype=bool)
    held[6:8] = True
    scheme = mb.FoldScheme(weights=weights, held_out=held, scheme="rolling", base_weights=np.ones(12))
    model = mb.glmboost("y ~ x", data, control=mb.boost_control(4))
    result = model.cvrisk(folds=scheme, preprocessing=preprocessing, _inner=True)
    assert np.max(result.risk_matrix) < 100


def test_refitted_cv_budget_and_truncation():
    from mboostrs._workflow import Workflow

    data = _data(mb.Gaussian())
    model = mb.glmboost("y ~ scale(x)", data, control=mb.boost_control(8))
    folds = (np.arange(36)[:, None] % 2 != np.arange(2)).astype(float)
    with Workflow(mb.CostBudget(fits=2)) as workflow:
        result = model.cvrisk(folds=folds, preprocessing="refit", at_mstop=4)
    assert workflow.report()["fit_charges"] == 2
    assert result.preprocessing == "refit"
    assert list(result.grid) == list(range(5))


def test_preprocessing_refusals():
    with pytest.raises(ValueError, match="fixed.*refit"):
        check_preprocessing("all-data")
    data = _data(mb.Gaussian())
    model = mb.glmboost("y ~ x", data, control=mb.boost_control(3))
    with pytest.raises(ValueError, match="training and evaluation"):
        model.cvrisk(folds=np.ones((36, 1)), preprocessing="refit")
    for bad in (float("nan"), -1.0):
        folds = np.r_[np.ones(24), np.zeros(12)][:, None]
        folds[0, 0] = bad
        with pytest.raises(ValueError, match="finite.*non-negative"):
            model.cvrisk(folds=folds, preprocessing="refit")


@pytest.mark.parametrize("preprocessing", ["fixed", "refit"])
def test_scheme_carries_complete_fit_and_evaluation_weights(preprocessing):
    data = _data(mb.Gaussian())
    parent_weights = np.tile([2.0, 5.0, 8.0], 12)
    model = mb.glmboost("y ~ x", data, center=False, weights=parent_weights,
                        weight_policy="raw", control=mb.boost_control(4))
    training = np.arange(36) < 24
    held = ~training
    scheme = mb.FoldScheme(weights=training[:, None].astype(float), held_out=held[:, None],
                           scheme="supplied", base_weights=np.arange(1, 37, dtype=float))
    result = model.cvrisk(folds=scheme, preprocessing=preprocessing, _inner=True)
    reference = mb.glmboost("y ~ x", {key: value[training] for key, value in data.items()},
                            center=False, control=mb.boost_control(4))
    prediction = reference.predict({"x": data["x"][held]})
    expected = np.average((data["y"][held] - prediction)**2, weights=scheme.base_weights[held])
    np.testing.assert_allclose(result.risk_matrix[0, 4], expected, atol=1e-12)


def test_glm_grid_prediction_without_intercept():
    data = _data(mb.Gaussian())
    model = mb.glmboost("y ~ x - 1", data, center=False, control=mb.boost_control(5))
    held = {"x": np.array([-1.0, 0.0, 1.0])}
    path = model.predict_path([0, 2, 5], newdata=held)
    for row, k in enumerate([0, 2, 5]):
        np.testing.assert_allclose(path[row], model.predict(held, mstop=k), atol=1e-14)
