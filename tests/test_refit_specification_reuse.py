"""One description of the fit, used by everything that refits it.

Continuation, bootstrap confidence intervals, conformal calibration and
stability selection all refit the model, and each rebuilt the call from
whatever attributes it happened to know about. Each knew a different subset,
so a fit with anything unusual about it was refitted as a plainer model under
the same name.

The sharpest case is `confint`. It drew bootstrap weights as
`binomial(1, 0.5, n)`, ignoring the model's own weights, where `mboost`'s
`confint.mboost` draws them with `cv(model.weights(object), B = B)` --
multinomial counts with probabilities proportional to those weights, so a row
the fit gave no weight is never drawn. Measured on a 300-row fit where the
first half has slope +2, the second half slope -2, and the second half carries
weight zero: the fit estimates +2.02 and the bootstrap interval came back as
`[-0.508, 0.241]`, centred on -0.13. It did not contain the estimate. The
replicates had fitted all 300 rows, where the two halves cancel.
"""

import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def split_weights():
    """Half the rows carry weight zero, and the two halves disagree."""
    rng = np.random.default_rng(5)
    n = 300
    x = rng.normal(size=n)
    weights = np.ones(n)
    weights[n // 2:] = 0.0
    y = np.where(np.arange(n) < n // 2, 2.0 * x, -2.0 * x) + rng.normal(size=n) * 0.5
    return {"x": x, "y": y}, weights


# ─── bootstrap resamples respect the model's weights ───────────────


def test_the_interval_contains_the_estimate_it_is_an_interval_for(split_weights):
    data, weights = split_weights
    model = mb.glmboost("y ~ x", data=data, weights=weights,
                        control=mb.boost_control(200, 0.1))
    slope = model.coef()["x"]
    assert slope == pytest.approx(2.0, abs=0.2), "the weighted fit sees only +2"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interval = model.confint(parm=["x"], B=100, B_mstop=0, seed=1)
    low, high = interval["lower"]["x"], interval["upper"]["x"]
    assert low <= slope <= high, (
        f"the interval [{low:.4f}, {high:.4f}] must contain the estimate "
        f"{slope:.4f} it is an interval for"
    )
    assert low > 0, "an interval built from the weighted rows cannot span zero"


def test_a_zero_weight_row_is_never_resampled(split_weights):
    """R's `cvboot` is `rmultinom(B, n, weights/sum(weights))`, so a row with
    probability zero is drawn zero times."""
    data, weights = split_weights
    n = len(weights)
    folds = mb.cv_weights(n, type="bootstrap", B=20, seed=1, weights=weights)
    assert np.all(folds[n // 2:] == 0.0)
    assert np.allclose(folds.sum(axis=0), n)


def test_the_bootstrap_result_says_whether_it_was_weighted(split_weights):
    data, weights = split_weights
    model = mb.glmboost("y ~ x", data=data, weights=weights,
                        control=mb.boost_control(50, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interval = model.confint(parm=["x"], B=20, B_mstop=0, seed=1)
    assert "weighted" in interval["weights"]


def test_a_gamboost_interval_uses_the_model_weights_too(split_weights):
    """The partial effect of `x` on the weighted half is `+2x`, so the
    interval must sit above zero where `x` is positive and below it where `x`
    is negative -- which it cannot do if the replicates fitted the unweighted
    rows, where the two halves cancel."""
    data, weights = split_weights
    model = mb.gamboost("y ~ bols(x)", data=data, weights=weights,
                        control=mb.boost_control(200, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interval = model.confint(parm=["x"], B=60, B_mstop=0, seed=1)
    lower, upper = interval["lower"]["x"], interval["upper"]["x"]
    fitted = model.partial_effects(which="x")["x"][1]
    assert np.all(lower <= fitted + 1e-9) and np.all(fitted <= upper + 1e-9)
    high = data["x"] > 1.0
    assert np.all(lower[high] > 0), (
        "a positive effect at large x should have an interval above zero"
    )


# ─── every refit carries every hyperparameter ──────────────────────


@pytest.fixture(scope="module")
def unusual():
    """A fit with several things about it that a hand-built call would drop."""
    rng = np.random.default_rng(7)
    n = 400
    x, z = rng.normal(size=n), rng.normal(size=n)
    return {"x": x, "z": z, "y": 2.0 * x + 0.5 * z + rng.normal(size=n)}


def test_continue_to_keeps_the_weight_policy(unusual):
    rng = np.random.default_rng(8)
    weights = rng.uniform(0.2, 0.9, size=len(unusual["y"]))
    model = mb.glmboost("y ~ x + z", data=unusual, weights=weights,
                        weight_policy="raw", control=mb.boost_control(30, 0.1))
    grown = model.continue_to(60)
    assert grown.weight_policy == "raw"
    assert grown.weights.sum() == pytest.approx(model.weights.sum())


def test_continue_to_keeps_the_binning(unusual):
    model = mb.gamboost("y ~ bbs(x, df=4, bins=32)", data=unusual,
                        control=mb.boost_control(30, 0.1))
    grown = model.continue_to(60)
    assert [spec.bins for spec in grown._learners_spec] == [32]


def test_continue_to_still_reproduces_the_shorter_fit(unusual):
    """The equivalence the method promises: growing then slicing back gives
    the model you started with."""
    model = mb.glmboost("y ~ x + z", data=unusual,
                        control=mb.boost_control(30, 0.1))
    back = model.continue_to(60)[:30]
    np.testing.assert_allclose(back.fitted(), model.fitted(), atol=1e-12)
    assert back.selected == model.selected


def test_continue_to_still_carries_an_interaction_recipe(unusual):
    rng = np.random.default_rng(9)
    data = dict(unusual)
    data["g"] = rng.integers(0, 4, size=len(data["y"])).astype(float)
    model = mb.gamboost("y ~ bbs(x, df=4) %X% brandom(g)", data=data,
                        control=mb.boost_control(40, 0.1))
    grown = model.continue_to(80)
    assert grown._interactions is not None
    assert np.all(np.isfinite(grown.predict(newdata=data)))


def test_stability_selection_refits_the_model_it_was_asked_about(unusual):
    """A fold refit used to carry five arguments; `bins` was not among them,
    so stability selection on a binned fit selected from the exact model."""
    model = mb.gamboost("y ~ bbs(x, df=4, bins=32) + bbs(z, df=4, bins=32)",
                        data=unusual, control=mb.boost_control(40, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.stabsel(q=1, PFER=1.0, B=6)
    assert result.n_fits == 12
    assert set(result.selected) <= {0, 1}


def test_the_refit_spec_carries_what_the_fit_accepted(unusual):
    from mboostrs._refit import refit_spec

    model = mb.gamboost("y ~ bbs(x, df=4)", data=unusual, bins=64,
                        na_action="omit", control=mb.boost_control(20, 0.1))
    spec = refit_spec(model)
    assert spec.arguments["bins"] == 64
    assert spec.arguments["na_action"] == "omit"
    assert spec.arguments["weight_policy"] == "upstream"


def test_the_spec_keeps_the_learners_even_when_the_call_cannot_take_them():
    """`gamboost` refuses `learners=` and `env=` together, but
    `row_subset_blockers()` reads the learners to see what a row subset would
    misalign -- so the spec keeps both and only the call chooses."""
    from mboostrs._refit import refit_spec

    rng = np.random.default_rng(10)
    n = 120
    data = {"x": rng.normal(size=n), "g": rng.integers(0, 3, size=n).astype(float)}
    data["y"] = data["x"] + rng.normal(size=n)
    model = mb.gamboost("y ~ bbs(x, df=4) %X% brandom(g)", data=data,
                        control=mb.boost_control(20, 0.1))
    spec = refit_spec(model)
    assert spec.arguments.get("env") is not None
    assert spec.arguments.get("learners") is not None
    assert spec.row_subset_blockers(), (
        "an interaction recipe blocks a row subset, and the blocker is found "
        "by reading the learners"
    )
    assert np.all(np.isfinite(spec.build().fitted()))


# ─── what a scoring-only artifact cannot do, said once ─────────────
# A full artifact and a pickle keep the fit's call and refit (issue #820);
# a scoring-only artifact drops it and refuses every refit.


@pytest.fixture
def loaded(unusual):
    model = mb.glmboost("y ~ x + z", data=unusual,
                        control=mb.boost_control(30, 0.1))
    path = Path(tempfile.mkdtemp()) / "model.json"
    mb.save_model(model, path, scoring_only=True)
    return mb.load_model(path)


def test_a_fitted_model_has_nothing_unavailable(unusual):
    model = mb.glmboost("y ~ x", data=unusual,
                        control=mb.boost_control(20, 0.1))
    assert model.unavailable_operations() == []


def test_a_loaded_artifact_lists_every_refit_operation(loaded):
    rows = loaded.unavailable_operations()
    assert {row["operation"] for row in rows} == {
        "continue_to", "cvrisk", "stabsel", "confint", "conformal_interval",
        "update",
    }
    for row in rows:
        assert row["needs"]
        assert "portable artifact" in row["reason"]


def test_an_unpickled_model_keeps_every_refit_operation(unusual):
    model = mb.glmboost("y ~ x", data=unusual,
                        control=mb.boost_control(20, 0.1))
    restored = pickle.loads(pickle.dumps(model))
    assert restored.unavailable_operations() == []


def test_every_listed_operation_actually_refuses(loaded, unusual):
    """A list that does not match what happens is worse than no list."""
    calls = {
        "continue_to": lambda: loaded.continue_to(60),
        "cvrisk": lambda: loaded.cvrisk(type="kfold", B=3),
        "stabsel": lambda: loaded.stabsel(q=1, PFER=1.0, B=4),
        "confint": lambda: loaded.confint(parm=["x"], B=4, B_mstop=0),
        "conformal_interval": lambda: loaded.conformal_interval(newdata=unusual),
        "update": lambda: loaded.update(mstop=60),
    }
    for operation, call in calls.items():
        with pytest.raises(ValueError, match="no record of the call|did not retain"):
            call()
        assert operation in {row["operation"]
                             for row in loaded.unavailable_operations()}


def test_the_refusals_no_longer_name_an_internal(loaded):
    """`stabsel` reached `_extract_column` on `None` and reported
    `Cannot extract column 'y' from <class 'NoneType'>`, which names nothing
    the caller can act on."""
    with pytest.raises(ValueError, match="stabsel is unavailable"):
        loaded.stabsel(q=1, PFER=1.0, B=4)
    with pytest.raises(ValueError, match="cvrisk is unavailable"):
        loaded.cvrisk(type="kfold", B=3)


def test_what_a_loaded_artifact_can_still_do(loaded, unusual):
    """The complement, so the list is read as a boundary and not a verdict on
    the artifact format."""
    assert np.all(np.isfinite(loaded.predict(newdata=unusual)))
    assert loaded.coef()
    assert loaded.update(mstop=10).mstop == 10
    assert loaded.summary().as_text()
    assert loaded.selected


def test_aic_refuses_rather_than_returning_none_after_a_load(loaded):
    """It used to answer `None`, which is a missing value with no reason
    attached -- the caller cannot tell "this family has no likelihood" from
    "this artifact carries no learner store"."""
    with pytest.raises(ValueError, match="hat-matrix trace"):
        loaded.aic()
    with pytest.raises(ValueError, match="hatvalues.. is unavailable"):
        loaded.hatvalues()


def test_hatvalues_refuses_by_name_for_a_learner_that_has_no_hat_matrix():
    rng = np.random.default_rng(11)
    n = 200
    data = {"x": rng.normal(size=n)}
    data["y"] = np.sin(data["x"]) + rng.normal(size=n) * 0.2
    model = mb.gamboost("y ~ bmono(x)", data=data,
                        control=mb.boost_control(20, 0.1))
    with pytest.raises(NotImplementedError, match="bmono"):
        model.hatvalues()
    with pytest.raises(NotImplementedError, match="not a linear map"):
        model.aic()


@pytest.mark.parametrize("output", ["polars", "pandas"])
def test_the_list_can_come_back_as_a_frame(loaded, output):
    frame = loaded.unavailable_operations(output=output)
    assert type(frame).__module__.split(".")[0] == output
