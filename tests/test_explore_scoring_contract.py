"""Independent fold loops for the specification assistants' scoring contract."""

from dataclasses import replace

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture
def data():
    rng = np.random.default_rng(24)
    x = rng.normal(size=100)
    return {"x": x, "y": x + rng.normal(size=100) * np.linspace(.2, 3, 100)}


def manual_mse(data, training_weights, held_out, evaluation_weights, family=None):
    scores = []
    for b in range(training_weights.shape[1]):
        train = training_weights[:, b] > 0
        held = held_out[:, b] & (evaluation_weights > 0)
        fit = mb.gamboost(
            "y ~ bols(x)", data={k: v[train] for k, v in data.items()},
            weights=training_weights[train, b], family=family,
            control=mb.boost_control(30),
        )
        prediction = fit.predict(newdata={"x": data["x"][held]}, type="response")
        observed = data["y"][held]
        if isinstance(family, mb.Binomial) and np.any(data["y"] < 0):
            observed = (observed + 1) / 2
        scores.append(np.average((observed - prediction) ** 2, weights=evaluation_weights[held]))
    return np.mean(scores)


@pytest.mark.parametrize("expanding", [True, False])
def test_rolling_score_uses_only_the_requested_horizons(data, expanding):
    scheme = mb.rolling_folds(100, B=3, initial=40, horizon=10, expanding=expanding)
    expected = manual_mse(data, scheme.weights, scheme.held_out, np.ones(100))
    actual = mb.cv_risk("bols(x)", data, "y", folds=scheme, mstop=30, score="mse")
    assert actual == pytest.approx(expected, rel=1e-12)
    # Rows 70 onwards are unused by every fit and evaluation in this design.
    changed = {**data, "y": data["y"].copy()}
    changed["y"][70:] = 1e6
    assert mb.cv_risk("bols(x)", changed, "y", folds=scheme, mstop=30,
                      score="mse") == actual


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("rolling", [False, True])
def test_structured_weights_are_used_once_for_training_and_evaluation(data, explicit, rolling):
    weights = np.linspace(.25, 4, 100)
    weights[::11] = 0
    scheme = (mb.rolling_folds(100, B=3, initial=40, horizon=10, weights=weights)
              if rolling else mb.grouped_folds(np.repeat(np.arange(20), 5), B=4, weights=weights))
    expected = manual_mse(data, scheme.weights, scheme.held_out, weights)
    actual = mb.cv_risk("bols(x)", data, "y", folds=scheme, mstop=30, score="mse",
                        weights=weights if explicit else None)
    assert actual == pytest.approx(expected, rel=1e-12)


def test_conflicting_base_weights_are_refused(data):
    scheme = mb.grouped_folds(np.repeat(np.arange(20), 5), B=4, weights=np.full(100, 2.))
    with pytest.raises(ValueError, match="weights.*base_weights"):
        mb.cv_risk("bols(x)", data, "y", folds=scheme, weights=np.ones(100))


def test_bootstrap_multipliers_compose_with_base_weights_once(data):
    weights = np.linspace(.3, 2, 100)
    weights[:10] = 0
    draws = mb.cv_weights(100, type="bootstrap", B=3, seed=13)
    expected = manual_mse(data, draws * weights[:, None], draws == 0, weights)
    actual = mb.cv_risk("bols(x)", data, "y", folds=draws, weights=weights,
                        mstop=30, score="mse")
    assert actual == pytest.approx(expected, rel=1e-12)


def test_judge_can_restrict_the_declared_held_out_rows(data):
    scheme = mb.blocked_folds(100, B=4)
    judge = scheme.held_out.copy()
    judge[::2] = False
    scheme = replace(scheme, judge=judge)
    expected = manual_mse(data, scheme.weights, judge, np.ones(100))
    assert mb.cv_risk("bols(x)", data, "y", folds=scheme, mstop=30,
                      score="mse") == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("family,coding", [(mb.Binomial(), "01"),
                                          (mb.Binomial(), "signed"), (mb.Poisson(), "count")])
def test_mse_compares_response_scale_values(data, family, coding):
    rng = np.random.default_rng(7)
    if coding == "count":
        y = rng.poisson(np.exp(.5 + .3 * data["x"])).astype(float)
    else:
        y = (rng.uniform(size=100) < 1 / (1 + np.exp(-data["x"]))).astype(float)
        if coding == "signed":
            y = 2 * y - 1
    data = {**data, "y": y}
    folds = mb.cv_weights(100, type="kfold", B=4, seed=13)
    expected = manual_mse(data, folds, folds == 0, np.ones(100), family)
    actual = mb.cv_risk("bols(x)", data, "y", family=family, folds=folds,
                        mstop=30, score="mse")
    assert actual == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("folds", [0, 1, 2.5, True, 101])
def test_invalid_fold_counts_are_rejected(data, folds):
    with pytest.raises((ValueError, TypeError), match="folds"):
        mb.cv_risk("bols(x)", data, "y", folds=folds, mstop=1)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
def test_invalid_fold_entries_are_rejected(data, bad):
    folds = mb.cv_weights(100, type="kfold", B=4)
    folds[0, 0] = bad
    with pytest.raises(ValueError, match="folds.*finite.*non-negative"):
        mb.cv_risk("bols(x)", data, "y", folds=folds, mstop=1)


@pytest.mark.parametrize("field", ["held_out", "judge"])
def test_invalid_mask_shapes_are_rejected(data, field):
    # Refused where the scheme is built, before any fold loop reads it.
    with pytest.raises(ValueError, match=field):
        replace(mb.blocked_folds(100, B=4), **{field: np.zeros((99, 4), dtype=bool)})


def test_training_and_evaluation_overlap_is_rejected(data):
    with pytest.raises(ValueError, match="holds out rows it also trains on"):
        replace(mb.blocked_folds(100, B=4), held_out=np.ones((100, 4), dtype=bool))


def test_empty_effective_evaluation_fold_is_rejected(data):
    # The first block holds only zero-weight rows, so fold 0 has nothing to score.
    weights = np.r_[np.zeros(25), np.ones(75)]
    with pytest.raises(ValueError, match="fold 0 scores no rows"):
        mb.blocked_folds(100, B=4, weights=weights)


def test_judge_outside_held_out_is_rejected(data):
    with pytest.raises(ValueError, match="judges rows it trains on|judge must be a subset"):
        replace(mb.blocked_folds(100, B=4), judge=np.ones((100, 4), dtype=bool))


def test_manual_scheme_uses_complete_training_weights(data):
    weights = np.linspace(.3, 2, 100)
    partition = mb.cv_weights(100, type="kfold", B=4, seed=1)
    complete = partition * weights[:, None]
    scheme = mb.FoldScheme(complete, partition == 0, "manual")
    expected = manual_mse(data, complete, partition == 0, weights)
    assert mb.cv_risk("bols(x)", data, "y", folds=scheme, weights=weights,
                      score="mse", mstop=30) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("weights", [np.ones((100, 1)), np.full(100, np.nan),
                                   np.full(100, np.inf), -np.ones(100), np.zeros(100)])
def test_invalid_observation_weights_are_rejected(data, weights):
    with pytest.raises(ValueError, match="weights"):
        mb.cv_risk("bols(x)", data, "y", folds=4, weights=weights, mstop=1)


def test_no_scored_rows_after_omission_is_reported(data):
    scheme = mb.rolling_folds(100, B=2, initial=40, horizon=10)
    data = {**data, "x": data["x"].copy()}
    data["x"][40:50] = np.nan
    with pytest.raises(ValueError, match="fold 0.*na_action"):
        mb.cv_risk("bols(x)", data, "y", folds=scheme, mstop=1, na_action="omit")


def test_base_weights_are_retained_for_never_trained_rows():
    weights = np.linspace(.2, 3, 100)
    scheme = mb.rolling_folds(100, B=2, initial=40, horizon=10, weights=weights)
    np.testing.assert_array_equal(scheme.base_weights, weights)
    assert np.all(scheme.weights[50:60] == 0)
    weights[:] = 0
    assert np.all(scheme.base_weights > 0)


def test_scheme_arrays_are_frozen_copies():
    weights = np.r_[np.ones(10), np.zeros(10)][:, None] * np.ones((1, 2))
    weights[:, 1] = 1 - weights[:, 0]
    scheme = mb.FoldScheme(weights, weights == 0, "manual")
    weights[0, 0] = 99.0
    assert scheme.weights[0, 0] == 1.0
    with pytest.raises(ValueError):
        scheme.weights[0, 0] = 5.0


@pytest.mark.parametrize("bad", [-1.0, np.nan])
def test_negative_or_nan_weights_are_refused_at_construction(bad):
    weights = np.ones((10, 2))
    weights[:5, 0] = 0
    weights[5:, 1] = 0
    weights[6, 0] = bad
    with pytest.raises(ValueError, match="finite and non-negative"):
        mb.FoldScheme(weights, weights == 0, "manual")
