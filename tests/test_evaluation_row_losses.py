"""Evaluation records reconstruct the family score; both averages are reported."""

import numpy as np
import pytest

import mboostrs as mb


def _data(family, n=103, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    if isinstance(family, mb.NBinomial):
        y = rng.poisson(np.exp(0.5 * x + 1)).astype(float)
    elif isinstance(family, mb.Binomial):
        y = (x + rng.normal(size=n) > 0).astype(float)
    elif isinstance(family, mb.GammaReg):
        y = rng.gamma(2.0, np.exp(0.3 * x) / 2.0)
    else:
        y = x + rng.normal(size=n)
    return {"x": x, "y": y}


@pytest.mark.parametrize("family", [mb.Gaussian(), mb.NBinomial(), mb.Binomial(),
                                    mb.GammaReg(), mb.Laplace()],
                         ids=lambda f: type(f).__name__)
def test_family_row_losses_reconstruct_each_fold_score(family):
    result = mb.evaluate_specification("bbs(x)", _data(family), "y", family=family,
                                       folds=5, mstop=50)
    for fold, expected in enumerate(result.fold_scores):
        rows = [r for r in result.records if r["fold"] == fold]
        weights = np.array([r["evaluation_weight"] for r in rows])
        losses = np.array([r["loss"] for r in rows])
        assert np.all(np.isfinite(losses))
        assert weights @ losses / weights.sum() == pytest.approx(expected, rel=1e-10)


def test_a_coupled_loss_has_no_row_losses():
    """The AUC loss compares pairs of observations, so no row has a loss of
    its own; the records say so with NaN rather than an invented share."""
    result = mb.evaluate_specification("bbs(x)", _data(mb.Binomial()), "y", family=mb.AUC(),
                                       folds=4, mstop=30)
    assert np.all(np.isnan([r["loss"] for r in result.records]))


def test_the_observation_weighted_mean_differs_for_unequal_folds():
    data = _data(mb.Gaussian(), n=100)
    groups = np.repeat(np.arange(4), [10, 20, 30, 40])
    result = mb.evaluate_specification("bbs(x)", data, "y", mstop=50,
                                       folds=mb.grouped_folds(groups, B=4))
    totals = np.array([row["evaluation_weight_total"] for row in result.folds])
    assert len(set(totals)) > 1
    expected = float(np.sum(result.fold_scores * totals) / totals.sum())
    assert result.observation_weighted_score == pytest.approx(expected)
    assert result.observation_weighted_score != pytest.approx(result.score)
    assert "observation-weighted mean" in result.describe()


def test_the_result_records_the_native_build():
    result = mb.evaluate_specification("bbs(x)", _data(mb.Gaussian()), "y", folds=3, mstop=20)
    assert result.build_id == mb.build_info()["build_id"]
