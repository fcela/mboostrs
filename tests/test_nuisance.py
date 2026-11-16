"""`nuisance()` returns the family's nuisance parameter at the model's step
count, as R's `nuisance()` does, and slicing keeps it."""

import numpy as np
import pytest

import mboostrs as mb


def _counts(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    mu = np.exp(0.5 + x)
    return {"x": x, "y": rng.negative_binomial(2, 2 / (2 + mu)).astype(float)}


@pytest.mark.parametrize("fit", [
    lambda d: mb.glmboost("y ~ x", data=d, family=mb.NBinomial()),
    lambda d: mb.gamboost("y ~ bbs(x)", data=d, family=mb.NBinomial()),
])
def test_nbinomial_dispersion_is_reported_and_follows_slicing(fit):
    model = fit(_counts())
    full = model.nuisance()
    assert full is not None and full.shape == (1,)
    assert 1.0 < full[0] < 4.0
    sliced = model[:30].nuisance()
    assert sliced is not None and sliced[0] != full[0]
    assert model[:30][:30].nuisance()[0] == sliced[0]


def test_a_sliced_proportional_odds_model_still_predicts_probabilities():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 400)
    latent = 1.5 * x + rng.logistic(size=400)
    y = 1 + (latent > -0.5).astype(int) + (latent > 0.8).astype(int)
    fit = mb.gamboost("y ~ bbs(x)", data={"x": x, "y": y.astype(float)},
                      family=mb.PropOdds(n_levels=3))
    probabilities = np.asarray(fit[:40].predict(type="response"))
    assert probabilities.shape == (400, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_gammareg_shape_and_gaussian_none():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 800)
    y = rng.gamma(3.0, np.exp(0.5 * x) / 3.0)
    shape = mb.gamboost("y ~ bbs(x)", data={"x": x, "y": y},
                        family=mb.GammaReg()).nuisance()
    assert shape is not None and 2.5 < shape[0] < 3.5
    assert mb.gamboost("y ~ bbs(x)", data={"x": x, "y": y}).nuisance() is None


def test_propodds_with_four_levels_keeps_its_thresholds_ordered():
    """Searching each threshold on its own let them cross with four levels; a
    middle class's probability went negative and the first threshold grew
    without bound. The thresholds are now parameterized as R's `d2s` does,
    and a linear fit matches R's (mboost 2.9, 100 steps) to 3e-4."""
    from pathlib import Path

    import pandas as pd

    data = pd.read_csv(Path(__file__).parent / "fixtures" / "propodds_four_levels.csv")
    fit = mb.glmboost("y ~ x1 + x2", data=data, family=mb.PropOdds(n_levels=4),
                      control=mb.boost_control(100, 0.1))
    thresholds = fit.nuisance()
    assert np.all(np.diff(thresholds) > 0)
    np.testing.assert_allclose(thresholds, [-1.0592464, -0.1517431, 0.7462317], atol=1e-3)
    coef = fit.coef()
    np.testing.assert_allclose([coef["x1"], coef["x2"]], [1.09469086, 0.37903859], atol=1e-3)
    probabilities = np.asarray(fit.predict(type="response"))
    assert np.all(probabilities >= 0) and np.allclose(probabilities.sum(axis=1), 1.0)


def _ordinal(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    code = np.digitize(1.5 * x + rng.logistic(size=n), [-0.5, 0.8])
    return x, code


@pytest.mark.parametrize("make", [
    lambda x, code: ({"x": x, "y": np.array(["low", "mid", "high"])[code]},
                     mb.PropOdds(levels=["low", "mid", "high"])),
    lambda x, code: ({"x": x, "y": code}, mb.PropOdds(levels=[0, 1, 2])),
    lambda x, code: (__import__("pandas").DataFrame({
        "x": x, "y": __import__("pandas").Categorical(
            np.array(["low", "mid", "high"])[code],
            categories=["low", "mid", "high"], ordered=True)}), mb.PropOdds()),
    lambda x, code: (__import__("polars").DataFrame({
        "x": x, "y": __import__("polars").Series(
            np.array(["low", "mid", "high"])[code],
            dtype=__import__("polars").Enum(["low", "mid", "high"]))}), mb.PropOdds()),
])
def test_propodds_takes_ordered_labels(make, tmp_path):
    x, code = _ordinal()
    reference = mb.glmboost("y ~ x", data={"x": x, "y": code + 1.0},
                            family=mb.PropOdds(n_levels=3)).nuisance()
    data, family = make(x, code)
    fit = mb.glmboost("y ~ x", data=data, family=family)
    np.testing.assert_allclose(fit.nuisance(), reference, rtol=1e-12)
    path = tmp_path / "po.json"
    mb.save_model(fit, path)
    np.testing.assert_allclose(np.asarray(mb.load_model(path).predict(type="response")),
                               np.asarray(fit.predict(type="response")))


def test_propodds_refuses_labels_without_an_order():
    x, code = _ordinal()
    with pytest.raises(ValueError, match="order is not known"):
        mb.gamboost("y ~ bbs(x)", data={"x": x, "y": np.array(["a", "b", "c"])[code]},
                    family=mb.PropOdds())


def test_glmboost_gaussian_nuisance_is_none_like_gamboost():
    """An empty nuisance path is not a parameter. glmboost used to return a
    zero-length array, which is not None, so nuisance()[0] raised IndexError
    for Gaussian while gamboost returned None."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 80)
    y = rng.normal(size=80)
    data = {"x": x, "y": y}
    model = mb.glmboost("y ~ x", data=data, family=mb.Gaussian(),
                        control=mb.boost_control(20, 0.1))
    assert model.nuisance() is None
    assert model[:10].nuisance() is None
