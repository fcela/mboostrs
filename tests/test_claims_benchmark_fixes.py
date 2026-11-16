"""Library defects found by the claims-discovery agent benchmark.

Each test reproduces what an analyst agent met: string labels through the
REML refits, conformal intervals and LSS prediction intervals; missing
values in distributional fits; and error messages that point at the wrong
remedy.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def labelled():
    rng = np.random.default_rng(0)
    n = 900
    df = pl.DataFrame({"x": rng.uniform(0, 1, n), "z": rng.uniform(0, 1, n),
                       "g": rng.choice(["a", "b", "c", "d"], n)})
    return df.with_columns(
        y=np.sin(3 * df["x"].to_numpy()) + 0.5 * (df["g"] == "d").to_numpy()
          + rng.normal(0, 0.2 + 0.4 * df["z"].to_numpy(), n))


@pytest.fixture(scope="module")
def incomplete(labelled):
    rng = np.random.default_rng(1)
    n = labelled.height
    x = labelled["x"].to_numpy()
    g = labelled["g"].to_numpy().astype(object)
    return labelled.with_columns(
        pl.Series("x", np.where(rng.uniform(size=n) < 0.1, np.nan, x)),
        pl.Series("g", np.where(rng.uniform(size=n) < 0.05, None, g).tolist(), dtype=pl.String),
    )


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def test_gamlss_refit_reads_string_labels(labelled):
    lss = _quiet(mb.gamboostLSS, {"mu": "y ~ bbs(x) + brandom(g, df=3)", "sigma": "y ~ bbs(z)"},
                 data=labelled, families=mb.GaussianLSS(), control=mb.boost_control(mstop=200))
    refit = mb.gamlss_refit(lss)
    grid = pl.DataFrame({"x": [0.5] * 4, "z": [0.5] * 4, "g": ["a", "b", "c", "d"]})
    mu = refit.predict(grid, parameter="mu")
    boosted = lss.predict(newdata=grid, parameter="mu", type="response")
    assert np.argmax(mu) == 3
    np.testing.assert_allclose(mu, boosted, atol=0.15)
    assert refit.draws(grid, n_draws=5)["mu"].shape == (5, 4)
    effects = refit.effects(grid, parameter="mu")
    assert set(effects) == {"bbs(x)", "brandom(g, df=3)"}


def test_confint_refit_with_string_brandom(labelled):
    model = _quiet(mb.gamboost, "y ~ bbs(x) + brandom(g)", labelled)
    ci = _quiet(model.confint, method="refit")
    assert set(ci["lower"]) == {"bbs(x)", "brandom(g)"}
    assert len(ci["lower"]["brandom(g)"]) == labelled.height


def test_lss_prediction_interval_with_string_factor(labelled):
    lss = _quiet(mb.gamboostLSS, {"mu": "y ~ bbs(x) + brandom(g)", "sigma": "y ~ bbs(z)"},
                 data=labelled, families=mb.GaussianLSS())
    band = lss.prediction_interval(level=0.9, newdata=labelled.head(4))
    assert band.height == 4
    assert np.all(band["lower"].to_numpy() < band["upper"].to_numpy())


@pytest.mark.parametrize("method", ["split", "cv+", "cqr"])
def test_conformal_with_string_factor(labelled, method):
    train, test = labelled[:600], labelled[600:]
    model = _quiet(mb.gamboost, "y ~ bbs(x) + brandom(g)", train)
    ci = _quiet(mb.conformal_interval, model, test, level=0.9, method=method, seed=0)
    frame = ci.to_frame()
    y = test["y"].to_numpy()
    covered = (y >= frame["lower"].to_numpy()) & (y <= frame["upper"].to_numpy())
    assert 0.8 < covered.mean() < 0.97


def test_conformal_held_out_calibration_with_string_factor(labelled):
    model = _quiet(mb.gamboost, "y ~ bbs(x) + brandom(g)", labelled[:400])
    ci = _quiet(mb.conformal_interval, model, labelled[600:], level=0.9,
                calibration=labelled[400:600])
    assert ci.to_frame().height == 300


def test_conformal_distributional_with_string_factor(labelled):
    lss = _quiet(mb.gamboostLSS, {"mu": "y ~ bbs(x) + brandom(g)", "sigma": "y ~ bbs(z)"},
                 data=labelled[:600], families=mb.GaussianLSS())
    ci = _quiet(mb.conformal_interval, lss, labelled[600:], level=0.9, seed=0)
    assert ci.to_frame().height == 300


def test_lss_refuses_missing_values_by_default(incomplete):
    with pytest.raises(ValueError, match="na_action='impute'"):
        _quiet(mb.gamboostLSS, {"mu": "y ~ bbs(x) + bols(g)", "sigma": "y ~ bbs(z)"},
               data=incomplete, families=mb.GaussianLSS())


def test_lss_impute_adds_indicators_in_every_parameter_and_predicts(incomplete):
    lss = _quiet(mb.gamboostLSS,
                 {"mu": "y ~ bbs(x) + brandom(g)", "sigma": "y ~ bbs(z) + bbs(x)"},
                 data=incomplete, families=mb.GaussianLSS(), na_action="impute",
                 control=mb.boost_control(mstop=300))
    names = {p: set(v) for p, v in lss.selected.items()}
    assert "x_missing" in names["mu"] | names["sigma"]
    new = pl.DataFrame({"x": [0.5, None], "z": [0.5, 0.5], "g": ["a", None]})
    mu = lss.predict(newdata=new, parameter="mu", type="response")
    assert np.all(np.isfinite(mu))
    refit = mb.gamlss_refit(lss)
    assert "bols(x_missing, intercept=False)" in refit.terms["mu"] + refit.terms["sigma"]
    assert np.all(np.isfinite(refit.predict(new, parameter="mu")))


def test_lss_omit_drops_rows_and_weights(incomplete):
    n = incomplete.height
    lss = _quiet(mb.gamboostLSS, {"mu": "y ~ bbs(x) + bols(g)", "sigma": "y ~ bbs(z)"},
                 data=incomplete, families=mb.GaussianLSS(), na_action="omit",
                 weights=np.ones(n))
    complete = incomplete.filter(pl.col("x").is_not_nan() & pl.col("g").is_not_null()).height
    assert len(lss.predict(parameter="mu")) == complete


def test_glmboostlss_impute(incomplete):
    lss = _quiet(mb.glmboostLSS, "y ~ x + z", data=incomplete.drop("g"),
                 families=mb.GaussianLSS(), na_action="impute")
    mu = lss.predict(newdata=pl.DataFrame({"x": [None, 0.3], "z": [0.5, 0.5]}),
                     parameter="mu", type="response")
    assert np.all(np.isfinite(mu))


def test_gam_refit_and_confint_on_imputed_gamboost(incomplete):
    model = _quiet(mb.gamboost, "y ~ bbs(x) + bbs(z) + brandom(g)", incomplete,
                   na_action="impute", control=mb.boost_control(mstop=300))
    refit = mb.gam_refit(model)
    assert "bols(x_missing, intercept=False)" in refit.terms
    new = pl.DataFrame({"x": [0.5, None], "z": [0.5, 0.5], "g": ["a", None]})
    np.testing.assert_allclose(refit.predict(new), model.predict(newdata=new), atol=0.1)
    ci = _quiet(model.confint, method="refit")
    assert "bols(x_missing, intercept=False)" in ci["lower"]


def test_missing_value_message_names_na_action(incomplete):
    with pytest.raises(ValueError, match="na_action='impute'"):
        _quiet(mb.gamboost, "y ~ bbs(x)", incomplete)


def test_unseen_level_message_flags_a_coding_mismatch():
    from mboostrs.interactions import _level_kind_hint

    assert "coded differently" in _level_kind_hint([0.0, 1.0, 2.0], ["a", "c"])
    assert _level_kind_hint(["a", "b"], ["z"]) is None


def test_empty_formula_points_at_intercept_only():
    df = pl.DataFrame({"x": [0.1, 0.5, 0.9] * 20, "y": np.arange(60, dtype=float)})
    with pytest.raises(ValueError, match="y ~ 1"):
        _quiet(mb.gamboostLSS, {"mu": "y ~ ", "sigma": "y ~ 1"}, data=df,
               families=mb.GaussianLSS())
    fit = _quiet(mb.gamboostLSS, {"mu": "y ~ 1", "sigma": "y ~ 1"}, data=df,
                 families=mb.GaussianLSS())
    assert np.allclose(fit.predict(newdata=df.head(2), parameter="mu", type="response"),
                       df["y"].mean(), atol=1.0)
