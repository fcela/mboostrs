"""Gamma step reliability under frequency weights and changing shape."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _tariff():
    root = Path(__file__).resolve().parents[1]
    data = pl.read_csv(root / "docs/book/case_studies/swedish_motor_1977.csv", comment_prefix="#")
    data = data.filter(pl.col("Claims") > 0).with_columns((pl.col("Payment") / pl.col("Claims")).alias("y"))
    names = []
    for column, reference in {"Kilometres": 1, "Zone": 4, "Bonus": 1, "Make": 1}.items():
        for level in sorted(data[column].unique().to_list()):
            if level != reference:
                name = f"{column.lower()}_{level}"
                data = data.with_columns((pl.col(column) == level).cast(pl.Float64).alias(name))
                names.append(name)
    return data, names


def test_tariff_fixed_step_warns_and_search_converges_to_gamma_glm():
    sm = pytest.importorskip("statsmodels.api")
    data, names = _tariff()
    weights = data["Claims"].to_numpy()
    y = data["y"].to_numpy()
    formula = "y ~ " + " + ".join(names)
    glm = sm.GLM(y, sm.add_constant(data.select(names).to_numpy()),
                 family=sm.families.Gamma(sm.families.links.Log()), freq_weights=weights).fit()
    with pytest.warns(RuntimeWarning, match="training risk rose"):
        unstable = mb.glmboost(formula, data, family=mb.GammaReg(), weights=weights,
                                weight_policy="raw", control=mb.boost_control(400, 0.1))
    assert unstable.risk_[-1] > np.min(unstable.risk_) + 1
    stable = mb.glmboost(formula, data, family=mb.GammaReg(), weights=weights,
                         weight_policy="raw", control=mb.boost_control(2000, 1, step="line_search"))
    np.testing.assert_allclose(stable.predict(type="response"), glm.fittedvalues, rtol=2e-5, atol=0.01)
    assert np.max(np.diff(stable.risk_)) < 1e-6


@pytest.mark.parametrize("kind", ["glm", "gam"])
def test_gamma_search_path_replays_and_cvrisk_keeps_rule(kind):
    rng = np.random.default_rng(44)
    x = rng.normal(size=90)
    data = {"x": x, "y": rng.gamma(20, 5000 * np.exp(0.2 * x) / 20)}
    weights = np.ones(90)
    weights[:8] = np.arange(500, 1300, 100)
    fit = mb.glmboost if kind == "glm" else mb.gamboost
    formula = "y ~ x" if kind == "glm" else "y ~ bols(x)"
    model = fit(formula, data, family=mb.GammaReg(), weights=weights, weight_policy="raw",
                 control=mb.boost_control(80, 0.5, step="line_search"))
    assert np.max(np.diff(model.risk_)) < 1e-7
    np.testing.assert_allclose(model.predict({"x": x}), model.fitted_, rtol=1e-13, atol=1e-13)
    folds = (np.arange(90)[:, None] % 3 != np.arange(3)).astype(float)
    result = model.cvrisk(folds=folds, _inner=True)
    assert np.all(np.isfinite(result.risk_matrix))
    assert model.control.step == "line_search"


@pytest.mark.r_parity
def test_gamma_fixed_path_retains_mboost_convention():
    import rpy2.robjects as ro

    data, names = _tariff()
    ro.r("suppressMessages(library(mboost))")
    ro.globalenv["d"] = ro.DataFrame({key: ro.FloatVector(data[key].to_numpy()) for key in ["y", *names]})
    ro.globalenv["w"] = ro.FloatVector(data["Claims"].to_numpy())
    formula = "y ~ " + " + ".join(names)
    ro.r(f"u <- mboost::glmboost({formula},d,weights=w,family=mboost::GammaReg(),"
         "control=mboost::boost_control(mstop=400,nu=.1))")
    with pytest.warns(RuntimeWarning, match="training risk rose"):
        model = mb.glmboost(formula, data, family=mb.GammaReg(), weights=data["Claims"].to_numpy(),
                             weight_policy="raw", control=mb.boost_control(400, 0.1))
    np.testing.assert_allclose(model.predict(), np.asarray(ro.r("as.numeric(predict(u))")),
                               rtol=0, atol=1e-7)


@pytest.mark.r_parity
@pytest.mark.parametrize("kind", ["glm", "gam"])
def test_gamma_search_and_cv_agree_between_frontends(kind):
    import rpy2.robjects as ro

    rng = np.random.default_rng(44)
    x = rng.normal(size=90)
    data = {"x": x, "y": rng.gamma(20, 5000 * np.exp(0.2 * x) / 20)}
    weights = np.ones(90)
    weights[:8] = np.arange(500, 1300, 100)
    folds = (np.arange(90)[:, None] % 3 != np.arange(3)).astype(float)
    fit = mb.glmboost if kind == "glm" else mb.gamboost
    formula = "y ~ x" if kind == "glm" else "y ~ bols(x)"
    model = fit(formula, data, family=mb.GammaReg(), weights=weights, weight_policy="raw",
                control=mb.boost_control(80, 0.5, step="line_search"))
    # Loading the namespace keeps unrelated mboost parity tests' S3 generics
    # and unqualified fitting functions on their upstream package.
    ro.r("stopifnot(requireNamespace('mboostrs', quietly=TRUE))")
    ro.globalenv["d"] = ro.DataFrame({key: ro.FloatVector(value) for key, value in data.items()})
    ro.globalenv["w"] = ro.FloatVector(weights)
    ro.globalenv["folds"] = ro.r.matrix(ro.FloatVector(folds.ravel(order="F")), nrow=90)
    ro.r(f"u <- mboostrs::{kind}boost({formula}, d, weights=w, weight_policy='raw',"
         "family=mboostrs::GammaReg(),"
         "control=mboostrs::boost_control(80,.5,step='line_search'))")
    np.testing.assert_allclose(model.predict(type="link"), np.asarray(ro.r("as.numeric(predict(u))")),
                               rtol=0, atol=1e-10)
    result = model.cvrisk(folds=folds, _inner=True)
    np.testing.assert_allclose(result.risk_matrix,
                               np.asarray(ro.r("mboostrs::cvrisk(u, folds=folds)$risk_matrix")),
                               rtol=1e-9, atol=1e-9)
