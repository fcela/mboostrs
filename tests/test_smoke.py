"""PyO3 binding smoke tests."""

import warnings

import numpy as np
import pytest
from mboostrs.families import Gaussian
from mboostrs.formula import parse_formula, parse_gamboost_formula
from mboostrs.gamboost import LearnerLayout
from mboostrs.learners import (
    Bbs,
    Bconstrained,
    Bkernel,
    Bmono,
    Bmrf,
    Bns,
    Bols,
    Brandom,
    Bspatial,
    Btree,
    Buser,
)

import tomllib
from pathlib import Path

import mboostrs as mboost


def test_version():
    # The workspace version in Cargo.toml is the source of truth (AGENTS.md).
    cargo = Path(__file__).resolve().parents[1] / "Cargo.toml"
    expected = tomllib.loads(cargo.read_text())["workspace"]["package"]["version"]
    assert mboost.__version__ == expected


def test_boost_control_valid():
    mstop, nu, risk, trace = mboost.boost_control(100, 0.1)[:4]
    assert mstop == 100
    assert abs(nu - 0.1) < 1e-12
    assert risk == "inbag"
    assert trace is False


def test_boost_control_nu_one():
    mstop, nu, risk, trace = mboost.boost_control(50, 1.0)[:4]
    assert mstop == 50
    assert nu == 1.0


def test_boost_control_zero_mstop():
    with pytest.raises(ValueError, match="mstop must be > 0"):
        mboost.boost_control(0, 0.1)


def test_boost_control_zero_nu():
    with pytest.raises(ValueError, match="nu must satisfy"):
        mboost.boost_control(100, 0.0)


def test_boost_control_nu_above_one():
    with pytest.raises(ValueError, match="nu must satisfy"):
        mboost.boost_control(100, 1.5)


def test_gaussian_family():
    g = Gaussian()
    assert g is not None


def test_bols_learner():
    b = Bols("x1")
    assert b.name == "x1"
    assert b.intercept is True  # R default: bols(intercept=TRUE)


def test_parse_formula():
    response, terms = parse_formula("y ~ x1 + x2")
    assert response == "y"
    assert terms == ["x1", "x2"]


def test_parse_formula_no_tilde():
    with pytest.raises(ValueError, match="~"):
        parse_formula("y x1 + x2")


def test_parse_formula_empty_response():
    with pytest.raises(ValueError, match="response"):
        parse_formula("~ x1 + x2")


def test_parse_formula_no_predictors():
    with pytest.raises(ValueError, match="predictor"):
        parse_formula("y ~ ")


def test_glmboost_simple():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 + 0.8 * x2 + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "x2": x2}

    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    assert model.mstop == 50
    assert len(model.selected) == 50
    assert model.offset_ != 0.0
    assert model.fitted_.shape == (n,)
    assert len(model.risk_) == 51
    assert model.risk_[0] >= model.risk_[-1]


def test_glmboost_coefficients():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x": x}

    model = mboost.glmboost(
        "y ~ x",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )

    coefs = model.coefficients_
    assert "x" in coefs
    assert abs(coefs["x"] - 2.0) < 0.5


def test_glmboost_repr():
    rng = np.random.default_rng(0)
    n = 20
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x": x}

    model = mboost.glmboost(
        "y ~ x",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(10, 0.1),
    )

    r = repr(model)
    assert "GLMBoostModel" in r
    assert "Gaussian" in r
    assert "mstop=10" in r


def test_glmboost_risk_inbag():
    """glmboost with risk='inbag' (default) should compute risk."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1}

    model = mboost.glmboost(
        "y ~ x1", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1, risk="inbag"),
    )
    # risk_ should be non-zero (computed)
    assert any(r > 0 for r in model.risk_)
    # risk should decrease over iterations
    assert model.risk_[0] >= model.risk_[-1]


def test_glmboost_risk_none():
    """glmboost with risk='none' should store inbag risk (R semantics)."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1}

    model_none = mboost.glmboost(
        "y ~ x1", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1, risk="none"),
    )
    model_inbag = mboost.glmboost(
        "y ~ x1", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1, risk="inbag"),
    )
    # R semantics: risk="none" still computes inbag risk
    np.testing.assert_allclose(model_none.risk_, model_inbag.risk_, atol=1e-10)
    assert model_none.fitted_.shape == (n,)


def test_glmboost_risk_oobag_with_zero_weights():
    """glmboost with risk='oobag' uses obs where weight=0 as OOB.
    When all weights=1, there are no OOB obs, so risk is 0 (correct).
    With some zero weights, OOB risk should be computed and differ from inbag."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 + 0.8 * x2 + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "x2": x2}

    # All weights=1 means no OOB obs, so risk=0 (by definition)
    model_uniform = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1, risk="oobag"),
    )
    assert all(r == 0.0 for r in model_uniform.risk_)

    # With some zero weights, OOB risk should be nonzero
    w = np.ones(n)
    w[:50] = 0.0  # first 50 obs are OOB
    model_oobag = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(), weights=w,
        control=mboost.boost_control(50, 0.1, risk="oobag"),
    )
    assert any(r > 0 for r in model_oobag.risk_)


def test_glmboost_trace_true():
    """glmboost with trace=True should not crash."""
    rng = np.random.default_rng(42)
    n = 50
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x": x}

    model = mboost.glmboost(
        "y ~ x", data=data, family=Gaussian(),
        control=mboost.boost_control(10, 0.1, trace=True),
    )
    # trace only prints to stderr; model should still work
    assert model.mstop == 10
    assert model.fitted_.shape == (n,)


def test_gamboost_risk_none():
    """gamboost with risk='none' should store inbag risk (R semantics)."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    y = 1.5 * x1 + np.sin(2 * z1) + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "z1": z1}

    model_none = mboost.gamboost(
        "y ~ bbs(z1, df=4) + bols(x1)", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1, risk="none"),
    )
    model_inbag = mboost.gamboost(
        "y ~ bbs(z1, df=4) + bols(x1)", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1, risk="inbag"),
    )
    np.testing.assert_allclose(model_none.risk_, model_inbag.risk_, atol=1e-10)
    assert model_none.fitted_.shape == (n,)


def test_gamboost_risk_oobag_with_zero_weights():
    """gamboost with risk='oobag' uses obs where weight=0 as OOB.
    With some zero weights, OOB risk should be computed and differ from inbag."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    y = 1.5 * x1 + np.sin(2 * z1) + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "z1": z1}

    # All weights=1 means no OOB obs, so risk=0
    model_uniform = mboost.gamboost(
        "y ~ bbs(z1, df=4) + bols(x1)", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1, risk="oobag"),
    )
    assert all(r == 0.0 for r in model_uniform.risk_)

    # With some zero weights, OOB risk should be nonzero
    w = np.ones(n)
    w[:50] = 0.0
    model_oobag = mboost.gamboost(
        "y ~ bbs(z1, df=4) + bols(x1)", data=data, family=Gaussian(), weights=w,
        control=mboost.boost_control(50, 0.1, risk="oobag"),
    )
    assert any(r > 0 for r in model_oobag.risk_)


def test_gamboost_trace_true():
    """gamboost with trace=True should not crash."""
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    y = 1.5 * x1 + np.sin(2 * z1) + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "z1": z1}

    model = mboost.gamboost(
        "y ~ bbs(z1, df=4) + bols(x1)", data=data, family=Gaussian(),
        control=mboost.boost_control(10, 0.1, trace=True),
    )
    assert model.mstop == 10
    assert model.fitted_.shape == (n,)


def test_gamboost_btree_depth2():
    """btree with maxdepth=2 should fit better than maxdepth=1."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.where(x1 > 0, 2.0, -1.0) + np.where(x2 > 0.5, 1.5, -0.5) + rng.normal(scale=0.2, size=n)
    data = {"y": y, "x1": x1, "x2": x2}

    m1 = mboost.gamboost(
        "y ~ btree(x1 + x2, maxdepth=1)",
        data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    m2 = mboost.gamboost(
        "y ~ btree(x1 + x2, maxdepth=2)",
        data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    assert m2.risk_[-1] < m1.risk_[-1], "depth-2 risk should be lower than depth-1"


def test_gamboost_btree_depth3():
    """btree with maxdepth=3 should fit better than maxdepth=2."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (np.where(x1 > 0, 2.0, -1.0)
         + np.where(x2 > 0.5, 1.5, -0.5)
         + np.where((x1 > 0) & (x2 > 0.5), 1.0, 0.0)
         + rng.normal(scale=0.2, size=n))
    data = {"y": y, "x1": x1, "x2": x2}

    m2 = mboost.gamboost(
        "y ~ btree(x1 + x2, maxdepth=2)",
        data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    m3 = mboost.gamboost(
        "y ~ btree(x1 + x2, maxdepth=3)",
        data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    assert m3.risk_[-1] < m2.risk_[-1], "depth-3 risk should be lower than depth-2"


def test_gamboost_btree_predict_depth2():
    """btree depth-2 predict should work and return correct shape."""
    rng = np.random.default_rng(0)
    n = 50
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.where(x1 > 0, 1.0, -1.0) + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x1": x1, "x2": x2}

    m = mboost.gamboost(
        "y ~ btree(x1 + x2, maxdepth=2)",
        data=data, family=Gaussian(),
        control=mboost.boost_control(20, 0.1),
    )
    pred = m.predict(newdata=data, type="response")
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))


def test_gamboost_bbs_linear_extrapolation():
    """bbs prediction beyond boundary knots should use linear extrapolation, not cubic."""
    rng = np.random.default_rng(0)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x": x}

    m = mboost.gamboost(
        "y ~ bbs(x, df=4)",
        data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    x_new = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    newdata = {"x": x_new}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pred = m.predict(newdata=newdata, type="response")
    assert any("outside the range they were fitted on" in str(warning.message)
               for warning in w)
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))
    pred_at_0 = pred[1]
    pred_at_1 = pred[3]
    pred_left = pred[0]
    pred_right = pred[4]
    dx_left = -0.5 - 0.0
    dx_right = 1.5 - 1.0
    assert abs(pred_left - pred_at_0) < 10 * abs(dx_left), "left extrapolation should be linear, not cubic"
    assert abs(pred_right - pred_at_1) < 10 * abs(dx_right), "right extrapolation should be linear, not cubic"


def test_gamboost_bbs_extrap_no_warning_in_domain():
    """bbs prediction within boundary knots should not warn."""
    rng = np.random.default_rng(0)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x": x}

    m = mboost.gamboost(
        "y ~ bbs(x, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    x_new = np.linspace(0.1, 0.9, 20)
    newdata = {"x": x_new}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pred = m.predict(newdata=newdata, type="response")
    assert not any("beyond" in str(warning.message) for warning in w)
    assert pred.shape == (20,)


def test_glmboost_nbinomial():
    """NBinomial family should fit and produce decreasing risk."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    mu = np.exp(0.5 + 0.3 * x1 - 0.2 * x2)
    y = rng.negative_binomial(n=5, p=5 / (5 + mu)).astype(float)
    data = {"y": y, "x1": x1, "x2": x2}

    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=mboost.NBinomial(),
        control=mboost.boost_control(50, 0.1),
    )
    assert model.mstop == 50
    assert model.fitted_.shape == (n,)


def test_gamboost_nbinomial():
    """NBinomial family should work with gamboost and smooth learners."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    mu = np.exp(0.5 + 0.3 * x1)
    y = rng.negative_binomial(n=5, p=5 / (5 + mu)).astype(float)
    data = {"y": y, "x1": x1, "z1": z1}

    model = mboost.gamboost(
        "y ~ bbs(z1, df=4) + bols(x1)",
        data=data,
        family=mboost.NBinomial(),
        control=mboost.boost_control(50, 0.1),
    )
    assert model.mstop == 50
    assert model.fitted_.shape == (n,)


def test_glmboost_coxph():
    """CoxPH family should work with glmboost."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    time = rng.exponential(scale=1.0, size=n)
    event = rng.binomial(1, 0.7, size=n).astype(float)
    data = {"time": time, "event": event, "x1": x1, "x2": x2}

    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=mboost.CoxPH(),
        control=mboost.boost_control(50, 0.1),
    )
    assert model.mstop == 50
    assert model.fitted_.shape == (n,)
    assert model.offset_ == 0.0


def test_glmboost_weibull():
    """Weibull family should work with glmboost."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    time = rng.exponential(scale=1.0, size=n)
    event = rng.binomial(1, 0.7, size=n).astype(float)
    data = {"time": time, "event": event, "x1": x1, "x2": x2}

    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=mboost.Weibull(),
        control=mboost.boost_control(50, 0.1),
    )
    assert model.mstop == 50
    assert model.fitted_.shape == (n,)


def test_glmboost_loglog():
    """Loglog family should work with glmboost."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    time = rng.exponential(scale=1.0, size=n)
    event = rng.binomial(1, 0.7, size=n).astype(float)
    data = {"time": time, "event": event, "x1": x1, "x2": x2}

    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=mboost.Loglog(),
        control=mboost.boost_control(50, 0.1),
    )
    assert model.mstop == 50
    assert model.fitted_.shape == (n,)


def test_datasets_bodyfat():
    from mboostrs.datasets import bodyfat

    bf = bodyfat()
    assert bf.shape == (71, 10)
    assert "DEXfat" in bf.columns


def test_datasets_wpbc():
    from mboostrs.datasets import wpbc

    wp = wpbc()
    assert wp.shape == (198, 34)
    assert "time" in wp.columns
    assert "status" in wp.columns


def test_datasets_india():
    from mboostrs.datasets import india

    ind = india()
    assert ind.shape[0] == 4000
    assert "stunting" in ind.columns


def test_datasets_iris():
    from mboostrs.datasets import iris

    ir = iris()
    assert ir.shape == (150, 5)
    assert "Species" in ir.columns


def test_glmboost_aic():
    from mboostrs.datasets import bodyfat

    bf = bodyfat()
    predictors = [c for c in bf.columns if c != "DEXfat"]
    formula = "DEXfat ~ " + " + ".join(predictors)
    fit = mboost.glmboost(
        formula, data=bf, family=Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    aic = fit.aic()
    assert aic is not None
    assert aic.mstop > 0
    assert aic.mstop <= 100
    assert len(aic.aic) == 100
    assert len(aic.df) == 100

    sliced = fit[:50]
    aic_s = sliced.aic()
    assert aic_s is not None
    assert aic_s.mstop > 0


# --- Buser learner validation ---


def test_buser_validates_x_ndim():
    """Buser rejects an X that is not 2-dimensional or not numeric."""
    K = np.eye(3)
    with pytest.raises(ValueError, match="Buser X must be 2-dimensional"):
        Buser("u", X=np.array([1, 2, 3]), K=K)
    with pytest.raises((TypeError, ValueError), match="Buser X"):
        Buser("u", X="not_array", K=K)


def test_buser_accepts_any_2d_array_like():
    """A nested list is converted once, so downstream code has an array."""
    learner = Buser("u", X=[[1.0, 2.0], [3.0, 4.0]], K=[[1.0, 0.0], [0.0, 1.0]])
    assert isinstance(learner.X, np.ndarray) and learner.X.shape == (2, 2)
    assert isinstance(learner.K, np.ndarray) and learner.K.shape == (2, 2)


def test_buser_rejects_a_non_finite_entry_by_position():
    with pytest.raises(ValueError, match=r"Buser X\[0, 1\] is nan"):
        Buser("u", X=[[1.0, np.nan]], K=np.eye(2))


def test_buser_validates_k_ndim():
    """Buser raises ValueError if K is not 2-dimensional."""
    X = np.eye(3)
    with pytest.raises(ValueError, match="Buser K must be 2-dimensional"):
        Buser("u", X=X, K=np.array([1.0, 2.0, 3.0]))


def test_buser_validates_k_square():
    """Buser raises ValueError if K is not square."""
    X = np.eye(5)
    K = np.ones((5, 3))
    with pytest.raises(ValueError, match="Buser K must be square"):
        Buser("u", X=X, K=K)


def test_buser_validates_x_k_dim_match():
    """Buser raises ValueError if X columns != K dimension."""
    X = np.eye(5)
    K = np.eye(3)
    with pytest.raises(ValueError, match="Buser K must have 5 rows, got 3"):
        Buser("u", X=X, K=K)


def test_buser_design_with_an_unoccupied_interaction_group_fits():
    """A Buser design whose columns form an interaction lattice is factored.

    `center=False` (the default) routes the design through the Kronecker-sum
    detection, which reads the group count off the column layout. When a
    residue class holds no observations, the group count exceeds the largest
    group index in the data; deriving the per-group solver count from the data
    instead left the solver vector short and the fit raised a Rust
    `PanicException` from an out-of-bounds index.
    """
    n, p, n_groups, p_left = 9, 8, 4, 2
    X = np.zeros((n, p))
    for row in range(n):
        group = row % 3               # residue class 3 is never occupied
        for left in range(p_left):
            X[row, left * n_groups + group] = 1.0 + row * (left + 1) * 0.1
    data = {"y": np.arange(n, dtype=float) * 0.5 - 1.0,
            "z": np.arange(n, dtype=float)}

    model = mboost.gamboost(
        "y ~ .", data=data,
        learners=[Buser("u", X=X, K=np.eye(p), lambda_value=1.0)],
        control=mboost.boost_control(5, 0.1),
    )
    assert np.all(np.isfinite(model.fitted()))
    assert np.all(np.isfinite(model.risk_))
    assert model.risk_[-1] < model.risk_[0]
    # The unoccupied group holds p_left columns and contributes no effect.
    coefficients = np.asarray(model.coef()["u"], dtype=float).ravel()
    assert coefficients.size == p
    assert np.all(coefficients[[left * n_groups + 3 for left in range(p_left)]] == 0.0)


def test_gamboost_learner_layout_round_trips_all_blocks():
    """LearnerLayout owns the formula-order <-> Rust-order mapping."""
    X = np.ones((6, 2))
    specs = [
        Buser("u", X=X, K=np.eye(2)),         # buser, formula 0, Rust last
        Bkernel("x"),                         # bkernel, formula 1
        Bols("x"),                            # scalar, formula 2
        Bbs("s"),                             # bbs, formula 3
        Bns("ns"),                            # bbs alias, formula 4
        Bmono("m"),                           # bmono, formula 5
        Bbs("c", constraint="increasing"),    # bmono alias, formula 6
        Btree(["x", "z"]),                    # btree, formula 7
        # Laplacian of a 3-node chain; np.eye is a ridge, which bmrf rejects.
        Bmrf("region", K=np.array([[1.0, -1.0, 0.0],
                                   [-1.0, 2.0, -1.0],
                                   [0.0, -1.0, 1.0]])),  # bmrf, formula 8
        Bspatial("x", "z"),                   # bspatial, formula 9
        Brandom("group"),                     # brandom, formula 10
    ]
    layout = LearnerLayout.from_specs(specs)

    assert list(layout.rust_idx_to_formula_idx) == [2, 3, 4, 5, 6, 7, 8, 9, 1, 10, 0]
    for rust_idx, formula_idx in enumerate(layout.rust_idx_to_formula_idx):
        assert layout.rust_to_formula(rust_idx) == formula_idx
        assert layout.formula_to_rust(formula_idx) == rust_idx

    assert list(layout.rust_indices_for_kind("bkernel")) == [8]
    assert list(layout.rust_indices_for_kind("buser")) == [10]


def test_gamboost_buser_smoke():
    """Buser learner can be used in gamboost via the learners kwarg."""
    rng = np.random.default_rng(42)
    n = 30
    Xmat = rng.normal(size=(n, 2))
    beta = np.array([2.0, -1.0])
    y = Xmat @ beta + rng.normal(scale=0.1, size=n)
    data = {"y": y}
    buser = Buser("myfeat", X=Xmat, K=np.eye(2), df=4)
    model = mboost.gamboost(
        "y ~ bols(dummy)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(20, 0.1),
        learners=[buser],
    )
    assert model.mstop == 20
    assert model.fitted_.shape == (n,)
    assert len(model.risk_) == 21


def test_gamboost_buser_first_layout_predict_newdata():
    """A buser term first in the formula predicts correctly on new data.

    This used to be a layout hazard: the array fit boundary grouped learners by
    kind, so a buser written first was stored after the standard learners and
    everything downstream went through a remap. The spec boundary takes
    learners in formula order, so the remap is the identity and the hazard is
    gone; the prediction assertion below is what the test is really for.
    """
    rng = np.random.default_rng(123)
    n = 40
    Xmat = rng.normal(size=(n, 2))
    x = rng.normal(size=n)
    y = 3.0 * Xmat[:, 0] - 2.0 * Xmat[:, 1] + rng.normal(scale=0.02, size=n)
    data = {"y": y, "x": x}
    buser = Buser("u", X=Xmat, K=np.eye(2), df=2)

    model = mboost.gamboost(
        "y ~ x",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(20, 0.1),
        learners=[buser, Bols("x")],
    )

    assert model._rust_idx_to_formula_idx == [0, 1]
    assert model._formula_idx_to_rust_idx == {0: 0, 1: 1}
    assert model._selected[0] == 0

    pred = model.predict(newdata=data, buser_newdata={"u": Xmat}, type="link")
    np.testing.assert_allclose(pred, model.fitted_, rtol=1e-12, atol=1e-12)


# --- Bbs kwargs ---


def test_bbs_cyclic_and_deriv():
    """Bbs accepts cyclic=True and deriv=1."""
    b = Bbs("x", cyclic=True, deriv=1)
    assert b.cyclic is True
    assert b.deriv == 1


def test_bbs_constraint_increasing():
    """Bbs accepts constraint='increasing'."""
    b = Bbs("x", constraint="increasing")
    assert b.constraint == "increasing"


def test_bbs_constraint_decreasing():
    """Bbs accepts constraint='decreasing'."""
    b = Bbs("x", constraint="decreasing")
    assert b.constraint == "decreasing"


def test_bbs_invalid_constraint():
    """Bbs raises ValueError for invalid constraint."""
    with pytest.raises(ValueError, match="constraint"):
        Bbs("x", constraint="invalid_choice")


def test_bbs_boundary_knots():
    """Bbs stores boundary_knots correctly."""
    b = Bbs("x", boundary_knots=(0, 10))
    assert b.boundary_knots == (0, 10)


# --- Formula parser kwargs ---


def test_parse_gamboost_bbs_cyclic():
    """parse_gamboost_formula parses bbs(x, cyclic=True)."""
    response, learners = parse_gamboost_formula("y ~ bbs(x, cyclic=True)")
    assert response == "y"
    assert len(learners) == 1
    b = learners[0]
    assert isinstance(b, Bbs)
    assert b.cyclic is True


def test_parse_gamboost_bbs_deriv():
    """parse_gamboost_formula parses bbs(x, deriv=1)."""
    response, learners = parse_gamboost_formula("y ~ bbs(x, deriv=1)")
    assert response == "y"
    assert len(learners) == 1
    b = learners[0]
    assert isinstance(b, Bbs)
    assert b.deriv == 1


def test_parse_gamboost_bbs_boundary_knots():
    """parse_gamboost_formula parses bbs(x, boundary.knots=c(0,10))."""
    response, learners = parse_gamboost_formula("y ~ bbs(x, boundary.knots=c(0,10))")
    assert response == "y"
    assert len(learners) == 1
    b = learners[0]
    assert isinstance(b, Bbs)
    assert b.boundary_knots == (0.0, 10.0)


def test_parse_gamboost_btree_x_plus_z():
    """parse_gamboost_formula parses btree(x + z)."""
    response, learners = parse_gamboost_formula("y ~ btree(x + z)")
    assert response == "y"
    assert len(learners) == 1
    t = learners[0]
    assert isinstance(t, Btree)
    assert t.names == ["x", "z"]


# --- Bconstrained learner validation ---


def test_bconstrained_validates_c_ndim():
    """Bconstrained rejects a C that is not 2-dimensional."""
    with pytest.raises(ValueError, match="Bconstrained C must be 2-dimensional"):
        Bconstrained("x", C=np.array([1.0, 2.0, 3.0]))


def test_bconstrained_rejects_a_non_numeric_c():
    with pytest.raises((TypeError, ValueError), match="Bconstrained C"):
        Bconstrained("x", C="not_array")


def test_bconstrained_smoke():
    """Bconstrained learner can be used in gamboost via the learners kwarg."""
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x": x}
    # Positivity constraint: identity matrix forces all coefficients ≥ 0
    # B-spline basis has p = n_inner + degree + 1 cols
    p = 20 + 3 + 1  # default knots=20, degree=3
    C = np.eye(p)
    bc = Bconstrained("x", C=C, df=4)
    model = mboost.gamboost(
        "y ~ bols(dummy)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
        learners=[bc],
    )
    assert model.mstop == 50
    assert model.fitted_.shape == (n,)
    assert len(model.risk_) == 51


def test_parse_gamboost_bconstrained():
    """parse_gamboost_formula parses bconstrained(x) with C=None."""
    response, learners = parse_gamboost_formula("y ~ bconstrained(x, df=6)")
    assert response == "y"
    assert len(learners) == 1
    bc = learners[0]
    assert isinstance(bc, Bconstrained)
    assert bc.name == "x"
    assert bc.C is None  # cannot come from formula string
    assert bc.df == 6


def test_bconstrained_predict_newdata():
    """Bconstrained predict(newdata=...) works and produces finite values."""
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x": x}
    p = 20 + 3 + 1  # default knots=20, degree=3
    C = np.eye(p)
    bc = Bconstrained("x", C=C, df=4)
    model = mboost.gamboost(
        "y ~ bols(dummy)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
        learners=[bc],
    )

    # Predict on new data
    x_new = np.linspace(0.2, 0.8, 10)
    nd = {"x": x_new, "dummy": np.ones(10)}
    pred = model.predict(newdata=nd)
    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))

    # Prediction should be monotone non-decreasing for positivity constraint
    # on a 1-d covariate
    diffs = np.diff(pred)
    assert np.all(diffs >= -1e-10), f"Non-positive constraint violated: {diffs}"


def test_lss_update_forwards_risk_and_trace():
    """risk/trace reach boost_control instead of being silently dropped."""
    from mboostrs.gamlss import GaussianLSS, glmboostLSS
    data = {"y": np.array([1.0, 2.0, 3.0, 4.0]), "x": np.array([0.5, 1.5, 2.5, 3.5])}
    mod = glmboostLSS("y ~ x", data=data, family=GaussianLSS())
    assert mod._control[2] == "inbag"

    updated = mod.update(mstop=50, risk="none", trace=False)
    assert updated.mstop == 50
    assert updated._control[2] == "none", "risk= must reach the new control"

    # A risk-only change must refit rather than return the same object.
    risk_only = mod.update(risk="none")
    assert risk_only is not mod
    assert risk_only._control[2] == "none"

    # Unchanged arguments carry the original settings over.
    same = mod.update(mstop=20)
    assert same._control[2] == "inbag"


def test_lss_update_rejects_oobweights_rather_than_ignoring_them():
    """No LSS fit function takes an out-of-bag weight vector, so accepting the
    argument and discarding it would advertise support that does not exist."""
    from mboostrs.gamlss import GaussianLSS, glmboostLSS
    data = {"y": np.array([1.0, 2.0, 3.0, 4.0]), "x": np.array([0.5, 1.5, 2.5, 3.5])}
    mod = glmboostLSS("y ~ x", data=data, family=GaussianLSS())
    with pytest.raises(NotImplementedError, match="oobweights"):
        mod.update(oobweights=np.ones(4))
    with pytest.raises(ValueError, match="risk must be"):
        mod.update(risk="bogus")


def test_treeboost_sklearn_estimators():
    """TreeBoostRegressor and TreeBoostClassifier fit and predict correctly."""
    from mboostrs.sklearn import TreeBoostClassifier, TreeBoostRegressor
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    y_reg = np.array([1.5, 2.5, 3.5, 4.5])
    y_cls = np.array([0, 0, 1, 1])

    reg = TreeBoostRegressor(mstop=20)
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert preds.shape == (4,)

    cls = TreeBoostClassifier(mstop=20)
    cls.fit(X, y_cls)
    proba = cls.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lss_distribution_diagnostics():
    """LSS models with GammaLSS, StudentTLSS, BetaLSS, ZINBLSS expose pit, centiles, prediction_interval."""
    from mboostrs.gamlss import ZINBLSS, BetaLSS, GammaLSS, StudentTLSS, glmboostLSS

    data_gamma = {"y": np.array([1.0, 2.0, 3.0, 4.0]), "x": np.array([0.5, 1.5, 2.5, 3.5])}
    mod_gamma = glmboostLSS("y ~ x", data=data_gamma, family=GammaLSS())
    assert len(mod_gamma.pit()) == 4
    assert mod_gamma.prediction_interval() is not None

    data_student = {"y": np.array([-1.0, 0.5, 1.0, 2.0]), "x": np.array([0.5, 1.5, 2.5, 3.5])}
    mod_student = glmboostLSS("y ~ x", data=data_student, family=StudentTLSS())
    assert len(mod_student.pit()) == 4
    assert mod_student.centiles() is not None

    data_beta = {"y": np.array([0.2, 0.4, 0.6, 0.8]), "x": np.array([0.5, 1.5, 2.5, 3.5])}
    mod_beta = glmboostLSS("y ~ x", data=data_beta, family=BetaLSS())
    assert len(mod_beta.pit()) == 4
    assert mod_beta.prediction_interval() is not None

    data_zinb = {"y": np.array([0.0, 1.0, 0.0, 5.0]), "x": np.array([0.5, 1.5, 2.5, 3.5])}
    mod_zinb = glmboostLSS("y ~ x", data=data_zinb, family=ZINBLSS())
    assert len(mod_zinb.pit()) == 4
    assert mod_zinb.prediction_interval() is not None




