import numpy as np

from mboostrs import gamboost
from mboostrs.families import Gaussian


def test_bbs_deriv_basic():
    n = 100
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, n)
    y = x**2 + 0.1 * rng.normal(size=n)
    data = dict(x=x, y=y)

    model = gamboost(
        "y ~ bbs(x, df=4, deriv=1)",
        data=data,
        family=Gaussian(),
        control=(50, 0.1),
    )
    fitted = model.predict()
    assert len(fitted) == n
    assert np.all(np.isfinite(fitted))

    risk = model._risk
    assert len(risk) == 51  # mstop + 1 (includes init)
    assert np.all(np.isfinite(risk))

    # cvrisk smoke test
    result = model.cvrisk(type="kfold", B=3, seed=42)
    assert result.grid, "cvrisk should report the grid it searched"
    cvrisk_vals = result.risk_matrix.mean(axis=1)
    assert len(cvrisk_vals) > 0
    assert np.all(np.isfinite(cvrisk_vals))


def test_bbs_deriv_cyclic():
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.default_rng(42).normal(size=n)
    data = dict(x=x, y=y)

    model = gamboost(
        "y ~ bbs(x, df=4, cyclic=True, deriv=1)",
        data=data,
        family=Gaussian(),
        control=(50, 0.1),
    )
    fitted = model.predict()
    assert len(fitted) == n
    assert np.all(np.isfinite(fitted))

    risk = model._risk
    assert len(risk) == 51  # mstop + 1
    assert np.all(np.isfinite(risk))


def test_bbs_deriv_slice():
    n = 100
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, n)
    y = x**2 + 0.1 * rng.normal(size=n)
    data = dict(x=x, y=y)

    model = gamboost(
        "y ~ bbs(x, df=4, deriv=1)",
        data=data,
        family=Gaussian(),
        control=(100, 0.1),
    )

    sub = model[:50]
    fitted_sub = sub.predict()
    assert len(fitted_sub) == n
    assert np.all(np.isfinite(fitted_sub))


def test_bbs_deriv_newdata():
    import polars as pl

    n_train = 100
    n_test = 20
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, n_train)
    y = x**2 + 0.1 * rng.normal(size=n_train)
    data = pl.DataFrame(dict(x=x, y=y))

    model = gamboost(
        "y ~ bbs(x, df=4, deriv=1)",
        data=data,
        family=Gaussian(),
        control=(50, 0.1),
    )

    new_x = np.linspace(0, 1, n_test)
    newdata = pl.DataFrame(dict(x=new_x))
    pred = model.predict(newdata=newdata)
    assert len(pred) == n_test
    assert np.all(np.isfinite(pred))


def test_bbs_deriv_pickle():
    import pickle

    n = 50
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, n)
    y = x**2 + 0.1 * rng.normal(size=n)
    data = dict(x=x, y=y)

    model = gamboost(
        "y ~ bbs(x, df=4, deriv=1)",
        data=data,
        family=Gaussian(),
        control=(50, 0.1),
    )
    pred_before = model.predict()

    serialized = pickle.dumps(model)
    model2 = pickle.loads(serialized)
    pred_after = model2.predict()

    assert len(pred_after) == n
    assert np.allclose(pred_before, pred_after)
    assert np.all(np.isfinite(pred_after))
