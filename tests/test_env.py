"""Tests for the `env` parameter in gamboost formula parsing."""
import numpy as np
import pandas as pd
import pytest

from mboostrs import gamboost
from mboostrs.formula import parse_gamboost_formula


def test_bmrf_env():
    """bmrf(region, K=K_mat) resolves via env dict."""
    # Two mutually adjacent regions: one neighbour each.
    K = np.array([[1.0, -1.0], [-1.0, 1.0]])
    n = 20
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "y": rng.normal(size=n),
        "region": rng.integers(0, 2, size=n),
    })
    data["region"] = data["region"].astype(int)
    resp, learners = parse_gamboost_formula(
        "y ~ bmrf(region, K=K_mat, df=4)",
        env={"K_mat": K},
    )
    assert len(learners) == 1
    assert learners[0].K is not None
    np.testing.assert_array_equal(learners[0].K, K)


def test_bmrf_env_missing():
    """bmrf with K= symbol but no env raises."""
    with pytest.raises(ValueError, match="requires an env="):
        parse_gamboost_formula("y ~ bmrf(region, K=K_mat, df=4)")


def test_buser_env():
    """buser(custom, X=X_basis, K=K_penalty) resolves via env dict."""
    n = 20
    p = 3
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, p))
    K = np.eye(p)
    resp, learners = parse_gamboost_formula(
        "y ~ buser(custom, X=X_basis, K=K_penalty, df=4)",
        env={"X_basis": X, "K_penalty": K},
    )
    assert len(learners) == 1
    assert learners[0].X.shape == (n, p)
    assert learners[0].K.shape == (p, p)
    np.testing.assert_array_equal(learners[0].K, K)


def test_buser_no_env():
    """buser without env raises."""
    # The point is that the formula cannot resolve `K` and `X` without an
    # env; binding them here would only shadow that.
    with pytest.raises(ValueError, match="requires an env="):
        parse_gamboost_formula(
            "y ~ buser(custom, X=X_basis, K=K_penalty, df=4)",
        )


def test_buser_missing_X():
    """buser without X= raises."""
    K = np.eye(3)
    with pytest.raises(ValueError, match="requires X=<symbol>"):
        parse_gamboost_formula(
            "y ~ buser(custom, K=K_penalty, df=4)",
            env={"K_penalty": K},
        )


def test_buser_missing_K():
    """buser without K= raises."""
    X = np.random.randn(10, 3)
    with pytest.raises(ValueError, match="requires K=<symbol>"):
        parse_gamboost_formula(
            "y ~ buser(custom, X=X_basis, df=4)",
            env={"X_basis": X},
        )


def test_bconstrained_env():
    """bconstrained(x, C=constr) resolves via env dict."""
    C = np.eye(24)
    # Parsing does not read the data, so this checks the env resolution alone.
    resp, learners = parse_gamboost_formula(
        "y ~ bconstrained(x, C=constr, knots=20, degree=3)",
        env={"constr": C},
    )
    assert len(learners) == 1
    assert learners[0].C is not None
    np.testing.assert_array_equal(learners[0].C, C)


def test_regular_learners_ignore_env():
    """bbs, bols, etc. work with or without env."""
    resp, learners = parse_gamboost_formula("y ~ bbs(x, df=4) + bols(z)", env={})
    assert len(learners) == 2


def test_gamboost_buser_fit():
    """End-to-end: gamboost with buser learner fits without error."""
    rng = np.random.default_rng(42)
    n = 50
    X_custom = rng.normal(size=(n, 3))
    K_custom = np.eye(3)
    beta_true = np.array([1.0, -0.5, 0.3])
    y = X_custom @ beta_true + rng.normal(size=n) * 0.1
    data = pd.DataFrame({"y": y, "dummy": np.zeros(n)})
    model = gamboost(
        "y ~ buser(custom, X=X_basis, K=K_penalty, df=4)",
        data=data,
        control=(200, 0.1),
        env={"X_basis": X_custom, "K_penalty": K_custom},
    )
    fitted = model.fitted()
    assert len(fitted) == n
    assert np.isfinite(fitted).all()
