"""Numeric-argument validation on base-learner constructors.

Regression for enhancement-hunt 12.0. Before this, `learners.py` validated only
the enum-valued arguments: `Bbs(df=-5)`, `Bbs(knots=0)`, and `Bols(df=-5)` were
accepted and fitted a different model with no complaint, while `Bbs(degree=-1)`
and `Bbs(deriv=-3)` surfaced a raw PyO3 `OverflowError: can't convert negative
int to unsigned` naming neither the parameter nor the constraint.

The accept/reject boundaries below were measured against `mboost` 2.9.11 and
match it, except where noted.
"""

import mboostrs as mb
import numpy as np
import pytest


# ── negative counts are rejected by name, not by OverflowError ───────────────
@pytest.mark.parametrize(
    "kwargs, param",
    [
        ({"knots": -3}, "knots"),
        ({"degree": -1}, "degree"),
        ({"differences": -1}, "differences"),
        ({"deriv": -3}, "deriv"),
    ],
)
def test_bbs_rejects_negative_counts(kwargs, param):
    with pytest.raises(ValueError, match=param):
        mb.Bbs("x", **kwargs)


def test_bbs_accepts_zero_counts_as_upstream_does():
    """`mboost::bbs` accepts `knots = 0`, `degree = 0`, and `differences = 0`;
    only `deriv` and the three above go negative. Rejecting zero would diverge
    from the package being mirrored."""
    assert mb.Bbs("x", knots=0).knots == 0
    assert mb.Bbs("x", degree=0, df=2).degree == 0
    assert mb.Bbs("x", differences=0).differences == 0


def test_count_arguments_must_be_integers():
    with pytest.raises(ValueError, match="knots must be an integer"):
        mb.Bbs("x", knots=20.5)
    with pytest.raises(ValueError, match="degree must be an integer"):
        mb.Bbs("x", degree=True)


# ── df ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cls, args, kwargs",
    [
        (mb.Bols, ("x",), {"df": -5}),
        (mb.Bbs, ("x",), {"df": -5}),
        (mb.Bns, ("x",), {"df": 0}),
        (mb.Brandom, ("g",), {"df": -1}),
        (mb.Bkernel, ("x",), {"df": float("nan")}),
        (mb.Bspatial, ("x", "z"), {"df": -6}),
    ],
)
def test_df_must_be_positive_and_finite(cls, args, kwargs):
    with pytest.raises(ValueError, match="df must be positive and finite"):
        cls(*args, **kwargs)


def test_bbs_rejects_df_below_the_penalty_null_space():
    """A difference penalty of order `d` leaves a `d`-dimensional unpenalized
    subspace, so `df` below `d` is unreachable and silently pins lambda at its
    maximum. `mboost::bbs` accepts df = 2 and rejects df = 1.99 at the default
    differences = 2; so does this."""
    assert mb.Bbs("x", df=2).df == 2
    with pytest.raises(ValueError, match="rank of the null space"):
        mb.Bbs("x", df=1.99)
    # differences = 1 lowers the rank, so df = 1 becomes reachable.
    assert mb.Bbs("x", df=1, differences=1).df == 1
    with pytest.raises(ValueError, match="rank of the null space"):
        mb.Bbs("x", df=0.9, differences=1)


def test_centering_removes_the_null_space_constraint():
    """`center=True` projects the null space out, so any positive df is
    attainable — matching `mboost::bbs(x, df = 1.5, center = TRUE)`."""
    assert mb.Bbs("x", df=1.5, center=True).df == 1.5


def test_bspatial_null_space_is_the_tensor_product_of_the_marginals():
    """The additive Kronecker penalty's null space has dimension
    `differences ** 2`. `mboost::bspatial` accepts df = 4 and rejects df = 3.99
    at the default differences = 2."""
    assert mb.Bspatial("x", "z", df=4).df == 4
    with pytest.raises(ValueError, match=r"df' \(>= 4\)"):
        mb.Bspatial("x", "z", df=3.99)


def test_ridge_penalised_learners_have_no_null_space():
    """`brandom` and `bns` substitute an identity ridge for the difference
    penalty, so a small df is legitimate — `mboost::brandom(f, df = 0.5)`
    is accepted."""
    assert mb.Brandom("g", df=0.5).df == 0.5
    assert mb.Bns("x", df=0.5).df == 0.5


# ── penalty weights ──────────────────────────────────────────────────────────
def test_lambda_value_must_be_non_negative():
    with pytest.raises(ValueError, match="lambda_value must be non-negative"):
        mb.Bbs("x", lambda_value=-1.0)
    with pytest.raises(ValueError, match="lambda_value must be non-negative"):
        mb.Bols("x", lambda_value=float("inf"))
    assert mb.Bbs("x", lambda_value=0.0).lambda_value == 0.0


def test_bmono_violation_penalty_and_iteration_count():
    with pytest.raises(ValueError, match="lambda2 must be positive"):
        mb.Bmono("x", lambda2=0.0)
    with pytest.raises(ValueError, match="niter must be >= 1"):
        mb.Bmono("x", niter=-1)
    with pytest.raises(ValueError, match="niter must be >= 1"):
        mb.Bmono("x", niter=0)


def test_btree_split_sizes_must_be_at_least_one():
    for param in ("minbucket", "minsplit", "maxdepth"):
        with pytest.raises(ValueError, match=f"{param} must be >= 1"):
            mb.Btree(["x"], **{param: 0})


def test_bkernel_bandwidth_must_be_positive():
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        mb.Bkernel("x", bandwidth=0.0)


# ── the enum lists the two P-spline learners share are one object ────────────
def test_shape_constraint_list_is_shared_not_duplicated():
    """`Bbs.VALID_CONSTRAINTS` and `Bmono.VALID_CONSTRAINTS` were two separate
    verbatim copies, which is how such lists come to disagree."""
    assert mb.Bbs.VALID_CONSTRAINTS is mb.Bmono.VALID_CONSTRAINTS


def test_every_learner_class_is_documented():
    """`Bols` — the most-used learner — was the only one without a docstring."""
    for name in ("Bols", "Bbs", "Bmono", "Btree", "Bmrf", "Bspatial",
                 "Brandom", "Bns", "Bkernel", "Bconstrained", "Buser"):
        cls = getattr(mb, name)
        assert cls.__doc__, f"{name} has no docstring"


# ── a valid construction still fits ──────────────────────────────────────────
def test_validation_does_not_disturb_a_normal_fit():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, n)
    d = {"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)}
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=d, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    assert np.asarray(fit.risk_)[-1] < np.asarray(fit.risk_)[0]
