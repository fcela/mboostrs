"""`.` expansion and term removal in formulas.

`y ~ .` is the first thing anyone types against a wide frame. These tests
cover the rewriting itself and then check that the rewritten formula actually
fits the model it claims to.
"""

import numpy as np
import polars as pl
import pytest
from mboostrs.formula import expand_formula

import mboostrs as mb


def _frame(n=300, p=5, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"x{j}": rng.normal(size=n) for j in range(p)}
    data["y"] = 1.5 * data["x0"] - 0.8 * data["x1"] + rng.normal(scale=0.3, size=n)
    return data


# --------------------------------------------------------------------------
# rewriting
# --------------------------------------------------------------------------


def test_dot_expands_to_every_non_response_column():
    data = {"a": [1.0], "b": [2.0], "y": [3.0]}
    assert expand_formula("y ~ .", data) == "y ~ a + b"


def test_dot_preserves_column_order():
    data = {"z": [1.0], "a": [2.0], "m": [3.0], "y": [4.0]}
    assert expand_formula("y ~ .", data) == "y ~ z + a + m"


def test_removal_drops_a_column_from_the_expansion():
    data = {"a": [1.0], "b": [2.0], "c": [3.0], "y": [4.0]}
    assert expand_formula("y ~ . - b", data) == "y ~ a + c"


def test_removal_without_a_dot_drops_an_explicit_term():
    data = {"a": [1.0], "b": [2.0], "y": [3.0]}
    assert expand_formula("y ~ a + b - b", data) == "y ~ a"


def test_dot_leaves_explicitly_named_columns_to_their_own_term():
    """A column given an explicit learner must not also appear bare."""
    data = {"a": [1.0], "b": [2.0], "y": [3.0]}
    assert expand_formula("y ~ . + bbs(a, df=6)", data) == "y ~ b + bbs(a, df=6)"


def test_formula_without_dot_or_removal_is_returned_unchanged():
    data = {"a": [1.0], "y": [2.0]}
    formula = "y ~ bbs(a, df=4) + bols(a)"
    assert expand_formula(formula, data) is formula


def test_dot_excludes_every_name_of_a_multi_column_response():
    data = {"time": [1.0], "status": [1.0], "a": [2.0], "b": [3.0]}
    assert expand_formula("Surv(time, status) ~ .", data) == "Surv(time, status) ~ a + b"


def test_dot_works_with_polars_and_pandas_frames():
    frame = pl.DataFrame({"a": [1.0], "b": [2.0], "y": [3.0]})
    assert expand_formula("y ~ .", frame) == "y ~ a + b"

    pd = pytest.importorskip("pandas")
    assert expand_formula("y ~ .", pd.DataFrame({"a": [1.0], "b": [2.0], "y": [3.0]})) == (
        "y ~ a + b"
    )


# --------------------------------------------------------------------------
# fitting through the rewritten formula
# --------------------------------------------------------------------------


def test_glmboost_dot_matches_the_explicit_formula():
    data = _frame()
    explicit = "y ~ x0 + x1 + x2 + x3 + x4"
    ctrl = mb.boost_control(150, 0.1)
    a = mb.glmboost("y ~ .", data=data, family=mb.Gaussian(), control=ctrl)
    b = mb.glmboost(explicit, data=data, family=mb.Gaussian(), control=ctrl)
    np.testing.assert_allclose(a.fitted_, b.fitted_, rtol=0, atol=1e-12)
    assert a.base_learner_names() == b.base_learner_names()


def test_gamboost_dot_matches_the_explicit_formula():
    data = _frame(p=3)
    ctrl = mb.boost_control(80, 0.1)
    a = mb.gamboost("y ~ .", data=data, family=mb.Gaussian(), control=ctrl)
    b = mb.gamboost("y ~ x0 + x1 + x2", data=data, family=mb.Gaussian(), control=ctrl)
    np.testing.assert_allclose(a.fitted_, b.fitted_, rtol=0, atol=1e-12)


def test_dot_with_removal_fits_the_reduced_model():
    data = _frame()
    ctrl = mb.boost_control(100, 0.1)
    a = mb.glmboost("y ~ . - x3 - x4", data=data, family=mb.Gaussian(), control=ctrl)
    b = mb.glmboost("y ~ x0 + x1 + x2", data=data, family=mb.Gaussian(), control=ctrl)
    np.testing.assert_allclose(a.fitted_, b.fitted_, rtol=0, atol=1e-12)


def test_mixed_dot_and_explicit_learner_fits():
    data = _frame(p=3)
    fit = mb.gamboost(
        "y ~ . + bbs(x0, df=6) - x2", data=data, family=mb.Gaussian(),
        control=mb.boost_control(80, 0.1),
    )
    assert sorted(fit.base_learner_names()) == ["x0", "x1"]
    assert np.all(np.isfinite(fit.fitted_))


def test_dot_model_predicts_on_new_data():
    data = _frame()
    fit = mb.glmboost("y ~ .", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(150, 0.1))
    newdata = _frame(n=50, seed=99)
    pred = fit.predict(newdata=newdata, type="response")
    assert np.asarray(pred).shape == (50,)
    assert np.all(np.isfinite(pred))


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


def test_dot_that_expands_to_nothing_is_rejected():
    data = {"y": [1.0, 2.0]}
    with pytest.raises(ValueError, match="expanded to no predictors"):
        expand_formula("y ~ .", data)


def test_removing_every_term_is_rejected():
    data = {"a": [1.0], "y": [2.0]}
    with pytest.raises(ValueError, match="no predictors left"):
        expand_formula("y ~ a - a", data)


def test_intercept_control_is_refused_explicitly_not_ignored():
    data = {"a": [1.0], "y": [2.0]}
    for formula in ("y ~ a - 1", "y ~ a + 0"):
        with pytest.raises(NotImplementedError, match="intercept control"):
            expand_formula(formula, data)


def test_minus_dot_is_rejected():
    data = {"a": [1.0], "y": [2.0]}
    with pytest.raises(ValueError, match="not a valid term"):
        expand_formula("y ~ a - .", data)


def test_formula_without_tilde_is_rejected():
    with pytest.raises(ValueError, match="must contain"):
        expand_formula("y + a", {"a": [1.0], "y": [2.0]})


def test_a_column_literally_named_0_or_1_is_not_mistaken_for_the_intercept():
    """sklearn's compliance checks build frames with columns named "0", "1", ...

    Those must keep referring to the column; only a `0`/`1` with no matching
    column means intercept control.
    """
    data = {"0": [1.0], "1": [2.0], "y": [3.0]}
    assert expand_formula("y ~ .", data) == "y ~ `0` + `1`"
    assert expand_formula("y ~ 0 + 1", data) == "y ~ 0 + 1"
    assert expand_formula("y ~ . - 1", data) == "y ~ `0`"

    # Without such a column the intercept message still applies.
    with pytest.raises(NotImplementedError, match="intercept control"):
        expand_formula("y ~ a - 1", {"a": [1.0], "y": [2.0]})


def test_no_intercept_workaround_matches_r_glmboost_minus_one():
    """The error message points at gamboost() + bols(); this pins that it works.

    R's `glmboost(y ~ x1 + x2 + x3 - 1)` drops the intercept learner from the
    candidate set and keeps the offset. gamboost() adds only the terms written,
    so plain bols() terms reproduce it. Reference values come from
    mboost 2.9.11 on the data constructed below (set.seed(7) in R).
    """
    import mboostrs as mb

    rng = np.random.default_rng(20260725)
    n = 400
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 3.0 + 1.2 * x1 - 0.8 * x2 + 0.4 * x3 + rng.normal(0.0, 0.3, n)
    data = {"x1": x1, "x2": x2, "x3": x3, "y": y}
    ctl = mb.boost_control(400, 0.1)

    no_int = mb.gamboost(
        "y ~ bols(x1) + bols(x2) + bols(x3)", data=data, family=mb.Gaussian(), control=ctl
    )
    coefs = {k: v for k, v in no_int.coef().items() if k != "(offset)"}
    # No separate intercept learner: only the three predictors appear, each
    # reported as mboost's bols reports it, [intercept, slope].
    assert set(coefs) == {"x1", "x2", "x3"}
    assert coefs["x1"][1] == pytest.approx(1.2, abs=0.05)
    assert coefs["x2"][1] == pytest.approx(-0.8, abs=0.05)
    assert coefs["x3"][1] == pytest.approx(0.4, abs=0.05)
    # The offset survives, as it does in R.
    assert no_int.offset == pytest.approx(y.mean(), abs=0.05)

    # glmboost, by contrast, estimates a separate intercept learner.
    with_int = mb.glmboost("y ~ x1 + x2 + x3", data=data, family=mb.Gaussian(), control=ctl)
    assert "(Intercept)" in with_int.coef()


def test_bols_intercept_false_is_a_different_operation():
    """Pins why the error message warns against it: it changes the fit rather
    than removing a separate intercept learner."""
    import mboostrs as mb

    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    data = {"x": x, "y": 2.0 + 1.5 * x + rng.normal(0.0, 0.2, n)}
    ctl = mb.boost_control(300, 0.1)
    plain = mb.gamboost("y ~ bols(x)", data=data, family=mb.Gaussian(), control=ctl)
    dropped = mb.gamboost(
        "y ~ bols(x, intercept=False)", data=data, family=mb.Gaussian(), control=ctl
    )
    # bols(x) reports [intercept, slope]; without an intercept only the slope.
    assert plain.coef()["x"][1] != pytest.approx(dropped.coef()["x"], abs=1e-6)


# ── 21.23: a linear formula must not silently discard a basis learner ────────
def test_glmboost_rejects_learners_it_cannot_fit():
    """`glmboost("y ~ bbs(x, df=4)")` used to reduce the term to a bare `x` and
    fit a linear model identical to `glmboost("y ~ x")`, with nothing to say the
    spline had been discarded."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 100
    data = {"x": rng.normal(size=n)}
    data["y"] = 1.0 + 2.0 * data["x"] + rng.normal(0, 0.5, n)

    for term in ("bbs(x, df=4)", "bkernel(x)", "btree(x)", "brandom(x)",
                 "bns(x)", "bmono(x)"):
        with pytest.raises(ValueError, match="cannot use it"):
            mb.glmboost(f"y ~ {term}", data=data, family=mb.Gaussian(),
                        control=mb.boost_control(5, 0.1))


def test_glmboost_still_accepts_bols_and_bare_names():
    """`bols(x)` means exactly what a bare `x` means, and is kept as a migration
    affordance even though upstream `mboost::glmboost` rejects it too."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 100
    data = {"x": rng.normal(size=n)}
    data["y"] = 1.0 + 2.0 * data["x"] + rng.normal(0, 0.5, n)
    bare = mb.glmboost("y ~ x", data=data, family=mb.Gaussian(),
                       control=mb.boost_control(30, 0.1))
    wrapped = mb.glmboost("y ~ bols(x)", data=data, family=mb.Gaussian(),
                          control=mb.boost_control(30, 0.1))
    np.testing.assert_array_equal(np.asarray(bare.risk_), np.asarray(wrapped.risk_))


def test_glmboostlss_rejects_learners_it_cannot_fit():
    """The same silent linearization reached `glmboostLSS`, where the manifest
    already recorded `Bkernel glmboostLSS = no`."""
    import numpy as np
    from mboostrs.gamlss import GaussianLSS

    rng = np.random.default_rng(0)
    n = 120
    data = {"x": rng.normal(size=n)}
    data["y"] = 1.0 + 2.0 * data["x"] + rng.normal(0, 0.5, n)
    with pytest.raises(ValueError, match="cannot use it"):
        mb.glmboostLSS({"mu": "y ~ bkernel(x)", "sigma": "y ~ x"}, data=data,
                       families=GaussianLSS(), control=mb.boost_control(5, 0.1))


def test_dot_expansion_still_sees_every_learner():
    """The restriction belongs to the linear parser only — `.` expansion has to
    recognise a column named by any learner so it does not add it twice."""
    from mboostrs.formula import expand_formula

    data = {"a": [1.0], "b": [2.0], "y": [3.0]}
    assert expand_formula("y ~ . + bbs(a, df=6)", data) == "y ~ b + bbs(a, df=6)"


def test_dot_leaves_out_a_constant_column_with_a_warning():
    """A column with one value in the rows (a split can make one) has no
    effect to fit, and a spline over it has no range for its knots."""
    import warnings

    import mboostrs as mb

    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    data = {"y": x + rng.normal(size=200), "x": x, "c": np.zeros(200)}
    with pytest.warns(UserWarning, match="leaves out 'c'"):
        model = mb.gamboost("y ~ .", data)
    assert set(model.selected) == {"x"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert expand_formula("y ~ .", {"y": data["y"], "x": x}) == "y ~ x"
