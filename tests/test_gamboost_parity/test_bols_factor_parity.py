"""`bols(f)` over a categorical column, against R's dummy-coded `bols(factor)`.

mboost builds `bols(f)` from the treatment-contrast design `[1, d_2, ..., d_K]`.
mboostrs fits the unpenalized level design `[d_1, ..., d_K]`, which spans the
same columns, so every boosting step and the fitted path agree; only the
coefficients are parametrized differently (level effects, not contrasts).
"""

import numpy as np
import pandas as pd
import pytest

import mboostrs as mb


def _data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "cat": rng.choice(["a", "b", "c", "d"], n)})
    frame["y"] = (frame["x1"] + (frame["cat"] == "b") - 0.5 * (frame["cat"] == "c")
                  + rng.normal(0, 0.3, n))
    return frame


def test_bols_on_a_factor_is_the_zero_penalty_level_design():
    frame = _data()
    control = mb.boost_control(mstop=200, nu=0.1)
    as_bols = mb.gamboost("y ~ bbs(x1) + bols(cat)", data=frame, control=control)
    as_levels = mb.gamboost("y ~ bbs(x1) + brandom(cat, lambda=0)", data=frame, control=control)
    np.testing.assert_allclose(np.asarray(as_bols.predict()).ravel(),
                               np.asarray(as_levels.predict()).ravel(), atol=1e-12)
    assert as_bols.base_learner_names() == ["x1", "bols(cat)"]


def test_bare_factor_terms_and_dot_expansion_fit_the_same_learner():
    frame = _data()
    control = mb.boost_control(mstop=200, nu=0.1)
    reference = np.asarray(mb.gamboost("y ~ bbs(x1) + bols(cat)", data=frame,
                                       control=control).predict()).ravel()
    for formula in ("y ~ bbs(x1) + cat", "y ~ ."):
        fitted = np.asarray(mb.gamboost(formula, data=frame, control=control).predict()).ravel()
        np.testing.assert_allclose(fitted, reference, atol=1e-12)


def test_new_rows_and_unseen_levels():
    frame = _data()
    fit = mb.gamboost("y ~ bbs(x1) + bols(cat)", data=frame,
                      control=mb.boost_control(mstop=300, nu=0.1))
    new = pd.DataFrame({"x1": [0.0, 0.0, 0.0], "cat": ["a", "b", "c"]})
    level_effect = np.asarray(fit.predict(newdata=new)).ravel()
    assert level_effect[1] - level_effect[0] == pytest.approx(1.0, abs=0.15)
    with pytest.raises(ValueError, match="not present"):
        fit.predict(newdata=pd.DataFrame({"x1": [0.0], "cat": ["z"]}))


def test_extra_arguments_fit_the_contrast_design():
    # Issue #159: mboost fits bols(factor, df=) on the treatment-contrast
    # design; tests/test_factor_bols_arguments.py compares it with mboost.
    frame = _data()
    fit = mb.gamboost("y ~ bols(cat, df=2)", data=frame)
    np.testing.assert_allclose(fit.predict(newdata=frame), fit.fitted_, atol=1e-10)


@pytest.mark.r_parity
def test_fitted_path_matches_r_bols_factor():
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    frame = _data()
    fit = mb.gamboost("y ~ bbs(x1) + bols(cat)", data=frame,
                      control=mb.boost_control(mstop=300, nu=0.1))
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        r_frame = ro.conversion.py2rpy(frame)
    fitted_r = ro.r("""
        function(d) {
            suppressMessages(library(mboost))
            d$cat <- factor(d$cat)
            m <- gamboost(y ~ bbs(x1) + bols(cat), data = d,
                          control = boost_control(mstop = 300, nu = 0.1))
            as.numeric(fitted(m))
        }
    """)(r_frame)
    np.testing.assert_allclose(np.asarray(fit.predict()).ravel(), np.asarray(fitted_r),
                               rtol=1e-8, atol=1e-8)
