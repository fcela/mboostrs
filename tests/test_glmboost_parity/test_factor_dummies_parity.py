"""`glmboost` with a factor column, against R's `model.matrix` expansion.

R expands a factor `f` with levels a < b < c into the treatment-contrast
columns `fb` and `fc`, each its own glmboost component. mboostrs does the same
through a `dummy` transform step that is replayed on new data.
"""

import numpy as np
import pandas as pd
import pytest

import mboostrs as mb


def _data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)})
    frame["y"] = (frame["x1"] + (frame["cat"] == "b") - 0.5 * (frame["cat"] == "c")
                  + rng.normal(0, 0.3, n))
    return frame


def test_a_factor_becomes_named_contrast_columns():
    fit = mb.glmboost("y ~ x1 + cat", data=_data(), control=mb.boost_control(mstop=500))
    coef = fit.coef()
    assert set(coef) == {"(Intercept)", "x1", "catb", "catc"}
    assert coef["catb"] == pytest.approx(1.0, abs=0.15)
    assert coef["catc"] == pytest.approx(-0.5, abs=0.15)


def test_new_rows_replay_and_unseen_levels_refuse(tmp_path):
    fit = mb.glmboost("y ~ x1 + cat", data=_data(), control=mb.boost_control(mstop=500))
    new = pd.DataFrame({"x1": [0.0, 0.0, 0.0], "cat": ["a", "b", "c"]})
    pred = np.asarray(fit.predict(newdata=new)).ravel()
    assert pred[1] - pred[0] == pytest.approx(fit.coef()["catb"])
    with pytest.raises(ValueError, match="not present"):
        fit.predict(newdata=pd.DataFrame({"x1": [0.0], "cat": ["z"]}))
    path = tmp_path / "m.json"
    mb.save_model(fit, str(path))
    np.testing.assert_allclose(np.asarray(mb.load_model(str(path)).predict(newdata=new)).ravel(), pred)


@pytest.mark.r_parity
def test_coefficients_match_r_glmboost_with_a_factor():
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    frame = _data()
    fit = mb.glmboost("y ~ x1 + cat", data=frame, control=mb.boost_control(mstop=500))
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_frame = ro.conversion.py2rpy(frame)
    r_coef = ro.r("""
        function(d) {
            suppressMessages(library(mboost))
            d$cat <- factor(d$cat)
            m <- glmboost(y ~ x1 + cat, data = d, control = boost_control(mstop = 500))
            coef(m, off2int = TRUE)
        }
    """)(r_frame)
    expected = dict(zip(list(r_coef.names), np.asarray(r_coef), strict=True))
    got = fit.coef(off2int=True)
    for name in ("(Intercept)", "x1", "catb", "catc"):
        assert got[name] == pytest.approx(expected[name], rel=1e-8, abs=1e-8)
