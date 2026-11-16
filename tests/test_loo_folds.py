"""Leave-one-out fold scheme (issue #100), mboost's cv(type = "kfold", B = n)."""

import numpy as np
import pytest

import mboostrs as mb


def test_each_fold_holds_out_one_row():
    scheme = mb.loo_folds(6)
    assert scheme.n_folds == 6
    np.testing.assert_array_equal(scheme.held_out, np.eye(6, dtype=bool))
    np.testing.assert_array_equal(scheme.weights, 1 - np.eye(6))


def test_zero_weight_rows_are_in_no_fold():
    scheme = mb.loo_folds(5, weights=[1, 0, 2, 1, 1])
    assert scheme.n_folds == 4
    assert not scheme.held_out[1].any()
    assert np.all(scheme.weights[1] == 0)


def test_loo_cvrisk_matches_manual_refits():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=25)
    data = {"x": x, "y": 2 * x + rng.normal(scale=0.2, size=25)}
    model = mb.glmboost("y ~ x", data=data, control=mb.boost_control(30))
    cv = model.cvrisk(folds=mb.loo_folds(25))
    # cvrisk refits each fold through in-bag weights on all rows, as mboost
    # does (glmboost then centres x by its full-data mean), so the manual
    # refits use weight zero for the held-out row.
    manual = []
    for i in range(25):
        weights = np.ones(25)
        weights[i] = 0.0
        fit = mb.glmboost("y ~ x", data=data, weights=weights,
                          control=mb.boost_control(30))
        pred = float(np.asarray(fit.predict(newdata={"x": x[i:i + 1]}))[0])
        manual.append((data["y"][i] - pred) ** 2)
    assert np.asarray(cv.risk_matrix)[:, 30].mean() == pytest.approx(np.mean(manual), rel=1e-8)


def test_too_few_rows_refused():
    with pytest.raises(ValueError):
        mb.loo_folds(1)


@pytest.mark.r_parity
def test_loo_cvrisk_matches_mboost():
    """mboost's leave-one-out is cv(type = "kfold", B = n). Its columns come
    in a random order, so the per-iteration risk is compared averaged over
    folds, which does not depend on that order."""
    from conftest import HAS_R, HAS_RPY2

    if not (HAS_R and HAS_RPY2):
        pytest.skip("requires R + rpy2")
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(3)
    n = 30
    x1, x2 = rng.uniform(size=n), rng.normal(size=n)
    y = 2 * x1 - x2 + rng.normal(scale=0.3, size=n)
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["x1"], ro.globalenv["x2"], ro.globalenv["y"] = x1, x2, y
        upstream = np.asarray(ro.r(
            "suppressMessages(library(mboost)); d <- data.frame(x1 = x1, x2 = x2, y = y); "
            "m <- mboost::glmboost(y ~ x1 + x2, data = d, "
            "control = mboost::boost_control(mstop = 40)); "
            "colMeans(mboost::cvrisk(m, folds = mboost::cv(model.weights(m), "
            "type = 'kfold', B = nrow(d)), papply = lapply))"))
    model = mb.glmboost("y ~ x1 + x2", data={"x1": x1, "x2": x2, "y": y},
                        control=mb.boost_control(40))
    ours = np.asarray(model.cvrisk(folds=mb.loo_folds(n)).risk_matrix).mean(axis=0)
    np.testing.assert_allclose(ours, upstream, rtol=1e-10)
