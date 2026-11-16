"""Linear design extraction and binary class predictions in both model types."""

import pickle
from contextlib import nullcontext

import numpy as np
import pytest

import mboostrs as mb


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("intercept", [True, False])
def test_glm_extract_fitted_design(center, intercept):
    rng = np.random.default_rng(7)
    x = rng.normal(3, 1, 80)
    z = rng.normal(-2, 1, 80)
    data = {"x": x, "z": z, "y": 2 * x + rng.normal(size=80)}
    formula = "y ~ x + z" + ("" if intercept else " - 1")
    with pytest.warns(UserWarning) if center and not intercept else nullcontext():
        model = mb.glmboost(formula, data, center=center, control=mb.boost_control(1))
    names = model.base_learner_names()
    explicit = model.extract("design", which=list(range(len(names))))
    assert list(explicit) == names
    np.testing.assert_allclose(explicit["x"], x - (x.mean() if center else 0))
    np.testing.assert_allclose(explicit["z"], z - (z.mean() if center else 0))
    if intercept:
        np.testing.assert_array_equal(explicit["(Intercept)"], np.ones(80))
    expected = set(model.selected)
    if intercept and center:
        expected.add("(Intercept)")
    assert set(model.extract("design")) == expected
    assert list(model.extract("design")) == model.extract("variable.names")
    assert list(model.extract("design", which=["z", "x"])) == ["z", "x"]
    for bad in ("missing", -1, len(names)):
        with pytest.raises(ValueError, match="unknown learner"):
            model.extract("design", which=bad)


@pytest.mark.parametrize("fit", [mb.glmboost, mb.gamboost])
@pytest.mark.parametrize("family", [mb.Binomial(), mb.AdaExp(), mb.HingeLoss(), mb.AUC()])
def test_binary_class_prediction(fit, family):
    rng = np.random.default_rng(12)
    x = rng.normal(size=100)
    y = (x + rng.normal(size=100) > 0).astype(float)
    model = fit("y ~ x", {"x": x, "y": y}, family=family, control=mb.boost_control(30))
    for newdata in (None, {"x": np.array([-2.0, 0.0, 2.0])}):
        for k in (5, 30):
            eta = model.predict(newdata, mstop=k)
            np.testing.assert_array_equal(model.predict(newdata, mstop=k, type="class"), eta > 0)
    restored = pickle.loads(pickle.dumps(model))
    newdata = {"x": np.array([-1.0, 1.0])}
    np.testing.assert_array_equal(restored.predict(newdata, type="class"),
                                  model.predict(newdata, type="class"))
    with pytest.raises(ValueError, match="whole model"):
        model.predict(type="class", which="x")
    with pytest.raises(ValueError, match="aggregate"):
        model.predict(type="class", aggregate="cumsum")


@pytest.mark.parametrize("fit", [mb.glmboost, mb.gamboost])
def test_class_ties_and_unsupported_family(fit):
    data = {"x": np.zeros(10), "y": np.tile([0.0, 1.0], 5)}
    formula = "y ~ bols(x)" if fit is mb.gamboost else "y ~ x"
    binary = fit(formula, data, family=mb.Binomial(), control=mb.boost_control(1))
    np.testing.assert_array_equal(binary.predict(type="class"), np.zeros(10, dtype=int))
    continuous = fit(formula, data, control=mb.boost_control(1))
    with pytest.raises(ValueError, match="binary family"):
        continuous.predict(type="class")


def test_asymmetric_hinge_class_boundary():
    from mboostrs.families import _class_prediction

    np.testing.assert_array_equal(
        _class_prediction(np.array([0.4, 0.5, 0.6]), mb.HingeLoss(alpha_cost=0.75)), [0, 0, 1]
    )


@pytest.mark.r_parity
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
def test_glm_extract_matches_mboost(center, intercept, weighted):
    import rpy2.robjects as ro

    ro.r("""suppressMessages(library(mboost)); set.seed(17)
        d <- data.frame(x = rnorm(80, 3), z = rnorm(80, -2))
        d$y <- 2 * d$x + rnorm(80); wt <- runif(80, 0.5, 2)""")
    data = {name: np.asarray(ro.r(f"d${name}")) for name in ("x", "z", "y")}
    formula = "y ~ x + z" + ("" if intercept else " - 1")
    weights = np.asarray(ro.r("wt")) if weighted else None
    ro.r(f"up <- mboost::glmboost({formula}, data = d, center = {'TRUE' if center else 'FALSE'}, "
         + ("weights = wt, " if weighted else "") + "control = mboost::boost_control(mstop = 1))")
    with pytest.warns(UserWarning) if center and not intercept else nullcontext():
        model = mb.glmboost(formula, data, center=center, weights=weights,
                            control=mb.boost_control(1))
    for which, rwhich in ((None, "NULL"), (model.base_learner_names(), "seq_len(3)" if intercept else "seq_len(2)")):
        ours = model.extract("design", which=which)
        upstream = np.asarray(ro.r(f"extract(up, 'design', which={rwhich}, asmatrix=TRUE)"))
        np.testing.assert_allclose(np.column_stack(list(ours.values())), upstream, rtol=0, atol=1e-14)
    np.testing.assert_allclose(model.extract("weights"), np.asarray(ro.r("extract(up, 'weights')")),
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(model.residuals(), np.asarray(ro.r("residuals(up)")).ravel(),
                               rtol=0, atol=1e-12)


@pytest.mark.r_parity
@pytest.mark.parametrize("fit_name", ["glmboost", "gamboost"])
@pytest.mark.parametrize("family_name", ["Binomial", "AdaExp", "HingeLoss", "AUC"])
def test_binary_classes_match_mboost(fit_name, family_name):
    import rpy2.robjects as ro

    ro.r("""suppressMessages(library(mboost)); set.seed(21)
        d <- data.frame(x = rnorm(100)); d$y <- factor(rbinom(100, 1, plogis(d$x)))
        nd <- data.frame(x = c(-2, -0.5, 0.5, 2))""")
    formula = "y ~ bols(x)" if fit_name == "gamboost" else "y ~ x"
    ro.r(f"up <- mboost::{fit_name}({formula}, d, family = get('{family_name}', asNamespace('mboost'))(), "
         "control = mboost::boost_control(mstop = 30))")
    model = getattr(mb, fit_name)(formula, {
        "x": np.asarray(ro.r("d$x")), "y": np.asarray(ro.r("as.integer(d$y) - 1")),
    }, family=getattr(mb, family_name)(), control=mb.boost_control(30))
    for k in (5, 30):
        ro.r(f"mboost::mstop(up) <- {k}")
        for newdata, rnew in ((None, "NULL"), ({"x": np.asarray(ro.r("nd$x"))}, "nd")):
            expected = np.asarray(ro.r(f"as.integer(predict(up, newdata={rnew}, type='class')) - 1"))
            np.testing.assert_array_equal(model.predict(newdata, mstop=k, type="class"), expected)
