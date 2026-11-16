"""A spatial radial fit retains its training basis at unseen locations."""

import pickle

import numpy as np
import pytest

import mboostrs as mb


def surface(n=160):
    rng = np.random.default_rng(9)
    x, z = rng.uniform(size=(2, n))
    by = rng.uniform(.2, 1.4, n)
    data = {"x": x, "z": z, "b": by, "y": np.sin(4*x)*np.cos(3*z)}
    return data


@pytest.mark.parametrize("center,by", [(False, None), (True, None), (True, "b")])
def test_formula_and_explicit_radial_fit_predict_at_new_locations(center, by, tmp_path):
    d = surface()
    weights = np.arange(len(d["x"])) % 4 + 1.0
    learner = mb.Brad(["x", "z"], d, knots=25, df=4, center=center, by=by)
    suffix = f", by={by}" if by else ""
    formula = f"y ~ brad(x, z, knots=25, df=4, center={center}{suffix})"
    model = mb.gamboost(formula, d, weights=weights, weight_policy="raw", control=mb.boost_control(150))
    explicit = mb.gamboost("y ~ bols(x)", d, learners=[learner], weights=weights,
                           weight_policy="raw", control=mb.boost_control(150))
    np.testing.assert_allclose(model.fitted_, explicit.fitted_, atol=1e-12)
    np.testing.assert_allclose(model.predict(d, type="link"), model.fitted_, atol=1e-12)
    new = surface(37)
    prediction = model.predict(new, type="link")
    assert prediction.shape == (37,)
    np.testing.assert_allclose(prediction, explicit.predict(new, type="link"), atol=1e-12)
    np.testing.assert_allclose(learner.design(d), learner.X, atol=1e-12)
    effects = model.partial_effects(newdata=new)
    np.testing.assert_allclose(sum(v for _, v in effects.values()) + model.offset, prediction, atol=1e-12)
    path = tmp_path / "radial.json"
    mb.save_model(model, path)
    for restored in [mb.load_model(path), pickle.loads(pickle.dumps(model))]:
        np.testing.assert_allclose(restored.predict(new, type="link"), prediction, atol=1e-12)
    sliced = model[:40]
    np.testing.assert_allclose(sliced.predict(new, type="link"), model.predict(new, mstop=40, type="link"), atol=1e-12)


def test_radial_formula_cvrisk_and_distributional_prediction():
    d = surface(90)
    model = mb.gamboost("y ~ brad(x, z, knots=15)", d, control=mb.boost_control(30))
    cv = model.cvrisk(B=3, type="kfold", _inner=True)
    assert np.all(np.isfinite(cv.risk_matrix))
    lss = mb.gamboostLSS({"mu": "y ~ brad(x, z, knots=15)", "sigma": "y ~ bols(x)"},
                         d, families=mb.GaussianLSS(), control=mb.boost_control(30))
    predictions = lss.predict(surface(23), type="response")
    assert all(np.asarray(v).shape == (23,) for v in predictions.values())
    assert np.all(np.isfinite(predictions["mu"]))


def test_radial_invalid_locations_and_formula_arguments():
    d = surface(20)
    learner = mb.Brad(["x", "z"], d, knots=10)
    with pytest.raises(ValueError, match="finite"):
        learner.design({"x": [np.nan], "z": [0.]})
    with pytest.raises(KeyError):
        learner.design({"x": [0.]})
    with pytest.raises(ValueError, match="unknown brad parameter"):
        mb.gamboost("y ~ brad(x, z, thetta=1)", d)


def test_one_coordinate_formula_and_three_coordinate_design():
    from mboostrs.formula import expand_formula

    d = surface(50)
    assert expand_formula("y ~ . + brad(x,z)", d) == "y ~ b + brad(x,z)"
    model = mb.gamboost("y ~ brad(x, knots=12)", d, control=mb.boost_control(30))
    assert model.predict(surface(11), type="link").shape == (11,)
    radial = mb.Brad(["x", "z", "b"], d, knots=12)
    assert radial.design(surface(17)).shape == (17, 12)


@pytest.mark.r_parity
def test_r_frontend_replays_the_same_weighted_radial_fit():
    import rpy2.robjects as ro

    d = surface(90)
    weights = np.arange(90) % 4 + 1.0
    formula = "y ~ brad(x,z,knots=20,df=4,center=True)"
    py = mb.gamboost(formula, d, weights=weights, weight_policy="raw", control=mb.boost_control(80))
    ro.r("stopifnot(requireNamespace('mboostrs', quietly=TRUE))")
    ro.globalenv["d"] = ro.DataFrame({k: ro.FloatVector(v) for k,v in d.items()})
    ro.globalenv["w"] = ro.FloatVector(weights)
    ro.r("r <- mboostrs::gamboost(y~brad(x,z,knots=20,df=4,center=TRUE), d, "
         "weights=w, weight_policy='raw', control=mboostrs::boost_control(80))")
    new = surface(27)
    ro.globalenv["new"] = ro.DataFrame({k: ro.FloatVector(v) for k,v in new.items()})
    np.testing.assert_allclose(py.fitted_, np.asarray(ro.r("r$fitted")), atol=1e-9)
    np.testing.assert_allclose(py.predict(new, type="link"), np.asarray(ro.r("as.numeric(predict(r,new))")), atol=1e-9)
