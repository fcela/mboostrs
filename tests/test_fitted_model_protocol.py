"""One protocol for every fitted model (issue #169).

`stopping_budget()`, `selected_terms()` and `predict_parameters()` answer the
same questions with the same return shapes for glmboost, gamboost and both
LSS schemes, using public calls only. A standard model has one predictor,
named "f".
"""

import numpy as np
import pytest

import mboostrs as mb


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return {"x": x, "y": rng.normal(x, np.exp(0.3 * x))}


def _models():
    d = _data()
    lss = {"mu": "y ~ bbs(x)", "sigma": "y ~ bols(x)"}
    return d, {
        "glmboost": mb.glmboost("y ~ x", data=d, control=mb.boost_control(20)),
        "gamboost": mb.gamboost("y ~ bbs(x) + bols(x)", data=d, control=mb.boost_control(20)),
        "glmboostLSS": mb.glmboostLSS({"mu": "y ~ x", "sigma": "y ~ x"}, data=d,
                                      families=mb.GaussianLSS(), control=mb.boost_control(6)),
        "cyclic": mb.gamboostLSS(lss, data=d, families=mb.GaussianLSS(),
                                 control=mb.boost_control(6)),
        "noncyclic": mb.gamboostLSS(lss, data=d, families=mb.GaussianLSS(),
                                    control=mb.boost_control(9), method="noncyclic"),
    }


@pytest.fixture(scope="module")
def fitted():
    return _models()


def test_stopping_budget(fitted):
    _, models = fitted
    for name, model in models.items():
        budget = model.stopping_budget()
        assert isinstance(budget, mb.StoppingBudget), name
        assert budget.total == sum(budget.per_parameter.values()), name
    assert models["glmboost"].stopping_budget().per_parameter == {"f": 20}
    assert models["cyclic"].stopping_budget() == mb.StoppingBudget(
        "cyclic", 12, {"mu": 6, "sigma": 6})
    noncyclic = models["noncyclic"].stopping_budget()
    assert noncyclic.scheme == "noncyclic" and noncyclic.total == 9


def test_selected_terms_agree_with_the_existing_accessors(fitted):
    _, models = fitted
    for name, model in models.items():
        rows = model.selected_terms(output="records")
        budget = model.stopping_budget()
        assert [r["order"] for r in rows] == list(range(1, budget.total + 1)), name
        for parameter, count in budget.per_parameter.items():
            mine = [r for r in rows if r["parameter"] == parameter]
            assert [r["iteration"] for r in mine] == list(range(1, count + 1)), name
    gam = models["gamboost"]
    assert [r["learner"] for r in gam.selected_terms(output="records")] == [
        gam.base_learner_names()[i] for i in gam._selected]
    for name in ("cyclic", "noncyclic", "glmboostLSS"):
        model = models[name]
        rows = model.selected_terms(output="records")
        for parameter, path in model.selected.items():
            assert [r["learner"] for r in rows if r["parameter"] == parameter] == list(path)
    first = models["cyclic"].selected_terms(output="records")[:2]
    assert [r["parameter"] for r in first] == ["mu", "sigma"]


def test_predict_parameters_has_one_shape(fitted):
    data, models = fitted
    for name, model in models.items():
        for scale in ("link", "response"):
            out = model.predict_parameters(data, type=scale)
            assert isinstance(out, dict) and out, name
            assert all(np.asarray(v).shape == (200,) for v in out.values()), name
    # The omitted type is the response scale (#326); an explicit type passes
    # through to predict() unchanged.
    np.testing.assert_array_equal(
        models["gamboost"].predict_parameters(data)["f"],
        models["gamboost"].predict(newdata=data, type="response"))
    np.testing.assert_array_equal(
        models["cyclic"].predict_parameters(data, type="response")["sigma"],
        models["cyclic"].predict(newdata=data, type="response")["sigma"])


def test_existing_calls_are_unchanged(fitted):
    _, models = fitted
    assert models["gamboost"].mstop == 20
    assert models["cyclic"].mstop == 6
    assert isinstance(models["cyclic"].predict(), dict)
    assert isinstance(models["gamboost"].predict(), np.ndarray)
