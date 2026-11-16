"""Named negative-binomial size keeps the fitted density and path semantics."""

import numpy as np
import pytest

import mboostrs as mb


@pytest.mark.parametrize("kind", ["glm", "gam"])
@pytest.mark.parametrize("step", ["fixed", "line_search"])
def test_size_scores_the_fitted_density_and_tracks_slices(kind, step, tmp_path):
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 250)
    mu = np.exp(0.5 + x)
    data = {"x": x, "y": rng.negative_binomial(2, 2 / (2 + mu)).astype(float)}
    weights = rng.integers(1, 5, len(x)).astype(float)
    fit = mb.glmboost if kind == "glm" else mb.gamboost
    formula = "y ~ x" if kind == "glm" else "y ~ bols(x)"
    model = fit(formula, data, family=mb.NBinomial(), weights=weights,
                weight_policy="raw", control=mb.boost_control(100, step=step))
    for current in [model, model[:20], model.continue_to(120)]:
        assert isinstance(current.size, float) and current.size > 0
        assert current.size == current.nuisance()[0]
        assert current.control.step == step
        mean = current.predict(type="response")
        loglik = scipy.nbinom.logpmf(data["y"], current.size, current.size / (current.size + mean))
        np.testing.assert_allclose(current.logLik(), np.sum(weights * loglik), rtol=1e-10)
    assert model[:20].size != model.size
    grown = model.continue_to(120)
    assert grown[:20].size == model[:20].size
    assert grown[:100].size == model.size
    cold = fit(formula, data, family=mb.NBinomial(), weights=weights,
               weight_policy="raw", control=mb.boost_control(120, step=step))
    np.testing.assert_allclose(grown.size, cold.size, rtol=1e-12)
    path = tmp_path / "nb.json"
    mb.save_model(model[:20], path)
    assert mb.load_model(path).size == model[:20].size


@pytest.mark.parametrize("keyword", ["alpha", "dispersion", "size"])
def test_constructor_explains_estimated_size_and_nb2_reciprocal(keyword):
    with pytest.raises(TypeError, match=r"size.*alpha = 1/size"):
        mb.NBinomial(**{keyword: 0.5})


@pytest.mark.parametrize("kind", ["glm", "gam"])
@pytest.mark.parametrize("family", [mb.Poisson(), mb.Hurdle()])
def test_size_does_not_label_other_families_nuisance(kind, family):
    fit = mb.glmboost if kind == "glm" else mb.gamboost
    model = fit("y ~ x" if kind == "glm" else "y ~ bols(x)",
                {"x": np.linspace(-1, 1, 30), "y": np.arange(30) % 5 + 1.0},
                family=family, control=mb.boost_control(10))
    with pytest.raises(AttributeError, match="NBinomial"):
        _ = model.size
