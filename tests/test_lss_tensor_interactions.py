"""`%X%` tensor interactions in `gamboostLSS` (X03).

`gamboostLSS("y ~ bbs(x) %X% bbs(z)")` used to fail inside the formula
parser with "got df= more than once" — the tensor was never expanded, so the
operator reached `bbs`'s argument parser as text. The expansion is the one
`gamboost` already runs: each tensor becomes a `buser` term whose design
matrix rides in `env`, the model keeps the interaction recipes, prediction
rebuilds the design at new rows, and `with_mstop`/`update`, `cvrisk`,
`stabsel` and artifacts carry it. The linear `glmboostLSS` keeps its
one-term-per-column contract and refuses `%X%` with the reason and the
supported alternative.
"""

import pathlib
import tempfile

import numpy as np
import pytest

import mboostrs as mb


def _surface_frame(seed=3, n=300):
    """A genuinely non-bilinear mean: sin(x)(z^2 + 1/2), which no bilinear
    form in x and z can span — the case a tensor exists for."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    y = np.sin(x) * (z ** 2 + 0.5) + rng.normal(size=n)
    return {"y": y, "x": x, "z": z}


TENSOR_SPEC = {"mu": "y ~ bbs(x, df=4) %X% bbs(z, df=4)", "sigma": "y ~ 1"}
BILINEAR_SPEC = {
    "mu": "y ~ bbs(x, df=4) + bbs(z, df=4) + bols(x, by=z, intercept=False)",
    "sigma": "y ~ 1",
}


def test_a_tensor_beats_the_bilinear_alternative_on_a_curved_surface():
    data = _surface_frame()
    fresh = {"x": np.random.default_rng(99).uniform(-2, 2, 400),
             "z": np.random.default_rng(98).uniform(-2, 2, 400)}
    truth = np.sin(fresh["x"]) * (fresh["z"] ** 2 + 0.5)
    control = mb.boost_control(3000, 0.05)
    tensor = mb.gamboostLSS(TENSOR_SPEC, data, family=mb.GaussianLSS(),
                            control=control)
    bilinear = mb.gamboostLSS(BILINEAR_SPEC, data, family=mb.GaussianLSS(),
                              control=control)
    tensor_mse = float(np.mean(
        (np.asarray(tensor.predict(newdata=fresh, type="response")["mu"])
         - truth) ** 2))
    bilinear_mse = float(np.mean(
        (np.asarray(bilinear.predict(newdata=fresh, type="response")["mu"])
         - truth) ** 2))
    assert tensor_mse < 0.5 * bilinear_mse, (tensor_mse, bilinear_mse)


def test_fit_predict_and_new_rows():
    data = _surface_frame(n=150)
    fit = mb.gamboostLSS(TENSOR_SPEC, data, family=mb.GaussianLSS(),
                         control=mb.boost_control(120, 0.1))
    assert fit.mstop == 120
    training = np.asarray(fit.params(type="response")["mu"])
    np.testing.assert_allclose(
        np.asarray(fit.predict(type="response")["mu"]), training, rtol=1e-12)
    fresh = {"x": np.array([-1.0, 0.0, 1.5]), "z": np.array([0.5, -1.0, 2.0])}
    predicted = fit.predict(newdata=fresh, type="response")
    assert np.all(np.isfinite(np.asarray(predicted["mu"])))
    # Fresh rows ride the rebuilt design, not the training one.
    assert len(predicted["mu"]) == 3


def test_truncation_and_cross_validation_carry_the_interaction():
    data = _surface_frame(n=150)
    fit = mb.gamboostLSS(TENSOR_SPEC, data, family=mb.GaussianLSS(),
                         control=mb.boost_control(80, 0.1))
    truncated = fit.with_mstop([20, 20])
    fresh = {"x": np.array([0.3, -0.4]), "z": np.array([1.0, 0.1])}
    assert np.all(np.isfinite(np.asarray(
        truncated.predict(newdata=fresh, type="response")["mu"])))
    folds = mb.cv_weights(150, type="kfold", B=3, seed=1)
    risk = fit.cvrisk(folds=folds)
    assert len(risk.optimal_mstop) == 2


def test_the_module_entry_accepts_a_raw_formula():
    data = _surface_frame(n=120)
    folds = mb.cv_weights(120, type="kfold", B=3, seed=2)
    grid = mb.diagonal_grid(60, 2, length_out=5)
    risk = mb.cvrisk_gamboostLSS(TENSOR_SPEC, data, family=mb.GaussianLSS(),
                                  control=mb.boost_control(60, 0.1),
                                  folds=folds, grid=grid)
    assert len(risk.optimal_mstop) == 2


def test_glmboostlss_refuses_a_tensor_with_the_alternative():
    data = _surface_frame(n=80)
    with pytest.raises(ValueError, match="one linear term per predictor"):
        mb.glmboostLSS("y ~ bbs(x, df=4) %X% bbs(z, df=4)", data,
                       family=mb.GaussianLSS())


def test_the_artifact_round_trips_the_interaction_recipes():
    data = _surface_frame(n=150)
    fit = mb.gamboostLSS(TENSOR_SPEC, data, family=mb.GaussianLSS(),
                         control=mb.boost_control(80, 0.1))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "tensor-lss.json"
        mb.save_model(fit, path)
        loaded = mb.load_model(path)
        fresh = {"x": np.array([-1.0, 0.5]), "z": np.array([0.0, 1.5])}
        reloaded = np.asarray(
            loaded.predict(newdata=fresh, type="response")["mu"])
        direct = np.asarray(
            fit.predict(newdata=fresh, type="response")["mu"])
        np.testing.assert_allclose(reloaded, direct, rtol=1e-12)


def test_predictions_equal_the_rebuilt_design_applied_by_hand():
    """The predictions are the tensor design at the new rows times the fit's
    aggregated coefficients — the expansion introduces no algebra of its
    own, and the replay the model runs is the one this recomputes.

    The reference the R comparison cannot give: R 2.9-11's `gamboostLSS`
    itself cannot fit `bbs(x) %X% bbs(z)` (it fails inside `knotf` with
    "boundary.knots must be a vector of length 2", while the same term
    fits in R `gamboost`), so mboostrs supports a specification upstream
    cannot, and the oracle is the design algebra itself.
    """
    from mboostrs.interactions import build_tensor_design

    data = _surface_frame(seed=7, n=150)
    fit = mb.gamboostLSS(TENSOR_SPEC, data, family=mb.GaussianLSS(),
                         control=mb.boost_control(100, 0.1))
    fresh = {"x": np.array([-1.2, 0.4, 1.7]), "z": np.array([0.6, -0.9, 1.1])}
    predicted = np.asarray(fit.predict(newdata=fresh, type="link")["mu"])

    interaction = next(r for r in fit._interactions)
    design = build_tensor_design(interaction, fresh)
    # Aggregate the mu path's coefficients over the steps that selected the
    # tensor term, scaled by nu, and add the parameter's offset.
    parameter = fit._params[fit._param_index("mu")]
    offset = np.full(len(fresh["x"]), float(parameter["offset"]))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for selected, coef in zip(parameter["selected"], parameter["coefs"],
                              strict=True):
        name = parameter["learners"][selected].name
        if name == interaction.derived:
            coefficients += fit._nu * np.asarray(coef, dtype=np.float64)
    by_hand = offset + design @ coefficients
    np.testing.assert_allclose(predicted, by_hand, rtol=1e-10, atol=1e-10)
