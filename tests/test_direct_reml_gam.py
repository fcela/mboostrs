"""Direct REML penalized GAMs: mb.gam and mb.gamlss, fitted from a formula."""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import direct_gam
from mboostrs.selection import EffectContrast, GamlssRefit, GamRefit, gam_refit


def _data(n=400, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    z = rng.uniform(0, 1, n)
    lat = rng.uniform(32.5, 42.0, n)
    lon = rng.uniform(-124.0, -114.0, n)
    group = rng.choice(["A", "B", "C", "D"], n)
    level = {"A": 0.0, "B": 0.4, "C": -0.2, "D": 0.9}
    shift = np.array([level[g] for g in group])
    f_x = np.sin(x / 10 * 2 * np.pi)
    bump = np.exp(-((lat - 37.0) ** 2 + (lon + 119.0) ** 2) / 4.0)
    mu = f_x + shift + 0.8 * bump + 0.2 * (z - 0.5) * x
    y = mu + rng.normal(0, 0.3, n)
    return pl.DataFrame({"y": y, "x": x, "z": z, "lat": lat, "lon": lon, "group": group})


FORMULA = ("y ~ bbs(x, knots=6) + bols(group) + bspatial(lon, lat, knots=4) "
           "+ bbs(x, knots=4) %X% bbs(z, knots=4)")


@pytest.fixture(scope="module")
def data():
    return _data()


@pytest.fixture(scope="module")
def fit(data):
    return direct_gam.gam(FORMULA, data)


def test_gam_returns_gam_refit_with_one_block_per_term(fit):
    assert isinstance(fit, GamRefit)
    assert fit.terms == (
        "bbs(x, knots=6)", "bols(group)", "bspatial(lon, lat, knots=4)",
        "bbs(x, knots=4) %X% bbs(z, knots=4)")
    assert set(fit.edf) == set(fit.terms)
    assert fit.conditional_on_selection is False
    assert np.isfinite(fit.reml_score)


def test_direct_fit_matches_gam_refit_on_the_same_terms(data):
    # A boosted model whose refit uses every formula term: the direct fit builds
    # the same designs and penalties, so the REML estimate is the same.
    boosted_formula = ("y ~ bbs(x, knots=6) + bols(group) + bspatial(lon, lat, knots=4) "
                       "+ bbs(x, knots=4) %X% bbs(z, knots=4)")
    boosted = mb.gamboost(boosted_formula, data=data, family=mb.Gaussian(),
                          control=mb.boost_control(mstop=5, nu=0.1))
    refit = gam_refit(boosted, which="all")
    direct = direct_gam.gam(boosted_formula, data)
    assert refit.terms == direct.terms
    # mb.gam forms X'WX once and factors it (the Gram path); gam_refit factors
    # the design by QR. The Gram squares the condition number, so the two agree
    # to about 1e-7 here, not to rounding. The REML criterion is flat at its
    # optimum, so how far each path's smoothing-parameter search gets depends on
    # rounding in the criterion: on AMD EPYC 7763 (AVX2 without AVX-512) the QR
    # path stops on the "flat" rule with a gradient of 9e-4 instead of 1e-7 and
    # its coefficients differ by 1.7e-5, while the two criterion values still
    # agree to 3e-9. The coefficient tolerance covers that spread.
    np.testing.assert_allclose(direct.coef()["estimate"].to_numpy(),
                               refit.coef()["estimate"].to_numpy(), rtol=0, atol=1e-4)
    np.testing.assert_allclose(direct.vcov(), refit.vcov(), rtol=1e-3, atol=1e-8)
    assert direct.reml_score == pytest.approx(refit.reml_score, abs=1e-5)
    for term in direct.terms:
        assert direct.edf[term] == pytest.approx(refit.edf[term], abs=1e-5)


def test_factor_term_is_unpenalized_and_recovers_level_differences(data, fit):
    # Level effects of group: D minus A is 0.9 in the simulation.
    contrast = fit.contrast("group", "A", "D", data=data)
    assert isinstance(contrast, EffectContrast)
    assert contrast.estimate == pytest.approx(0.9, abs=0.1)
    assert contrast.lower < 0.9 < contrast.upper
    coef = fit.coef()
    group_rows = coef.filter(pl.col("term") == "bols(group)")
    assert group_rows.height == 3  # four levels, one coefficient fewer after the constraint
    assert fit.edf["bols(group)"] == pytest.approx(3.0, abs=1e-6)


def test_smooth_term_tracks_the_curve(data, fit):
    grid = pl.DataFrame({"x": np.linspace(0.5, 9.5, 50), "z": np.full(50, 0.5),
                         "lat": np.full(50, 37.0), "lon": np.full(50, -119.0),
                         "group": ["A"] * 50, "y": np.zeros(50)})
    effects = fit.effects(grid, centre=True)
    estimate = effects["bbs(x, knots=6)"][0]
    truth = np.sin(grid["x"].to_numpy() / 10 * 2 * np.pi)
    truth = truth - truth.mean()
    assert np.sqrt(np.mean((estimate - truth) ** 2)) < 0.2


def test_spatial_and_tensor_terms_have_their_own_smoothing(fit):
    assert fit.smoothing["bspatial(lon, lat, knots=4)"] is not None
    tensor_lambda = fit.smoothing["bbs(x, knots=4) %X% bbs(z, knots=4)"]
    assert isinstance(tensor_lambda, tuple) and len(tensor_lambda) == 2
    assert fit.edf["bspatial(lon, lat, knots=4)"] > 1.0


def test_predict_interval_has_bounds_and_extrapolation_policy(data, fit):
    frame = fit.predict(data.head(20), interval=True, level=0.9)
    assert {"estimate", "se", "lower", "upper", "outside_range"} <= set(frame.columns)
    assert np.all(frame["lower"].to_numpy() <= frame["estimate"].to_numpy())
    assert np.all(frame["estimate"].to_numpy() <= frame["upper"].to_numpy())
    far = data.head(3).with_columns(pl.lit(50.0).alias("x"))
    with pytest.warns(Warning):
        fit.predict(far)
    with pytest.raises(ValueError):
        fit.predict(far, extrapolation="error")
    assert np.isfinite(fit.predict(far, extrapolation="allow")).all()


def test_contrast_is_the_mean_of_the_predicted_change(data, fit):
    # The average change equals the mean of predict(hi) - predict(lo) over the rows.
    lo_frame = data.with_columns(pl.lit(2.0).alias("x"))
    hi_frame = data.with_columns(pl.lit(7.0).alias("x"))
    expected = np.mean(fit.predict(hi_frame) - fit.predict(lo_frame))
    contrast = fit.contrast("x", 2.0, 7.0, data=data)
    assert contrast.estimate == pytest.approx(expected, abs=1e-8)
    assert contrast.lower <= contrast.estimate <= contrast.upper
    assert contrast.se > 0


def test_contrast_of_a_variable_no_term_reads_is_exactly_zero(data, fit):
    contrast = fit.contrast("unused_column", 0.0, 1.0, data=data.with_columns(
        pl.lit(0.0).alias("unused_column")))
    assert contrast.estimate == 0.0 and contrast.se == 0.0


def test_contrast_moves_several_covariates_together(data, fit):
    contrast = fit.contrast(("lat", "lon"), (37.8, -122.3), (34.2, -117.4), data=data)
    moved_lo = data.with_columns(pl.lit(37.8).alias("lat"), pl.lit(-122.3).alias("lon"))
    moved_hi = data.with_columns(pl.lit(34.2).alias("lat"), pl.lit(-117.4).alias("lon"))
    expected = np.mean(fit.predict(moved_hi) - fit.predict(moved_lo))
    assert contrast.estimate == pytest.approx(expected, abs=1e-8)


def test_contrast_rejects_mismatched_value_lengths(data, fit):
    with pytest.raises(ValueError, match="same length"):
        fit.contrast(("lat", "lon"), (37.8,), (34.2, -117.4), data=data)


def test_gamlss_fits_mu_and_sigma_jointly(data):
    rng = np.random.default_rng(9)
    sd = np.exp(-0.5 + 1.2 * data["z"].to_numpy())
    y = data["x"].to_numpy() * 0.3 + rng.normal(0, 1, data.height) * sd
    lss_data = data.with_columns(pl.Series("y", y))
    model = direct_gam.gamlss({"mu": "y ~ bbs(x, knots=5)",
                               "sigma": "y ~ bbs(z, knots=5)"}, lss_data)
    assert isinstance(model, GamlssRefit)
    assert model.parameters == ("mu", "sigma")
    low = model.predict(lss_data.head(1).with_columns(pl.lit(0.05).alias("z")),
                        parameter="sigma", type="response")
    high = model.predict(lss_data.head(1).with_columns(pl.lit(0.95).alias("z")),
                         parameter="sigma", type="response")
    assert float(np.asarray(high).ravel()[0]) > float(np.asarray(low).ravel()[0])
    interval = model.predict(lss_data.head(10), parameter="mu", interval=True)
    assert {"estimate", "se", "lower", "upper"} <= set(interval.columns)


def test_gamlss_contrast_per_parameter(data):
    lss_data = data.with_columns(pl.Series("y", data["y"]))
    model = direct_gam.gamlss({"mu": "y ~ bbs(x, knots=5)", "sigma": "y ~ bbs(z, knots=5)"},
                              lss_data)
    both = model.contrast("x", 2.0, 7.0, data=lss_data)
    assert set(both) == {"mu", "sigma"}
    mu = model.contrast("x", 2.0, 7.0, data=lss_data, parameter="mu")
    assert mu.estimate == pytest.approx(both["mu"].estimate)
    assert both["sigma"].se == 0.0  # sigma has no term that reads x
    assert mu.lower <= mu.estimate <= mu.upper


def test_binomial_and_poisson_families_fit(data):
    binary = data.with_columns((pl.col("y") > 0.5).cast(pl.Float64).alias("y"))
    logit = direct_gam.gam("y ~ bbs(x, knots=5) + bols(group)", binary, family="binomial")
    assert logit.link == "logit"
    counts = data.with_columns(pl.col("y").exp().round().alias("y"))
    pois = direct_gam.gam("y ~ bbs(x, knots=5)", counts, family=mb.Poisson())
    assert pois.link == "log"


def test_gamma_family_fits_a_positive_response(data):
    positive = data.with_columns(pl.col("y").exp().alias("y"))
    fit = direct_gam.gam("y ~ bbs(x, knots=5)", positive, family="gamma")
    assert fit.link == "log"
    assert np.all(fit.predict(positive.head(5), type="response") > 0)


def test_na_action_error_names_the_column_and_remedy():
    rng = np.random.default_rng(1)
    frame = pl.DataFrame({"y": rng.normal(size=60), "x": [None] + list(rng.uniform(0, 1, 59))})
    with pytest.raises(ValueError, match="'x'.*na_action='impute'"):
        direct_gam.gam("y ~ bbs(x, knots=4)", frame)


def test_na_action_omit_drops_rows_and_impute_keeps_them():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 80)
    y = rng.normal(size=80) + x
    x[:5] = np.nan
    frame = pl.DataFrame({"y": y, "x": x})
    omitted = direct_gam.gam("y ~ bbs(x, knots=4)", frame, na_action="omit")
    assert len(omitted._training_data) == 75
    imputed = direct_gam.gam("y ~ bbs(x, knots=4)", frame, na_action="impute")
    assert any("x_missing" in t for t in imputed.terms)
    assert np.isfinite(imputed.predict(frame.head(3))).all()


def test_method_other_than_reml_is_refused(data):
    with pytest.raises(NotImplementedError, match="REML"):
        direct_gam.gam("y ~ bbs(x, knots=4)", data, method="ML")


def test_unsupported_learner_is_named(data):
    with pytest.raises(NotImplementedError, match="btree"):
        direct_gam.gam("y ~ btree(group)", data)


def test_gam_has_no_boosting_argument(data):
    with pytest.raises(TypeError):
        direct_gam.gam("y ~ bbs(x, knots=4)", data, mstop=50)


def test_fit_emits_no_convergence_warning_on_the_test_model(data):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        direct_gam.gam("y ~ bbs(x, knots=5) + bols(group)", data)
    assert not [w for w in seen if "did not converge" in str(w.message)]


@pytest.mark.parametrize("basis", ["bols(x)", "bbs(x, knots=6)"])
@pytest.mark.parametrize("weighted", [False, True])
def test_numeric_by_preserves_the_constant_in_x_direction(basis, weighted):
    # The raw constant becomes g, independent of the global intercept. A
    # constraint on f(x) before multiplication would silently remove it.
    rng = np.random.default_rng(619211)
    x = rng.uniform(-1, 1, 400)
    g = rng.uniform(.5, 2, 400)
    y = .4 + (2 + 3*x)*g + rng.normal(0, .01, 400)
    data = pl.DataFrame(dict(y=y, x=x, g=g))
    weights = rng.uniform(.5, 2, 400) if weighted else None
    term = basis[:-1] + ", by=g)"
    fit = mb.gam("y ~ " + term, data, weights=weights)
    assert np.mean((fit.predict(data) - y)**2) < .0002
    nd = pl.DataFrame({"x": np.linspace(-.9, .9, 30), "g": np.linspace(.6, 1.9, 30)})
    truth = .4 + (2 + 3*nd["x"].to_numpy())*nd["g"].to_numpy()
    assert np.max(np.abs(fit.predict(nd) - truth)) < .02
    interval = fit.predict(nd, interval=True)
    assert np.isfinite(interval["se"].to_numpy()).all()
    assert (interval["se"].to_numpy() > 0).all()
    if basis == "bols(x)":
        # Independent weighted least-squares oracle verifies the complete
        # function space, its covariance and the out-of-sample design map.
        w = np.ones(400) if weights is None else weights
        design = np.column_stack([np.ones(400), g, x*g])
        beta = np.linalg.lstsq(design*np.sqrt(w[:, None]), y*np.sqrt(w), rcond=None)[0]
        pred = design @ beta
        np.testing.assert_allclose(fit.predict(data), pred, atol=1e-8, rtol=0)
        phi = np.sum(w*(y-pred)**2)/(400-3)
        new_design = np.column_stack([np.ones(30), nd["g"], nd["x"]*nd["g"]])
        cov = phi*np.linalg.inv(design.T @ (w[:, None]*design))
        se = np.sqrt(np.einsum("ij,jk,ik->i", new_design, cov, new_design))
        np.testing.assert_allclose(interval["se"], se, atol=1e-8, rtol=1e-6)
        assert fit.edf[term] == pytest.approx(2, abs=1e-8)


@pytest.mark.parametrize("basis", ["bols(x)", "bbs(x, knots=6)"])
def test_constant_numeric_by_retains_ordinary_centering(basis):
    rng = np.random.default_rng(619212)
    x = rng.uniform(-1, 1, 250)
    data = pl.DataFrame(dict(x=x, g=np.full(250, 2.), y=1+3*x+rng.normal(0, .03, 250)))
    plain = mb.gam("y ~ " + basis, data)
    by = mb.gam("y ~ " + basis[:-1] + ", by=g)", data)
    np.testing.assert_allclose(by.predict(data), plain.predict(data), atol=2e-5, rtol=0)


def test_numeric_by_removes_actual_global_and_nested_overlap():
    rng = np.random.default_rng(619213)
    x = rng.uniform(.5, 2, 300)
    g = 1/x
    y = .4 + 2*g + rng.normal(0, .01, 300)
    data = pl.DataFrame(dict(x=x, g=g, y=y))
    inverse = mb.gam("y ~ bols(x, by=g)", data)
    with pytest.raises(ValueError, match="reproduced entirely"):
        mb.gam("y ~ bols(g) + bols(x, by=g)", data)
    oracle = mb.gam("y ~ bols(g)", data)
    np.testing.assert_allclose(inverse.predict(data), oracle.predict(data), atol=1e-8)


def test_numeric_by_explicit_main_effect_and_refit_agree():
    rng = np.random.default_rng(619214)
    x = rng.uniform(-1, 1, 300)
    g = rng.uniform(.5, 2, 300)
    data = pl.DataFrame(dict(x=x, g=g, y=(2+3*x)*g+rng.normal(0, .01, 300)))
    formula = "y ~ bols(x, by=g)"
    direct = mb.gam(formula, data)
    explicit = mb.gam("y ~ bols(g) + bols(x, by=g)", data)
    boosted = mb.gamboost(formula, data=data, control=mb.boost_control(mstop=5))
    refit = mb.gam_refit(boosted, which="all")
    np.testing.assert_allclose(direct.predict(data), explicit.predict(data), atol=1e-8)
    np.testing.assert_allclose(direct.predict(data), refit.predict(data), atol=1e-8)
