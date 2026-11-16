import json
from math import factorial, lgamma

import numpy as np
import pytest
from mboostrs.gamlss import (
    ZINBLSS,
    BetaLSS,
    BinomialLSS,
    GammaLSS,
    GaussianLSS,
    GeometricLSS,
    LssCvRiskResult,
    LssStabSelResult,
    NBinomialLSS,
    NegativeBinomialLSS,
    PoissonLSS,
    StudentTLSS,
    ZIPoLSS,
    cvrisk_glmboostLSS,
    stabsel_glmboostLSS,
)

import mboostrs as mb


def _data(n=48, seed=2026):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    sigma = np.exp(0.1 + 0.2 * x2)
    y = 1.0 + 0.8 * x1 + sigma * rng.normal(size=n)
    return {"y": y, "x1": x1, "x2": x2}


def _folds(n, b=3):
    folds = np.ones((n, b), dtype=np.float64)
    for j in range(b):
        folds[j::b, j] = 0.0
    return folds


def test_lss_cvrisk_result_object_keeps_dict_compatibility():
    data = _data()
    grid = [[1, 1], [2, 1], [2, 2]]

    res = cvrisk_glmboostLSS(
        "y ~ x1 + x2",
        data=data,
        families=GaussianLSS(),
        control=mb.boost_control(4, 0.1),
        folds=_folds(len(data["y"])),
        grid=grid,
    )

    assert isinstance(res, LssCvRiskResult)
    assert res.risk_matrix.shape == (3, 3)
    assert res["risk"].shape == (3, 3)
    assert res["grid"] == grid
    assert res["mstop"] in grid
    np.testing.assert_allclose(res.aggregate, res.risk_matrix.mean(axis=0))

    frame = res.to_frame()
    assert {"fold", "grid_index", "mstop", "risk", "aggregate", "optimal", "mstop_0", "mstop_1"} <= set(frame.columns)
    assert len(frame) == 9


def test_lss_stabsel_result_object_keeps_dict_compatibility():
    data = _data(n=40)
    res = stabsel_glmboostLSS(
        "y ~ x1 + x2",
        data=data,
        families=GaussianLSS(),
        control=mb.boost_control(5, 0.1),
        folds=_folds(len(data["y"]), b=4),
        cutoff=0.5,
        q=2,
        sampling_type="MB",
    )

    assert isinstance(res, LssStabSelResult)
    assert res.phat.shape[0] == len(res.learner_names)
    assert set(res["selected"]) == set(res.selected)
    assert set(res["max"]) == set(res.learner_names)
    assert res["names"] == res.learner_names
    assert res["cutoff"] == res.cutoff
    assert res["q"] == res.q

    frame = res.to_frame()
    assert {"learner", "max", "selected", "cutoff", "q", "PFER"} <= set(frame.columns)
    assert len(frame) == len(res.learner_names)


def test_lss_model_inspection_methods_and_update():
    data = _data(n=36)
    mod = mb.glmboostLSS(
        "y ~ x1 + x2",
        data=data,
        families=GaussianLSS(),
        control=mb.boost_control(5, 0.1),
    )

    selected = mod.selected
    assert set(selected) == {"mu", "sigma"}
    assert mod.selected_which(parameter="mu") == selected["mu"]
    assert all(name.endswith((".mu", ".sigma")) for name in mod.selected_which(merge=True))

    np.testing.assert_allclose(
        mod.fitted(parameter="mu"),
        mod.predict(parameter="mu", type="link"),
    )
    np.testing.assert_allclose(mod.risk, mod.risk_)
    assert mod.risk is not mod.risk_
    with pytest.raises(NotImplementedError, match="parameter-specific"):
        mod.risk(parameter="mu")

    weights = mod.model_weights()
    np.testing.assert_allclose(weights, np.ones(len(data["y"])))
    info = mod.summary(as_dict=True)
    json.dumps(info)
    assert info["model_kind"] == "glmboostLSS"
    assert info["family"] == "GaussianLSS"
    assert mod.logLik() == -float(mod.risk_[-1])
    gaic_val = mod.gaic()
    assert isinstance(gaic_val, float)
    assert gaic_val > 0

    same = mod.update()
    assert same is mod
    updated = mod.update(mstop=3)
    assert updated.mstop == 3
    assert updated.summary(as_dict=True)["mstop"] == 3
    pred_all = mod.predict_all()
    assert {"y", "mu", "sigma"} <= set(pred_all.columns)
    assert len(pred_all) == len(data["y"])
    pred_dict = mod.predict_all(output="dict")
    assert set(pred_dict) == {"mu", "sigma"}
    coef_all = mod.coef_all()
    assert {"parameter", "learner", "coefficient"} <= set(coef_all.columns)
    pit = mod.pit(randomized=False)
    assert pit.shape == data["y"].shape
    assert np.all((pit > 0.0) & (pit < 1.0))
    res_frame = mod.residual_frame(kind="quantile", randomized=False)
    assert {"y", "mu", "sigma", "residual", "pit", "deviance"} <= set(res_frame.columns)
    assert len(res_frame) == len(data["y"])
    response_frame = mod.residual_frame(kind="response")
    assert {"residual_kind", "residual"} <= set(response_frame.columns)
    q_stats = mod.q_stats(xvar="x1", n_intervals=4, randomized=False)
    assert {"interval", "xvar", "x_min", "x_max", "n", "mean", "variance", "skewness", "kurtosis"} <= set(
        q_stats.columns
    )
    assert len(q_stats) == 4


def test_smooth_lss_model_inspection_methods_and_update():
    data = _data(n=40)
    mod = mb.gamboostLSS(
        "y ~ bbs(x1, df=4)",
        data=data,
        families=GaussianLSS(),
        control=mb.boost_control(5, 0.1),
    )

    selected = mod.selected
    assert set(selected) == {"mu", "sigma"}
    assert mod.selected_which(parameter="sigma") == selected["sigma"]
    assert all(name.endswith((".mu", ".sigma")) for name in mod.selected_which(merge=True))

    np.testing.assert_allclose(
        mod.fitted(parameter="mu"),
        mod.predict(parameter="mu", type="link"),
    )
    truncated = mod.fitted(parameter="mu", mstop=2)
    assert truncated.shape == data["y"].shape
    assert mod.summary(as_dict=True)["model_kind"] == "gamboostLSS"

    updated = mod.update(mstop=2)
    assert updated.mstop == 2
    np.testing.assert_allclose(
        updated.fitted(parameter="sigma"),
        updated.predict(parameter="sigma", type="link"),
    )
    pred_all = mod.predict_all()
    assert {"y", "mu", "sigma"} <= set(pred_all.columns)
    coef_all = mod.coef_all()
    assert {"parameter", "learner", "coefficient"} <= set(coef_all.columns)
    res_frame = mod.residual_frame(kind="response")
    assert {"y", "mu", "sigma", "residual"} <= set(res_frame.columns)
    q_stats = mod.q_stats(n_intervals=4, randomized=False)
    assert {"interval", "mean", "variance", "skewness", "kurtosis"} <= set(q_stats.columns)


def test_gaussian_lss_distribution_functions_and_model_deviance():
    family = GaussianLSS()
    y = np.array([0.0, 1.0, 2.0])
    params = {"mu": np.array([0.0, 0.5, 1.0]), "sigma": np.array([1.0, 2.0, 0.5])}

    z = (y - params["mu"]) / params["sigma"]
    expected_pdf = np.exp(-0.5 * z * z) / (params["sigma"] * np.sqrt(2.0 * np.pi))
    np.testing.assert_allclose(family.pdf(y, params), expected_pdf)
    np.testing.assert_allclose(family.cdf(params["mu"], params), np.full(3, 0.5))
    np.testing.assert_allclose(family.quantile(np.full(3, 0.5), params), params["mu"])
    np.testing.assert_allclose(family.deviance_increment(y, params), -2.0 * np.log(expected_pdf))
    assert family.simulate(params, rng=np.random.default_rng(1)).shape == (3,)
    with pytest.raises(ValueError, match="probabilities"):
        family.quantile([0.0], {"mu": [0.0], "sigma": [1.0]})

    data = _data(n=30)
    mod = mb.glmboostLSS(
        "y ~ x1 + x2",
        data=data,
        families=GaussianLSS(),
        control=mb.boost_control(4, 0.1),
    )
    model_params = mod.params()
    assert set(model_params) == {"mu", "sigma"}
    np.testing.assert_allclose(model_params["mu"], mod.predict(parameter="mu", type="response"))
    deviance = mod.deviance_increment()
    expected = family.deviance_increment(data["y"], model_params)
    np.testing.assert_allclose(deviance, expected)
    interval = mod.prediction_interval(level=0.8)
    assert {"lower", "median", "upper", "level", "lower_prob", "upper_prob"} <= set(interval.columns)
    assert len(interval) == len(data["y"])
    centiles = mod.centiles(probs=[0.1, 0.5, 0.9])
    assert {"q10", "q50", "q90"} <= set(centiles.columns)
    residuals = mod.quantile_residuals(randomized=False)
    assert residuals.shape == data["y"].shape
    assert np.isfinite(residuals).all()


def test_lss_density_mass_and_deviance_for_initial_family_batch():
    y_gamma = np.array([0.5, 1.0, 2.0])
    gamma_params = {"mu": np.array([1.0, 1.5, 2.5]), "sigma": np.array([2.0, 3.0, 4.0])}
    gamma_scale = gamma_params["mu"] / gamma_params["sigma"]
    gamma_logpdf = (
        (gamma_params["sigma"] - 1.0) * np.log(y_gamma)
        - y_gamma / gamma_scale
        - np.vectorize(lgamma)(gamma_params["sigma"])
        - gamma_params["sigma"] * np.log(gamma_scale)
    )
    np.testing.assert_allclose(GammaLSS().pdf(y_gamma, gamma_params), np.exp(gamma_logpdf))
    np.testing.assert_allclose(GammaLSS().deviance_increment(y_gamma, gamma_params), -2.0 * gamma_logpdf)

    y_beta = np.array([0.2, 0.5, 0.8])
    beta_params = {"mu": np.array([0.3, 0.5, 0.7]), "phi": np.array([5.0, 8.0, 10.0])}
    alpha = beta_params["mu"] * beta_params["phi"]
    beta = (1.0 - beta_params["mu"]) * beta_params["phi"]
    beta_logpdf = (
        (alpha - 1.0) * np.log(y_beta)
        + (beta - 1.0) * np.log1p(-y_beta)
        + np.vectorize(lgamma)(alpha + beta)
        - np.vectorize(lgamma)(alpha)
        - np.vectorize(lgamma)(beta)
    )
    np.testing.assert_allclose(BetaLSS().pdf(y_beta, beta_params), np.exp(beta_logpdf))
    assert BetaLSS().simulate(beta_params, rng=np.random.default_rng(1)).shape == (3,)

    y_nb = np.array([0.0, 2.0, 5.0])
    nb_params = {"mu": np.array([1.0, 3.0, 4.0]), "sigma": np.array([2.0, 5.0, 3.0])}
    nb_logpmf = (
        np.vectorize(lgamma)(y_nb + nb_params["sigma"])
        - np.vectorize(lgamma)(nb_params["sigma"])
        - np.vectorize(lgamma)(y_nb + 1.0)
        + nb_params["sigma"] * (np.log(nb_params["sigma"]) - np.log(nb_params["sigma"] + nb_params["mu"]))
        + y_nb * (np.log(nb_params["mu"]) - np.log(nb_params["sigma"] + nb_params["mu"]))
    )
    np.testing.assert_allclose(NBinomialLSS().pmf(y_nb, nb_params), np.exp(nb_logpmf))
    np.testing.assert_allclose(NBinomialLSS().deviance_increment(y_nb, nb_params), -2.0 * nb_logpmf)
    nb_cdf = NBinomialLSS().cdf(y_nb, nb_params)
    assert np.all((nb_cdf > 0.0) & (nb_cdf < 1.0))
    nb_q = NBinomialLSS().quantile([0.25, 0.5, 0.75], nb_params)
    assert np.all(nb_q >= 0.0)
    np.testing.assert_array_less(
        NBinomialLSS().cdf(nb_q - 1.0, nb_params),
        np.array([0.25, 0.5, 0.75]),
    )
    assert np.all(NBinomialLSS().cdf(nb_q, nb_params) >= np.array([0.25, 0.5, 0.75]))

    y_pois = np.array([0.0, 2.0, 4.0])
    pois_params = {"mu": np.array([1.0, 3.0, 5.0])}
    pois_logpmf = y_pois * np.log(pois_params["mu"]) - pois_params["mu"] - np.vectorize(lgamma)(y_pois + 1.0)
    np.testing.assert_allclose(PoissonLSS().pmf(y_pois, pois_params), np.exp(pois_logpmf))
    np.testing.assert_allclose(PoissonLSS().deviance_increment(y_pois, pois_params), -2.0 * pois_logpmf)
    pois_cdf = PoissonLSS().cdf(y_pois, pois_params)
    expected_pois_cdf = np.array(
        [
            np.exp(-1.0),
            np.exp(-3.0) * (1.0 + 3.0 + 3.0**2 / 2.0),
            np.exp(-5.0) * sum(5.0**k / factorial(k) for k in range(5)),
        ]
    )
    np.testing.assert_allclose(pois_cdf, expected_pois_cdf)
    pois_q = PoissonLSS().quantile([0.25, 0.5, 0.75], pois_params)
    np.testing.assert_array_less(
        PoissonLSS().cdf(pois_q - 1.0, pois_params),
        np.array([0.25, 0.5, 0.75]),
    )
    assert np.all(PoissonLSS().cdf(pois_q, pois_params) >= np.array([0.25, 0.5, 0.75]))
    assert PoissonLSS().simulate(pois_params, rng=np.random.default_rng(1)).shape == (3,)


def test_poisson_lss_model_interval_and_quantile_residuals():
    rng = np.random.default_rng(2027)
    n = 32
    x1 = rng.normal(size=n)
    mu = np.exp(0.5 + 0.2 * x1)
    data = {"y": rng.poisson(mu), "x1": x1}
    mod = mb.glmboostLSS(
        "y ~ x1",
        data=data,
        families=PoissonLSS(),
        control=mb.boost_control(4, 0.1),
    )

    interval = mod.predint(level=0.8)
    assert len(interval) == n
    assert {"lower", "median", "upper"} <= set(interval.columns)
    assert np.all(np.asarray(interval["lower"], dtype=float) <= np.asarray(interval["upper"], dtype=float))
    centiles = mod.centiles(probs=[0.1, 0.9])
    assert {"q10", "q90"} <= set(centiles.columns)
    residuals = mod.quantile_residuals(randomized=False)
    assert residuals.shape == (n,)
    assert np.isfinite(residuals).all()
    pit = mod.pit(randomized=True, random_state=123)
    np.testing.assert_allclose(pit, mod.pit(randomized=True, random_state=123))
    assert np.all((pit > 0.0) & (pit < 1.0))
    res_frame = mod.residual_frame(kind="quantile", randomized=False)
    assert {"y", "mu", "residual", "pit", "deviance"} <= set(res_frame.columns)
    q_stats = mod.q_stats(xvar="x1", n_intervals=4, randomized=False)
    assert len(q_stats) == 4


def test_second_batch_discrete_distribution_functions():
    """Verify pmf/cdf/quantile/simulate/deviance for the second family batch."""

    # --- NegativeBinomialLSS (NBII parameterization) ---
    y = np.array([0.0, 2.0, 5.0])
    params = {"mu": np.array([2.0, 3.0, 4.0]), "sigma": np.array([1.0, 2.0, 3.0])}
    nb = NegativeBinomialLSS()
    pmf = nb.pmf(y, params)
    assert np.all(pmf > 0) and np.all(pmf < 1)
    cdf = nb.cdf(y, params)
    assert np.all((cdf > 0) & (cdf <= 1))
    q = nb.quantile([0.3, 0.5, 0.7], params)
    assert np.all(q >= 0)
    np.testing.assert_array_less(nb.cdf(q - 1.0, params), [0.3, 0.5, 0.7])
    assert np.all(nb.cdf(q, params) >= [0.3, 0.5, 0.7])
    assert nb.simulate(params, rng=np.random.default_rng(0)).shape == (3,)
    dev = nb.deviance_increment(y, params)
    np.testing.assert_allclose(dev, -2.0 * np.log(pmf))

    # --- GeometricLSS ---
    y = np.array([0.0, 1.0, 3.0])
    params = {"mu": np.array([1.0, 3.0, 5.0])}
    geom = GeometricLSS()
    pmf = geom.pmf(y, params)
    assert np.all(pmf > 0) and np.all(pmf < 1)
    cdf = geom.cdf(y, params)
    assert np.all((cdf > 0) & (cdf <= 1))
    q = geom.quantile([0.3, 0.5, 0.7], params)
    assert np.all(q >= 0)
    assert geom.simulate(params, rng=np.random.default_rng(0)).shape == (3,)
    dev = geom.deviance_increment(y, params)
    np.testing.assert_allclose(dev, -2.0 * np.log(pmf))

    # --- BinomialLSS (single trial, Bernoulli) ---
    y = np.array([0.0, 1.0, 0.0])
    params = {"mu": np.array([0.3, 0.6, 0.8])}
    bl = BinomialLSS()
    pmf = bl.pmf(y, params)
    assert np.all(pmf > 0) and np.all(pmf < 1)
    cdf = bl.cdf(y, params)
    assert np.all((cdf > 0) & (cdf <= 1))
    q = bl.quantile([0.3, 0.5, 0.7], params)
    assert np.all(q >= 0)
    assert bl.simulate(params, rng=np.random.default_rng(0)).shape == (3,)
    dev = bl.deviance_increment(y, params)
    np.testing.assert_allclose(dev, -2.0 * np.log(pmf))

    # --- ZIPoLSS ---
    y = np.array([0.0, 1.0, 3.0])
    params = {"mu": np.array([1.0, 3.0, 5.0]), "sigma": np.array([0.1, 0.3, 0.5])}
    zipo = ZIPoLSS()
    pmf = zipo.pmf(y, params)
    assert np.all(pmf > 0) and np.all(pmf < 1)
    cdf = zipo.cdf(y, params)
    assert np.all((cdf > 0) & (cdf <= 1))
    q = zipo.quantile([0.3, 0.5, 0.7], params)
    assert np.all(q >= 0)
    assert zipo.simulate(params, rng=np.random.default_rng(0)).shape == (3,)
    dev = zipo.deviance_increment(y, params)
    np.testing.assert_allclose(dev, -2.0 * np.log(pmf))

    # --- ZINBLSS ---
    y = np.array([0.0, 2.0, 5.0])
    params = {
        "mu": np.array([2.0, 3.0, 4.0]),
        "sigma": np.array([1.0, 2.0, 3.0]),
        "nu": np.array([0.1, 0.3, 0.5]),
    }
    zinb = ZINBLSS()
    pmf = zinb.pmf(y, params)
    assert np.all(pmf > 0) and np.all(pmf < 1)
    cdf = zinb.cdf(y, params)
    assert np.all((cdf > 0) & (cdf <= 1))
    assert zinb.simulate(params, rng=np.random.default_rng(0)).shape == (3,)
    dev = zinb.deviance_increment(y, params)
    np.testing.assert_allclose(dev, -2.0 * np.log(pmf))


def test_studentt_and_dirichlet_distribution_functions():
    """Verify pdf/simulate/deviance for continuous second-batch families."""

    # --- StudentTLSS ---
    y = np.array([0.0, 1.0, -1.0])
    params = {"mu": np.array([0.0, 0.5, -0.5]), "sigma": np.array([1.0, 1.5, 0.8]), "df": np.array([3.0, 5.0, 10.0])}
    st = StudentTLSS()
    dens = st.pdf(y, params)
    assert np.all(dens > 0)
    assert st.simulate(params, rng=np.random.default_rng(0)).shape == (3,)
    dev = st.deviance_increment(y, params)
    np.testing.assert_allclose(dev, -2.0 * np.log(dens))

    # --- DirichletLSS (K=3) ---
    dl = mb.DirichletLSS(K=3)
    y = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    params = {"a1": np.array([2.0, 3.0]), "a2": np.array([3.0, 2.0]), "a3": np.array([5.0, 1.0])}
    dens = dl.pdf(y, params)
    assert np.all(dens >= 0)
    samples = dl.simulate(params, rng=np.random.default_rng(0), size=2)
    assert samples.shape == (2, 3)
    assert np.allclose(samples.sum(axis=-1), 1.0)


def test_negativebinomiallss_risk_includes_log_y_factorial():
    """Regression for enhancement-hunt 21.20: `NegativeBinomialLSS::risk()`
    must include the ``Σ log(y!)`` term so the cross-family GAIC / logLik /
    scan_lss_families comparisons are on the same scale as the other LSS
    count families. The Rust family uses the gamlss.dist ``NBI()`` parameterization
    (variance = mu + sigma * mu**2, sigma is the dispersion), so the recomputed
    negative log-likelihood uses size = 1/sigma.
    """
    from scipy.special import gammaln as sgammaln

    rng = np.random.default_rng(42)
    y = rng.negative_binomial(2, 0.3, size=300).astype(float)
    x = rng.normal(size=300)
    d = {"y": y, "x": x}
    fit = mb.gamboostLSS(
        "y ~ bbs(x, df=4)",
        data=d,
        families=NegativeBinomialLSS(),
        control=mb.boost_control(50, 0.1),
    )
    risk = float(np.asarray(fit.risk)[-1])

    mu = np.asarray(fit.fitted(parameter="mu", type="response"))
    sig = np.asarray(fit.fitted(parameter="sigma", type="response"))
    t = sig * mu
    # NBI negative log-likelihood, matching crates/mboostrs-core/src/lss.rs
    # `NegativeBinomialLSS::risk`:
    #   NLL_i = -y*log(t) + (y + 1/sig)*log(1 + t)
    #           - lgamma(y + 1/sig) + lgamma(1/sig) + lgamma(y + 1)
    #   (where t = sig*mu, size = 1/sig)
    nll = (
        -y * np.log(t)
        + (y + 1.0 / sig) * np.log(1.0 + t)
        - sgammaln(y + 1.0 / sig)
        + sgammaln(1.0 / sig)
        + sgammaln(y + 1.0)
    )
    expected = float(nll.sum())

    # Without the lgamma(y+1) term the risk would be ~451 smaller; the
    # tolerance is tight enough that the regression would fail loudly if the
    # term were dropped again.
    np.testing.assert_allclose(risk, expected, rtol=1e-10, atol=1e-8)

    # And confirm the dropped-term version (the pre-fix behaviour) disagrees
    # by the documented amount, so this test fails if risk ever regresses.
    dropped = expected - float(np.sum(sgammaln(y + 1.0)))
    assert abs(risk - dropped) > 1.0  # Σ lgamma(y+1) ≈ 1752 on this seed


def test_noncyclic_cvrisk_tunes_the_total_step_count():
    data = _data(n=120)
    fit = mb.glmboostLSS("y ~ x1 + x2", data=data, families=GaussianLSS(),
                         control=mb.boost_control(80, 0.1), method="noncyclic")
    folds = _folds(120)
    result = fit.cvrisk(folds=folds, grid=[10, 40, 80])

    assert result.grid == [[10], [40], [80]]
    assert result.risk_matrix.shape == (3, 3)
    # Each entry is the held-out Gaussian NLL of that fold's fit, read at
    # the grid total.
    fold_fit = fit.update(mstop=80, weights=folds[:, 0])
    held = folds[:, 0] == 0
    mu = fold_fit.predict(newdata=data, parameter="mu", type="response", mstop=40)
    sigma = fold_fit.predict(newdata=data, parameter="sigma", type="response",
                             mstop=40)
    z = (data["y"] - mu) / sigma
    nll = 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * z**2
    assert result.risk_matrix[0, 1] == pytest.approx(nll[held].mean(), rel=1e-10)

    best = result.optimal_mstop
    assert len(best) == 1
    refit = fit.with_mstop(best)
    assert sum(refit.mstop.values()) == best[0]


def test_noncyclic_cvrisk_rejects_per_parameter_grid_points():
    data = _data(n=60)
    fit = mb.glmboostLSS("y ~ x1 + x2", data=data, families=GaussianLSS(),
                         control=mb.boost_control(40, 0.1), method="noncyclic")
    with pytest.raises(ValueError, match="total step count"):
        fit.cvrisk(folds=_folds(60), grid=[[10, 10]])


def _factor_data(n=600, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    g = rng.choice(["a", "b", "c"], n)
    effect = {"a": 0.0, "b": 1.0, "c": -1.0}
    scale = {"a": 0.5, "b": 1.0, "c": 2.0}
    y = (x1 + np.array([effect[v] for v in g])
         + rng.normal(size=n) * np.array([scale[v] for v in g]))
    return {"y": y, "x1": x1, "g": g}


def test_glmboostlss_expands_a_string_factor_into_contrasts():
    data = _factor_data()
    fit = mb.glmboostLSS("y ~ x1 + g", data=data, families=GaussianLSS(),
                         control=mb.boost_control(500, 0.1))
    coef = fit.coef()
    assert set(coef["mu"]) == {"(Intercept)", "x1", "gb", "gc"}
    new = {"x1": np.zeros(3), "g": np.array(["a", "b", "c"])}
    sigma = fit.predict(newdata=new, parameter="sigma", type="response")
    assert sigma[0] < sigma[1] < sigma[2]


@pytest.mark.parametrize("term", ["bols(g)", "brandom(g)"])
def test_gamboostlss_encodes_a_string_factor(term):
    data = _factor_data()
    fit = mb.gamboostLSS(f"y ~ bbs(x1) + {term}", data=data,
                         families=GaussianLSS(),
                         control=mb.boost_control(300, 0.1))
    new = {"x1": np.zeros(3), "g": np.array(["a", "b", "c"])}
    mu = fit.predict(newdata=new, parameter="mu", type="response")
    assert mu[1] > mu[0] > mu[2]


def test_lss_fits_a_factor_by():
    # Issue #176: lowered per parameter as gamboost lowers it, so prediction on
    # the training rows reproduces the fit.
    data = _factor_data()
    fit = mb.gamboostLSS("y ~ bbs(x1, by=g, df=6)", data=data, families=GaussianLSS())
    for name in fit.param_names:
        np.testing.assert_allclose(fit.predict(newdata=data, type="link")[name],
                                   fit.predict(type="link")[name], atol=1e-10)
