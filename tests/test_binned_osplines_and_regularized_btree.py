"""Tests for Milestone 2 (C1 quantile-knot binned bbs + auto-binning, C3 regularized btree, C7-Rust multi-quantile CV+fit)."""

from __future__ import annotations

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust
from mboostrs.gamlss import learner_spec


# ─── C1: Quantile-knot binned bbs & non-Gaussian auto-binning ───────────────


def test_c1_bbs_quantile_binned_matches_unbinned_and_supports_binomial_and_lss():
    rng = np.random.default_rng(123)
    n = 1500
    x = np.exp(rng.normal(0.0, 1.0, n))
    z = rng.normal(size=n)
    eta = 0.4 * np.sin(x) + 0.3 * z
    prob = 1.0 / (1.0 + np.exp(-eta))
    y_bin = rng.binomial(1, prob).astype(float)
    data_bin = {"y": y_bin, "x": x, "z": z}

    m_exact = mb.gamboost(
        "y ~ bbs(x, knots=15, df=4, knot_placement='quantile') + bbs(z, knots=15, df=4, knot_placement='quantile')",
        data=data_bin,
        family=mb.Binomial(),
        control=mb.boost_control(60, 0.2),
    )
    m_binned = mb.gamboost(
        "y ~ bbs(x, knots=15, df=4, knot_placement='quantile', bins=256) + bbs(z, knots=15, df=4, knot_placement='quantile', bins=256)",
        data=data_bin,
        family=mb.Binomial(),
        control=mb.boost_control(60, 0.2),
    )
    report = {r["learner"]: r for r in m_binned.binning_report(output="records")}
    assert report["bbs(x)"]["binned"] is True
    assert report["bbs(x)"]["effective_bins"] == 256
    assert report["bbs(z)"]["binned"] is True

    p_exact = np.asarray(m_exact.predict(data_bin, type="response"), dtype=float)
    p_binned = np.asarray(m_binned.predict(data_bin, type="response"), dtype=float)
    assert float(np.mean(np.abs(p_exact - p_binned))) < 0.01
    assert float(np.corrcoef(p_exact, p_binned)[0, 1]) > 0.995

    # Exact match when bins >= n
    m_full_bins = mb.gamboost(
        f"y ~ bbs(x, knots=15, df=4, knot_placement='quantile', bins={n}) + bbs(z, knots=15, df=4, knot_placement='quantile', bins={n})",
        data=data_bin,
        family=mb.Binomial(),
        control=mb.boost_control(60, 0.2),
    )
    p_full_bins = np.asarray(m_full_bins.predict(data_bin, type="response"), dtype=float)
    assert float(np.max(np.abs(p_exact - p_full_bins))) < 1e-8

    # Centered quantile O-spline binned + cvrisk works end-to-end
    m_cent = mb.gamboost(
        "y ~ bols(x) + bbs(x, knots=15, df=3, center=True, knot_placement='quantile', bins=256)",
        data=data_bin,
        family=mb.Binomial(),
        control=mb.boost_control(40, 0.2),
    )
    cv = m_cent.cvrisk(folds=3, seed=7)
    assert cv.optimal_mstop >= 1
    assert np.all(np.isfinite(cv.aggregate))
    assert float(cv.aggregate[cv.optimal_mstop]) < float(cv.aggregate[0])

    # Works inside GaussianLSS too
    y_cont = np.sin(x) + rng.normal(0.0, 0.3 + 0.1 * np.abs(z))
    m_lss = mb.gamboostLSS(
        {
            "mu": "y ~ bbs(x, knots=12, df=4, knot_placement='quantile', bins=256)",
            "sigma": "y ~ bbs(z, knots=12, df=3, knot_placement='quantile', bins=256)",
        },
        data={"y": y_cont, "x": x, "z": z},
        family=mb.GaussianLSS(),
        control=mb.boost_control(30, 0.1),
    )
    preds = m_lss.predict({"x": x[:20], "z": z[:20]}, type="response")
    assert np.all(np.isfinite(preds["mu"]))
    assert np.all(np.isfinite(preds["sigma"])) and np.all(preds["sigma"] > 0.0)


def test_c1_fast_autospec_non_gaussian_uses_bins_and_completes_cleanly():
    rng = np.random.default_rng(321)
    n = 900
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    eta = 1.1 * np.sin(x1) + 0.8 * x2 - 0.5 * x3
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3}

    rec = mb.autospec(data, "y", family=mb.Binomial(), budget=40, search="fast", seed=0)
    assert rec.mstop >= 1
    assert isinstance(rec.family, mb.Binomial)


def test_c4_fast_autospec_binary_separable_tree_adopts_line_search():
    rng = np.random.default_rng(424)
    n = 900
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    margin = 2.0 * (x1 * x2 > 0).astype(float) - 1.0 + 0.6 * np.sin(x3)
    y = (margin > 0).astype(float)
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3}

    rec = mb.autospec(
        data,
        "y",
        family=mb.Binomial(),
        budget=64,
        search="fast",
        residual_tree=True,
        seed=0,
    )
    assert "btree(" in rec.formula
    assert rec.control.step == "line_search"
    model = rec.fit(data)
    probs = np.asarray(model.predict(data, type="response"), dtype=float)
    acc = float(np.mean((probs >= 0.5) == (y == 1.0)))
    assert acc > 0.90


# ─── C3: Regularized Btree (lambda_leaf, deterministic colsample) ────────────


def test_c3_btree_formula_parsing_determinism_and_leaf_shrinkage():
    # Validation & repr/eq on Btree
    with pytest.raises(ValueError, match="lambda_leaf"):
        mb.Btree(["x1", "x2"], lambda_leaf=-1.0)
    with pytest.raises(ValueError, match="colsample"):
        mb.Btree(["x1", "x2"], colsample=0.0)
    with pytest.raises(ValueError, match="colsample"):
        mb.Btree(["x1", "x2"], colsample=1.5)

    t1 = mb.Btree(["x1", "x2"], maxdepth=4, split_method="histogram", lambda_leaf=1.5, colsample=0.5, colsample_seed=42)
    t2 = mb.Btree(["x1", "x2"], maxdepth=4, split_method="histogram", lambda_leaf=1.5, colsample=0.5, colsample_seed=42)
    assert t1 == t2
    assert "lambda_leaf=1.5" in repr(t1)
    assert "colsample=0.5" in repr(t1)
    assert "colsample_seed=42" in repr(t1)

    rng = np.random.default_rng(555)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    x4 = rng.normal(size=n)
    y = 2.0 * (x1 * x2 > 0).astype(float) + rng.normal(0.0, 0.8, size=n)
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4}

    f_reg = "y ~ btree(x1, x2, x3, x4, maxdepth=4, split_method='histogram', lambda_leaf=1.5, colsample=0.5, colsample_seed=42)"
    m1 = mb.gamboost(f_reg, data=data, control=mb.boost_control(30, 0.1))
    m2 = mb.gamboost(f_reg, data=data, control=mb.boost_control(30, 0.1))

    p1 = np.asarray(m1.predict(data), dtype=float)
    p2 = np.asarray(m2.predict(data), dtype=float)
    assert np.array_equal(p1, p2)

    # Higher lambda_leaf shrinks leaf predictions and fitted excursion toward 0
    f_unreg = "y ~ btree(x1, x2, x3, x4, maxdepth=4, split_method='histogram', lambda_leaf=0.0)"
    f_heavy = "y ~ btree(x1, x2, x3, x4, maxdepth=4, split_method='histogram', lambda_leaf=100.0)"
    m_unreg = mb.gamboost(f_unreg, data=data, control=mb.boost_control(5, 0.1))
    m_heavy = mb.gamboost(f_heavy, data=data, control=mb.boost_control(5, 0.1))
    exc_unreg = float(np.std(np.asarray(m_unreg.fitted(), dtype=float) - m_unreg.offset))
    exc_heavy = float(np.std(np.asarray(m_heavy.fitted(), dtype=float) - m_heavy.offset))
    assert exc_heavy < exc_unreg * 0.65

    # Exact split_method also supports lambda_leaf and colsample deterministically
    f_exact = "y ~ btree(x1, x2, x3, x4, maxdepth=3, split_method='exact', lambda_leaf=2.0, colsample=0.5, colsample_seed=7)"
    me1 = mb.gamboost(f_exact, data=data, control=mb.boost_control(15, 0.1))
    me2 = mb.gamboost(f_exact, data=data, control=mb.boost_control(15, 0.1))
    assert np.array_equal(np.asarray(me1.predict(data)), np.asarray(me2.predict(data)))


# ─── C7-Rust: Rayon-parallel shared-blueprint multi-quantile CV & fit ────────


def test_c7_cv_and_fit_multi_quantile_specs_9_quantiles():
    rng = np.random.default_rng(777)
    n = 600
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * np.sin(x1) + 0.8 * x2 + (0.4 + 0.3 * np.abs(x1)) * rng.normal(size=n)
    data = {"y": y, "x1": x1, "x2": x2}

    learners = [mb.Bbs("x1", df=4, knots=12), mb.Bbs("x2", df=4, knots=12)]
    specs = [learner_spec(lr, data, bols_form="centered") for lr in learners]
    w = np.ones(n, dtype=np.float64)
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    folds = np.ascontiguousarray(mb.cv_weights(n, type="kfold", B=5, seed=0), dtype=np.float64)

    res = _rust.cv_and_fit_multi_quantile_specs(
        specs,
        np.ascontiguousarray(y, dtype=np.float64),
        w,
        taus,
        folds=folds,
        max_mstop=150,
        nu=0.3,
        response_scale=0.0,
    )
    assert len(res) == len(taus)

    median_fitted = None
    fitted_means = []
    for q_idx, item in enumerate(res):
        assert item["tau"] == pytest.approx(taus[q_idx])
        assert 1 <= item["opt_mstop"] <= 150
        cv_risk = np.asarray(item["cv_risk"], dtype=float)
        assert cv_risk.shape == (151,)
        assert np.all(np.isfinite(cv_risk))
        assert cv_risk[item["opt_mstop"]] <= cv_risk[0]

        fit_tuple = item["fit"]
        assert len(fit_tuple) == 10
        offset_val, selected, coefs_vecs, risk, fitted, nuisance_vec, nuisance_vec_path, store, report, residual = fit_tuple
        assert len(selected) == item["opt_mstop"]
        assert len(coefs_vecs) == item["opt_mstop"]
        fitted_arr = np.asarray(fitted, dtype=float)
        assert fitted_arr.shape == (n,)
        assert np.all(np.isfinite(fitted_arr))
        fitted_means.append(float(np.mean(fitted_arr)))
        if taus[q_idx] == 0.5:
            median_fitted = fitted_arr

    # Higher tau should yield higher average quantile prediction
    assert all(fitted_means[i] < fitted_means[i + 1] for i in range(len(fitted_means) - 1))

    # Matches single-quantile gamboost at same opt_mstop
    opt_50 = res[4]["opt_mstop"]
    single_50 = mb.gamboost(
        "y ~ bbs(x1, df=4, knots=12) + bbs(x2, df=4, knots=12)",
        data=data,
        family=mb.Quantile(0.5),
        control=mb.boost_control(opt_50, 0.3),
    )
    assert np.allclose(median_fitted, np.asarray(single_50.fitted(), dtype=float), atol=1e-9)

    # Also test folds=None path
    res_no_cv = _rust.cv_and_fit_multi_quantile_specs(
        specs,
        np.ascontiguousarray(y, dtype=np.float64),
        w,
        [0.25, 0.75],
        folds=None,
        max_mstop=40,
        nu=0.3,
        response_scale=0.0,
    )
    assert len(res_no_cv) == 2
    assert res_no_cv[0]["opt_mstop"] == 40
    assert res_no_cv[0]["cv_risk"] == []
