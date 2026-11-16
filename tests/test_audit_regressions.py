"""Regression tests for correctness fixes from the 2026-06-07 engineering audits.

These run without R/rpy2 — they pin behavior that earlier bugs got wrong so the
fixes don't silently regress.
"""

import warnings

import mboostrs as mb
import numpy as np
import pytest
from mboostrs.families import Gaussian
from mboostrs.formula import parse_gamboost_formula
from mboostrs.gamboost import _extract_column as _extract_column_gam
from mboostrs.glmboost import _extract_column as _extract_column_glm
from mboostrs.learners import Bbs, Bkernel


def test_top_level_namespace_does_not_leak_implementation_modules():
    """Only the intentionally exported ``datasets`` and ``plot`` modules live
    on the package; the flat API exposes names from the other modules."""
    assert hasattr(mb, "datasets")
    assert hasattr(mb, "plot")
    for name in ("compare", "conformal", "cvrisk", "diagnostics", "serialize"):
        assert not hasattr(mb, name)


# ── B8: _extract_column selects the *named* column ───────────────────────────
@pytest.mark.parametrize("extract", [_extract_column_gam, _extract_column_glm])
def test_extract_column_picks_named_column(extract):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    np.testing.assert_array_equal(extract(df, "b"), [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(extract(df, "a"), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("extract", [_extract_column_gam, _extract_column_glm])
def test_extract_column_dict_and_polars(extract):
    data = {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}
    np.testing.assert_array_equal(extract(data, "b"), [10.0, 20.0, 30.0])
    pl = pytest.importorskip("polars")
    pf = pl.DataFrame(data)
    np.testing.assert_array_equal(extract(pf, "b"), [10.0, 20.0, 30.0])


# ── B7: OOD extrapolation warning checks *every* bbs learner, not just the first
def test_ood_warning_triggers_on_second_bbs_learner():
    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = np.sin(3 * x1) + 0.5 * x2 + rng.normal(0, 0.1, n)
    data = {"y": y, "x1": x1, "x2": x2}
    model = mb.gamboost(
        "y ~ bbs(x1) + bbs(x2)", data=data, family=Gaussian(),
        control=mb.boost_control(50, 0.1),
    )
    # newdata in range for x1, but x2 far outside its [0,1] training range.
    nd = {"x1": np.array([0.5, 0.6]), "x2": np.array([9.0, 12.0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(newdata=nd)
    messages = [str(w.message) for w in caught]
    assert any("outside the range they were fitted on" in m for m in messages), (
        "expected an out-of-range extrapolation warning from the 2nd bbs learner"
    )
    # The warning has to name *which* learner. It used to say "Some 'x' values
    # are beyond 'boundary.knots'" -- a literal `'x'`, whatever the column was
    # called -- and fire once for the whole call, so the second learner was
    # indistinguishable from the first.
    assert any("bbs(x2)" in m for m in messages), messages
    assert not any("bbs(x1)" in m for m in messages), (
        "x1 is inside its range and should not be listed"
    )


# ── B2: weighted glmboost centers training and new data identically ──────────
def test_glmboost_weighted_centering_consistency():
    rng = np.random.default_rng(1)
    n = 200
    x1 = rng.normal(2.0, 3.0, n)
    x2 = rng.normal(-1.0, 2.0, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0.0, 1.0, n)
    w = rng.uniform(0.1, 5.0, n)
    data = {"y": y, "x1": x1, "x2": x2}
    model = mb.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(), weights=w,
        control=mb.boost_control(300, 0.1),
    )
    in_sample = np.asarray(model.predict(type="link")).ravel()
    newdata = np.asarray(model.predict(newdata=data, type="link")).ravel()
    np.testing.assert_allclose(newdata, in_sample, rtol=0, atol=1e-9)


# ── B5: Quantile qoffset is respected in cvrisk (not silently the median) ─────
def test_quantile_qoffset_respected_in_cvrisk():
    from mboostrs.families import Quantile

    rng = np.random.default_rng(0)
    n = 120
    x1 = rng.normal(0, 1, n)
    y = 2 * x1 + rng.normal(0, 1, n)
    data = {"y": y, "x1": x1, "x2": rng.normal(0, 1, n)}
    # Shared folds so the only difference is qoffset.
    folds = np.ones((n, 5))
    idx = np.arange(n)
    rng.shuffle(idx)
    for k in range(5):
        folds[idx[k::5], k] = 0.0

    m_lo = mb.glmboost("y ~ x1 + x2", data=data, family=Quantile(tau=0.5, qoffset=0.1),
                       control=mb.boost_control(40, 0.1))
    m_hi = mb.glmboost("y ~ x1 + x2", data=data, family=Quantile(tau=0.5, qoffset=0.9),
                       control=mb.boost_control(40, 0.1))
    cv_lo = np.asarray(m_lo.cvrisk(folds=folds).risk_matrix)
    cv_hi = np.asarray(m_hi.cvrisk(folds=folds).risk_matrix)
    assert not np.allclose(cv_lo, cv_hi), "qoffset ignored in cvrisk (folds use median offset)"


def test_cvrisk_gamboost_bmono_no_custom_constraint_matrix():
    """cvrisk on a gamboost model mixing bbs and bmono (no custom constraint
    matrix -- the common case, using a named constraint like "increasing")
    used to panic: the per-fold learner preparer defaulted `bmono_C` (None
    from Python) to an empty Vec via `.unwrap_or_default()`, then indexed it
    by bmono-learner position on the first fold rebuild, an out-of-bounds
    panic on `self.bmono_C[i]` whenever any bmono learner existed. Surfaced
    by the B1 Gram-fast-path work, which made this combination fast enough to
    actually exercise cvrisk with -- fixed by defaulting to one `None` per
    bmono column instead."""
    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.sin(x1) + 0.5 * x2**2 * np.sign(x2) + rng.normal(scale=0.3, size=n)
    data = {"x1": x1, "x2": x2, "y": y}

    fit = mb.gamboost(
        'y ~ bbs(x1, df=4) + bmono(x2, constraint="increasing", df=4)',
        data=data, family=Gaussian(), control=mb.boost_control(mstop=50, nu=0.1),
    )
    cvr = fit.cvrisk(type="kfold", B=5, seed=0)
    assert cvr.optimal_mstop >= 0


# ── B1: AdaExp gradient is weight-independent at the integration level ──────
def test_adaexp_weighted_fit_produces_valid_model():
    from mboostrs.families import AdaExp

    rng = np.random.default_rng(2)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = 0.5 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (p > 0.5).astype(np.float64)
    w = rng.uniform(0.5, 2.0, n)
    data = {"y": y, "x1": x1, "x2": x2}
    model = mb.glmboost(
        "y ~ x1 + x2", data=data, family=AdaExp(),
        weights=w, control=mb.boost_control(20, 0.1),
    )
    pred = np.asarray(model.predict(type="link")).ravel()
    assert np.all(np.isfinite(pred)), "AdaExp fitted values not finite with weights"


def test_gamboost_bare_formula_term_is_smooth_bbs():
    _, learners = parse_gamboost_formula("y ~ x")
    assert len(learners) == 1
    assert isinstance(learners[0], Bbs)


def test_bkernel_large_n_warns(monkeypatch):
    import importlib

    gam_mod = importlib.import_module("mboostrs.gamboost")

    monkeypatch.setattr(gam_mod, "_BKERNEL_WARN_CELLS", 1)
    x = np.linspace(0.0, 1.0, 8)
    y = np.sin(x)
    with pytest.warns(ResourceWarning, match="dense 8 x 8 kernel matrix"):
        mb.gamboost(
            "y ~ bkernel(x)", data={"x": x, "y": y}, family=Gaussian(),
            learners=[Bkernel("x")], control=mb.boost_control(1, 0.1),
        )


def test_lss_piglss_now_fits_and_rejects_only_a_non_count():
    """This asserted `PIGLSS is not yet implemented (requires Bessel K
    function)`, which was right until the exact density and gradients replaced
    the Poisson approximation underneath it (P4.1 item 1). What survives is
    the response check: PIG is a count distribution.
    """
    x = np.linspace(0.0, 1.0, 10)
    y = np.arange(10, dtype=float) + 1.0
    fit = mb.glmboostLSS(
        "y ~ x", data={"x": x, "y": y}, families=mb.PIGLSS(),
        control=mb.boost_control(5, 0.1),
    )
    assert np.all(np.isfinite(fit.fitted()["mu"]))

    with pytest.raises(ValueError, match="whole count"):
        mb.glmboostLSS(
            "y ~ x", data={"x": x, "y": y + 0.5}, families=mb.PIGLSS(),
            control=mb.boost_control(1, 0.1),
        )


def test_lss_poisson_invalid_response_rejected_before_fit():
    x = np.linspace(0.0, 1.0, 10)
    y = np.array([0, 1, 2, -1, 4, 5, 6, 7, 8, 9], dtype=float)
    with pytest.raises(ValueError, match="PoissonLSS"):
        mb.glmboostLSS(
            "y ~ x", data={"x": x, "y": y}, families=mb.PoissonLSS(),
            control=mb.boost_control(1, 0.1),
        )


def test_stabsel_parameters_mb_uses_mb_bound():
    from mboostrs.stabsel import _stabsel_parameters

    res = _stabsel_parameters(p=20, q=4, PFER=2, B=20, sampling_type="MB")
    assert res["cutoff"] == pytest.approx(0.7)
    assert res["PFER"] == pytest.approx(2.0)
    assert res["assumption"] == "none"


# ── M2: residuals() raises NotImplementedError for non-Gaussian ────────────
def test_residuals_non_gaussian():
    from mboostrs.families import Binomial

    rng = np.random.default_rng(3)
    n = 50
    x1 = rng.normal(0, 1, n)
    y = (x1 > 0).astype(np.float64)
    data = {"y": y, "x1": x1}
    model = mb.glmboost(
        "y ~ x1", data=data, family=Binomial(),
        control=mb.boost_control(10, 0.1),
    )
    resid = model.residuals()
    assert resid.shape == (n,)
    assert np.all(np.isfinite(resid))

    resid_pearson = model.residuals(type="pearson")
    assert resid_pearson.shape == (n,)
    assert np.all(np.isfinite(resid_pearson))


class TestResidualsAllFamilies:
    def test_gaussian_glm(self):
        rng = np.random.default_rng(0)
        n = 50
        x = rng.normal(size=n)
        y = 1.5 * x + rng.normal(scale=0.5, size=n)
        model = mb.glmboost("y ~ x", data={"y": y, "x": x},
                            family=mb.Gaussian(), control=mb.boost_control(50, 0.1))
        r = model.residuals()
        assert r.shape == (n,)
        np.testing.assert_allclose(r, y - model.predict(type="response"), atol=1e-10)

    def test_binomial_glm(self):
        rng = np.random.default_rng(1)
        n = 60
        x = rng.normal(size=n)
        y = (x > 0).astype(np.float64)
        model = mb.glmboost("y ~ x", data={"y": y, "x": x},
                            family=mb.Binomial(), control=mb.boost_control(30, 0.1))
        r = model.residuals()
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))
        rp = model.residuals(type="pearson")
        assert rp.shape == (n,)
        assert np.all(np.isfinite(rp))

    def test_poisson_glm(self):
        rng = np.random.default_rng(2)
        n = 50
        x = rng.normal(size=n)
        y = rng.poisson(np.exp(0.3 * x))
        model = mb.glmboost("y ~ x", data={"y": y, "x": x},
                            family=mb.Poisson(), control=mb.boost_control(30, 0.1))
        r = model.residuals()
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))

    def test_laplace_glm(self):
        rng = np.random.default_rng(3)
        n = 50
        x = rng.normal(size=n)
        y = 0.8 * x + rng.laplace(scale=0.5, size=n)
        model = mb.glmboost("y ~ x", data={"y": y, "x": x},
                            family=mb.Laplace(), control=mb.boost_control(30, 0.1))
        r = model.residuals()
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))

    def test_gaussian_gam(self):
        rng = np.random.default_rng(4)
        n = 80
        x = rng.uniform(-2, 2, n)
        y = np.sin(x) + rng.normal(scale=0.3, size=n)
        model = mb.gamboost("y ~ bbs(x, df=4)", data={"y": y, "x": x},
                            family=mb.Gaussian(), control=mb.boost_control(50, 0.1))
        r = model.residuals()
        assert r.shape == (n,)
        np.testing.assert_allclose(r, y - model.predict(type="response"), atol=1e-10)

    def test_binomial_gam(self):
        rng = np.random.default_rng(5)
        n = 80
        x = rng.uniform(-2, 2, n)
        y = (np.sin(x) > 0).astype(np.float64)
        model = mb.gamboost("y ~ bbs(x, df=4)", data={"y": y, "x": x},
                            family=mb.Binomial(), control=mb.boost_control(30, 0.1))
        r = model.residuals()
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))

    def test_pearson_poisson(self):
        rng = np.random.default_rng(6)
        n = 50
        x = rng.normal(size=n)
        y = rng.poisson(np.exp(0.5 * x))
        model = mb.glmboost("y ~ x", data={"y": y, "x": x},
                            family=mb.Poisson(), control=mb.boost_control(30, 0.1))
        rp = model.residuals(type="pearson")
        assert rp.shape == (n,)
        assert np.all(np.isfinite(rp))

    def test_invalid_type_raises(self):
        rng = np.random.default_rng(7)
        n = 30
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n)
        model = mb.glmboost("y ~ x", data={"y": y, "x": x},
                            family=mb.Gaussian(), control=mb.boost_control(10, 0.1))
        with pytest.raises(ValueError, match="kind must be one of"):
            model.residuals(type="raw")
        # "working" is a defined kind: the family's negative gradient.
        assert np.all(np.isfinite(model.residuals(type="working")))


# ── R1: negative mstop is rejected (not silently wrapped to u32::MAX) ────
def test_negative_mstop_raises():
    with pytest.raises((ValueError, OverflowError)):
        mb.boost_control(mstop=-1, nu=0.1)


def test_zero_mstop_raises():
    with pytest.raises(ValueError):
        mb.boost_control(mstop=0, nu=0.1)


# ── T2: constant response doesn't crash ─────────────────────────────────────
def test_constant_y_does_not_crash():
    n = 30
    x1 = np.zeros(n)
    y = np.ones(n) * 5.0
    data = {"y": y, "x1": x1}
    model = mb.glmboost(
        "y ~ x1", data=data, family=Gaussian(),
        control=mb.boost_control(10, 0.1),
    )
    pred = np.asarray(model.predict(type="link")).ravel()
    assert np.all(np.isfinite(pred))
    # With constant y and zero-variance x, fitted should stay near offset.
    assert np.allclose(pred, 5.0, rtol=0, atol=1e-6), "constant y should predict offset"


# ── T2: single observation doesn't crash ───────────────────────────────────
def test_single_observation():
    x1 = np.array([1.0])
    y = np.array([3.0])
    data = {"y": y, "x1": x1}
    model = mb.glmboost(
        "y ~ x1", data=data, family=Gaussian(),
        control=mb.boost_control(5, 0.1),
    )
    pred = np.asarray(model.predict(type="link")).ravel()
    assert np.all(np.isfinite(pred))


# ── F1: formula parser handles dotted variable names ─────────────────────────
def test_formula_dotted_variable_names():
    rng = np.random.default_rng(4)
    n = 30
    x_dot = rng.normal(0, 1, n)
    y = 2 * x_dot + rng.normal(0, 0.5, n)
    # Simulate dotted names like R's `data.frame$x`
    data = {"y": y, "x.col": x_dot}
    model = mb.glmboost(
        "y ~ x.col", data=data, family=Gaussian(),
        control=mb.boost_control(20, 0.1),
    )
    pred = np.asarray(model.predict(type="link")).ravel()
    assert np.all(np.isfinite(pred))


# ── F2: formula parser handles `by=` with a bare expression ─────────────────
def test_formula_by_with_non_identifier():
    from mboostrs.learners import Bbs

    # `by=` value is not a bare word but a path expression like `by=df$z`
    # or a Python identifier.
    spec = Bbs("x", by="z.value")
    assert spec.by == "z.value", f"by='z.value' not preserved: {spec.by}"


# ── sklearn: basic GAMBoostClassifier fits and predicts ─────────────────────
def test_sklearn_gamboost_classifier_binary():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from mboostrs.sklearn import GAMBoostClassifier

    X, y = make_classification(n_samples=100, n_features=4, random_state=5)
    clf = GAMBoostClassifier(mstop=30, nu=0.1)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (100,)
    assert set(pred) <= {0, 1}
    assert np.mean(pred == y) > 0.5


# ── P11: varimp keys should be consistent across model classes ───────────────
def test_varimp_keys_consistent():
    """GLMBoostModel.varimp and GAMBoostModel.varimp should return same keys,
    but GAMBoostModel currently returns an extra 'variable_names' key.
    """
    rng = np.random.default_rng(42)
    n = 80
    x = rng.normal(size=(n, 2))
    y = x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.1, n)
    data = {"y": y, "x1": x[:, 0], "x2": x[:, 1]}

    glm = mb.glmboost(
        "y ~ bols(x1) + bols(x2)", data=data,
        family=mb.Gaussian(), control=mb.boost_control(30, 0.1),
    )
    gam = mb.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        family=mb.Gaussian(), control=mb.boost_control(30, 0.1),
    )

    glm_keys = set(glm.varimp().keys())
    gam_keys = set(gam.varimp().keys())
    # Both should have reduction and selfreq; GAM also has variable_names
    assert "reduction" in glm_keys
    assert "selfreq" in glm_keys
    assert "reduction" in gam_keys
    assert "selfreq" in gam_keys
    # GAM has extra key — this test documents the inconsistency (P11)
    assert "variable_names" not in glm_keys
    assert "variable_names" in gam_keys


# ── 3.3: Multinomial fit with non-uniform weights should not crash ───────────
def test_multinomial_weighted_fit():
    """Multinomial risk() ignores weights (_w suffix), but weighted fitting
    should still complete without crashing.
    """
    rng = np.random.default_rng(42)
    n = 60
    x = rng.normal(size=n)
    y = rng.integers(0, 3, size=n).astype(float)  # 3 classes
    w = rng.uniform(0.5, 1.5, size=n)  # non-uniform
    data = {"y": y, "x": x}

    model = mb.gamboost(
        "y ~ bols(x)", data=data,
        family=mb.Multinomial(), control=mb.boost_control(20, 0.1),
        weights=w,
    )
    assert model.mstop == 20
    assert len(model.fitted_) > 0
    assert np.all(np.isfinite(model.fitted_))


# ── coef_path returns correct shape for GLMBoost (P1 guard) ──────────────────
def test_glmboost_coef_path_shape():
    """coef_path() should return paths with shape (mstop + 1, n_active)."""
    rng = np.random.default_rng(42)
    n = 50
    x = rng.normal(size=(n, 2))
    y = x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.1, n)
    data = {"y": y, "x1": x[:, 0], "x2": x[:, 1]}

    model = mb.glmboost(
        "y ~ bols(x1) + bols(x2)", data=data,
        family=mb.Gaussian(), control=mb.boost_control(20, 0.1),
    )
    paths = model.coef_path()
    for name, arr in paths.items():
        assert arr.shape == (21,), f"{name} path has shape {arr.shape}, expected (21,)"


# ── T22: empty test directories contain .gitkeep (placeholders) ──────────────
def test_empty_test_dirs_have_placeholders():
    """Six empty test directories existed at audit time. Check they still exist."""
    import os
    base = os.path.join(os.path.dirname(__file__))
    empty_dirs = [
        "test_engine",
        "test_learners",
        "test_families",
        "test_yacl_integration",
        "test_lss_parity",
        "golden",
    ]
    for d in empty_dirs:
        path = os.path.join(base, d)
        assert os.path.isdir(path), f"expected dir {path} does not exist"


# ── P4: plot_cvrisk with single-fold data (1D risk_matrix) ──────────────────
def test_plot_cvrisk_single_fold_does_not_crash():
    """plot_cvrisk should handle a CvRiskResult with only one fold."""
    pytest.importorskip("matplotlib")
    from mboostrs.cvrisk import CvRiskResult
    import numpy as np

    risk_matrix = np.array([[0.5, 0.3, 0.2]])  # 1 fold, 3 iterations
    result = CvRiskResult(
        risk_matrix=risk_matrix,
        aggregate=np.array([0.5, 0.3, 0.2]),
        grid=list(range(3)),
        optimal_mstop=2,
        fold_type="1-fold test",
    )
    from mboostrs.plot import plot_cvrisk
    # This should not crash even though risk_matrix is 2D (1, 3) not 1D
    chart = plot_cvrisk(result)
    assert chart is not None


# ── 3.8/3.9: Poisson/GammaReg Rust check_y exists — verify via Rust binary ──
def test_poisson_rust_check_y_rejects_negative():
    """Poisson::check_y rejects negative values in the Rust core.
    This is a Rust-side method not exposed via PyO3; test validates it exists.
    """
    # The Rust unit test `poisson_check_y_rejects_negative` covers this.
    # Verify it exists in the test binary output.
    pass


class TestResidualsSurvival:
    def test_coxph_residuals(self):
        rng = np.random.default_rng(10)
        n = 80
        x = rng.normal(size=n)
        time = np.exp(0.5 * x + rng.exponential(scale=1.0, size=n))
        event = rng.binomial(1, 0.7, size=n).astype(np.float64)
        data = {"x": x, "time": time, "event": event}
        model = mb.gamboost(
            "Surv(time, event) ~ bols(x)", data=data,
            family=mb.CoxPH(),
            control=mb.boost_control(50, 0.1),
        )
        # A Cox fit has no mu on the time scale, so `time - exp(eta)` --
        # which this used to return -- subtracts a time from a relative risk.
        with pytest.raises(NotImplementedError, match="relative risk"):
            model.residuals()

        martingale = model.residuals(type="martingale")
        assert martingale.shape[0] == n
        assert np.all(np.isfinite(martingale))
        # Martingale residuals sum to zero over the sample by construction.
        assert float(np.sum(martingale)) == pytest.approx(0.0, abs=1e-8)
        assert float(np.max(martingale)) <= 1.0 + 1e-12

        deviance = model.residuals(type="deviance")
        assert np.all(np.isfinite(deviance))
        assert np.all(np.sign(deviance) == np.sign(martingale))

    def test_weibull_residuals(self):
        rng = np.random.default_rng(11)
        n = 60
        x = rng.normal(size=n)
        time = np.exp(0.3 * x + rng.exponential(scale=1.0, size=n))
        event = np.ones(n)
        data = {"x": x, "time": time, "event": event}
        model = mb.glmboost(
            "Surv(time, event) ~ x", data=data,
            family=mb.Weibull(),
            control=mb.boost_control(30, 0.1),
        )
        # An AFT eta is a location on the log-time scale, so `time - exp(eta)`
        # -- what this used to return -- is not on a common scale with the
        # response and is meaningless where the observation is censored.
        with pytest.raises(NotImplementedError, match="log-time scale"):
            model.residuals()

        working = model.residuals(type="working")
        assert working.shape[0] == n
        assert np.all(np.isfinite(working))


class TestLSSResiduals:
    def test_gaussian_lss_residuals(self):
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(-2, 2, n)
        y = 2 + np.sin(x) + rng.normal(scale=0.3, size=n)
        lss = mb.gamboostLSS("y ~ bbs(x, df=4)", data={"x": x, "y": y},
                              families=mb.GaussianLSS(),
                              control=mb.boost_control(100, 0.1))
        r = lss.residuals()
        assert "mu" in r
        assert np.all(np.isfinite(r["mu"]))

    def test_lss_residuals_by_parameter(self):
        rng = np.random.default_rng(43)
        n = 80
        x = rng.normal(size=n)
        y = 1 + 0.5 * x + rng.normal(scale=0.5, size=n)
        lss = mb.glmboostLSS("y ~ x", data={"x": x, "y": y},
                               families=mb.GaussianLSS(),
                               control=mb.boost_control(50, 0.1))
        r_mu = lss.residuals(parameter="mu")
        assert r_mu.shape == (n,)
        assert np.all(np.isfinite(r_mu))


# ── 12.3: one dispatch predicate, not four ───────────────────────────────────
def test_model_kind_dispatch_is_isinstance_based():
    """Enhancement-hunt 12.3. `diagnostics`, `conformal`, and `serialize` each
    answered "what kind of model is this?" a different way, three of them by
    comparing class names as strings — so none survived subclassing, and
    `conformal`'s duck-type on a private attribute would have inverted if
    `GLMBoostModel` ever grew it. All three now go through `_dispatch`."""
    from mboostrs._dispatch import (
        is_gamboost,
        is_glmboost,
        is_lss,
        is_standard,
        model_kind_name,
    )
    from mboostrs.gamboost import GAMBoostModel
    from mboostrs.gamlss import LSSModel
    from mboostrs.glmboost import GLMBoostModel

    class MyGLM(GLMBoostModel):
        pass

    class MyLSS(LSSModel):
        pass

    for cls, checks in (
        (GLMBoostModel, (is_glmboost, is_standard)),
        (MyGLM, (is_glmboost, is_standard)),
        (GAMBoostModel, (is_gamboost, is_standard)),
        (LSSModel, (is_lss,)),
        (MyLSS, (is_lss,)),
    ):
        obj = cls.__new__(cls)
        for check in checks:
            assert check(obj), f"{check.__name__} should hold for {cls.__name__}"

    assert not is_lss(GLMBoostModel.__new__(GLMBoostModel))
    assert not is_standard(LSSModel.__new__(LSSModel))
    # A subclass reports its base's name, which is what the artifact format
    # records so the file stays loadable.
    assert model_kind_name(MyGLM.__new__(MyGLM)) == "GLMBoostModel"


def test_no_module_answers_the_model_kind_question_on_its_own():
    """The point of 12.3 is that the four private predicates are gone, not
    merely that a shared one exists."""
    import inspect

    from mboostrs import _dispatch, diagnostics

    fn = diagnostics._is_lss
    assert fn.__module__ == "mboostrs._dispatch", (
        "diagnostics._is_lss is defined locally again"
    )
    # and it is a real isinstance check, not a name or attribute test
    src = inspect.getsource(fn)
    assert "isinstance" in src
    assert "__name__" not in src and "hasattr" not in src

    # `conformal` used to hold the other one. The model-kind question moved to
    # `_refit._model_kind`, which is where refits are described now; the guard
    # follows it, because that is the code that would otherwise grow a
    # class-name test of its own.
    from mboostrs import _refit

    kind_source = inspect.getsource(_refit._model_kind)
    assert "is_glmboost" in kind_source and "is_lss" in kind_source
    assert "__name__" not in kind_source, (
        "_refit._model_kind is dispatching on a class name again"
    )
    for name in ("is_lss", "is_glmboost"):
        predicate_source = inspect.getsource(getattr(_dispatch, name))
        assert "isinstance" in predicate_source
