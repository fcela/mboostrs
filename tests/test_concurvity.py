"""Concurvity as mgcv's concurvity() measures it (issue #118): refit terms
against the rest of the model, and the fitted boosted model's terms through
GAMBoostModel.concurvity (summary and pairwise)."""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2
from mboostrs import _rust


def _data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    return {"x": x, "z": x + rng.normal(0, 0.1, n), "w": rng.uniform(size=n),
            "y": np.sin(3 * x) + rng.uniform(size=n) + rng.normal(0, 0.3, n)}


def test_collinear_terms_have_high_concurvity_and_an_unrelated_one_low():
    data = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x) + bbs(z) + bbs(w)", data=data,
                            control=mb.boost_control(200))
        table = mb.gam_refit(model, which="all").concurvity(output="records")
    by_term = {row["term"]: row for row in table}
    assert by_term["bbs(x)"]["worst"] > 0.8 and by_term["bbs(z)"]["worst"] > 0.8
    assert by_term["bbs(w)"]["worst"] < by_term["bbs(x)"]["worst"]
    for row in table:
        assert 0 <= row["estimate"] <= row["worst"] <= 1 + 1e-12


def test_exact_duplicate_is_fully_concurved():
    rng = np.random.default_rng(1)
    n = 100
    x1 = rng.normal(size=n)
    design = np.column_stack([np.ones(n), x1, x1, rng.normal(size=n)])
    worst, observed, _ = _rust.concurvity_measures(design, [(1, 2), (2, 3), (3, 4)],
                                                   np.ones(4))[0]
    assert worst == pytest.approx(1.0, abs=1e-8)
    assert observed == pytest.approx(1.0, abs=1e-8)


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_matches_mgcv_on_its_own_model_matrix():
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    ro.r("""suppressMessages(library(mgcv)); set.seed(0); n <- 300; x <- runif(n)
    z <- x + rnorm(n, 0, .1); w <- runif(n); y <- sin(3*x) + w + rnorm(n, 0, .3)
    g <- gam(y ~ s(x, k = 10) + s(z, k = 10) + s(w, k = 8) + I(x > 0.5), method = "REML")
    X <- model.matrix(g); b <- coef(g)
    spans <- t(sapply(g$smooth, function(s) c(s$first.para - 1, s$last.para)))
    cc <- concurvity(g, full = TRUE)""")
    with localconverter(ro.default_converter + numpy2ri.converter):
        design = np.asarray(ro.r("X"))
        beta = np.asarray(ro.r("b"))
        spans = np.asarray(ro.r("spans")).astype(int)
        expected = np.asarray(ro.r("cc"))[:, 1:]
    ours = np.array(_rust.concurvity_measures(
        np.ascontiguousarray(design), [tuple(map(int, s)) for s in spans],
        np.ascontiguousarray(beta))).T
    np.testing.assert_allclose(ours, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Boosted models: GAMBoostModel.concurvity
# ---------------------------------------------------------------------------

def _boosted_data(seed=4, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    return {"x": x, "x_copy": x.copy(), "z": x + rng.normal(0, 0.05, n),
            "w": rng.uniform(size=n),
            "y": np.sin(3 * x) + rng.uniform(size=n) + rng.normal(0, 0.3, n)}


def _boosted_model(data, formula="y ~ bbs(x) + bbs(x_copy) + bbs(w) + bbs(z)"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboost(formula, data=data, control=mb.boost_control(120))


def test_boosted_duplicate_terms_are_fully_concurved_and_an_independent_one_less():
    data = _boosted_data()
    model = _boosted_model(data)
    rows = {row["term"]: row for row in model.concurvity(output="records")}
    assert set(rows) == {"bbs(x)", "bbs(x_copy)", "bbs(z)", "bbs(w)"}
    assert rows["bbs(x_copy)"]["worst"] > 0.99
    assert rows["bbs(z)"]["worst"] > 0.9
    # mgcv's "worst" is a pessimistic bound -- rich unrelated bases still
    # share their low-frequency directions -- so compare magnitudes.
    assert rows["bbs(w)"]["worst"] < rows["bbs(z)"]["worst"]
    for row in rows.values():
        assert 0 <= row["estimate"] <= row["worst"] + 1e-12


def test_boosted_pairwise_rows_are_one_per_ordered_pair():
    data = _boosted_data()
    model = _boosted_model(data, formula="y ~ bbs(x) + bbs(z) + bbs(w)")
    rows = model.concurvity(pairwise=True, output="records")
    assert len(rows) == 6  # 3 terms, both directions, no diagonal
    pairs = {(row["term"], row["against"]) for row in rows}
    assert pairs == {("bbs(x)", "bbs(z)"), ("bbs(z)", "bbs(x)"),
                     ("bbs(x)", "bbs(w)"), ("bbs(w)", "bbs(x)"),
                     ("bbs(z)", "bbs(w)"), ("bbs(w)", "bbs(z)")}
    by_pair = {(row["term"], row["against"]): row for row in rows}
    # z tracks x, so the x/z pair concurves far more than either does with w.
    assert by_pair[("bbs(z)", "bbs(x)")]["worst"] > 0.9
    assert by_pair[("bbs(z)", "bbs(w)")]["worst"] < by_pair[("bbs(z)", "bbs(x)")]["worst"]


def test_boosted_concurvity_matches_direct_measures_on_assembled_design():
    """The store method's summary equals a direct `_rust.concurvity_measures`
    call on the design the core assembles: intercept first, then every
    term's constant-reduced columns, with the path-accumulated coefficients.

    Measured on single-column `bols` terms, whose centered designs stay
    full-rank; a spline basis partitions unity, so its centered design is
    rank-deficient by one direction, and which column the core's pivoted
    pass drops is not something a test should replicate.
    """
    data = _boosted_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bols(x) + bols(z) + bols(w)", data=data,
                            control=mb.boost_control(120))
    store = model._replay_store()
    included, summary, _ = store.concurvity(model._selected_rust(),
                                            model._coefs_rust())
    # Assemble the same matrix in Python: center each term's columns, drop
    # numerical constants, accumulate beta = nu * sum of the term's steps.
    designs, coefs = [], []
    for rust_idx in included:
        design = np.asarray(store.extract_design(rust_idx), dtype=float)
        mean = design.mean(axis=0, keepdims=True)
        centered = design - mean
        keep = (centered ** 2).sum(axis=0) > 1e-20 * (design ** 2).sum(axis=0)
        designs.append(centered[:, keep])
        beta = np.zeros(design.shape[1])
        for sel, step in zip(model._selected_rust(), model._coefs_rust(),
                             strict=False):
            if sel == rust_idx:
                beta += model._control.nu * np.asarray(step, dtype=float)
        coefs.append(beta[keep])
    x = np.column_stack([np.ones(len(data["y"]))] + designs)
    spans = []
    start = 1
    for design in designs:
        spans.append((start, start + design.shape[1]))
        start += design.shape[1]
    beta_flat = np.concatenate([[0.0]] + coefs)
    direct = _rust.concurvity_measures(np.ascontiguousarray(x), spans,
                                       np.ascontiguousarray(beta_flat))
    for got, (worst, observed, estimate) in zip(summary, direct, strict=True):
        assert got[0] == pytest.approx(worst, abs=1e-9)
        if np.isnan(observed):
            assert np.isnan(got[1])
        else:
            assert got[1] == pytest.approx(observed, abs=1e-9)
        assert got[2] == pytest.approx(estimate, abs=1e-9)


def test_boosted_concurvity_omits_trees_and_reports_nan_for_never_selected():
    data = _boosted_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x) + btree(w)", data=data,
                            control=mb.boost_control(80))
    rows = model.concurvity(output="records")
    assert [row["term"] for row in rows] == ["bbs(x)"]
    assert model.concurvity(pairwise=True, output="records") == []


def test_boosted_concurvity_survives_pickle_roundtrip():
    import pickle

    data = _boosted_data()
    model = _boosted_model(data, formula="y ~ bbs(x) + bbs(z)")
    before = model.concurvity(output="records")
    restored = pickle.loads(pickle.dumps(model))
    after = restored.concurvity(output="records")
    assert before == after


def _glmboost_data():
    """y driven by x; x2 a near-exact multiple of x; z, w orthogonal noise."""
    rng = np.random.default_rng(11)
    n = 200
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    return {"y": x + z + rng.normal(0, .1, n),
            "x": x, "x2": 2.0 * x + 0.01 * rng.normal(size=n),
            "z": z, "w": w}


def test_glmboost_concurvity_is_the_collinearity_of_the_centered_columns():
    """A glmboost term is one column, so its concurvity is ordinary
    collinearity: the centered column's squared-multiple-correlation with
    the intercept and the other centered columns."""
    data = _glmboost_data()
    model = mb.glmboost("y ~ x + x2 + z + w", data, family=mb.Gaussian())
    rows = {row["term"]: row for row in model.concurvity(output="records")}
    assert rows["x2"]["worst"] > 0.99
    assert rows["w"]["worst"] < 0.05
    # For a single column the three measures coincide up to rounding: there
    # is only one coefficient direction to take a worst or an average over.
    # The intercept reports NaN (no function space after centering).
    for row in rows.values():
        if np.isnan(row["worst"]):
            continue
        assert np.isclose(row["worst"], row["estimate"], rtol=1e-9)
    # The measure equals the R^2 of regressing the centered column on the
    # other centered columns, the identity collinearity reduces to.
    xc = data["x2"] - data["x2"].mean()
    others = np.column_stack([
        np.ones(len(xc)),
        data["x"] - data["x"].mean(),
        data["z"] - data["z"].mean(),
        data["w"] - data["w"].mean()])
    r2 = 1 - np.sum((xc - others @ np.linalg.lstsq(others, xc, rcond=None)[0]) ** 2) \
        / np.sum(xc ** 2)
    assert np.isclose(rows["x2"]["worst"], r2, rtol=1e-9)


def test_glmboost_intercept_reports_nan_and_never_selected_observed_nan():
    """The formula's intercept is constant after centering, so it has no
    function space left and reports NaN; a term boosting never selected
    has no fitted coefficients, so its observed value is NaN while its
    worst-case value stays finite."""
    data = _glmboost_data()
    control = mb.boost_control(mstop=1)
    model = mb.glmboost("y ~ x + z + w", data, family=mb.Gaussian(), control=control)
    rows = {row["term"]: row for row in model.concurvity(output="records")}
    assert np.isnan(rows["(Intercept)"]["worst"])
    selected_name = model._learner_names[model._selected[0]]
    for name, row in rows.items():
        if name in ("(Intercept)", selected_name):
            continue
        assert np.isnan(row["observed"]), name
        assert np.isfinite(row["worst"]), name
    assert np.isfinite(rows[selected_name]["observed"])


def test_glmboost_pairwise_rows_name_every_ordered_pair():
    data = _glmboost_data()
    model = mb.glmboost("y ~ x + x2 + w", data, family=mb.Gaussian())
    rows = model.concurvity(pairwise=True, output="records")
    terms = {"(Intercept)", "x", "x2", "w"}
    assert {(row["term"], row["against"]) for row in rows} == {
        (a, b) for a in terms for b in terms if a != b}
    x_vs_x2 = next(row for row in rows
                   if {row["term"], row["against"]} == {"x", "x2"})
    assert x_vs_x2["worst"] > 0.99
    intercept_rows = [row for row in rows if "(Intercept)" in
                      (row["term"], row["against"])]
    assert len(intercept_rows) == 6
    assert all(np.isnan(row["worst"]) for row in intercept_rows)


def test_glmboost_concurvity_after_pickle_raises_a_clean_error():
    """The learner store cannot be pickled and a glmboost model keeps no
    replay specifications, so a restored model says so rather than
    returning numbers or crashing."""
    import pickle

    data = _glmboost_data()
    model = mb.glmboost("y ~ x + z", data, family=mb.Gaussian())
    restored = pickle.loads(pickle.dumps(model))
    with pytest.raises(ValueError, match="learner store"):
        restored.concurvity()
