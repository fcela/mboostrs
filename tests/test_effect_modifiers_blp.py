"""Effect modifiers from AIPW scores: BLP slopes, GATES, projections and ranking."""

import warnings
from functools import partial

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import _rust
from mboostrs.causal import AIPWResult
from mboostrs.effect_modifiers import BLPProjection
from mboostrs.effect_modifiers import effect_modifiers as _effect_modifiers

# The synthetic results below carry the true effect as their scores and no outcome
# model, so the score regressions are tested without the R-learner gate.
effect_modifiers = partial(_effect_modifiers, gate=False)


def spec(covariates=("x1", "x2", "x3")):
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=tuple(covariates),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="the covariates are the common causes",
    )


def synthetic(n=3000, seed=0):
    """A known effect tau(x) = 1000 + 800 x1 - 600 x2 with x3 unrelated to it.

    The returned AIPWResult carries the true effect plus independent noise as its
    scores, so the tests exercise the regression and ranking, not the nuisance fits.
    ``x4`` is a binary covariate that changes the effect by +500 when it is 1."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=(3, n))
    x4 = np.where(rng.uniform(size=n) < 0.5, "yes", "no")
    e = np.clip(1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2))), 0.1, 0.9)
    tau = 1000 + 800 * x1 - 600 * x2 + 500 * (x4 == "yes")
    scores = tau + rng.normal(0, 3000, n)
    a = rng.binomial(1, e).astype(float)
    data = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "a": a})
    result = AIPWResult(
        estimate=float(scores.mean()), standard_error=None, interval=None, level=0.95,
        spec=spec(("x1", "x2", "x3", "x4")), treatment=a, outcome=np.zeros(n),
        propensity=e, mu0=np.zeros(n), mu1=np.zeros(n), scores=scores,
        influence=scores - scores.mean(), fold_id=np.zeros(n, dtype=int),
        diagnostics={}, nuisance={},
    )
    return result, data, tau, e


def hc1_reference(y, x, w):
    """Textbook WLS with HC1 covariance, written out with numpy."""
    xtwx = x.T @ (x * w[:, None])
    bread = np.linalg.inv(xtwx)
    beta = bread @ (x.T @ (w * y))
    resid = y - x @ beta
    meat = (x * (w * resid)[:, None] ** 2).T @ x
    n, p = x.shape
    cov = (n / (n - p)) * bread @ meat @ bread
    return beta, cov


def test_robust_wls_matches_the_textbook_hc1_sandwich():
    rng = np.random.default_rng(1)
    n = 400
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(size=n)])
    y = x @ np.array([1.0, 2.0, -1.0]) + rng.standard_t(4, size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    coef, se, cov, n_used = _rust.effect_robust_wls(y, x.ravel(), 3, w)
    beta_ref, cov_ref = hc1_reference(y, x, w)
    assert n_used == n
    np.testing.assert_allclose(coef, beta_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cov).reshape(3, 3), cov_ref, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(se, np.sqrt(np.diag(cov_ref)), rtol=1e-8)


def test_wald_test_matches_the_quadratic_form():
    coef = np.array([0.5, 2.0, -1.0])
    cov = np.array([[0.04, 0.0, 0.01], [0.0, 0.09, 0.0], [0.01, 0.0, 0.16]])
    chi2, p = _rust.effect_wald_test(coef, cov.ravel(), 3, [1, 2])
    sub = cov[np.ix_([1, 2], [1, 2])]
    expected = coef[[1, 2]] @ np.linalg.solve(sub, coef[[1, 2]])
    assert chi2 == pytest.approx(expected, rel=1e-10)
    assert 0 < p < 1


def test_holm_adjustment_matches_the_step_down_definition():
    adj = np.asarray(_rust.effect_holm_adjust(np.array([0.01, 0.04, 0.03, 0.005])))
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    with_nan = np.asarray(_rust.effect_holm_adjust(np.array([np.nan, 0.02, 0.5])))
    assert np.isnan(with_nan[0])
    assert with_nan[1] == pytest.approx(0.04)


def test_gates_contrast_is_top_minus_bottom_group_mean():
    score = np.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0])
    weights = np.ones(6)
    means, se, counts, ordered, extreme, omnibus = _rust.effect_gates(
        score, weights, [0, 0, 1, 1, 2, 2], 3)
    assert list(means) == pytest.approx([0.0, 1.0, 3.0])
    assert counts == [2, 2, 2]
    assert ordered[0] == pytest.approx(3.0)
    assert (ordered[3], ordered[4]) == (2, 0)
    assert omnibus[1] == 2


def test_blp_slopes_recover_the_known_effect_and_separate_the_noise():
    result, data, tau, _ = synthetic()
    out = effect_modifiers(result, data.drop("a"), method="blp")
    rec = {r["covariate"]: r for r in out.records}
    # The slope on a standardized covariate is the change in tau per standard deviation.
    assert rec["x1"]["blp_slope"] == pytest.approx(800 * np.std(data["x1"].to_numpy(), ddof=1),
                                                   rel=0.15)
    assert rec["x1"]["decision"] and rec["x1"]["direction"] == "increases"
    assert rec["x2"]["decision"] and rec["x2"]["direction"] == "decreases"
    assert not rec["x3"]["decision"]
    assert {"x1", "x2"} <= set(out.selected())


def test_categorical_covariate_gets_level_coefficients_and_a_joint_test():
    result, data, _, _ = synthetic()
    out = effect_modifiers(result, data.drop("a"), method="gates")
    rec = {r["covariate"]: r for r in out.records}["x4"]
    assert rec["kind"] == "categorical"
    assert rec["blp_slope"] is None
    assert rec["blp_levels"][0]["level"] == "yes"
    assert rec["blp_levels"][0]["coefficient"] == pytest.approx(500, abs=350)
    assert rec["decision"]
    assert rec["gates_top"] == "yes"


def test_gates_method_reports_quantile_group_contrasts_in_order():
    result, data, _, _ = synthetic()
    out = effect_modifiers(result, data.drop("a"), method="gates", groups=4)
    rec = {r["covariate"]: r for r in out.records}["x1"]
    assert rec["gates_top"] == "group 4 of 4"
    assert rec["gates_bottom"] == "group 1 of 4"
    assert rec["gates_contrast"] > 0
    assert len(rec["gates_means"]) == 4
    assert rec["gates_lower"] < rec["gates_contrast"] < rec["gates_upper"]


def test_only_rows_inside_the_overlap_bounds_are_used():
    result, data, _, e = synthetic()
    inside = (e >= 0.2) & (e <= 0.8)
    out = effect_modifiers(result, data.drop("a"), overlap_bounds=(0.2, 0.8))
    assert out.n_inside == int(inside.sum())
    assert out.n == len(e)


def test_adjust_none_keeps_raw_p_values_and_holm_never_lowers_them():
    result, data, _, _ = synthetic()
    raw = effect_modifiers(result, data.drop("a"), adjust="none")
    holm = effect_modifiers(result, data.drop("a"), adjust="holm")
    for r, h in zip(raw.records, holm.records, strict=True):
        assert r["blp_p_adjusted"] == pytest.approx(r["blp_p"])
        assert h["blp_p_adjusted"] >= h["blp_p"] - 1e-12


def test_joint_table_reports_each_design_column():
    result, data, _, _ = synthetic()
    out = effect_modifiers(result, data.drop("a"))
    columns = [row["column"] for row in out.joint]
    assert columns == ["x1", "x2", "x3", "x4[yes]"]
    assert not any(row["collinear"] for row in out.joint)


def test_collinear_covariates_are_dropped_from_the_joint_fit_with_a_warning():
    result, data, _, _ = synthetic()
    data = data.with_columns((pl.col("x1") + pl.col("x2")).alias("x5"))
    res2 = AIPWResult(**{**result.__dict__, "spec": spec(("x1", "x2", "x3", "x4", "x5"))})
    with pytest.warns(UserWarning, match="collinear"):
        out = effect_modifiers(res2, data.drop("a"))
    dropped = [row for row in out.joint if row["collinear"]]
    assert len(dropped) == 1 and dropped[0]["column"] == "x5"
    assert np.isnan(dropped[0]["coefficient"])


def test_projection_predicts_the_fitted_effect_and_ranks_within_overlap():
    result, data, tau, e = synthetic()
    out = effect_modifiers(result, data.drop("a"))
    proj = out.projection("all")
    assert isinstance(proj, BLPProjection)
    pred = proj.predict(data.drop("a"))
    assert pred.shape == (len(e),)
    assert np.corrcoef(pred, tau)[0, 1] > 0.8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranking = mb.rank_by_effect(proj, data.drop("a"), share=0.2, propensity=e,
                                    bounds=(0.02, 0.98))
    chosen = np.asarray(ranking.chosen)
    assert chosen.size == round(0.2 * len(e))
    assert np.all((e[chosen] >= 0.02) & (e[chosen] <= 0.98))
    assert tau[chosen].mean() > tau.mean()


def test_rank_by_effect_on_a_projection_needs_propensity():
    result, data, _, _ = synthetic(n=600)
    proj = effect_modifiers(result, data.drop("a")).projection("all")
    with pytest.raises(ValueError, match="needs propensity"):
        mb.rank_by_effect(proj, data.drop("a"))


def test_passing_projection_uses_only_the_covariates_that_pass():
    result, data, _, _ = synthetic()
    out = effect_modifiers(result, data.drop("a"), method="blp")
    proj = out.projection("passing")
    assert proj.covariates == out.selected()
    assert "x3" not in proj.covariates


def test_passing_projection_with_no_passing_covariate_is_refused():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    e = np.full(n, 0.5)
    scores = rng.normal(size=n)
    result = AIPWResult(
        estimate=0.0, standard_error=None, interval=None, level=0.95, spec=spec(("x1",)),
        treatment=np.zeros(n), outcome=np.zeros(n), propensity=e, mu0=np.zeros(n),
        mu1=np.zeros(n), scores=scores, influence=scores, fold_id=np.zeros(n, dtype=int),
        diagnostics={}, nuisance={})
    out = effect_modifiers(result, pl.DataFrame({"x1": x}))
    assert out.selected() == ()
    with pytest.raises(ValueError, match="no covariate has a True decision"):
        out.projection("passing")


def test_invalid_arguments_are_refused_with_a_remedy():
    result, data, _, _ = synthetic(n=400)
    with pytest.raises(TypeError, match="AIPWResult"):
        effect_modifiers({"not": "a result"}, data)
    with pytest.raises(ValueError, match="method must be"):
        effect_modifiers(result, data, method="tree")
    with pytest.raises(ValueError, match="adjust must be"):
        effect_modifiers(result, data, adjust="bonferroni")
    with pytest.raises(ValueError, match="groups must be"):
        effect_modifiers(result, data, groups=1)
    with pytest.raises(ValueError, match="level must be"):
        effect_modifiers(result, data, level=1.0)
    with pytest.raises(ValueError, match="rows but the result has"):
        effect_modifiers(result, data.head(10))


def test_constant_covariate_inside_the_overlap_is_refused():
    result, data, _, _ = synthetic(n=400)
    data = data.with_columns(pl.lit(1.0).alias("const"))
    with pytest.raises(ValueError, match="constant inside the overlap"):
        effect_modifiers(result, data.drop("a"), covariates=["const"])


def test_projection_refuses_an_unseen_level_at_prediction():
    result, data, _, _ = synthetic(n=600)
    proj = effect_modifiers(result, data.drop("a")).projection(["x4"])
    newdata = data.drop("a").with_columns(pl.lit("maybe").alias("x4"))
    with pytest.raises(ValueError, match="levels"):
        proj.predict(newdata)


def test_table_and_describe_are_available():
    result, data, _, _ = synthetic(n=400)
    out = effect_modifiers(result, data.drop("a"))
    frame = out.table()
    assert frame.height == 4
    assert {"covariate", "blp_slope", "gates_contrast", "adjusted_p", "decision"} <= set(
        frame.columns)
    assert out.joint_table().height == 4
    assert "x1" in out.describe()
    assert out.to_dict()["selected"] == list(out.selected())
