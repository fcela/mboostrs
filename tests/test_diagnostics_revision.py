"""Residual checks that name a revision: clusters, time, influence.

Each check runs on planted structure it must detect (clustered residuals,
AR(1) residuals over time, one high-influence row) and on data with nothing
to find, where it must report nothing. The planted fits use the default
Gaussian family, and the checks use fewer permutations than the defaults so
the module runs in seconds.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import residual_checks as rc
from mboostrs.families import Poisson

FAST = dict(permutations=199, draws=100, seed=3)


def _fit(frame, formula="y ~ bbs(x, df=4)", mstop=150):
    return mb.gamboost(formula, frame, control=mb.boost_control(mstop=mstop, nu=0.1))


def _clustered(seed=1, groups=20, per_group=15, sd_cluster=1.0):
    rng = np.random.default_rng(seed)
    n = groups * per_group
    x = rng.uniform(-2, 2, n)
    site = np.repeat(np.arange(groups), per_group)
    effect = rng.normal(0.0, sd_cluster, groups)[site]
    y = np.sin(x) + effect + rng.normal(0.0, 0.5, n)
    return pl.DataFrame({"x": x, "y": y, "site": site})


def _ar1_series(seed=2, n=300, phi=0.7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + rng.normal(0.0, 0.5)
    return pl.DataFrame({"x": x, "y": np.sin(x) + noise, "t": np.arange(n, dtype=float)})


def _iid(seed=4, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    return pl.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0.0, 0.5, n),
                         "t": np.arange(n, dtype=float),
                         "site": np.repeat(np.arange(20), n // 20)})


def _planted_outlier(seed=5, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = np.sin(x) + rng.normal(0.0, 0.5, n)
    y[7] += 8.0
    x[7] = 2.5
    return pl.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


def test_clustered_residuals_are_flagged_with_a_random_intercept_revision():
    model = _fit(_clustered())
    finding = rc.cluster_residual_check(model, "site", **FAST)
    assert isinstance(finding, rc.RevisionFinding)
    assert finding.check == "cluster"
    assert finding.flagged
    assert finding.value > 0.3
    assert finding.p_value < 0.01
    assert finding.evidence["clusters"] == 20
    assert "brandom(site)" in finding.revision
    assert "grouped_folds" in finding.held_out_check
    assert 0.0 <= finding.false_alarm_rate <= 0.2
    assert finding.false_alarm_draws == FAST["draws"]


def test_adding_the_cluster_intercept_removes_the_cluster_flag():
    # Boosting shrinks a random intercept by nu per step, so the term needs a
    # larger mstop than the default test fits to absorb the cluster effect.
    frame = _clustered()
    before = rc.cluster_residual_check(_fit(frame, mstop=400), "site", **FAST)
    after = rc.cluster_residual_check(
        _fit(frame, "y ~ bbs(x, df=4) + brandom(site)", mstop=400), "site", **FAST)
    assert before.flagged
    assert not after.flagged
    assert abs(after.value) < abs(before.value)


def test_independent_residuals_across_clusters_report_nothing():
    model = _fit(_clustered(seed=8, sd_cluster=0.0))
    finding = rc.cluster_residual_check(model, "site", **FAST)
    assert not finding.flagged
    assert abs(finding.value) < 0.2
    assert "no cluster term is indicated" in finding.revision


def test_cluster_column_can_be_an_array_and_must_match_the_rows():
    frame = _clustered()
    model = _fit(frame)
    by_array = rc.cluster_residual_check(model, frame["site"].to_numpy(), **FAST)
    by_name = rc.cluster_residual_check(model, "site", **FAST)
    assert by_array.value == pytest.approx(by_name.value)
    with pytest.raises(ValueError, match="one value per row"):
        rc.cluster_residual_check(model, np.arange(10), **FAST)


def test_a_single_cluster_is_refused_with_a_reason():
    model = _fit(_clustered())
    with pytest.raises(ValueError, match="at least two clusters"):
        rc.cluster_residual_check(model, np.zeros(300, dtype=int), **FAST)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def test_ar1_residuals_over_time_are_flagged_with_a_time_smooth_revision():
    model = _fit(_ar1_series())
    finding = rc.time_residual_check(model, "t", **FAST)
    assert finding.check == "time"
    assert finding.flagged
    assert finding.p_value < 0.01
    assert finding.evidence["lag_one"] > 0.4
    assert 1 in finding.evidence["lags_outside_band"]
    assert "bbs(t)" in finding.revision
    assert "blocked_folds" in finding.held_out_check


def test_white_noise_over_time_reports_nothing():
    model = _fit(_iid())
    finding = rc.time_residual_check(model, "t", **FAST)
    assert not finding.flagged
    assert finding.p_value > 0.05
    assert "no time term is indicated" in finding.revision


def test_a_time_term_already_in_the_formula_points_at_the_error_process():
    model = _fit(_ar1_series(), "y ~ bbs(x, df=4) + bbs(t, df=4)")
    finding = rc.time_residual_check(model, "t", **FAST)
    assert finding.flagged
    assert "error process" in finding.revision


def test_text_time_is_refused_with_the_remedy():
    frame = _ar1_series().with_columns(label=pl.lit("a"))
    model = _fit(frame)
    with pytest.raises(ValueError, match="must be converted"):
        rc.time_residual_check(model, "label", **FAST)


# ---------------------------------------------------------------------------
# Influence
# ---------------------------------------------------------------------------


def test_planted_high_influence_row_is_flagged_and_refit_without_it_moves_the_fit():
    model = _fit(_planted_outlier())
    finding = rc.influence_check(model, draws=100, seed=3)
    assert isinstance(finding, rc.InfluenceFinding)
    assert finding.flagged
    assert finding.rows == (7,)
    assert finding.cooks[7] == max(finding.cooks)
    assert finding.leverage[7] > 2 * finding.evidence["df"] / 300
    assert len(finding.refits) == 1
    refit = finding.refits[0]
    assert refit["row"] == 7
    assert refit["rows_kept"] == 299
    assert refit["max_change"] > 0.1
    assert "refit without rows [7]" in finding.revision
    assert finding.false_alarm_rate <= 0.1


def test_clean_gaussian_data_flags_no_row():
    model = _fit(_iid())
    finding = rc.influence_check(model, draws=100, seed=3)
    assert not finding.flagged
    assert finding.rows == ()
    assert finding.refits == ()
    assert "keep all rows" in finding.revision


def test_influence_refuses_a_non_gaussian_family_by_name():
    rng = np.random.default_rng(6)
    x = rng.uniform(-2, 2, 200)
    frame = pl.DataFrame({"x": x, "y": rng.poisson(np.exp(0.5 * x)).astype(float)})
    model = mb.gamboost("y ~ bbs(x, df=4)", frame, family=Poisson(),
                        control=mb.boost_control(mstop=50, nu=0.1))
    with pytest.raises(ValueError, match="Poisson"):
        rc.influence_check(model, draws=100)


def test_refit_top_zero_skips_the_exact_deletion_refits():
    model = _fit(_planted_outlier())
    finding = rc.influence_check(model, refit_top=0, draws=100, seed=3)
    assert finding.flagged
    assert finding.refits == ()
    assert "refit without rows" not in finding.revision


# ---------------------------------------------------------------------------
# All checks together, and the report
# ---------------------------------------------------------------------------


def test_revision_checks_runs_all_three_and_lists_the_flagged_revisions():
    frame = _clustered().with_columns(t=pl.Series(np.arange(300, dtype=float)))
    model = _fit(frame)
    report = rc.revision_checks(model, cluster="site", time="t", **FAST)
    assert [f.check for f in report.findings] == ["cluster", "time", "influence"]
    assert report.flagged
    assert any("brandom(site)" in text for text in report.revisions)
    assert "revision:" in report.summary


def test_revision_checks_skips_the_checks_whose_column_is_not_given():
    model = _fit(_planted_outlier())
    report = rc.revision_checks(model, draws=100, permutations=199)
    assert [f.check for f in report.findings] == ["influence"]
    assert report.flagged[0].check == "influence"


def test_the_summary_states_the_guide_fields_for_every_finding():
    model = _fit(_clustered())
    text = rc.cluster_residual_check(model, "site", **FAST).summary
    for label in ("evidence:", "false-alarm rate", "assumption challenged:",
                  "alternatives:", "revision:", "held-out check:"):
        assert label in text


def test_a_column_name_that_is_not_in_the_data_is_refused():
    rng = np.random.default_rng(9)
    frame = pl.DataFrame({"x": rng.uniform(-2, 2, 200), "y": rng.normal(size=200)})
    model = _fit(frame)
    with pytest.raises(ValueError, match="no column"):
        rc.cluster_residual_check(model, "missing_column", **FAST)
