"""Every reader of a CV curve's optimum excludes iteration 0 (issue #63).

Iteration 0 is the offset-only model. `optimal_mstop` and `mstop()` never
return it (AGENTS gotchas 18 and 22); the one-standard-error rule, the
boundary check, the summary table and `truncated_at_mstop` used an
unrestricted argmin and could disagree with it.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.cvrisk import CvRiskResult, _optimum_index, truncated_at_mstop
from mboostrs.summary import _cv_table

# Offset-only risk is lowest; among real fits, iteration 1 is best.
RISK = np.array([[1.00, 1.05, 1.10, 1.20],
                 [1.00, 1.02, 1.15, 1.20],
                 [1.00, 1.08, 1.05, 1.30]])


def _result():
    return CvRiskResult(RISK, RISK.mean(0), [0, 1, 2, 3], 1, "kfold")


def test_optimum_index_skips_iteration_zero():
    assert _optimum_index(RISK.mean(0), [0, 1, 2, 3]) == 1


def test_one_standard_error_rule_never_returns_zero():
    assert _result().mstop_1se == 1


def test_boundary_check_agrees_with_optimal_mstop():
    falling = np.array([[3.0, 2.0, 1.5, 1.0]] * 3)
    result = CvRiskResult(falling, falling.mean(0), [0, 1, 2, 3], 3, "kfold")
    assert result.at_boundary()
    assert not _result().at_boundary()


def test_truncation_reports_a_positive_optimum():
    assert truncated_at_mstop(_result(), 3).optimal_mstop == 1


def test_summary_table_reads_risk_at_the_reported_optimum():
    rng = np.random.default_rng(1)
    data = {"x": rng.normal(size=60), "y": rng.normal(size=60)}
    model = mb.glmboost("y ~ x", data=data, control=mb.boost_control(3))
    table = _cv_table(model, _result())
    rows = dict((row[0], row[1]) for row in table.rows)
    assert int(rows["Optimal mstop"]) == 1
    assert float(rows["CV risk at optimum"]) == pytest.approx(RISK.mean(0)[1], rel=1e-3)
