"""Do not rank mismatched inputs or silently incomplete study panels."""
import pytest

from benches.report_competitive_shootout import aggregate, md_table, validate_pairing


def test_reporting_refuses_unpaired_inputs():
    document = {"protocol": {"specs": [1, 2]}, "results": [
        {"_status": "success", "task": "housing", "seed": 0, "tier": "small",
         "results": [{"tier": "small"}], "input_digest": "a", "partition_digest": "split"},
        {"_status": "success", "task": "housing", "seed": 0, "tier": "small",
         "results": [{"tier": "small"}], "input_digest": "b", "partition_digest": "split"},
    ]}
    with pytest.raises(ValueError, match="identical inputs"):
        validate_pairing(document)


def test_an_incomplete_panel_cannot_be_counted_as_competitive():
    points = [
        {"task": "housing", "method": "catboost", "tier": "tuned", "seed": seed,
         "loss": 1., "seconds": 1.}
        for seed in range(3)
    ] + [{"task": "housing", "method": "mboost_additive", "tier": "tuned",
          "seed": 0, "loss": .8, "seconds": .1}]
    result = aggregate(points)
    mboost = next(c for c in result if c["method"] == "mboost_additive")
    assert not mboost["complete"]
    assert not mboost["competitive"]


def test_material_gaps_follow_the_prespecified_relative_loss_threshold():
    points = [
        {"task": "housing", "method": method, "tier": "tuned", "seed": seed,
         "loss": loss, "seconds": 1.}
        for method, loss in (("catboost", 1.), ("mboost_additive", 1.2), ("mboost_hybrid", 1.04))
        for seed in range(3)
    ]
    result = {c["method"]: c for c in aggregate(points)}
    assert result["mboost_additive"]["material_gap"]
    assert result["mboost_hybrid"]["competitive"]


def test_markdown_report_marks_the_best_value_and_keeps_counts_whole():
    records = [{"method": "catboost", "loss": 0.5, "n": 3},
               {"method": "mboost_hybrid", "loss": 0.6, "n": 3}]
    rendered = md_table(records, "loss", "lower is better", rowname="method",
                        better={"loss": "min"})
    assert "| catboost | **0.500** | 3 |" in rendered
    assert "| mboost_hybrid | 0.600 | 3 |" in rendered


def test_remediation_table_is_absent_without_both_confirmation_artifacts(monkeypatch, tmp_path):
    from benches import report_competitive_shootout as report
    monkeypatch.setattr(report, "DEST", tmp_path)
    assert report.remediation_blocks() == []


def test_markdown_report_states_that_a_metric_is_missing():
    records = [{"method": "ebm", "coverage90": None}, {"method": "xgboost", "coverage90": 0.9}]
    rendered = md_table(records, "coverage", "", rowname="method", better={"coverage90": "min"})
    assert "| ebm | — |" in rendered
