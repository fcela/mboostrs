"""Regression tests for the #509 stopping-rule ablation harness.

The harness itself is bench code, but the semantics it must preserve are
load-bearing for the experiment's conclusions: the iteration-zero contract,
the status-quo replication of the shootout's fixed-budget argmin, the
exposure of an unresolved boundary, and the pairing of rule-versus-status-quo
losses on identical test frames. These tests pin those semantics on small
panels so a library change that breaks the ablation's assumptions is caught
before results are recorded.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benches.shootout_models import formula  # noqa: E402
from benches.stopping_ablation_509 import (  # noqa: E402
    RULES,
    configuration_set,
    configuration_differences,
    fold_manifest,
    grouped_summary,
    paired_differences,
    stopping_runs,
    summarize,
    summary_document,
    validation_scheme,
)

SMALL = dict(budget=60, cap=120, patience=20, k=3, repeats=2)


def panel(n=240, noise=0.1, seed=7):
    """A small smooth panel split like the shootout: fit/val/cal/test."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 3))
    y = 2 * np.sin(3 * x[:, 0]) + x[:, 1] ** 2 + noise * rng.normal(size=n)
    frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})
    cuts = np.array_split(rng.permutation(n), [n // 2, 3 * n // 4, 7 * n // 8])
    return [frame[c] for c in cuts]


def falling_panel(n=240, seed=13):
    """Strong smooth signal: with a small budget the risk is still falling at
    the cap, which is the budget-exhaustion regime the record must expose."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 3))
    y = 6 * x[:, 0] + 4 * (x[:, 1] ** 2 - 1 / 3) + 0.05 * rng.normal(size=n)
    frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})
    cuts = np.array_split(rng.permutation(n), [n // 2, 3 * n // 4, 7 * n // 8])
    return [frame[c] for c in cuts]


@pytest.fixture(scope="module")
def smooth_result():
    outputs, tiers, scheme = configuration_set("mboost_additive", panel(), [], "regression", 0, **SMALL)
    return outputs, tiers, scheme


def test_every_rule_preserves_the_iteration_zero_contract(smooth_result):
    outputs, _, _ = smooth_result
    for row in outputs:
        for rule in RULES:
            record = row["rules"][rule]
            assert record["mstop"] >= 1
            assert record["at_boundary"] in (True, False)
            assert record["largest"] >= record["mstop"]
            assert len(record["aggregate"]) == record["largest"] + 1
            assert record["selection_loss"] is not None
            assert record["seconds"] > 0


def test_fixed_argmin_replicates_the_shootout_selection(smooth_result):
    """`fixed_argmin` must be the shootout's own route and argmin, so the
    ablation's baseline is the status quo rather than a near-copy."""
    import mboostrs as mb

    outputs, _, _ = smooth_result
    train, val, _, _ = panel()
    dev = pl.concat([train, val])
    weights = np.r_[np.ones(len(train)), np.zeros(len(val))][:, None]
    for row in outputs:
        config = row["config"]
        rhs, _ = formula(dev, [], config["df"])
        model = mb.gamboost(rhs, dev, family=mb.Gaussian(), control=mb.boost_control(SMALL["budget"], nu=config["nu"]))
        cv = model.cvrisk(folds=weights, preprocessing="refit")
        assert row["rules"]["fixed_argmin"]["mstop"] == int(cv.optimal_mstop)


def test_growth_rules_share_the_first_grid_and_record_their_extensions(smooth_result):
    outputs, _, _ = smooth_result
    for row in outputs:
        fixed = row["rules"]["fixed_argmin"]
        for rule in ("boundary_growth", "patience_60", "repeated_cv_growth"):
            record = row["rules"][rule]
            assert record["grid_ends"][0] == fixed["grid_ends"][0]
            assert record["preprocessing"] == "fixed"
            assert record["extended"] == (record["largest"] > record["grid_ends"][0])


def test_nonextended_growth_matches_its_preprocessing_control():
    outputs, _, _ = configuration_set(
        "mboost_additive", panel(noise=2.0), [], "regression", 0, budget=200, cap=400, patience=20, k=3, repeats=2
    )
    checked = 0
    for row in outputs:
        for baseline, growth in (("fixed_preprocessing", "boundary_growth"), ("fold_local_fixed", "fold_local_growth")):
            fixed, grown = row["rules"][baseline], row["rules"][growth]
            assert fixed["preprocessing"] == grown["preprocessing"]
            assert fixed["score"] == grown["score"]
            if not grown["extended"]:
                assert fixed["aggregate"] == grown["aggregate"]
                assert fixed["mstop"] == grown["mstop"]
                assert fixed["test_loss"] == pytest.approx(grown["test_loss"], abs=1e-12)
                checked += 1
    assert checked > 0, "the test must exercise a nonextended path"


def test_fold_local_curves_preserve_fold_scores(smooth_result):
    outputs, _, scheme = smooth_result
    for row in outputs:
        for rule in ("fold_local_fixed", "fold_local_growth", "fold_local_cv_growth"):
            record = row["rules"][rule]
            curves = np.asarray(record["fold_risk_curves"])
            nfolds = scheme.weights.shape[1] if rule == "fold_local_cv_growth" else 1
            assert curves.shape == (nfolds, record["largest"] + 1)
            np.testing.assert_allclose(curves[:, record["mstop"]], record["fold_scores"])
            np.testing.assert_allclose(curves.mean(axis=0), record["aggregate"], rtol=1e-5)


def test_temporal_folds_keep_whole_times_and_judge_only_the_next_horizon():
    times = np.repeat(np.arange(40), 3)[::-1]
    scheme = validation_scheme(len(times), 5, design="temporal", groups=times, k=3, repeats=2)
    manifest = fold_manifest(scheme)
    assert scheme.ordered
    assert manifest["seed"] is None
    assert scheme.weights.shape[1] == 6
    for b, fold in enumerate(manifest["folds"]):
        train, held = scheme.weights[:, b] > 0, scheme.held_out[:, b]
        assert times[train].max() < times[held].min()
        assert set(times[train]).isdisjoint(times[held])
        np.testing.assert_array_equal(scheme.judge[:, b], held)
        assert fold["judge_rows"] == fold["held_out_rows"]
    assert np.any((scheme.weights[:, 0] == 0) & ~scheme.judge[:, 0])


def test_bike_future_uses_its_actual_day_groups():
    from benches.shootout_data import CACHE, payload

    if not (CACHE / "bike_future.parquet").exists():
        pytest.skip("local frozen bike data not available")
    data = payload("bike_future", 0)
    times = np.asarray(data["metadata"]["auxiliary"]["groups"])[data["indices"][0] + data["indices"][1]].astype(
        "datetime64[D]"
    )
    scheme = validation_scheme(len(times), 0, design="temporal", groups=times)
    for b in range(scheme.weights.shape[1]):
        assert times[scheme.weights[:, b] > 0].max() < times[scheme.judge[:, b]].min()


@pytest.mark.parametrize("design", ["spatial", "grouped"])
def test_grouped_designs_keep_dependency_units_intact(design):
    groups = np.repeat(np.arange(20), 4)
    scheme = validation_scheme(len(groups), 0, design=design, groups=groups, k=3, repeats=2)
    for b in range(scheme.weights.shape[1]):
        assert set(groups[scheme.weights[:, b] > 0]).isdisjoint(groups[scheme.held_out[:, b]])


def test_structured_designs_require_dependency_units():
    with pytest.raises(ValueError, match="dependency unit"):
        validation_scheme(80, 0, design="temporal")


def test_temporal_scheme_reaches_tuner_with_its_judge_mask(monkeypatch):
    import mboostrs as mb

    train, val, _, _ = panel(n=160)
    dev = pl.concat([train, val])
    scheme = validation_scheme(len(dev), 0, design="temporal", groups=np.arange(len(dev)), k=3, repeats=2)
    split = np.r_[np.ones(len(train)), np.zeros(len(val))][:, None]
    original = mb.tune_mstop
    passed = []

    def checked(*args, **kwargs):
        if kwargs["folds"] is scheme:
            passed.append(kwargs["folds"])
        return original(*args, **kwargs)

    monkeypatch.setattr(mb, "tune_mstop", checked)
    stopping_runs("y ~ bbs(x0, df=4)", dev, mb.Gaussian(), 0.1, split, scheme, budget=20, cap=40, patience=10)
    assert passed == [scheme, scheme]


def test_null_response_selects_first_positive_iteration_under_every_rule():
    frames = [frame.with_columns(pl.lit(1.0).alias("y")) for frame in panel(n=160)]
    outputs, _, _ = configuration_set(
        "mboost_additive", frames, [], "regression", 0, budget=10, cap=20, patience=5, k=3, repeats=2
    )
    for row in outputs:
        for record in row["rules"].values():
            assert record["mstop"] == 1
            assert record["test_loss"] == pytest.approx(0.0, abs=1e-20)


def test_unresolved_boundary_is_exposed_not_hidden():
    """A curve still falling at the cap must surface as an unresolved boundary
    in every rule that can still see one, with the library's own warning
    captured in the record rather than only printed."""
    outputs, _, _ = configuration_set(
        "mboost_additive", falling_panel(), [], "regression", 0, budget=40, cap=80, patience=10, k=3, repeats=2
    )
    for row in outputs:
        fixed = row["rules"]["fixed_argmin"]
        growth = row["rules"]["boundary_growth"]
        assert fixed["at_boundary"], "still falling: argmin sits at the budget edge"
        assert fixed["mstop"] == 40, "the fixed budget is the truncation point"
        assert growth["largest"] == 80, "still falling: the grid must reach the cap"
        assert growth["at_boundary"], "cap reached while falling is unresolved"
        assert any("boundary" in message.lower() or "falling" in message.lower() for message in growth["warnings"]), (
            growth["warnings"]
        )
        assert row["rules"]["repeated_cv_growth"]["mstop"] >= 1


def test_records_are_deterministic_for_a_fixed_seed():
    frames = panel(n=180, seed=3)
    first, _, _ = configuration_set(
        "mboost_additive",
        frames,
        [],
        "regression",
        5,
        budget=40,
        cap=80,
        patience=10,
        k=SMALL["k"],
        repeats=SMALL["repeats"],
    )
    second, _, _ = configuration_set(
        "mboost_additive",
        frames,
        [],
        "regression",
        5,
        budget=40,
        cap=80,
        patience=10,
        k=SMALL["k"],
        repeats=SMALL["repeats"],
    )
    for one, two in zip(first, second, strict=True):
        for rule in RULES:
            assert one["rules"][rule]["mstop"] == two["rules"][rule]["mstop"]
            assert one["rules"][rule]["aggregate"] == two["rules"][rule]["aggregate"]


def test_stopping_runs_reports_the_shared_split_and_scheme():
    """All four rules read the same development rows; the single-split rules
    share the zero-weighted validation column and the repeated rule shares
    the recorded scheme."""
    import mboostrs as mb
    from mboostrs.folds import repeated_kfold_folds

    train, val, _, _ = panel(n=160, seed=2)
    dev = pl.concat([train, val])
    weights = np.r_[np.ones(len(train)), np.zeros(len(val))][:, None]
    scheme = repeated_kfold_folds(len(dev), k=3, repeats=2, seed=555)
    rhs = "y ~ bbs(x0, df=4) + bbs(x1, df=4) + bbs(x2, df=4)"
    records = stopping_runs(rhs, dev, mb.Gaussian(), 0.1, weights, scheme, budget=40, cap=80, patience=10)
    assert set(records) == set(RULES)
    assert records["fixed_argmin"]["fold_fits"] == 1
    assert records["repeated_cv_growth"]["fold_fits"] == 6
    assert all(records[rule]["model"] is not None for rule in RULES)


def test_paired_differences_pair_within_seed_and_task():
    def rule_tiers(small_loss, tuned_loss):
        return {
            "small": {"config_index": 0, "cal_loss": 1.0, "test_loss": small_loss, "mstop": 10},
            "tuned": {"config_index": 1, "cal_loss": 0.9, "test_loss": tuned_loss, "mstop": 12},
        }

    row = {
        "task": "toy",
        "seed": 0,
        "rules": list(RULES),
        "method_results": [
            {
                "method": "mboost_hybrid",
                "configurations": [],
                "tiers": {
                    rule: (rule_tiers(2.0, 2.0) if rule == "fixed_argmin" else rule_tiers(2.1, 2.2)) for rule in RULES
                },
            }
        ],
    }
    differences = paired_differences([row])
    assert len(differences) == (len(RULES) - 1) * 2
    for difference in differences:
        assert difference["task"] == "toy"
        assert difference["paired_test_loss"] == 2.0
        if difference["tier"] == "small":
            assert difference["difference"] == pytest.approx(0.1)
        else:
            assert difference["difference"] == pytest.approx(0.2)
            assert difference["relative"] == pytest.approx(0.1)
    none_row = {
        "task": "toy",
        "seed": 1,
        "rules": list(RULES),
        "method_results": [
            {
                "method": "mboost_hybrid",
                "configurations": [],
                "tiers": {
                    rule: {
                        "small": {"config_index": 0, "cal_loss": None, "test_loss": None, "mstop": 5},
                        "tuned": {"config_index": 0, "cal_loss": None, "test_loss": None, "mstop": 5},
                    }
                    for rule in RULES
                },
            }
        ],
    }
    for difference in paired_differences([none_row]):
        assert difference["difference"] is None
        assert difference["relative"] is None


def test_summarize_reports_every_rule_and_protocol_notes():
    document = {
        "created_at": "2026-10-12T00:00:00+00:00",
        "commit": "deadbeef",
        "results": [
            {
                "task": "toy",
                "seed": 0,
                "rules": list(RULES),
                "method_results": [
                    {
                        "method": "mboost_hybrid",
                        "configurations": [
                            {
                                "config_index": 0,
                                "config": {"df": 4, "nu": 0.05, "depth": 3},
                                "rules": {rule: {"mstop": 10, "at_boundary": False, "largest": 60} for rule in RULES},
                            }
                        ],
                        "tiers": {
                            rule: {
                                "small": {"config_index": 0, "cal_loss": 1.0, "test_loss": 2.0, "mstop": 10},
                                "tuned": {"config_index": 0, "cal_loss": 1.0, "test_loss": 2.0, "mstop": 10},
                            }
                            for rule in RULES
                        },
                    }
                ],
            }
        ],
    }
    text = summarize(document)
    assert "# Stopping-rule ablation (issue #509)" in text
    for rule in RULES:
        assert rule in text
    assert "| toy |" in text
    assert "Protocol notes" in text
    assert "claim that they were absent" in text


def test_descriptive_summaries_keep_harmful_seeds_and_denominators():
    pairs = [
        {
            "task": "cancer",
            "method": "mboost_hybrid",
            "tier": "small",
            "rule": "growth",
            "reference": "fixed",
            "seed": seed,
            "paired_test_loss": 2.0,
            "test_loss": 2.0 * (1.0 + relative),
            "relative": relative,
            "difference": 2.0 * relative,
        }
        for seed, relative in enumerate([-0.1, -0.05, 0.091])
    ]
    summary = grouped_summary(pairs)[0]
    assert summary["n"] == summary["n_finite"] == 3
    assert summary["median_relative"] == -0.05
    assert summary["min_relative"] == -0.1
    assert summary["max_relative"] == 0.091
    assert summary["worst_seed"] == 2
    assert summary["worst_reference_loss"] == 2.0
    assert summary["worst_test_loss"] == pytest.approx(2.182)


def test_saved_v1_contradicts_the_withdrawn_universal_claims():
    import json

    path = Path(__file__).resolve().parents[1] / "docs/benchmarks/stopping_ablation_509/results.json"
    document = json.loads(path.read_text())
    summary = summary_document(document)
    growth = next(
        r
        for r in summary["historical"]["pairs"]
        if (r["task"], r["seed"], r["method"], r["tier"], r["rule"])
        == ("cancer", 2, "mboost_hybrid", "small", "boundary_growth")
    )
    assert growth["relative"] == pytest.approx(0.091308, abs=1e-5)
    patience = paired_differences(document["results"], reference="boundary_growth", rules=["patience_60"])
    patience = next(
        r for r in patience if (r["task"], r["seed"], r["method"], r["tier"]) == ("cancer", 2, "mboost_hybrid", "small")
    )
    assert patience["relative"] < -0.05
    tuning = configuration_differences(document["results"])
    tuning = next(
        r
        for r in tuning
        if (r["task"], r["seed"], r["method"], r["rule"]) == ("cancer", 2, "mboost_hybrid", "fixed_argmin")
    )
    assert tuning["relative"] > 1.07


def test_saved_v2_report_and_summary_are_derived_from_the_artifact():
    import gzip
    import hashlib
    import json
    from benches.benchmark_contract import data_digest

    base = Path(__file__).resolve().parents[1] / "docs/benchmarks/stopping_ablation_509"
    path = base / "v2/results.json.gz"
    if not path.exists():
        pytest.skip("versioned rerun not yet recorded")
    raw = gzip.decompress(path.read_bytes())
    document = json.loads(raw)
    manifest = json.loads((base / "v2/manifest.json").read_text())
    assert len(raw) == manifest["uncompressed_bytes"]
    assert hashlib.sha256(raw).hexdigest() == manifest["uncompressed_sha256"]
    assert len(document["results"]) == len(document["protocol"]["specs"]) == 25
    assert json.loads((base / "v2/summary.json").read_text()) == summary_document(document)
    assert (base / "v2/README.md").read_text() == summarize(document) + "\n"
    legacy = json.loads((base / "results.json").read_text())
    old = {(row["task"], row["seed"]): row for row in legacy["results"]}
    for row in document["results"]:
        assert row["_status"] == "success"
        before = old[row["task"], row["seed"]]
        assert row["input_digest"] == before["input_digest"]
        assert row["partition_digest"] == before["partition_digest"]
        assert row["source_hash"] == before["source_hash"]
        assert row["fold_scheme_digest"] == data_digest(row["fold_scheme"])
        if row["design"] == "temporal":
            times = np.array(row["fold_scheme"]["times"], dtype="datetime64[D]")
            for fold in row["fold_scheme"]["folds"]:
                assert times[fold["training_rows"]].max() < times[fold["judge_rows"]].min()
                assert fold["judge_rows"] == fold["held_out_rows"]
        for method in row["method_results"]:
            for config in method["configurations"]:
                assert set(config["rules"]) == set(RULES)
                for reference, rule in (
                    ("fixed_preprocessing", "boundary_growth"),
                    ("fold_local_fixed", "fold_local_growth"),
                ):
                    fixed, grown = config["rules"][reference], config["rules"][rule]
                    if not grown["extended"]:
                        assert fixed["aggregate"] == grown["aggregate"]
                        assert fixed["mstop"] == grown["mstop"]
