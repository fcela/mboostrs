"""PR 1 scaffolding for the unified modelling procedure (issue #474).

Covers the decision-trail schema, the ``DecisionTrail`` builder, the seeding
streams, the ``fit``/``Fit``/``report`` skeleton, and the export wiring
(shadowing ``mboostrs.report``). Pure Python; nothing here touches the Rust
extension beyond what ``autospec``/``gamboost`` do already.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import mboostrs as mb


def _gaussian_data(n: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.5, size=n)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3}


def _section5_trail() -> dict:
    """The abridged example trail from section 5 of the design note."""
    return {
        "schema_version": "1.0",
        "provenance": {"mboostrs": "0.x", "git_sha": "660f1ce", "seed": 1, "blas_threads": 1},
        "data": {"n": 1254, "response": "y", "columns": {"x1": "numeric", "school": "group"}},
        "folds": {"scheme": "grouped", "K": 5, "R": 1, "hash": "9f2c..."},
        "criterion": "oof_log_score",
        "steps": [
            {
                "step": 2,
                "decision": "family",
                "incumbent": "Gaussian",
                "candidates": [
                    {"spec": "Gaussian", "score": 1.412, "status": "ok"},
                    {"spec": "StudentT", "score": 1.371, "d": -0.041, "se": 0.012, "status": "ok"},
                    {"spec": "GammaLSS", "score": None, "status": "failed",
                     "reason": "PanicException: digamma(0)"},
                ],
                "threshold": 1.38,
                "adopted": "StudentT",
                "rule": "d < -c_m*SE, m=2",
            },
            {
                "step": 5,
                "decision": "scale_equation",
                "incumbent": "sigma ~ 1",
                "candidates": [{"spec": "sigma ~ bbs(x1)", "d": -0.006, "se": 0.009}],
                "adopted": None,
                "reason": "tie: |d| <= SE, kept simpler",
            },
            {
                "step": "7b",
                "decision": "reference_gap",
                "reference": "btree(all, depth=4)",
                "d": 0.003,
                "se": 0.010,
                "verdict": "no_gap",
            },
        ],
        "final": {
            "formula": "y ~ bbs(x1) + bols(school)",
            "family": "StudentT",
            "mstop": 412,
        },
        "uncertainty": {
            "method": "stability",
            "B": 100,
            "pi_thr": 0.75,
            "terms": [{"term": "bbs(x1)", "pi": 0.98}, {"term": "bols(x2)", "pi": 0.41}],
            "expected_false_selections_bound": 0.8,
        },
        "budget": {"setting": "standard", "spent": 1.9e7, "degradations": []},
    }


# -- exports ---------------------------------------------------------------


def test_new_names_are_exported_and_typed():
    assert "fit" in mb.__all__
    assert "report" in mb.__all__
    assert "Fit" in mb.__all__
    assert "DecisionTrail" in mb.__all__
    assert callable(mb.fit)
    assert callable(mb.report)
    assert isinstance(mb.Fit, type)
    assert isinstance(mb.DecisionTrail, type)


def test_report_function_shadows_the_submodule():
    import inspect
    import sys

    assert inspect.isfunction(mb.report)  # the entry point, not the submodule
    assert "mboostrs.report" in sys.modules  # submodule still importable
    from mboostrs.report import ExecutionReport, PathRow  # noqa: PLC0415

    assert PathRow.__name__ == "PathRow"
    assert ExecutionReport.__name__ == "ExecutionReport"


# -- schema ----------------------------------------------------------------


def test_section5_example_validates_against_the_schema():
    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed")
    import pathlib

    schema = json.loads(
        pathlib.Path("docs/schemas/decision_trail.schema.json").read_text()
    )
    jsonschema.validate(_section5_trail(), schema)


def test_schema_pins_the_required_top_level_keys():
    schema = json.load(open("docs/schemas/decision_trail.schema.json"))
    required = set(schema["required"])
    assert {
        "schema_version",
        "provenance",
        "data",
        "folds",
        "criterion",
        "steps",
        "final",
        "uncertainty",
        "budget",
    } <= required


# -- DecisionTrail builder -------------------------------------------------


def test_to_json_from_dict_round_trip():
    data = _gaussian_data()
    trail = mb.DecisionTrail()
    trail.start(data, "y", seed=0)
    trail.add_step(0, "autospec_delegate", adopted="y ~ bbs(x1)", reason="scaffold")
    trail.set_final("y ~ bbs(x1)", "Gaussian", 100)
    trail.set_uncertainty("none")
    trail.set_budget("standard", 0.0, [])
    restored = mb.DecisionTrail.from_dict(trail.to_json())
    assert restored.to_dict() == trail.to_dict()


def test_to_dict_is_json_safe_with_numpy_scalars_and_nan():
    data = {
        "y": np.array([1.5, np.nan, 3.0]),
        "x": np.array([np.int64(7), np.float32(8.5), np.float64(np.inf)]),
    }
    trail = mb.DecisionTrail()
    trail.start(data, "y", seed=0)
    trail.set_final("y ~ bbs(x)", "Gaussian", np.int64(50))
    trail.set_budget("standard", np.float64(np.nan), ["nope"])
    record = trail.to_dict()
    assert record["final"]["mstop"] == 50
    assert isinstance(record["final"]["mstop"], int)
    assert record["budget"]["spent"] is None  # NaN -> null
    dumped = json.dumps(record, allow_nan=False)  # would fail on NaN/Infinity
    rebuilt = json.loads(dumped)
    assert rebuilt == record


# -- seeding ---------------------------------------------------------------


def test_streams_are_deterministic():
    from mboostrs._procedure._seeding import STREAMS, streams

    assert tuple(streams(1).keys()) == STREAMS
    one = {name: rng.random(20) for name, rng in streams(42).items()}
    two = {name: rng.random(20) for name, rng in streams(42).items()}
    for name in STREAMS:
        assert np.array_equal(one[name], two[name])


def test_streams_are_pairwise_independent():
    from mboostrs._procedure._seeding import STREAMS, streams

    rngs = streams(7)
    for left in STREAMS:
        for right in STREAMS:
            if left == right:
                continue
            a = rngs[left].random(64)
            b = rngs[right].random(64)
            assert not np.array_equal(a, b)


def test_blas_threads_reports_a_positive_integer_or_none():
    from mboostrs._procedure._seeding import blas_threads

    value = blas_threads()
    assert value is None or (isinstance(value, int) and value >= 1)


# -- fit / Fit / report ----------------------------------------------------


def test_fit_returns_a_valid_trail_and_predict_matches_the_model():
    data = _gaussian_data()
    fitted = mb.fit(data, "y", seed=0)
    assert isinstance(fitted, mb.Fit)
    assert fitted.trail.validate()
    record = fitted.trail.to_dict()
    assert [s["decision"] for s in record["steps"]] == ["inputs", "autospec_delegate"]
    assert record["final"]["family"] == "Gaussian"
    new = {"x1": data["x1"][:5], "x2": data["x2"][:5], "x3": data["x3"][:5]}
    assert np.allclose(fitted.predict(new), fitted.model.predict(new))
    assert fitted.to_dict() == record


def test_two_fits_with_the_same_seed_give_identical_trail_dicts():
    data = _gaussian_data()
    first = mb.fit(data, "y", seed=11)
    second = mb.fit(data, "y", seed=11)
    assert first.trail.to_dict() == second.trail.to_dict()


def test_uncertainty_other_than_none_raises_not_implemented():
    data = _gaussian_data()
    with pytest.raises(NotImplementedError):
        mb.fit(data, "y", seed=0, uncertainty="stability")
    with pytest.raises(NotImplementedError):
        mb.fit(data, "y", seed=0, uncertainty="bootstrap")


def test_invalid_budget_setting_raises_value_error():
    data = _gaussian_data()
    with pytest.raises(ValueError, match="budget"):
        mb.fit(data, "y", seed=0, budget="exhaustive")


def test_reports_renders_all_formats_without_fitting():
    data = _gaussian_data()
    fitted = mb.fit(data, "y", seed=0)

    text = mb.report(fitted, format="text")
    assert isinstance(text, str)
    assert "mboostrs fit:" in text and "mstop:" in text and "autospec_delegate" in text
    assert "not computed" in text  # uncertainty is none

    markdown = mb.report(fitted, format="markdown")
    assert isinstance(markdown, str)
    assert markdown.startswith("#")
    assert "autospec_delegate" in markdown

    as_json = mb.report(fitted, format="json")
    assert isinstance(as_json, dict)
    assert as_json == fitted.trail.to_dict()

    # a bare DecisionTrail renders too, and nothing fits along the way
    assert mb.report(fitted.trail) == mb.report(fitted, format="text")

    with pytest.raises(ValueError, match="format"):
        mb.report(fitted, format="xml")


def test_reports_default_is_text():
    data = _gaussian_data()
    fitted = mb.fit(data, "y", seed=0)
    assert mb.report(fitted) == mb.report(fitted, format="text")

# -- stage 0: inputs (docs/design/procedure-stages.md, first PR) ----------


def test_inputs_stage_records_response_hygiene_and_missing_rows():
    data = _gaussian_data()
    data["const"] = np.ones_like(data["y"])
    data["x1"] = data["x1"].copy()
    data["x1"][:3] = np.nan
    fitted = mb.fit(data, "y", seed=0)
    assert fitted.trail.validate()
    step = fitted.trail.to_dict()["steps"][0]
    assert (step["step"], step["stage"], step["fits"]) == (0, "inputs", 0)
    assert step["adopted"] == "Gaussian" and step["rule"] == "inferred"
    evidence = step["evidence"]
    assert evidence["response"]["kind"] == "continuous"
    assert evidence["incomplete_rows"] == 3
    assert any(f["column"] == "const" and f["kind"] == "constant"
               for f in evidence["hygiene"])


def test_inputs_stage_records_a_declared_family_and_groups():
    data = _gaussian_data()
    data["g"] = np.repeat(np.arange(10), 20)
    fitted = mb.fit(data, "y", family=mb.Gaussian(), groups="g", seed=0)
    step = fitted.trail.to_dict()["steps"][0]
    assert step["rule"] == "declared"
    assert step["evidence"]["groups"] == ["g"]
    assert json.loads(fitted.trail.to_json())["steps"][0] == step
