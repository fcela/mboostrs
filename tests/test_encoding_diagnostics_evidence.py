"""Observed grids and exact-prefix claims require complete runtime evidence."""
import copy

import pytest

from benches.bench_encoding_tracks import ENCODINGS
from benches.report_encoding_diagnostics import cap_contract, conditioning


@pytest.fixture
def contracts():
    low = [{"task": t, "seed": s, "method": m, "_status": "success", "mse": 1.,
            "input_digest": f"input-{t}-{s}", "latent_digest": f"latent-{s}",
            "prediction_digest": f"pred-{t}-{s}-{m}"}
           for t in ENCODINGS for s in range(8) for m in ("mboostrs_known", "mboostrs_discover")]
    before, after = [], []
    for seed in range(8):
        common = {"task": "shifted", "seed": seed, "method": "mboostrs_discover", "_status": "success",
                  "input_digest": f"input-shifted-{seed}", "formula": "bbs(x, by=g)", "mse": 1.}
        before.append({**common, "mstop": 10000,
                       "tuning_stages": [{"declared_grid": 12800, "actual_largest": 10000, "public_maximum": 10000}]})
        after.append({**common, "mstop": 11000,
                      "tuning_stages": [{"declared_grid": 12800, "actual_largest": 12800, "public_maximum": 25600}]})
    return {"results": before}, {"results": after}, {"results": copy.deepcopy(low)}, {"results": low}


def test_correct_grid_and_all_eighty_original_replays_are_required(contracts):
    result = cap_contract(*contracts)
    assert result["passed"] and len(result["low_cap_exact_replays"]) == 80
    assert len(result["high_rows"]) == 8 and all(r["before_mismatch"] for r in result["high_rows"])


@pytest.mark.parametrize("mutation", ["empty", "low_prediction", "missing"])
def test_incomplete_or_changed_contract_evidence_is_rejected(contracts, mutation):
    if mutation == "empty":
        contracts[1]["results"][0]["tuning_stages"] = []
    elif mutation == "low_prediction":
        contracts[2]["results"][0]["prediction_digest"] = "changed"
    else:
        contracts[1]["results"].pop()
    with pytest.raises(ValueError):
        cap_contract(*contracts)


def test_a_still_clipped_after_grid_cannot_pass(contracts):
    contracts[1]["results"][0]["tuning_stages"][0]["actual_largest"] = 10000
    assert not cap_contract(*contracts)["passed"]


def test_prefix_replay_and_positive_path_minima_are_verified():
    rows, reference, gaps = [], [], []
    for task in ("indicator", "shifted"):
        gaps.append({"task": task, "best_competitor": "catboost"})
        for seed in range(8):
            common = {"task": task, "seed": seed, "_status": "success", "input_digest": f"i-{task}-{seed}", "latent_digest": f"l-{seed}"}
            choices = [{"pool": p, "mstop": 2, "validation_mse": .5, "mse": 1., "prediction_digest": "same"}
                       for p in ("prefix1600", "longer_search")]
            rows.append({**common, "largest": 4, "risk": [2., 1., .5, 1., 2.], "choices": choices})
            reference.extend([{**common, "method": "mboostrs_known", "prediction_digest": "same", "mse": 1.},
                              {**common, "method": "catboost", "mse": 2.}])
    result = conditioning({"results": rows}, {"results": reference}, {"gaps": gaps})
    assert len(result["exact_prefix_replays"]) == 16
    rows[0]["choices"][0]["prediction_digest"] = "different"
    with pytest.raises(ValueError, match="replay"):
        conditioning({"results": rows}, {"results": reference}, {"gaps": gaps})
