"""New measurements carry build, dataset and completion-time identities."""

from pathlib import Path

import pytest

from benches import _quality_provenance as provenance


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(provenance, "machine_metadata", lambda: {
        "commit": "test-commit", "versions": {}, "threads": {"workers": 1},
        "platform": {"machine": "test"}, "native_build": {"build_id": "test-build"},
    })
    source = Path(__file__).resolve().parents[1] / "benches/bench_gamma.py"
    provenance.begin_run(source, lambda seed, n, task: [seed, n, task], 10, 20, ["linear"])
    return provenance.result_rows()


def test_each_new_row_records_inputs_outputs_and_build(run):
    row = {"task": "linear", "seed": 0, "method": "fit", "seconds": 0.1,
           "own_nll": float("nan")}
    run.append(row)
    identity = row["_identity"]
    assert identity["fingerprint"] and identity["input_digest"] and identity["output_digest"]
    assert identity["measured_at"] and identity["native_build_id"] == "test-build"
    run.append({"task": "linear", "seed": 1, "method": "fit", "seconds": 0.3})
    assert run[1]["_identity"]["input_digest"] != identity["input_digest"]
    document = {"design": {"seeds": [0, 1]}, "results": run}
    provenance.attach_provenance(document)
    meta = document["provenance"]
    assert meta["commit"] == "test-commit"
    assert meta["uncertainty"]["groups"][0]["seconds_se"] == pytest.approx(0.1)
    assert meta["repeats"]["training_seeds"] == [0, 1]


def test_extending_a_result_list_preserves_original_identity(run):
    row = {"task": "linear", "seed": 0, "method": "fit", "seconds": 0.1}
    run.append(row)
    original = dict(row["_identity"])
    collected = provenance.result_rows()
    collected.extend(run)
    assert collected[0]["_identity"] == original


def test_seed_offsets_and_independent_test_draws(run):
    source = Path(__file__).resolve().parents[1] / "benches/bench_zero_inflated.py"
    calls = []
    def simulate(seed, n, task):
        calls.append((seed, n, task))
        return [seed, n, task]
    provenance.begin_run(source, simulate, 10, 20, ["linear"],
                          train_seed_offset=10_000, test_seed_offset=80_000, test_shared=False)
    rows = provenance.result_rows()
    rows.append({"task": "linear", "seed": 3, "method": "fit", "seconds": 0.1})
    assert calls == [(10_003, 10, "linear"), (80_003, 20, "linear")]
    document = {"results": rows}
    provenance.attach_provenance(document)
    assert "per seed/task" in document["provenance"]["repeats"]["test"]
