"""Verify the schema-only repair restores reports without changing predictions."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from benches.bench_factor_reference import CASES, DIRECTORY, data
from benches.benchmark_contract import data_digest


def assess(before, after):
    expected = {(case, seed) for case in CASES for seed in range(6)}
    documents = []
    for document in (before, after):
        rows = {}
        for row in document["results"]:
            key = (row["case"], row["seed"]) if "case" in row else tuple(row["job"])
            if key in rows or key not in expected:
                raise ValueError("duplicate or unexpected control")
            rows[key] = row
        if set(rows) != expected:
            raise ValueError("missing planned control")
        documents.append(rows)
    if before["protocol"]["document_sha256"] != after["protocol"]["document_sha256"]:
        raise ValueError("control protocol changed")
    paired, recovered, failures = [], [], []
    for key in sorted(expected):
        b, a = (rows[key] for rows in documents)
        if a["_status"] != "success":
            failures.append({"case": key[0], "seed": key[1], "status": a["_status"]})
            continue
        train, test = data(*key)
        expected_data = data_digest([train.to_dict("list"), test.to_dict("list")])
        if a["input_digest"] != expected_data:
            raise ValueError("replayed control rows changed")
        if b["_status"] == "success":
            if b["input_digest"] != a["input_digest"] or b["prediction_digest"] != a["prediction_digest"] or b["mse"] != a["mse"]:
                raise ValueError("schema-only change altered an existing prediction")
            paired.append({"case": key[0], "seed": key[1], "prediction_unchanged": True})
        else:
            recovered.append({"case": key[0], "seed": key[1], "before": b["_status"], "after": "success", "after_mse": a["mse"]})
    return {"planned_per_phase": len(expected), "unchanged_successful_predictions": paired,
            "recovered_failures": recovered, "after_failures": failures,
            "schema_replay_passed": not failures,
            "original_factor_reference_confirmation_gate_passed": len(paired) == len(expected) and not failures,
            "scope": "Schema-only replay, not promotion of the separate factor-reference dictionary policy."}


def main():
    before, after = (json.loads((DIRECTORY / f"{phase}.json").read_text()) for phase in ("before", "after"))
    summary = assess(before, after)
    summary.update(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   before_sha256=hashlib.sha256((DIRECTORY / "before.json").read_bytes()).hexdigest(),
                   after_sha256=hashlib.sha256((DIRECTORY / "after.json").read_bytes()).hexdigest())
    (DIRECTORY / "schema_replay.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
