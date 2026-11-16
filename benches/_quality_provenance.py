"""Provenance for seeded comparisons of different statistical methods.

These runs measure predictive quality and descriptive elapsed time. They are
not paired timing trials of equivalent implementations. Metadata is captured
while the run executes; historical rows are never stamped retrospectively.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, UTC
from pathlib import Path

import numpy as np

from benches.benchmark_contract import data_digest, machine_metadata

_RUN = ContextVar("quality_benchmark_run", default=None)


def _now():
    return datetime.now(UTC).isoformat()


def _serializable(value):
    if hasattr(value, "to_numpy") and hasattr(value, "columns"):
        return {"columns": list(value.columns), "data": value.to_numpy()}
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


class _Run:
    def __init__(self, source, simulate, n_train, n_test, tasks,
                 train_seed_offset=0, test_seed_offset=10_000, test_shared=True):
        self.created_at = _now()
        self.metadata = machine_metadata()
        for name in ("scipy", "statsmodels", "scikit-learn", "polars", "patsy"):
            try:
                self.metadata["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.metadata["versions"][name] = None
        self.source = str(Path(source).resolve().relative_to(Path(__file__).resolve().parents[1]))
        self.source_digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        self.simulate = simulate
        self.n_train, self.n_test, self.tasks = n_train, n_test, tuple(tasks)
        self.train_seed_offset, self.test_seed_offset = train_seed_offset, test_seed_offset
        self.test_shared = test_shared
        self.input_digests = {}

    def stamp(self, row):
        key = (row["task"], int(row["seed"]))
        if key not in self.input_digests:
            task, seed = key
            # Recreate the deterministic simulation outside the timed region
            # to fingerprint exactly the data and truth arrays used by the run.
            test_seed = self.test_seed_offset + (self.tasks.index(task) if self.test_shared else seed)
            inputs = (self.simulate(self.train_seed_offset + seed, self.n_train, task),
                      self.simulate(test_seed, self.n_test, task))
            self.input_digests[key] = data_digest(_serializable(inputs))
        identity = {
            "created_at": self.created_at, "metadata": self.metadata,
            "runner_digest": self.source_digest, "input_digest": self.input_digests[key],
            "task": key[0], "seed": key[1], "method": row["method"],
        }
        row["_identity"] = {
            "fingerprint": data_digest(identity), "measured_at": _now(),
            "time_scope": "completion of the scored result row, after its timed operation",
            "input_digest": self.input_digests[key],
            "output_digest": data_digest(_serializable(row)),
            "native_build_id": self.metadata["native_build"]["build_id"],
        }


class _Results(list):
    def append(self, row):
        run = _RUN.get()
        if run is not None and "_identity" not in row:
            run.stamp(row)
        super().append(row)

    def extend(self, rows):
        for row in rows:
            self.append(row)


def begin_run(source, simulate, n_train, n_test, tasks, *,
              train_seed_offset=0, test_seed_offset=10_000, test_shared=True):
    """Capture the actual build/environment before any method is timed."""
    _RUN.set(_Run(source, simulate, n_train, n_test, tasks,
                  train_seed_offset, test_seed_offset, test_shared))


def result_rows():
    """A list that records each result's identity when it is appended."""
    return _Results()


def attach_provenance(document):
    """Record the protocol and observed spread of the newly measured rows."""
    run = _RUN.get()
    if run is None:
        raise RuntimeError("begin_run must precede a benchmark publication")
    groups = defaultdict(list)
    for row in document["results"]:
        groups[(row["task"], row["method"])].append(row)
    uncertainty = []
    for (task, method), rows in groups.items():
        values = [row["seconds"] for row in rows if math.isfinite(row.get("seconds", math.nan))]
        uncertainty.append({"task": task, "method": method, "n": len(values),
            "seconds_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "seconds_se": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None})
    document["provenance"] = {
        **run.metadata, "created_at": run.created_at,
        "runner": run.source, "runner_sha256": run.source_digest,
        "protocol": {
            "kind": "seeded predictive-method comparison",
            "design": document.get("design", {key: document[key] for key in (
                "n_train", "n_test", "seeds", "nu", "folds", "mstop_cap", "mstop_extend", "tasks"
            ) if key in document}),
            "timing": "runner perf_counter regions; method-specific fit/tuning/prediction work",
            "order": "sequential task, seed, method order in the recorded runner",
            "timing_claim": "descriptive elapsed times; no paired equal-work speedup or timing interval",
            "warmup": "no dedicated warmup; imports and provenance collection excluded from timers",
        },
        "repeats": {"training_seeds": sorted({int(row["seed"]) for row in document["results"]}),
                    "test": ("one independently generated test sample per task, shared across methods and seeds"
                             if run.test_shared else
                             "one independently generated test sample per seed/task, shared across methods")},
        "uncertainty": {"scope": "descriptive timing spread across training seeds, not repeated identical-input trials",
                        "groups": uncertainty},
    }
