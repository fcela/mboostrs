"""Small evidence helpers shared by the book's comparative studies."""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from benches.benchmark_contract import data_digest, machine_metadata


def gaussian_scores(y, mu, sigma):
    """Proper Gaussian scores; sigma is standard deviation, never variance."""
    y, mu, sigma = (np.asarray(a, dtype=float) for a in (y, mu, sigma))
    if y.ndim != 1 or y.shape != mu.shape or y.shape != sigma.shape:
        raise ValueError("observations, means and scales must be aligned vectors")
    if not all(np.isfinite(a).all() for a in (y, mu, sigma)) or np.any(sigma <= 0):
        raise ValueError("Gaussian predictions must be finite with positive scale")
    z = (y - mu) / sigma
    log_score = .5 * np.log(2 * np.pi) + np.log(sigma) + .5 * z**2
    crps = sigma * (z * (2 * ndtr(z) - 1)
                    + 2 * np.exp(-.5 * z**2) / np.sqrt(2 * np.pi)
                    - 1 / np.sqrt(np.pi))
    return {"log_score": log_score, "crps": crps, "squared_error": (y - mu)**2,
            "covered_90": np.abs(z) <= 1.6448536269514722,
            "width_90": 2 * 1.6448536269514722 * sigma}


DETERMINISTIC_UNCERTAINTY = {
    "unit": "fixed input fixtures shared across configurations; no independent sampling replicates",
    "summary": "numerical agreement with fixed references; no sampling standard error",
    "timing": "diagnostic elapsed times are descriptive; no controlled speed comparison",
}


def evidence_document(protocol, repeats, scripts, *, uncertainty=None):
    """Record source/native identities and the study's uncertainty contract.

    The historical default describes independent simulation draws. Deterministic
    numerical audits must pass their actual contract, for example
    ``uncertainty=DETERMINISTIC_UNCERTAINTY``. Copy the supplied mapping so later
    mutation of a caller's settings cannot change recorded evidence.
    """
    import mboostrs as mb

    metadata = machine_metadata()
    for name in ("scipy", "pandas", "ngboost", "mssm", "scikit-learn", "catboost", "lightgbm"):
        try:
            metadata["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metadata["versions"][name] = None
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        **{key: metadata[key] for key in
           ("commit", "working_tree", "platform", "versions", "threads")},
        "build_info": dict(mb.build_info()),
        "scripts": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in scripts},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "protocol": protocol, "repeats": repeats,
        "uncertainty": dict(uncertainty) if uncertainty is not None else {
            "unit": "one independent training and test draw; pair methods within a draw",
            "summary": "mean and Monte Carlo standard error across draws",
            "timing": "elapsed whole-workflow costs, not controlled implementation speed ratios",
        },
        "memory": {"status": "not measured"},
        "results": [],
    }


def write_evidence(document, path):
    """Checkpoint complete rows without allowing nonstandard NaN JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")
    temporary.replace(target)


__all__ = ["DETERMINISTIC_UNCERTAINTY", "data_digest", "evidence_document", "gaussian_scores", "write_evidence"]
