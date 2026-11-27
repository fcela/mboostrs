"""One-command reproduction of a ``fit()`` decision trail and discovery report.

Implements D10 Anchor 4: given a saved :class:`DecisionTrail` (file path, JSON
string, dict, ``DecisionTrail``, or ``Fit``) and the dataset, :func:`reproduce`
replays the exact modelling procedure (same response, seed, budget, and
declared structure), renders the complete discovery report, computes the
prediction SHA-256 digest, and checks whether the reproduced final
specification matches the recorded trail.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mboostrs._procedure._fit import Fit, fit
from mboostrs._procedure._report import report as _render_report
from mboostrs._procedure._trail import DecisionTrail

__all__ = ["ReproductionResult", "reproduce"]


def _load_trail_dict(source: Any) -> dict[str, Any]:
    if isinstance(source, Fit):
        return source.trail.to_dict()
    if isinstance(source, DecisionTrail):
        return source.to_dict()
    if isinstance(source, dict):
        return DecisionTrail.from_dict(source).to_dict()
    if isinstance(source, Path):
        return DecisionTrail.from_dict(source.read_text(encoding="utf-8")).to_dict()
    if isinstance(source, str):
        stripped = source.strip()
        if stripped.startswith("{"):
            return DecisionTrail.from_dict(stripped).to_dict()
        path = Path(source)
        if path.is_file():
            return DecisionTrail.from_dict(path.read_text(encoding="utf-8")).to_dict()
        raise FileNotFoundError(f"decision trail file not found: {source!r}")
    raise TypeError(
        "reproduce() expects a Fit, DecisionTrail, dict, JSON string, or path "
        f"to a trail JSON file; got {type(source).__name__}"
    )


def _prediction_digest(model: Any, data: Any) -> str:
    """Stable SHA-256 of the fitted model's response-scale predictions on ``data``."""
    pred = model.predict(data, type="response")
    if isinstance(pred, dict):
        pred = model.predict(data, type="mean")
    arr = np.ascontiguousarray(np.asarray(pred, dtype=np.float64).ravel())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _compare_trails(original: dict[str, Any], reproduced: dict[str, Any]) -> tuple[str, ...]:
    diffs: list[str] = []
    orig_final = original.get("final", {})
    repr_final = reproduced.get("final", {})
    for key in ("formula", "family", "mstop"):
        if orig_final.get(key) != repr_final.get(key):
            diffs.append(
                f"final.{key}: recorded={orig_final.get(key)!r} != "
                f"reproduced={repr_final.get(key)!r}"
            )
    orig_n = original.get("data", {}).get("n")
    repr_n = reproduced.get("data", {}).get("n")
    if orig_n is not None and repr_n is not None and orig_n != repr_n:
        diffs.append(f"data.n: recorded={orig_n} != reproduced={repr_n}")
    return tuple(diffs)


@dataclass(frozen=True)
class ReproductionResult:
    """Outcome of replaying a saved :class:`DecisionTrail` on a dataset."""

    fit: Fit
    report: str
    matched: bool
    differences: tuple[str, ...]
    prediction_sha256: str
    original_trail: dict[str, Any]
    reproduced_trail: dict[str, Any]

    @property
    def model(self) -> Any:
        return self.fit.model

    @property
    def trail(self) -> DecisionTrail:
        return self.fit.trail

    def predict(self, newdata: Any, **kwargs: Any) -> Any:
        return self.fit.predict(newdata, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "differences": list(self.differences),
            "prediction_sha256": self.prediction_sha256,
            "original_final": dict(self.original_trail.get("final", {})),
            "reproduced_final": dict(self.reproduced_trail.get("final", {})),
            "reproduced_trail": self.reproduced_trail,
        }

    def __repr__(self) -> str:
        status = "matched" if self.matched else f"MISMATCH ({len(self.differences)})"
        return (
            f"ReproductionResult(status={status!r}, "
            f"sha256={self.prediction_sha256[:12]!r}, fit={self.fit!r})"
        )


def reproduce(
    trail: Fit | DecisionTrail | dict[str, Any] | str | Path,
    data: Any,
    *,
    report_format: str = "markdown",
    verify: bool = True,
    **overrides: Any,
) -> ReproductionResult:
    """Reproduce a recorded :func:`fit` and its discovery report in one call.

    Parameters
    ----------
    trail : Fit, DecisionTrail, dict, str, or Path
        The recorded decision trail (or a ``Fit`` holding one, or a path to a
        saved ``trail.json`` file).
    data : DataFrame or dict
        The dataset to fit on.
    report_format : {"markdown", "text", "json"}, default "markdown"
        Format for the reproduced discovery report stored in ``result.report``.
    verify : bool, default True
        When True, compare the reproduced final specification against the
        recorded trail and populate ``matched`` and ``differences``.
    **overrides
        Optional keyword arguments forwarded to :func:`fit` (overriding values
        read from the trail).
    """
    recorded = _load_trail_dict(trail)
    response = recorded["data"]["response"]
    seed = int(recorded["provenance"]["seed"])
    budget = str(recorded.get("budget", {}).get("setting", "standard"))

    # Restore declared grouping columns when the recorded trail typed any as "group".
    cols = recorded.get("data", {}).get("columns", {})
    group_cols = [c for c, k in cols.items() if k == "group"]
    kwargs: dict[str, Any] = {"seed": seed, "budget": budget}
    if group_cols:
        kwargs["groups"] = group_cols[0] if len(group_cols) == 1 else tuple(group_cols)
    kwargs.update(overrides)

    reproduced_fit = fit(data, response, **kwargs)
    reproduced_dict = reproduced_fit.trail.to_dict()
    rendered = _render_report(reproduced_fit, format=report_format)
    if isinstance(rendered, dict):
        rendered = json.dumps(rendered, indent=2, sort_keys=True)
    digest = _prediction_digest(reproduced_fit.model, data)
    diffs = _compare_trails(recorded, reproduced_dict) if verify else ()
    return ReproductionResult(
        fit=reproduced_fit,
        report=rendered,
        matched=(len(diffs) == 0),
        differences=diffs,
        prediction_sha256=digest,
        original_trail=recorded,
        reproduced_trail=reproduced_dict,
    )
