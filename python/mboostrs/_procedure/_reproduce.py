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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mboostrs._procedure._fit import Fit, fit
from mboostrs._procedure._report import report as _render_report
from mboostrs._procedure._trail import DecisionTrail
from mboostrs._provenance import fingerprint, provenance

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
    data_verified: bool | None = None
    environment_matches: bool | None = None
    environment_differences: dict[str, Any] = field(default_factory=dict)

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
            "data_verified": self.data_verified,
            "environment_matches": self.environment_matches,
            "environment_differences": dict(self.environment_differences),
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


def _check_data_lock(
    recorded: dict[str, Any], data: Any
) -> tuple[bool | None, list[str]]:
    """Verify supplied ``data`` against recorded ``data.schema`` and ``data.sha256``."""
    data_block = recorded.get("data", {})
    recorded_sha = data_block.get("sha256")
    recorded_schema = data_block.get("schema")
    if recorded_sha is None and recorded_schema is None:
        return None, []
    fp_now = fingerprint(data)
    schema_now = {col["name"]: col["dtype"] for col in fp_now["columns"]}
    diffs: list[str] = []
    if recorded_schema is not None and recorded_schema != schema_now:
        diffs.append("data.schema")
    if recorded_sha is not None and recorded_sha != fp_now["sha256"]:
        diffs.append("data.sha256")
    if recorded_sha is None:
        return (False if diffs else None), diffs
    return (len(diffs) == 0), diffs


def _check_environment(
    recorded: dict[str, Any], data: Any, seed: int
) -> tuple[bool | None, dict[str, Any]]:
    """Compare recorded ``provenance.environment`` against current runtime."""
    recorded_env = recorded.get("provenance", {}).get("environment")
    if not isinstance(recorded_env, dict):
        return None, {}
    prov_now = provenance(data, seed=seed)
    current_env = {
        "python": prov_now["python_version"],
        "platform": prov_now["platform"],
        "packages": dict(prov_now["packages"]),
    }
    env_diffs: dict[str, Any] = {}
    for key in ("python", "platform"):
        if key in recorded_env and recorded_env[key] != current_env[key]:
            env_diffs[key] = {
                "recorded": recorded_env[key],
                "current": current_env[key],
            }
    if isinstance(recorded_env.get("packages"), dict):
        rec_pkgs = recorded_env["packages"]
        cur_pkgs = current_env["packages"]
        pkg_diffs = {
            pkg: {"recorded": rec_pkgs.get(pkg), "current": cur_pkgs.get(pkg)}
            for pkg in sorted(set(rec_pkgs) | set(cur_pkgs))
            if rec_pkgs.get(pkg) != cur_pkgs.get(pkg)
        }
        if pkg_diffs:
            env_diffs["packages"] = pkg_diffs
    return (len(env_diffs) == 0), env_diffs


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
    time_col = recorded.get("folds", {}).get("time_column")
    if isinstance(time_col, str) and time_col:
        kwargs["time"] = time_col
    unc_method = recorded.get("uncertainty", {}).get("method")
    if unc_method in ("stability", "bootstrap", "conformal"):
        kwargs["uncertainty"] = unc_method
    kwargs.update(overrides)

    reproduced_fit = fit(data, response, **kwargs)
    reproduced_dict = reproduced_fit.trail.to_dict()
    rendered = _render_report(reproduced_fit, format=report_format)
    if isinstance(rendered, dict):
        rendered = json.dumps(rendered, indent=2, sort_keys=True)
    digest = _prediction_digest(reproduced_fit.model, data)
    data_verified, data_diffs = _check_data_lock(recorded, data)
    env_matches, env_diffs = _check_environment(recorded, data, seed)
    diffs = (*_compare_trails(recorded, reproduced_dict), *data_diffs) if verify else ()
    return ReproductionResult(
        fit=reproduced_fit,
        report=rendered,
        matched=(len(diffs) == 0),
        differences=diffs,
        prediction_sha256=digest,
        original_trail=recorded,
        reproduced_trail=reproduced_dict,
        data_verified=data_verified,
        environment_matches=env_matches,
        environment_differences=env_diffs,
    )
