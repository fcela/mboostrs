"""The decision trail: a JSON-serialisable record of one ``fit``.

The schema lives in ``docs/schemas/decision_trail.schema.json`` and is
described in section 5 of ``docs/design/unified-procedure.md``. This module is
the builder: ``start`` opens a trail with the data, folds and provenance,
``add_step`` appends a decision, and ``set_final``/``set_uncertainty``/
``set_budget`` close it. ``to_dict`` returns plain Python types only -- NumPy
scalars become Python numbers, and NaN/infinity become ``None`` -- so
``json.dumps`` never has to know about NumPy and never emits the non-standard
``NaN``/``Infinity`` tokens.

``validate`` checks the record against the schema with ``jsonschema`` when it
is importable, and otherwise with a small structural check that verifies the
required keys and their types. The built-in check is deliberately weaker than
the schema; it exists so the package does not gain a hard ``jsonschema``
dependency, not so the two agree.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SCHEMA_VERSION", "DecisionTrail"]

SCHEMA_VERSION = "1.0"
CRITERION = "oof_log_score"

_REQUIRED_TOP = (
    "schema_version",
    "provenance",
    "data",
    "folds",
    "criterion",
    "steps",
    "final",
    "uncertainty",
    "budget",
)

def _jsonable(value: Any) -> Any:
    """Plain-Python, JSON-safe copy of ``value``.

    NumPy scalars become their Python equivalents, arrays and tuples become
    lists, mappings become ``str``-keyed dicts, and a non-finite float becomes
    ``None``. The conversion is idempotent.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


@lru_cache(maxsize=1)
def _schema_path() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "docs" / "schemas" / "decision_trail.schema.json"
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any] | None:
    path = _schema_path()
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else None


def _structural_check(record: dict[str, Any]) -> None:
    """Required keys and types, without ``jsonschema``."""
    missing = [key for key in _REQUIRED_TOP if key not in record]
    if missing:
        raise ValueError(f"decision trail is missing required keys: {missing}")
    expected = {
        "schema_version": str,
        "provenance": dict,
        "data": dict,
        "folds": dict,
        "criterion": str,
        "steps": list,
        "final": dict,
        "uncertainty": dict,
        "budget": dict,
    }
    for key, kind in expected.items():
        if not isinstance(record[key], kind):
            raise ValueError(
                f"decision trail field {key!r} must be {kind.__name__}, "
                f"got {type(record[key]).__name__}"
            )
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported decision trail schema_version "
            f"{record['schema_version']!r}, expected {SCHEMA_VERSION!r}"
        )
    for index, step in enumerate(record["steps"]):
        if not isinstance(step, dict):
            raise ValueError(f"decision trail step {index} is not an object")
        for key in ("step", "decision"):
            if key not in step:
                raise ValueError(f"decision trail step {index} is missing {key!r}")
    budget = record["budget"]
    if budget.get("setting") not in ("quick", "standard", "thorough"):
        raise ValueError(f"invalid budget setting {budget.get('setting')!r}")
    if not isinstance(budget.get("degradations"), list):
        raise ValueError("budget.degradations must be a list")
    uncertainty = record["uncertainty"]
    if uncertainty.get("method") not in ("stability", "bootstrap", "conformal", "none"):
        raise ValueError(f"invalid uncertainty method {uncertainty.get('method')!r}")


def summarize_data(data: Any, response: str, *, column_kinds: dict[str, str] | None = None) -> dict[str, Any]:
    """The ``data`` record: row count, response name, and column kinds."""
    from mboostrs._frame import column_names, read_column

    names = list(column_names(data))
    kinds: dict[str, str] = {}
    for name in names:
        if column_kinds is not None and name in column_kinds:
            kinds[name] = column_kinds[name]
            continue
        try:
            kinds[name] = read_column(data, name).kind
        except Exception:  # noqa: BLE001 - a trail records, it does not refuse
            kinds[name] = "unknown"
    n = _row_count(data, names)
    return {"n": n, "response": str(response), "columns": kinds}


def _row_count(data: Any, names: list[str]) -> int:
    from mboostrs._frame import read_column

    for name in names:
        try:
            return int(read_column(data, name).values.shape[0])
        except Exception:  # noqa: BLE001
            continue
    try:
        return int(len(data))
    except Exception:  # noqa: BLE001
        return 0


def _fold_hash(folds: Any) -> str:
    """A stable digest of a fold assignment, for ``folds.hash``."""
    digest = hashlib.sha256()
    if hasattr(folds, "weights"):
        weights = np.asarray(folds.weights, dtype=np.float64)
        digest.update(weights.tobytes())
    else:
        try:
            digest.update(np.asarray(folds, dtype=np.float64).tobytes())
        except Exception:  # noqa: BLE001
            digest.update(repr(folds).encode("utf-8"))
    return digest.hexdigest()[:16]


def _folds_info(folds: Any, seed: int, n: int) -> dict[str, Any]:
    """The ``folds`` record: scheme, K, R and a hash, from a scheme or a matrix."""
    if folds is None:
        digest = hashlib.sha256(f"kfold:{int(seed)}:{int(n)}".encode())
        return {"scheme": "kfold", "K": 5, "R": 2, "hash": digest.hexdigest()[:16]}
    scheme = str(getattr(folds, "scheme", "cv_weights"))
    try:
        weights = np.asarray(folds)
        k = int(weights.shape[1]) if weights.ndim > 1 else 1
    except Exception:  # noqa: BLE001
        k = 0
    r = int(getattr(folds, "repeats", 1) or 1)
    return {"scheme": scheme, "K": k, "R": r, "hash": _fold_hash(folds)}


@dataclass
class DecisionTrail:
    """A builder for the record in section 5 of the design note."""

    schema_version: str = SCHEMA_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    folds: dict[str, Any] = field(default_factory=dict)
    criterion: str = CRITERION
    steps: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(
        default_factory=lambda: {"formula": "", "family": "", "mstop": 0}
    )
    uncertainty: dict[str, Any] = field(
        default_factory=lambda: {
            "method": "none",
            "B": None,
            "pi_thr": None,
            "terms": [],
            "expected_false_selections_bound": None,
        }
    )
    budget: dict[str, Any] = field(
        default_factory=lambda: {
            "setting": "standard",
            "spent": 0.0,
            "degradations": [],
        }
    )

    # -- builder -------------------------------------------------------
    def start(
        self,
        data: Any,
        response: str,
        seed: int,
        folds_info: dict[str, Any] | None = None,
        *,
        column_kinds: dict[str, str] | None = None,
        folds: Any = None,
    ) -> DecisionTrail:
        """Open a trail with the data, folds and provenance.

        ``folds_info`` is used as given; when it is ``None`` the record is
        derived from ``folds`` (a :class:`~mboostrs.folds.FoldScheme`, a weight
        matrix, or ``None`` for the default scheme), which is what ``fit``
        does.
        """
        from mboostrs import __version__
        from mboostrs._procedure._seeding import blas_threads

        try:
            from mboostrs.build import build_info

            git_sha = build_info().get("git_revision")
        except Exception:  # noqa: BLE001
            git_sha = None

        self.provenance = {
            "mboostrs": __version__,
            "git_sha": None if git_sha is None else str(git_sha),
            "seed": int(seed),
            "blas_threads": blas_threads(),
        }
        self.data = summarize_data(data, response, column_kinds=column_kinds)
        if folds_info is None:
            folds_info = _folds_info(folds, seed, self.data.get("n", 0))
        self.folds = dict(folds_info)
        self.criterion = CRITERION
        self.steps = []
        return self

    def add_step(self, step: int | str, decision: str, **fields: Any) -> DecisionTrail:
        """Append one decision. Extra keyword arguments become its fields."""
        record: dict[str, Any] = {"step": step, "decision": str(decision)}
        record.update(fields)
        self.steps.append(record)
        return self

    def set_final(self, formula: str, family: Any, mstop: int,
                  unseen_levels: str | None = None) -> DecisionTrail:
        """Record the final specification, and the model's unseen-level policy
        when it is not the library default ``"error"``."""
        if not isinstance(family, str):
            family = type(family).__name__
        self.final = {"formula": str(formula), "family": family, "mstop": int(mstop)}
        if unseen_levels is not None:
            self.final["unseen_levels"] = str(unseen_levels)
        return self

    def set_uncertainty(self, method: str = "none", **fields: Any) -> DecisionTrail:
        record: dict[str, Any] = {
            "method": method,
            "B": fields.pop("B", None),
            "pi_thr": fields.pop("pi_thr", None),
            "terms": list(fields.pop("terms", [])),
            "expected_false_selections_bound": fields.pop(
                "expected_false_selections_bound", None
            ),
        }
        record.update(fields)
        self.uncertainty = record
        return self

    def set_budget(
        self, setting: str = "standard", spent: float = 0.0,
        degradations: Any = (), limit: int | None = None,
        seconds_limit: float | None = None,
    ) -> DecisionTrail:
        """Record the budget setting, its fit and time limits and the fits spent."""
        self.budget = {
            "setting": setting,
            "limit": None if limit is None else int(limit),
            "seconds_limit": None if seconds_limit is None else float(seconds_limit),
            "spent": float(spent),
            "degradations": [str(item) for item in degradations],
        }
        return self

    # -- serialisation -------------------------------------------------
    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """The trail as plain Python types, JSON-safe (NaN/inf become null)."""
        del include_arrays  # accepted for the shared to_dict() convention
        return _jsonable(
            {
                "schema_version": self.schema_version,
                "provenance": self.provenance,
                "data": self.data,
                "folds": self.folds,
                "criterion": self.criterion,
                "steps": self.steps,
                "final": self.final,
                "uncertainty": self.uncertainty,
                "budget": self.budget,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | str) -> DecisionTrail:
        """Rebuild a trail from :meth:`to_dict` output or a JSON string."""
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise TypeError(
                f"decision trail must be a mapping or JSON object, "
                f"got {type(value).__name__}"
            )
        return cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            provenance=dict(value.get("provenance", {})),
            data=dict(value.get("data", {})),
            folds=dict(value.get("folds", {})),
            criterion=value.get("criterion", CRITERION),
            steps=[dict(step) for step in value.get("steps", [])],
            final=dict(value.get("final", {})),
            uncertainty=dict(value.get("uncertainty", {})),
            budget=dict(value.get("budget", {})),
        )

    # -- validation ----------------------------------------------------
    def validate(self) -> bool:
        """Check the record against the schema, raising on failure."""
        record = self.to_dict()
        try:
            import jsonschema
        except ImportError:
            jsonschema = None
        schema = _schema()
        if jsonschema is not None and schema is not None:
            jsonschema.validate(record, schema)
            return True
        _structural_check(record)
        return True
