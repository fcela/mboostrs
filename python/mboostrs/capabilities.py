"""Public views of the checked capability and validation contract.

The canonical data lives in :mod:`mboostrs.capability_manifest.json`. Python,
R, the README, and the book are generated from that one file; numerical
dispatch deliberately remains in the implementation modules.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from importlib.resources import files
from typing import Any

FAMILY_COLUMNS = (
    "name",
    "kind",
    "r_package",
    "r_name",
    "parameters",
    "links",
    "fit",
    "predict_link",
    "predict_response",
    "predict_class",
    "coef",
    "residuals",
    "aic",
    "loglik",
    "cdf",
    "pdf_or_pmf",
    "quantile",
    "simulate",
    "deviance_increment",
    "diagnostics",
    "known_deviation",
    "notes",
)
LEARNER_COLUMNS = (
    "name",
    "r_name",
    "glmboost",
    "gamboost",
    "glmboostLSS",
    "gamboostLSS",
    "newdata",
    "cvrisk",
    "stabsel",
    "extract_design",
    "penalty",
    "notes",
)
METHOD_COLUMNS = (
    "model_class",
    "method",
    "implemented",
    "r_equivalent",
    "partial",
    "notes",
)
OPERATIONS = ("fit", "predict", "cv", "stability_selection", "diagnostics")


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    resource = files("mboostrs").joinpath("capability_manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _table(rows: Iterable[dict[str, Any]], *, as_frame: bool = True, output: str | None = None) -> Any:
    from mboostrs._output import _normalize_output, output_frame

    fmt = _normalize_output(output, as_frame)
    return output_frame([dict(row) for row in rows], fmt)


def _flatten_entity(row: dict[str, Any], columns: tuple[str, ...]) -> dict[str, str]:
    result = {column: str(row.get(column, "")) for column in columns}
    validation_ids: set[str] = set()
    for frontend in ("python", "r"):
        support = row.get("support", {}).get(frontend, {})
        operations = support.get("operations", {})
        for operation in OPERATIONS:
            result[f"{frontend}_{operation}_status"] = str(operations.get(operation, "unsupported"))
        validation_ids.update(support.get("validation_ids", []))
    result["validation_ids"] = "; ".join(sorted(validation_ids))
    return result


def _flatten_method(row: dict[str, Any], frontend: str) -> dict[str, str]:
    result = {column: str(row.get(column, "")) for column in METHOD_COLUMNS}
    support = row.get("support", {}).get(frontend, {})
    result["status"] = str(support.get("status", "unsupported"))
    result["validation_ids"] = "; ".join(sorted(support.get("validation_ids", [])))
    return result


def family_capabilities(*, as_frame: bool = True, output: str | None = None) -> Any:
    """Return family support and evidence rows for both frontends.

    The legacy yes/no columns remain for compatibility. The
    ``python_*_status`` and ``r_*_status`` columns use the checked vocabulary:
    ``verified``, ``smoke``, ``intentional-deviation``, ``unsupported``, or
    ``unverified``.
    """
    rows = (_flatten_entity(row, FAMILY_COLUMNS) for row in _manifest()["tables"]["families"])
    return _table(rows, as_frame=as_frame, output=output)


def learner_capabilities(*, as_frame: bool = True, output: str | None = None) -> Any:
    """Return base-learner support and evidence rows for both frontends."""
    rows = (_flatten_entity(row, LEARNER_COLUMNS) for row in _manifest()["tables"]["learners"])
    return _table(rows, as_frame=as_frame, output=output)


def method_capabilities(*, frontend: str = "python", as_frame: bool = True, output: str | None = None) -> Any:
    """Return public method support for ``frontend='python'`` or ``'r'``."""
    if frontend not in {"python", "r"}:
        raise ValueError("frontend must be 'python' or 'r'")
    rows = (_flatten_method(row, frontend) for row in _manifest()["tables"]["methods"][frontend])
    return _table(rows, as_frame=as_frame, output=output)


def validation_capabilities(*, as_frame: bool = True, output: str | None = None) -> Any:
    """Return the explicit validation ledger behind advertised capabilities."""
    rows = []
    for record in _manifest()["validations"]:
        row = dict(record)
        row["compared_outputs"] = "; ".join(record["compared_outputs"])
        rows.append(row)
    return _table(rows, as_frame=as_frame, output=output)


def capabilities(*, as_frame: bool = True, output: str | None = None) -> dict[str, Any]:
    """Return family, learner, and Python-method capability tables."""
    return {
        "families": family_capabilities(as_frame=as_frame, output=output),
        "learners": learner_capabilities(as_frame=as_frame, output=output),
        "methods": method_capabilities(as_frame=as_frame, output=output),
    }


__all__ = [
    "capabilities",
    "family_capabilities",
    "learner_capabilities",
    "method_capabilities",
    "validation_capabilities",
]
