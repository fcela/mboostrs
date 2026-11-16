"""Strict result JSON preserving unbounded, undefined and missing values."""

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

_SCHEMA = "mboostrs.result-json/1"
_TAGS = {"nan": float("nan"), "positive_infinity": float("inf"), "negative_infinity": float("-inf")}


def _child(path, key):
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def dump_result_json(data: Any, *, indent: int | None = None) -> str:
    """Serialize a result dictionary or JSON data with reversible nonfinite values.

    Pass ``result.to_dict(include_arrays=True)`` to retain row-level results.
    NumPy scalars/arrays and tuples become plain values/lists. Mappings need
    string keys. Positive/negative infinity and NaN become null placeholders,
    with their categories recorded by JSON Pointer in a versioned envelope.
    Genuine missing values remain null and have no nonfinite entry. User keys
    cannot collide with the envelope because the entire input is its ``data``.

    Decode with :func:`load_result_json` to recover plain Python data with IEEE
    nonfinite floats. This is result-data export, not fitted-model serialization.
    Existing ``to_dict()`` shapes and numerical floats are unchanged.
    """
    locations = {}

    def visit(value, path):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("strict result JSON mappings need string keys")
            return {key: visit(item, _child(path, key)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item, _child(path, index)) for index, item in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            locations[path] = "nan" if math.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"
            return None
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"strict result JSON cannot encode {type(value).__name__}")

    payload = visit(data, "")
    return json.dumps({"schema": _SCHEMA, "data": payload, "nonfinite": locations}, allow_nan=False, indent=indent)


def load_result_json(document: str) -> Any:
    """Decode a :func:`dump_result_json` document into plain Python result data.

    Restores nonfinite categories without conflating them with genuine nulls.
    Rejects unknown schemas, duplicate keys, malformed/non-null placeholders,
    unknown categories, unmatched paths and nonstandard JSON float constants.
    Does not instantiate a model or execute code from the document.
    """
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate result JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonstandard JSON constant: {value}")

    envelope = json.loads(document, object_pairs_hook=unique, parse_constant=reject_constant)
    if (not isinstance(envelope, dict) or set(envelope) != {"schema", "data", "nonfinite"}
            or envelope["schema"] != _SCHEMA):
        raise ValueError("expected mboostrs.result-json/1 envelope")
    locations = envelope["nonfinite"]
    if (not isinstance(locations, dict) or any(not isinstance(path, str) or not isinstance(tag, str)
                                              or tag not in _TAGS for path, tag in locations.items())):
        raise ValueError("invalid nonfinite location metadata")
    used = set()

    def visit(value, path):
        if path in locations:
            if value is not None:
                raise ValueError("a nonfinite placeholder must be null")
            used.add(path)
            return _TAGS[locations[path]]
        if isinstance(value, dict):
            return {key: visit(item, _child(path, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, _child(path, index)) for index, item in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON numbers require explicit location metadata")
        return value

    decoded = visit(envelope["data"], "")
    if used != set(locations):
        raise ValueError("nonfinite metadata contains unmatched paths")
    return decoded
