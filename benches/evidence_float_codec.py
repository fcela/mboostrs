"""Strict JSON evidence with explicit, reversible nonfinite float locations."""

from collections.abc import Mapping
import math

import numpy as np


def encode_record(record):
    """Keep finite values, replace nonfinite leaves by null and tag their paths."""
    if "_nonfinite" in record:
        raise ValueError("reserved evidence metadata key: _nonfinite")
    locations = {}

    def visit(value, path):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, Mapping):
            return {
                str(key): visit(item, path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [visit(item, path + "/" + str(i)) for i, item in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            locations[path] = "nan" if math.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"
            return None
        return value

    encoded = visit(record, "")
    if locations:
        encoded["_nonfinite"] = locations
    return encoded


def decode_record(record):
    """Recover the distinguished nonfinite values without changing genuine nulls."""
    locations = record.get("_nonfinite", {})
    values = {"nan": float("nan"), "positive_infinity": float("inf"), "negative_infinity": float("-inf")}

    def visit(value, path):
        if path in locations:
            return values[locations[path]]
        if isinstance(value, dict):
            return {
                key: visit(item, path + "/" + key.replace("~", "~0").replace("/", "~1"))
                for key, item in value.items()
                if key != "_nonfinite" or path != ""
            }
        if isinstance(value, list):
            return [visit(item, path + "/" + str(i)) for i, item in enumerate(value)]
        return value

    return visit(record, "")
