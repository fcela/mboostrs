"""Strict evidence JSON preserves nonfinite values without changing nulls."""

import json

import numpy as np
import pytest

from benches.evidence_float_codec import encode_record, decode_record


def test_nested_nonfinite_roundtrip_including_escaped_paths():
    original = {
        "q": float("inf"),
        "curve": np.array([1.0, np.nan, -np.inf]),
        "missing": None,
        "a/b~c": {"tuple": (np.float64(np.inf), False, 3)},
    }
    encoded = encode_record(original)
    text = json.dumps(encoded, allow_nan=False)
    decoded = decode_record(json.loads(text))
    assert np.isposinf(decoded["q"]) and decoded["missing"] is None
    assert np.isnan(decoded["curve"][1]) and np.isneginf(decoded["curve"][2])
    assert np.isposinf(decoded["a/b~c"]["tuple"][0])
    assert decoded["a/b~c"]["tuple"][1:] == [False, 3]
    assert encode_record({"a": [1.0, 2.0], "missing": None}) == {"a": [1.0, 2.0], "missing": None}
    with pytest.raises(ValueError, match="reserved"):
        encode_record({"_nonfinite": {}})
