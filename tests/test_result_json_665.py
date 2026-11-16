"""Strict JSON preserves missing, undefined and unbounded result semantics."""

import json

import numpy as np
import pytest

import mboostrs as mb


def test_nested_numpy_nonfinite_categories_and_literal_metadata_keys():
    data = {
        "schema": "a user value",
        "data": {"a/b~": np.array([np.inf, -np.inf, np.nan, None], dtype=object)},
        "nonfinite": None,
        "_nonfinite": {"": "literal"},
        "finite": (np.int64(3), np.float32(0.5), np.bool_(True)),
    }
    document = mb.dump_result_json(data)
    envelope = json.loads(document, parse_constant=lambda value: pytest.fail(value))
    assert envelope["schema"] == "mboostrs.result-json/1"
    assert len(envelope["nonfinite"]) == 3
    decoded = mb.load_result_json(document)
    values = decoded["data"]["a/b~"]
    assert values[0] == np.inf and values[1] == -np.inf and np.isnan(values[2]) and values[3] is None
    assert decoded["nonfinite"] is None and decoded["_nonfinite"] == {"": "literal"}
    assert decoded["finite"] == [3, 0.5, True]
    assert decoded["schema"] == "a user value"


@pytest.mark.parametrize("value", [np.float64(np.inf), np.float32(-np.inf), np.float64(np.nan), None, 3, "abc"])
def test_root_scalar_categories(value):
    decoded = mb.load_result_json(mb.dump_result_json(value))
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        assert np.isnan(decoded)
    else:
        assert decoded == value


def test_unbounded_interval_round_trip_preserves_numerics_and_default_dictionary():
    interval = mb.ConformalInterval(
        np.array([0.0, -1.0]),
        np.array([np.inf, -1.0]),
        np.array([1.0, 0.0]),
        0.9,
        "split",
        20,
        quantile=np.inf,
        support="nonnegative_integers",
    )
    legacy = interval.to_dict(include_arrays=True)
    assert np.isposinf(legacy["upper"][0])
    decoded = mb.load_result_json(interval.to_json(include_arrays=True, indent=2))
    for name in ("lower", "upper", "predicted", "count_size"):
        np.testing.assert_array_equal(decoded[name], legacy[name])
    assert decoded["reference_distribution"] is None
    assert np.isposinf(decoded["quantile"]) and np.isposinf(decoded["mean_width"])
    assert interval.contains([1000000, 0]).tolist() == [True, False]


@pytest.mark.parametrize(
    "document",
    [
        '{"schema":"unknown","data":null,"nonfinite":{}}',
        '{"schema":"mboostrs.result-json/1","data":1,"nonfinite":{"":"positive_infinity"}}',
        '{"schema":"mboostrs.result-json/1","data":null,"nonfinite":{"/missing":"nan"}}',
        '{"schema":"mboostrs.result-json/1","data":null,"nonfinite":{"":"unknown"}}',
        '{"schema":"mboostrs.result-json/1","data":Infinity,"nonfinite":{}}',
        '{"schema":"mboostrs.result-json/1","data":1e999,"nonfinite":{}}',
        '{"schema":"mboostrs.result-json/1","data":{"x":1,"x":2},"nonfinite":{}}',
        '{"schema":"mboostrs.result-json/1","data":null,"nonfinite":[],"extra":1}',
    ],
)
def test_reject_malformed_or_nonstandard_documents(document):
    with pytest.raises(ValueError):
        mb.load_result_json(document)


def test_strict_encoder_refuses_ambiguous_keys_and_unsupported_objects():
    with pytest.raises(TypeError, match="string keys"):
        mb.dump_result_json({1: "a", "1": "b"})
    with pytest.raises(TypeError, match="complex"):
        mb.dump_result_json(np.complex128(1 + 2j))
