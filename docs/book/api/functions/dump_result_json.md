# `dump_result_json`

Defined in `mboostrs.result_json`; exported as `mboostrs.dump_result_json`.


*Function*

```python
dump_result_json(data: Any, *, indent: int | None = None) -> str
```

Serialize a result dictionary or JSON data with reversible nonfinite values.

Pass `result.to_dict(include_arrays=True)` to retain row-level results.
NumPy scalars/arrays and tuples become plain values/lists. Mappings need
string keys. Positive/negative infinity and NaN become null placeholders,
with their categories recorded by JSON Pointer in a versioned envelope.
Genuine missing values remain null and have no nonfinite entry. User keys
cannot collide with the envelope because the entire input is its `data`.

Decode with `load_result_json` to recover plain Python data with IEEE
nonfinite floats. This is result-data export, not fitted-model serialization.
Existing `to_dict()` shapes and numerical floats are unchanged.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`

Source: [`python/mboostrs/result_json.py` lines 18-53](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/result_json.py#L18-L53)
