# `load_result_json`

Defined in `mboostrs.result_json`; exported as `mboostrs.load_result_json`.


*Function*

```python
load_result_json(document: str) -> Any
```

Decode a `dump_result_json` document into plain Python result data.

Restores nonfinite categories without conflating them with genuine nulls.
Rejects unknown schemas, duplicate keys, malformed/non-null placeholders,
unknown categories, unmatched paths and nonstandard JSON float constants.
Does not instantiate a model or execute code from the document.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/result_json.py` lines 56-102](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/result_json.py#L56-L102)
