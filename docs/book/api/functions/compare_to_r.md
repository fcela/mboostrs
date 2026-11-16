# `compare_to_r`

Defined in `mboostrs.compare`; exported as `mboostrs.compare_to_r`.


*Function*

```python
compare_to_r(
    spec: mboostrs.compare.CompareSpec,
    data: dict[str, numpy.ndarray] | None = None,
    backend: str = 'mboost'
) -> CompareReport
```

Compare a mboostrs fit against R mboost.

**Parameters**

- `spec` (CompareSpec): Model specification.
- `data` (dict, optional): Pre-generated data. If None, synthetic data is generated from the spec.
- `backend` (str): R backend to compare against (currently only "mboost").

**Returns**

CompareReport
    Structured report with diffs.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/compare.py` lines 448-515](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L448-L515)
