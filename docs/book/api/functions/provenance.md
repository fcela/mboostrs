# `provenance`

Defined in `mboostrs._provenance`; exported as `mboostrs.provenance`.


*Function*

```python
provenance(data: Any = None, *, seed: int | None = None) -> dict[str, Any]
```

Return build, environment and optional data identity metadata for a result.

**Parameters**

- `data` (DataFrame, dict of arrays, optional): The rows the result was computed on, as a Polars frame, a pandas frame
  or a mapping from column name to array. When given, the record carries
  a `"data"` entry with the row count, the column names with their
  dtypes, and a SHA-256 digest of the values. Equal content gives an
  equal digest whichever container holds it, and a changed value gives a
  different digest.
- `seed` (int, optional): The random seed the computation used, recorded as given.

**Returns**

dict
    `mboostrs_version`, `build` (the mapping from `build_info`),
    `python_version`, `platform`, `packages` (the NumPy and Polars
    versions, and pandas when it is installed), `seed` when a seed was
    given, and `data` when data was given. The record has no timestamp.

**Notes**

The digest reads every value once in Python for text columns, so a frame
with millions of text cells takes seconds to hash. Numeric columns are
hashed as one byte string each.

Source: [`python/mboostrs/_provenance.py` lines 39-90](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_provenance.py#L39-L90)
