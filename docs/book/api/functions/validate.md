# `validate`

Defined in `mboostrs._preflight`; exported as `mboostrs.validate`.


*Function*

```python
validate(model, newdata, *, offset=None, output='auto')
```

The issues that `predict(newdata)` would meet, before it runs.

**Parameters**

- `model` (GAMBoostModel, GLMBoostModel or LSSModel): A fitted model.
- `newdata` (dict or DataFrame): The rows that will be predicted, with raw columns as `predict` takes
  them.
- `offset` (number or array-like, optional): The offset `predict` would receive. Checked against the rows of
  `newdata` by the rules `predict(offset=)` uses.
- `output` ({"auto", "pandas", "polars", "records"}, default "auto"): `"records"` returns a list of dicts. The other values return a frame
  with one row per issue, following the package's `output=` convention.

**Returns**

list of dict, or DataFrame
    One record per issue, with the keys in `RECORD_COLUMNS`. An empty
    result means the checks listed in the module documentation found
    nothing. Every value is a plain Python type, so the records can be
    passed to `json.dumps`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`

**Notes**

`validate` replays the transforms, `%X%` designs and factor encodings
that `predict` replays, so a model whose data cannot be replayed raises
here rather than returning issues. It never predicts and never modifies
`newdata`.

The range checks of `predict` do not cover `btree`, `bkernel`,
`buser` and `brad` terms, because those have no training-range
boundary. `validate` lists each of them as `not_range_checked`.

Source: [`python/mboostrs/_preflight.py` lines 110-243](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_preflight.py#L110-L243)
