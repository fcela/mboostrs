# `report`

Defined in `mboostrs._procedure._report`; exported as `mboostrs.report`.


*Function*

```python
report(fit_obj: Any, format: str = 'text') -> Any
```

Render a `Fit` (or `DecisionTrail`) as text, markdown or JSON.

Returns a `str` for `"text"`/`"markdown"` and the trail `dict` for
`"json"`. Purely a reader of the trail -- it never fits anything.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/_procedure/_report.py` lines 138-151](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_report.py#L138-L151)
