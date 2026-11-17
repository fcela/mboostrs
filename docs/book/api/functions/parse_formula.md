# `parse_formula`

Defined in `mboostrs.formula`; exported as `mboostrs.parse_formula`.


*Function*

```python
parse_formula(formula: str) -> tuple[str | tuple[str, ...], list[str]]
```

Split a linear formula into its response and predictor names.

**Parameters**

- `formula` (str): `"y ~ x1 + x2"`. A `Surv(time, status)` response returns as a tuple
  of the two column names.

**Returns**

tuple
    `(response, predictors)`, where `response` is a column name or a
    tuple for a survival response, and `predictors` is a list of names.

**Examples**

```python
>>> parse_formula("y ~ a + b")
('y', ['a', 'b'])
```

Source: [`python/mboostrs/formula.py` lines 157-180](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/formula.py#L157-L180)
