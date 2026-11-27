# `expand_formula`

Defined in `mboostrs.formula`; exported as `mboostrs.expand_formula`.


*Function*

```python
expand_formula(formula: str, data) -> str
```

Resolve `.` and `- term` against the data, returning a plain formula.

* `y ~ .` becomes every column except the response.
* `y ~ . - z` drops `z` from that expansion.
* `y ~ . + bbs(z, df=4)` expands `.` to the remaining columns and keeps
  the explicit term, so a single predictor can be given a different learner
  without naming all the others.

Formulas containing neither `.` nor a `-` are returned unchanged, so
this is a no-op for existing code.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/formula.py` lines 481-590](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/formula.py#L481-L590)
