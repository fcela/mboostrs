# `expand_interactions`

Defined in `mboostrs.interactions`; exported as `mboostrs.expand_interactions`.


*Function*

```python
expand_interactions(formula: str, data, *, na_action: str = 'error')
```

Rewrite `a %X% b` terms into buser references.

Returns `(rewritten_formula, env, interactions)` where `env` maps the
injected design/penalty matrix names, and `interactions` is the list of
`TensorInteraction` recipes to store for prediction. A formula with
no `%X%` returns unchanged with an empty env and list.

`na_action="impute"` fills a numeric margin's missing values with its
training median, recorded in the recipe so new rows are filled the same
way. Unlike a main effect, a tensor margin gets no `<x>_missing`
indicator. With the default `"error"`, a missing margin value raises,
naming the column.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/interactions.py` lines 956-1021](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/interactions.py#L956-L1021)
