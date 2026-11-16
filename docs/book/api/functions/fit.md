# `fit`

Defined in `mboostrs._procedure._fit`; exported as `mboostrs.fit`.


*Function*

```python
fit(
    data: Any,
    response: str,
    family: Any = None,
    folds: Any = None,
    groups: Any = None,
    time: Any = None,
    seed: int = 0,
    budget: str = 'standard',
    uncertainty: str = 'none',
    **kwargs: Any
) -> Fit
```

Fit one model and return it with the trail of how it was chosen.

`fit` uses `autospec` to choose the specification and returns the
fitted model with a trail containing the input declarations, specification
decision, final formula, family and stopping iteration.

**Parameters**

- `data` (DataFrame or dict): Polars/pandas frame or mapping of column name to array.
- `response` (str): The response column.
- `family` (Family, optional): Pinned family, passed to `autospec`.
- `folds` (optional): Fold scheme or weight matrix, passed to `autospec`.
- `groups, time` (optional): Declared dependence structure. `groups` is passed to `autospec`;
  `time` currently raises `NotImplementedError`.
- `seed` (int, default 0): The single seed for every random stream.
- `budget` ({"quick", "standard", "thorough"}): Accepted values are `"quick"`, `"standard"` and `"thorough"`.
  The setting is validated but does not currently change the compute
  budget.
- `uncertainty` ({"none", "stability", "bootstrap"}): `"none"` is currently supported. `"stability"` and
  `"bootstrap"` raise `NotImplementedError`.
- `**kwargs`: Passed through to `autospec`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/_procedure/_fit.py` lines 56-151](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L56-L151)
