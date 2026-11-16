# `profile_mstop`

Defined in `mboostrs.model_selection`; exported as `mboostrs.profile_mstop`.


*Function*

```python
profile_mstop(
    model,
    grid: list[int] | None = None,
    metric: str = 'risk',
    k: float = 2.0
) -> list[dict[str, Any]]
```

Evaluate a model at different mstop values.

**Parameters**

- `model`: A fitted GLMBoostModel, GAMBoostModel, or LSSModel.
- `grid` (list of int, optional): mstop values to evaluate. If None, uses 10 evenly-spaced points
  from 1 to model.mstop.
- `metric` ({"risk", "GAIC", "logLik"}): Metric to evaluate at each mstop.
- `k` (float): Penalty for GAIC.

**Returns**

list of dict
    One row per grid point with mstop and metric value.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/model_selection.py` lines 891-976](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/model_selection.py#L891-L976)
