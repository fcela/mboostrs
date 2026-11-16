# `StoppingBudget`

Defined in `mboostrs._model_api`; exported as `mboostrs.StoppingBudget`.


*Class*

```python
StoppingBudget(scheme: str, total: int, per_parameter: dict) -> None
```

How far a fitted model was boosted, in the same shape for every model.

`scheme` is `"single"` for a standard model (one predictor `f`),
`"cyclic"` for a distributional model that updates every parameter once
per iteration, and `"noncyclic"` for one that picks a parameter per
update. `per_parameter` maps each predictor to its number of updates;
`total` is their sum, which is the budget a non-cyclic refit takes.

Source: [`python/mboostrs/_model_api.py` lines 371-384](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_model_api.py#L371-L384)
