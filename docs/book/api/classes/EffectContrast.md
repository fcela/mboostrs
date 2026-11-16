# `EffectContrast`

Defined in `mboostrs.selection`; exported as `mboostrs.EffectContrast`.


*Class*

```python
EffectContrast(
    estimate: float,
    se: float,
    lower: float,
    upper: float,
    level: float,
    scale: str = 'link'
) -> None
```

The average change in a linear predictor when covariates move, with its interval.

`estimate` is the mean, over the rows of the data given to `contrast`,
of the change in the predictor when the covariates move from their
`from` values to their `to` values with every other column at the
row's own value. `se` is its standard error from the coefficient
covariance, and `lower` and `upper` are `estimate -/+ z * se` at
`level`. `scale` is the link scale the change is on (`"link"`).

Source: [`python/mboostrs/selection.py` lines 730-747](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L730-L747)
