# `BLPProjection`

Defined in `mboostrs.effect_modifiers`; exported as `mboostrs.BLPProjection`.


*Class*

```python
BLPProjection(
    covariates: tuple,
    plans: tuple,
    coefficients: tuple,
    kept: tuple,
    labels: tuple,
    n: int,
    level: float,
    recipes: dict = <factory>
) -> None
```

A linear projection of the effect on covariates, usable as a ranking source.

The projection is the best linear predictor of the AIPW score on the
covariates it was fit with: `predict(newdata)` returns the fitted value
`a + sum_j b_j z_j(x)`, where `z_j` is a covariate standardized over the
overlap region of the fit (or the dummy of a categorical level). Its values
are estimated effects in outcome units at each row. Passing it to
`mboostrs.rank_by_effect` ranks the rows inside the overlap region by
this estimate.

Source: [`python/mboostrs/effect_modifiers.py` lines 287-325](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L287-L325)

## Members

| Member | Kind | Summary |
|---|---|---|
| `predict` | Method | The projected effect of each row of ``newdata``, one value per row. |

(blpprojection-predict)=
## `BLPProjection.predict`

*Method* of `BLPProjection`

```python
BLPProjection.predict(newdata)
```

The projected effect of each row of `newdata`, one value per row.

A covariate written as a transform (`"log(prior_earnings)"`) is
computed from the columns of `newdata` with the parameters of the
fitting rows.

Source: [`python/mboostrs/effect_modifiers.py` lines 311-322](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L311-L322)
