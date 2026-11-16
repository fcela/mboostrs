# `gamlss_refit`

Defined in `mboostrs.selection`; exported as `mboostrs.gamlss_refit`.


*Function*

```python
gamlss_refit(model, terms=None, *, which='selected', factor_ridge=False)
```

Refit a distributional model's terms as one penalized likelihood with
REML smoothing parameters.

Every distribution parameter keeps the terms its boosting formula gave it
(by default those selected at least once, `which="selected"`; all of
them with `which="all"`; or `terms={"mu": [...], "sigma": [...]}`).
All coefficients are estimated jointly by penalized maximum likelihood,
and the smoothing parameters minimize the Laplace-approximate restricted
likelihood (Wood, Pya and Saefken 2016). A parameter with no terms keeps
an intercept. Linear terms of a `glmboostLSS` formula are refitted as
`bols`.

**Parameters**

- `model` (LSSModel): A `gamboostLSS` or `glmboostLSS` fit.
- `terms` (dict, optional): Terms per distribution parameter.
- `which` ({"selected", "all"})
- `factor_ridge` (bool, default False): Categorical `bols` terms are unpenalized: each level's effect is
  estimated from that level's rows alone, relative to the reference
  level. `True` puts an identity (ridge) penalty on the level
  indicators, with its strength chosen by REML, which shrinks every
  level's effect toward the reference level.

**Returns**

GamlssRefit

The intervals are conditional on the chosen terms. Survival and
multi-column responses are not supported.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/selection.py` lines 2341-2472](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2341-L2472)
