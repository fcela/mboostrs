# `r_squared`

Defined in `mboostrs.model_selection`; exported as `mboostrs.r_squared`.


*Function*

```python
r_squared(model, kind='both')
```

Generalized R-squared for a distributional model, as `gamlss::Rsq`.

Both forms are likelihood ratios against the model's own null — for a
boosted fit, iteration zero, where the predictor is the offset alone and
nothing has been selected:

* Cox-Snell: `1 - exp((2/n) * (loglik_null - loglik_fit))`. Bounded above
  by `1 - exp((2/n) * loglik_null)`, which is below 1 for a discrete
  response, so it cannot reach 1 however good the fit.
* Cragg-Uhler (also called Nagelkerke): the same quantity divided by that
  bound, so 1 is attainable.

**Parameters**

- `model` (LSSModel): A distributional fit. The measure needs a risk path that is a negative
  log-likelihood, which is what an LSS risk is.
- `kind` ({"both", "cox-snell", "cragg-uhler"})

**Returns**

float or dict

**Raises**

- `NotImplementedError`: For a standard `glmboost` or `gamboost` fit. Their `risk_` is the
family's loss, and for most families that is not a log-likelihood:
Gaussian's is the residual sum of squares, so `-risk_` is short of the
log-likelihood by the normalizing terms, and `Quantile`, `Expectile`
and `Huber` are not likelihoods at all. `logLik()` provides the
value at the fitted `mstop`, but no matching likelihood is available
at iteration zero for the ratio.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

Descriptive, not adjusted for the number of selected terms. Boosting
selects terms from the data, so this reads higher than a cross-validated
estimate would; use `cvrisk` to compare models.

Source: [`python/mboostrs/model_selection.py` lines 979-1056](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/model_selection.py#L979-L1056)
