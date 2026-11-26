# `NBinomial`

Defined in `mboostrs.families`; exported as `mboostrs.NBinomial`.


*Class*

```python
NBinomial(**kwargs)
```

Negative binomial log-likelihood with a log link, for overdispersed counts.

`nuisance()` is the size `sigma` in `Var(Y) = mu + mu**2 / sigma`,
profiled at each iteration by a one-dimensional Brent search matching R's
`optimize`. `Hurdle` uses the reciprocal parameterization (its
`sigma` is the dispersion, and the size is `1 / sigma`). A statsmodels
NB2 fit reports the reciprocal as `alpha`
(`Var = mu + alpha * mu**2`); using `nuisance()` as that alpha
overstates the extra-Poisson variance by `sigma**2`. The negative
gradient depends on `sigma`, so it is re-estimated as the fit proceeds
rather than fixed in advance.

The fitted model's `size` property names this value explicitly.
`NBinomial()` takes no fixed size, `alpha` or `dispersion` argument;
the core estimates the size at every step.

Source: [`python/mboostrs/families.py` lines 220-245](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L220-L245)
