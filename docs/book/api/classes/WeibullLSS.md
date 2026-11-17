# `WeibullLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.WeibullLSS`.


*Class*

```python
WeibullLSS(stabilization: str = 'none')
```

Weibull AFT family. Takes a right-censored `Surv(time, status)`
response, or a plain positive response treated as uncensored.

Negative log-likelihood, gradients, and risk match R
`gamboostLSS::WeibullLSS()`. The automatic `mu` offset is an
identity-link location minimizing the Weibull negative log-likelihood;
its value can differ from the log-location initialization used by R's
`gamboostLSS`.

Source: [`python/mboostrs/gamlss.py` lines 2906-2935](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2906-L2935)

## Members

| Member | Kind | Summary |
|---|---|---|
| `mean` | Method | Uncensored AFT Weibull mean ``exp(mu) * Gamma(1 + sigma)``. |

(weibulllss-mean)=
## `WeibullLSS.mean`

*Method* of `WeibullLSS`

```python
WeibullLSS.mean(params) -> ndarray
```

Uncensored AFT Weibull mean `exp(mu) * Gamma(1 + sigma)`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2919-2925](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2919-L2925)
