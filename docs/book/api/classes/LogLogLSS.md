# `LogLogLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.LogLogLSS`.


*Class*

```python
LogLogLSS(stabilization: str = 'none')
```

Log-logistic AFT family. Takes a right-censored `Surv(time, status)`
response, or a plain positive response treated as uncensored.

Negative log-likelihood, gradients, and risk match R
`gamboostLSS::LogLogLSS()`. For censored data, the automatic sigma
offset minimizes the log-logistic negative log-likelihood; its value can
differ from the initialization used by R's `gamboostLSS`.

Source: [`python/mboostrs/gamlss.py` lines 2871-2903](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2871-L2903)

## Members

| Member | Kind | Summary |
|---|---|---|
| `mean` | Method | Uncensored log-logistic mean, defined for ``0 < sigma < 1``. |

(logloglss-mean)=
## `LogLogLSS.mean`

*Method* of `LogLogLSS`

```python
LogLogLSS.mean(params) -> ndarray
```

Uncensored log-logistic mean, defined for `0 < sigma < 1`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2883-2889](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2883-L2889)
