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

Source: [`python/mboostrs/gamlss.py` lines 2869-2901](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2869-L2901)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`mean`](#logloglss-mean) | Method | Uncensored log-logistic mean, defined for ``0 < sigma < 1``. |

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

Source: [`python/mboostrs/gamlss.py` lines 2881-2887](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2881-L2887)
