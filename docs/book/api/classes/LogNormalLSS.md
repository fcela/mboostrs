# `LogNormalLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.LogNormalLSS`.


*Class*

```python
LogNormalLSS(stabilization: str = 'none')
```

Log-normal AFT family. Matches R `gamboostLSS::LogNormalLSS()` for a
right-censored `Surv(time, status)` response.

A plain positive response is treated as uncensored (every status 1). Its
risk is the negative log-likelihood of the response on its own scale,
comparable with the other `realplus` families.

Source: [`python/mboostrs/gamlss.py` lines 2894-2920](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2894-L2920)

## Members

| Member | Kind | Summary |
|---|---|---|
| `mean` | Method | Uncensored survival-time mean ``exp(mu + sigma**2 / 2)``. |

(lognormallss-mean)=
## `LogNormalLSS.mean`

*Method* of `LogNormalLSS`

```python
LogNormalLSS.mean(params) -> ndarray
```

Uncensored survival-time mean `exp(mu + sigma**2 / 2)`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2904-2910](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2904-L2910)
