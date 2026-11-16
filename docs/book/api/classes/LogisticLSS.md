# `LogisticLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.LogisticLSS`.


*Class*

```python
LogisticLSS(stabilization: str = 'none')
```

Logistic family: `mu` (identity link, location), `sigma` (log link,
scale). `y` unrestricted. Matches `gamlss.dist::LO()`.

Heavier-tailed than Gaussian at the same scale, so it is the usual drop-in
when residuals are symmetric but mildly outlier-prone. Note `sigma` is the
logistic scale, not the standard deviation: `sd = sigma * pi / sqrt(3)`.

Source: [`python/mboostrs/gamlss.py` lines 2082-2130](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2082-L2130)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#logisticlss-response) | Method |  |
| [`mean`](#logisticlss-mean) | Method | Conditional response mean. |
| [`pdf`](#logisticlss-pdf) | Method |  |
| [`cdf`](#logisticlss-cdf) | Method | Logistic CDF, ``1 / (1 + exp(-(y - mu) / sigma))``. |
| [`quantile`](#logisticlss-quantile) | Method | Logistic quantile, ``mu + sigma * log(p / (1 - p))``. |

(logisticlss-response)=
## `LogisticLSS.response`

*Method* of `LogisticLSS`

```python
LogisticLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2095-2101](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2095-L2101)

(logisticlss-mean)=
## `LogisticLSS.mean`

*Method* of `LogisticLSS`

```python
LogisticLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 2103-2105](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2103-L2105)

(logisticlss-pdf)=
## `LogisticLSS.pdf`

*Method* of `LogisticLSS`

```python
LogisticLSS.pdf(y, params)
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2107-2114](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2107-L2114)

(logisticlss-cdf)=
## `LogisticLSS.cdf`

*Method* of `LogisticLSS`

```python
LogisticLSS.cdf(y, params) -> ndarray
```

Logistic CDF, `1 / (1 + exp(-(y - mu) / sigma))`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2116-2126](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2116-L2126)

(logisticlss-quantile)=
## `LogisticLSS.quantile`

*Method* of `LogisticLSS`

```python
LogisticLSS.quantile(p, params) -> ndarray
```

Logistic quantile, `mu + sigma * log(p / (1 - p))`.

Source: [`python/mboostrs/gamlss.py` lines 2128-2130](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2128-L2130)
