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

Source: [`python/mboostrs/gamlss.py` lines 2129-2177](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2129-L2177)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional response mean. |
| `pdf` | Method |  |
| `cdf` | Method | Logistic CDF, ``1 / (1 + exp(-(y - mu) / sigma))``. |
| `quantile` | Method | Logistic quantile, ``mu + sigma * log(p / (1 - p))``. |

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

Source: [`python/mboostrs/gamlss.py` lines 2142-2148](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2142-L2148)

(logisticlss-mean)=
## `LogisticLSS.mean`

*Method* of `LogisticLSS`

```python
LogisticLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 2150-2152](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2150-L2152)

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

Source: [`python/mboostrs/gamlss.py` lines 2154-2161](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2154-L2161)

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

Source: [`python/mboostrs/gamlss.py` lines 2163-2173](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2163-L2173)

(logisticlss-quantile)=
## `LogisticLSS.quantile`

*Method* of `LogisticLSS`

```python
LogisticLSS.quantile(p, params) -> ndarray
```

Logistic quantile, `mu + sigma * log(p / (1 - p))`.

Source: [`python/mboostrs/gamlss.py` lines 2175-2177](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2175-L2177)
