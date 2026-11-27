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

Source: [`python/mboostrs/gamlss.py` lines 2136-2184](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2136-L2184)

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

Source: [`python/mboostrs/gamlss.py` lines 2149-2155](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2149-L2155)

(logisticlss-mean)=
## `LogisticLSS.mean`

*Method* of `LogisticLSS`

```python
LogisticLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 2157-2159](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2157-L2159)

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

Source: [`python/mboostrs/gamlss.py` lines 2161-2168](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2161-L2168)

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

Source: [`python/mboostrs/gamlss.py` lines 2170-2180](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2170-L2180)

(logisticlss-quantile)=
## `LogisticLSS.quantile`

*Method* of `LogisticLSS`

```python
LogisticLSS.quantile(p, params) -> ndarray
```

Logistic quantile, `mu + sigma * log(p / (1 - p))`.

Source: [`python/mboostrs/gamlss.py` lines 2182-2184](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2182-L2184)
