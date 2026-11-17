# `GumbelLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GumbelLSS`.


*Class*

```python
GumbelLSS(stabilization: str = 'none')
```

Gumbel location-scale family for maxima — the standard extreme-value
type I: `mu` (identity link, location), `sigma` (log link, scale).

Matches `gamlss.dist::RG()`, which gamlss labels "Reverse Gumbel", **not**
`GU()`. gamlss's naming inverts the common one: its `GU` is the Gumbel
for *minima*, and its `RG` is the right-skewed form usually meant by
"Gumbel". For `y = [0.4, 1.3, -0.7, 2.1, 0.0]` at
`mu = 0.5, sigma = 2`, the risk is 9.042800226406863, matching
`-sum(dRG(..., log = TRUE))`; `dGU` gives 9.161943.

Source: [`python/mboostrs/gamlss.py` lines 2135-2186](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2135-L2186)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional response mean, ``mu + EulerGamma * sigma``. |
| `cdf` | Method | CDF of the extreme-value distribution for maxima, |
| `quantile` | Method | Gumbel (maxima) quantile, ``mu - sigma * log(-log(p))``. |
| `pdf` | Method | Density of the extreme-value distribution for maxima. Matches |

(gumbellss-response)=
## `GumbelLSS.response`

*Method* of `GumbelLSS`

```python
GumbelLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2151-2157](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2151-L2157)

(gumbellss-mean)=
## `GumbelLSS.mean`

*Method* of `GumbelLSS`

```python
GumbelLSS.mean(params) -> ndarray
```

Conditional response mean, `mu + EulerGamma * sigma`.

Source: [`python/mboostrs/gamlss.py` lines 2159-2161](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2159-L2161)

(gumbellss-cdf)=
## `GumbelLSS.cdf`

*Method* of `GumbelLSS`

```python
GumbelLSS.cdf(y, params) -> ndarray
```

CDF of the extreme-value distribution for maxima,
`exp(-exp(-(y - mu) / sigma))`. Matches `gamlss.dist::pRG`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2163-2171](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2163-L2171)

(gumbellss-quantile)=
## `GumbelLSS.quantile`

*Method* of `GumbelLSS`

```python
GumbelLSS.quantile(p, params) -> ndarray
```

Gumbel (maxima) quantile, `mu - sigma * log(-log(p))`.

Source: [`python/mboostrs/gamlss.py` lines 2173-2175](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2173-L2175)

(gumbellss-pdf)=
## `GumbelLSS.pdf`

*Method* of `GumbelLSS`

```python
GumbelLSS.pdf(y, params) -> ndarray
```

Density of the extreme-value distribution for maxima. Matches
`gamlss.dist::dRG`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2177-2186](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2177-L2186)
