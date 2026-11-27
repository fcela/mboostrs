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

Source: [`python/mboostrs/gamlss.py` lines 2187-2238](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2187-L2238)

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

Source: [`python/mboostrs/gamlss.py` lines 2203-2209](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2203-L2209)

(gumbellss-mean)=
## `GumbelLSS.mean`

*Method* of `GumbelLSS`

```python
GumbelLSS.mean(params) -> ndarray
```

Conditional response mean, `mu + EulerGamma * sigma`.

Source: [`python/mboostrs/gamlss.py` lines 2211-2213](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2211-L2213)

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

Source: [`python/mboostrs/gamlss.py` lines 2215-2223](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2215-L2223)

(gumbellss-quantile)=
## `GumbelLSS.quantile`

*Method* of `GumbelLSS`

```python
GumbelLSS.quantile(p, params) -> ndarray
```

Gumbel (maxima) quantile, `mu - sigma * log(-log(p))`.

Source: [`python/mboostrs/gamlss.py` lines 2225-2227](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2225-L2227)

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

Source: [`python/mboostrs/gamlss.py` lines 2229-2238](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2229-L2238)
