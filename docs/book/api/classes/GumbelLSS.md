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

Source: [`python/mboostrs/gamlss.py` lines 2133-2184](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2133-L2184)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#gumbellss-response) | Method |  |
| [`mean`](#gumbellss-mean) | Method | Conditional response mean, ``mu + EulerGamma * sigma``. |
| [`cdf`](#gumbellss-cdf) | Method | CDF of the extreme-value distribution for maxima, |
| [`quantile`](#gumbellss-quantile) | Method | Gumbel (maxima) quantile, ``mu - sigma * log(-log(p))``. |
| [`pdf`](#gumbellss-pdf) | Method | Density of the extreme-value distribution for maxima. Matches |

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

Source: [`python/mboostrs/gamlss.py` lines 2149-2155](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2149-L2155)

(gumbellss-mean)=
## `GumbelLSS.mean`

*Method* of `GumbelLSS`

```python
GumbelLSS.mean(params) -> ndarray
```

Conditional response mean, `mu + EulerGamma * sigma`.

Source: [`python/mboostrs/gamlss.py` lines 2157-2159](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2157-L2159)

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

Source: [`python/mboostrs/gamlss.py` lines 2161-2169](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2161-L2169)

(gumbellss-quantile)=
## `GumbelLSS.quantile`

*Method* of `GumbelLSS`

```python
GumbelLSS.quantile(p, params) -> ndarray
```

Gumbel (maxima) quantile, `mu - sigma * log(-log(p))`.

Source: [`python/mboostrs/gamlss.py` lines 2171-2173](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2171-L2173)

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

Source: [`python/mboostrs/gamlss.py` lines 2175-2184](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2175-L2184)
