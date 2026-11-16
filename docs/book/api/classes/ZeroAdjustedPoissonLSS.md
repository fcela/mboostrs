# `ZeroAdjustedPoissonLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.ZeroAdjustedPoissonLSS`.


*Class*

```python
ZeroAdjustedPoissonLSS(stabilization: str = 'none')
```

Zero-adjusted Poisson family: `mu` (log link, Poisson mean), `sigma`
(logit link, zero-inflation probability). Matches R
`gamboostLSS::ZeroAdjustedPoissonLSS()` (wrapping gamlss.dist `ZAP`).
Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 2250-2312](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2250-L2312)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#zeroadjustedpoissonlss-response) | Method |  |
| [`mean`](#zeroadjustedpoissonlss-mean) | Method | Mixture mean ``(1-sigma) * mu / (1-exp(-mu))``. |
| [`cdf`](#zeroadjustedpoissonlss-cdf) | Method | Zero-adjusted Poisson CDF: ``sigma`` at zero, then the |
| [`quantile`](#zeroadjustedpoissonlss-quantile) | Method | Zero-adjusted Poisson quantile; zero for ``p <= sigma``. |
| [`pdf`](#zeroadjustedpoissonlss-pdf) | Method | Zero-adjusted Poisson mass: ``sigma`` at zero, the zero-truncated |

(zeroadjustedpoissonlss-response)=
## `ZeroAdjustedPoissonLSS.response`

*Method* of `ZeroAdjustedPoissonLSS`

```python
ZeroAdjustedPoissonLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2264-2271](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2264-L2271)

(zeroadjustedpoissonlss-mean)=
## `ZeroAdjustedPoissonLSS.mean`

*Method* of `ZeroAdjustedPoissonLSS`

```python
ZeroAdjustedPoissonLSS.mean(params) -> ndarray
```

Mixture mean `(1-sigma) * mu / (1-exp(-mu))`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2273-2279](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2273-L2279)

(zeroadjustedpoissonlss-cdf)=
## `ZeroAdjustedPoissonLSS.cdf`

*Method* of `ZeroAdjustedPoissonLSS`

```python
ZeroAdjustedPoissonLSS.cdf(y, params) -> ndarray
```

Zero-adjusted Poisson CDF: `sigma` at zero, then the
zero-truncated Poisson scaled by `1 - sigma`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2281-2294](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2281-L2294)

(zeroadjustedpoissonlss-quantile)=
## `ZeroAdjustedPoissonLSS.quantile`

*Method* of `ZeroAdjustedPoissonLSS`

```python
ZeroAdjustedPoissonLSS.quantile(p, params) -> ndarray
```

Zero-adjusted Poisson quantile; zero for `p <= sigma`.

Source: [`python/mboostrs/gamlss.py` lines 2296-2298](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2296-L2298)

(zeroadjustedpoissonlss-pdf)=
## `ZeroAdjustedPoissonLSS.pdf`

*Method* of `ZeroAdjustedPoissonLSS`

```python
ZeroAdjustedPoissonLSS.pdf(y, params) -> ndarray
```

Zero-adjusted Poisson mass: `sigma` at zero, the zero-truncated
Poisson scaled by `1 - sigma` above. Matches `gamlss.dist::dZAP`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2300-2312](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2300-L2312)
