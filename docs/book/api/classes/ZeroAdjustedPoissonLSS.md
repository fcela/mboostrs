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

Source: [`python/mboostrs/gamlss.py` lines 2301-2363](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2301-L2363)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Mixture mean ``(1-sigma) * mu / (1-exp(-mu))``. |
| `cdf` | Method | Zero-adjusted Poisson CDF: ``sigma`` at zero, then the |
| `quantile` | Method | Zero-adjusted Poisson quantile; zero for ``p <= sigma``. |
| `pdf` | Method | Zero-adjusted Poisson mass: ``sigma`` at zero, the zero-truncated |

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

Source: [`python/mboostrs/gamlss.py` lines 2315-2322](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2315-L2322)

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

Source: [`python/mboostrs/gamlss.py` lines 2324-2330](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2324-L2330)

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

Source: [`python/mboostrs/gamlss.py` lines 2332-2345](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2332-L2345)

(zeroadjustedpoissonlss-quantile)=
## `ZeroAdjustedPoissonLSS.quantile`

*Method* of `ZeroAdjustedPoissonLSS`

```python
ZeroAdjustedPoissonLSS.quantile(p, params) -> ndarray
```

Zero-adjusted Poisson quantile; zero for `p <= sigma`.

Source: [`python/mboostrs/gamlss.py` lines 2347-2349](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2347-L2349)

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

Source: [`python/mboostrs/gamlss.py` lines 2351-2363](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2351-L2363)
