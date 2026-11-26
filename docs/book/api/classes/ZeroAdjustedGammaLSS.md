# `ZeroAdjustedGammaLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.ZeroAdjustedGammaLSS`.


*Class*

```python
ZeroAdjustedGammaLSS(stabilization: str = 'none')
```

Zero-adjusted Gamma: `mu` (log, mean of the positive part), `sigma`
(log, Gamma dispersion), `nu` (logit, probability of an exact zero).
Requires `y >= 0`. Matches `gamlss.dist::ZAGA()`.

The semicontinuous case: insurance claims, daily rainfall, household spend
— positive and right-skewed *when it happens*, exactly zero otherwise.
Because `nu` is a fitted parameter you can put learners on, "what drives
whether it happens" and "what drives how large it is" become two separate
predictors rather than one blended variance assumption.

An all-zero response is refused: the Gamma part would have no data to
identify `mu` and `sigma` from.

Source: [`python/mboostrs/gamlss.py` lines 1976-2066](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1976-L2066)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Unconditional response mean, ``(1 - nu) * mu``. |
| `pdf` | Method | Mixture density: ``nu`` at zero, ``(1-nu) * Gamma`` above it. |
| `cdf` | Method | Zero-adjusted gamma CDF: ``nu`` at zero, then ``nu + (1 - nu) * F_gamma``. |
| `cdf_left` | Method | Left CDF limit: zero at the atom, ordinary CDF above zero. |
| `quantile` | Method | Zero-adjusted gamma quantile; zero for ``p <= nu``. |

(zeroadjustedgammalss-response)=
## `ZeroAdjustedGammaLSS.response`

*Method* of `ZeroAdjustedGammaLSS`

```python
ZeroAdjustedGammaLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1996-2002](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1996-L2002)

(zeroadjustedgammalss-mean)=
## `ZeroAdjustedGammaLSS.mean`

*Method* of `ZeroAdjustedGammaLSS`

```python
ZeroAdjustedGammaLSS.mean(params) -> ndarray
```

Unconditional response mean, `(1 - nu) * mu`.

Source: [`python/mboostrs/gamlss.py` lines 2004-2006](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2004-L2006)

(zeroadjustedgammalss-pdf)=
## `ZeroAdjustedGammaLSS.pdf`

*Method* of `ZeroAdjustedGammaLSS`

```python
ZeroAdjustedGammaLSS.pdf(y, params)
```

Mixture density: `nu` at zero, `(1-nu) * Gamma` above it.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2008-2029](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2008-L2029)

(zeroadjustedgammalss-cdf)=
## `ZeroAdjustedGammaLSS.cdf`

*Method* of `ZeroAdjustedGammaLSS`

```python
ZeroAdjustedGammaLSS.cdf(y, params) -> ndarray
```

Zero-adjusted gamma CDF: `nu` at zero, then `nu + (1 - nu) * F_gamma`.

Note the parameterization: here `sigma` is the gamma coefficient of
variation, as in `gamlss.dist::ZAGA`, so the shape is `1 / sigma^2`
and the scale `mu * sigma^2`. This differs from `GammaLSS`,
whose `sigma` is the shape itself.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2031-2057](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2031-L2057)

(zeroadjustedgammalss-cdf_left)=
## `ZeroAdjustedGammaLSS.cdf_left`

*Method* of `ZeroAdjustedGammaLSS`

```python
ZeroAdjustedGammaLSS.cdf_left(y, params) -> ndarray
```

Left CDF limit: zero at the atom, ordinary CDF above zero.

Source: [`python/mboostrs/gamlss.py` lines 2059-2062](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2059-L2062)

(zeroadjustedgammalss-quantile)=
## `ZeroAdjustedGammaLSS.quantile`

*Method* of `ZeroAdjustedGammaLSS`

```python
ZeroAdjustedGammaLSS.quantile(p, params) -> ndarray
```

Zero-adjusted gamma quantile; zero for `p <= nu`.

Source: [`python/mboostrs/gamlss.py` lines 2064-2066](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2064-L2066)
