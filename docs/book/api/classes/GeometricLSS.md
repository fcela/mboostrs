# `GeometricLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GeometricLSS`.


*Class*

```python
GeometricLSS(stabilization: str = 'none')
```

:math:`\text{Geom}(\pi)` geometric location family: `mu` (log link,
mean). Matches R `gamboostLSS::GeometricLSS()` (wrapping gamlss.dist
`GEOM`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1683-1758](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1683-L1758)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#geometriclss-response) | Method |  |
| [`mean`](#geometriclss-mean) | Method | Mean failures before the first success, ``mu``. |
| [`pmf`](#geometriclss-pmf) | Method | Geometric mass with response-scale mean ``mu`` (number of failures |
| [`cdf`](#geometriclss-cdf) | Method | Geometric CDF: :math:`1 - (1-\pi)^{\lfloor y \rfloor + 1}`. |
| [`quantile`](#geometriclss-quantile) | Method | Geometric quantile: smallest ``k`` with ``CDF(k) >= p``. |
| [`simulate`](#geometriclss-simulate) | Method | Draw geometric samples (number of failures before first success). |
| [`deviance_increment`](#geometriclss-deviance_increment) | Method |  |
| [`pdf`](#geometriclss-pdf) | Method | Geometric probability mass. Matches ``gamlss.dist::dGEOM``, whose |

(geometriclss-response)=
## `GeometricLSS.response`

*Method* of `GeometricLSS`

```python
GeometricLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1696-1697](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1696-L1697)

(geometriclss-mean)=
## `GeometricLSS.mean`

*Method* of `GeometricLSS`

```python
GeometricLSS.mean(params) -> ndarray
```

Mean failures before the first success, `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1699-1701](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1699-L1701)

(geometriclss-pmf)=
## `GeometricLSS.pmf`

*Method* of `GeometricLSS`

```python
GeometricLSS.pmf(y, params) -> ndarray
```

Geometric mass with response-scale mean `mu` (number of failures
before the first success).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1703-1713](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1703-L1713)

(geometriclss-cdf)=
## `GeometricLSS.cdf`

*Method* of `GeometricLSS`

```python
GeometricLSS.cdf(y, params) -> ndarray
```

Geometric CDF: :math:`1 - (1-\pi)^{\lfloor y \rfloor + 1}`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1715-1727](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1715-L1727)

(geometriclss-quantile)=
## `GeometricLSS.quantile`

*Method* of `GeometricLSS`

```python
GeometricLSS.quantile(p, params) -> ndarray
```

Geometric quantile: smallest `k` with `CDF(k) >= p`.

Source: [`python/mboostrs/gamlss.py` lines 1729-1731](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1729-L1731)

(geometriclss-simulate)=
## `GeometricLSS.simulate`

*Method* of `GeometricLSS`

```python
GeometricLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw geometric samples (number of failures before first success).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1733-1740](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1733-L1740)

(geometriclss-deviance_increment)=
## `GeometricLSS.deviance_increment`

*Method* of `GeometricLSS`

```python
GeometricLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1742-1744](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1742-L1744)

(geometriclss-pdf)=
## `GeometricLSS.pdf`

*Method* of `GeometricLSS`

```python
GeometricLSS.pdf(y, params) -> ndarray
```

Geometric probability mass. Matches `gamlss.dist::dGEOM`, whose
`mu` is the mean, so the success probability is `1 / (1 + mu)`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1746-1758](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1746-L1758)
