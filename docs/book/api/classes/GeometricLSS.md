# `GeometricLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GeometricLSS`.


*Class*

```python
GeometricLSS(stabilization: str = 'none')
```

:math:`\text{Geom}(\pi)` geometric location family: `mu` (log link,
mean). Matches R `gamboostLSS::GeometricLSS()` (wrapping gamlss.dist
`GEOM`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1737-1812](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1737-L1812)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Mean failures before the first success, ``mu``. |
| `pmf` | Method | Geometric mass with response-scale mean ``mu`` (number of failures |
| `cdf` | Method | Geometric CDF: :math:`1 - (1-\pi)^{\lfloor y \rfloor + 1}`. |
| `quantile` | Method | Geometric quantile: smallest ``k`` with ``CDF(k) >= p``. |
| `simulate` | Method | Draw geometric samples (number of failures before first success). |
| `deviance_increment` | Method |  |
| `pdf` | Method | Geometric probability mass. Matches ``gamlss.dist::dGEOM``, whose |

(geometriclss-response)=
## `GeometricLSS.response`

*Method* of `GeometricLSS`

```python
GeometricLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1750-1751](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1750-L1751)

(geometriclss-mean)=
## `GeometricLSS.mean`

*Method* of `GeometricLSS`

```python
GeometricLSS.mean(params) -> ndarray
```

Mean failures before the first success, `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1753-1755](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1753-L1755)

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

Source: [`python/mboostrs/gamlss.py` lines 1757-1767](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1757-L1767)

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

Source: [`python/mboostrs/gamlss.py` lines 1769-1781](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1769-L1781)

(geometriclss-quantile)=
## `GeometricLSS.quantile`

*Method* of `GeometricLSS`

```python
GeometricLSS.quantile(p, params) -> ndarray
```

Geometric quantile: smallest `k` with `CDF(k) >= p`.

Source: [`python/mboostrs/gamlss.py` lines 1783-1785](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1783-L1785)

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

Source: [`python/mboostrs/gamlss.py` lines 1787-1794](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1787-L1794)

(geometriclss-deviance_increment)=
## `GeometricLSS.deviance_increment`

*Method* of `GeometricLSS`

```python
GeometricLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1796-1798](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1796-L1798)

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

Source: [`python/mboostrs/gamlss.py` lines 1800-1812](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1800-L1812)
