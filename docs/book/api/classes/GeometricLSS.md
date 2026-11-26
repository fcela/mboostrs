# `GeometricLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GeometricLSS`.


*Class*

```python
GeometricLSS(stabilization: str = 'none')
```

:math:`\text{Geom}(\pi)` geometric location family: `mu` (log link,
mean). Matches R `gamboostLSS::GeometricLSS()` (wrapping gamlss.dist
`GEOM`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1730-1805](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1730-L1805)

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

Source: [`python/mboostrs/gamlss.py` lines 1743-1744](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1743-L1744)

(geometriclss-mean)=
## `GeometricLSS.mean`

*Method* of `GeometricLSS`

```python
GeometricLSS.mean(params) -> ndarray
```

Mean failures before the first success, `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1746-1748](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1746-L1748)

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

Source: [`python/mboostrs/gamlss.py` lines 1750-1760](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1750-L1760)

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

Source: [`python/mboostrs/gamlss.py` lines 1762-1774](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1762-L1774)

(geometriclss-quantile)=
## `GeometricLSS.quantile`

*Method* of `GeometricLSS`

```python
GeometricLSS.quantile(p, params) -> ndarray
```

Geometric quantile: smallest `k` with `CDF(k) >= p`.

Source: [`python/mboostrs/gamlss.py` lines 1776-1778](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1776-L1778)

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

Source: [`python/mboostrs/gamlss.py` lines 1780-1787](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1780-L1787)

(geometriclss-deviance_increment)=
## `GeometricLSS.deviance_increment`

*Method* of `GeometricLSS`

```python
GeometricLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1789-1791](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1789-L1791)

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

Source: [`python/mboostrs/gamlss.py` lines 1793-1805](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1793-L1805)
