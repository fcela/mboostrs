# `GeometricLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GeometricLSS`.


*Class*

```python
GeometricLSS(stabilization: str = 'none')
```

:math:`\text{Geom}(\pi)` geometric location family: `mu` (log link,
mean). Matches R `gamboostLSS::GeometricLSS()` (wrapping gamlss.dist
`GEOM`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1685-1760](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1685-L1760)

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

Source: [`python/mboostrs/gamlss.py` lines 1698-1699](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1698-L1699)

(geometriclss-mean)=
## `GeometricLSS.mean`

*Method* of `GeometricLSS`

```python
GeometricLSS.mean(params) -> ndarray
```

Mean failures before the first success, `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1701-1703](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1701-L1703)

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

Source: [`python/mboostrs/gamlss.py` lines 1705-1715](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1705-L1715)

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

Source: [`python/mboostrs/gamlss.py` lines 1717-1729](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1717-L1729)

(geometriclss-quantile)=
## `GeometricLSS.quantile`

*Method* of `GeometricLSS`

```python
GeometricLSS.quantile(p, params) -> ndarray
```

Geometric quantile: smallest `k` with `CDF(k) >= p`.

Source: [`python/mboostrs/gamlss.py` lines 1731-1733](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1731-L1733)

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

Source: [`python/mboostrs/gamlss.py` lines 1735-1742](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1735-L1742)

(geometriclss-deviance_increment)=
## `GeometricLSS.deviance_increment`

*Method* of `GeometricLSS`

```python
GeometricLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1744-1746](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1744-L1746)

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

Source: [`python/mboostrs/gamlss.py` lines 1748-1760](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1748-L1760)
