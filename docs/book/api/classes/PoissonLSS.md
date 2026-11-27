# `PoissonLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.PoissonLSS`.


*Class*

```python
PoissonLSS(stabilization: str = 'none')
```

:math:`P(\mu)` Poisson location family: `mu` (log link, mean).
Matches R `gamboostLSS::PoissonLSS()` (wrapping gamlss.dist `PO`).
Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1667-1731](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1667-L1731)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional count mean ``mu``. |
| `pmf` | Method | Poisson mass with response-scale mean ``mu``. |
| `cdf` | Method | Poisson CDF with response-scale mean ``mu``. |
| `quantile` | Method | Poisson quantile for probabilities ``p``. |
| `simulate` | Method | Draw Poisson samples with response-scale mean ``mu``. |
| `deviance_increment` | Method |  |
| `pdf` | Method | Poisson probability mass at the integer counts ``y``. |

(poissonlss-response)=
## `PoissonLSS.response`

*Method* of `PoissonLSS`

```python
PoissonLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1680-1681](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1680-L1681)

(poissonlss-mean)=
## `PoissonLSS.mean`

*Method* of `PoissonLSS`

```python
PoissonLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1683-1685](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1683-L1685)

(poissonlss-pmf)=
## `PoissonLSS.pmf`

*Method* of `PoissonLSS`

```python
PoissonLSS.pmf(y, params) -> ndarray
```

Poisson mass with response-scale mean `mu`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1687-1695](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1687-L1695)

(poissonlss-cdf)=
## `PoissonLSS.cdf`

*Method* of `PoissonLSS`

```python
PoissonLSS.cdf(y, params) -> ndarray
```

Poisson CDF with response-scale mean `mu`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1697-1703](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1697-L1703)

(poissonlss-quantile)=
## `PoissonLSS.quantile`

*Method* of `PoissonLSS`

```python
PoissonLSS.quantile(p, params) -> ndarray
```

Poisson quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1705-1707](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1705-L1707)

(poissonlss-simulate)=
## `PoissonLSS.simulate`

*Method* of `PoissonLSS`

```python
PoissonLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw Poisson samples with response-scale mean `mu`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1709-1715](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1709-L1715)

(poissonlss-deviance_increment)=
## `PoissonLSS.deviance_increment`

*Method* of `PoissonLSS`

```python
PoissonLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1717-1719](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1717-L1719)

(poissonlss-pdf)=
## `PoissonLSS.pdf`

*Method* of `PoissonLSS`

```python
PoissonLSS.pdf(y, params) -> ndarray
```

Poisson probability mass at the integer counts `y`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1721-1731](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1721-L1731)
