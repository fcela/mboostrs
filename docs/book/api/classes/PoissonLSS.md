# `PoissonLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.PoissonLSS`.


*Class*

```python
PoissonLSS(stabilization: str = 'none')
```

:math:`P(\mu)` Poisson location family: `mu` (log link, mean).
Matches R `gamboostLSS::PoissonLSS()` (wrapping gamlss.dist `PO`).
Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1670-1734](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1670-L1734)

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

Source: [`python/mboostrs/gamlss.py` lines 1683-1684](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1683-L1684)

(poissonlss-mean)=
## `PoissonLSS.mean`

*Method* of `PoissonLSS`

```python
PoissonLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1686-1688](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1686-L1688)

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

Source: [`python/mboostrs/gamlss.py` lines 1690-1698](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1690-L1698)

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

Source: [`python/mboostrs/gamlss.py` lines 1700-1706](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1700-L1706)

(poissonlss-quantile)=
## `PoissonLSS.quantile`

*Method* of `PoissonLSS`

```python
PoissonLSS.quantile(p, params) -> ndarray
```

Poisson quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1708-1710](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1708-L1710)

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

Source: [`python/mboostrs/gamlss.py` lines 1712-1718](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1712-L1718)

(poissonlss-deviance_increment)=
## `PoissonLSS.deviance_increment`

*Method* of `PoissonLSS`

```python
PoissonLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1720-1722](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1720-L1722)

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

Source: [`python/mboostrs/gamlss.py` lines 1724-1734](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1724-L1734)
