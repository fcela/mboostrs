# `PoissonLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.PoissonLSS`.


*Class*

```python
PoissonLSS(stabilization: str = 'none')
```

:math:`P(\mu)` Poisson location family: `mu` (log link, mean).
Matches R `gamboostLSS::PoissonLSS()` (wrapping gamlss.dist `PO`).
Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1618-1682](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1618-L1682)

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

Source: [`python/mboostrs/gamlss.py` lines 1631-1632](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1631-L1632)

(poissonlss-mean)=
## `PoissonLSS.mean`

*Method* of `PoissonLSS`

```python
PoissonLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1634-1636](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1634-L1636)

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

Source: [`python/mboostrs/gamlss.py` lines 1638-1646](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1638-L1646)

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

Source: [`python/mboostrs/gamlss.py` lines 1648-1654](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1648-L1654)

(poissonlss-quantile)=
## `PoissonLSS.quantile`

*Method* of `PoissonLSS`

```python
PoissonLSS.quantile(p, params) -> ndarray
```

Poisson quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1656-1658](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1656-L1658)

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

Source: [`python/mboostrs/gamlss.py` lines 1660-1666](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1660-L1666)

(poissonlss-deviance_increment)=
## `PoissonLSS.deviance_increment`

*Method* of `PoissonLSS`

```python
PoissonLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1668-1670](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1668-L1670)

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

Source: [`python/mboostrs/gamlss.py` lines 1672-1682](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1672-L1682)
