# `PoissonLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.PoissonLSS`.


*Class*

```python
PoissonLSS(stabilization: str = 'none')
```

:math:`P(\mu)` Poisson location family: `mu` (log link, mean).
Matches R `gamboostLSS::PoissonLSS()` (wrapping gamlss.dist `PO`).
Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1616-1680](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1616-L1680)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#poissonlss-response) | Method |  |
| [`mean`](#poissonlss-mean) | Method | Conditional count mean ``mu``. |
| [`pmf`](#poissonlss-pmf) | Method | Poisson mass with response-scale mean ``mu``. |
| [`cdf`](#poissonlss-cdf) | Method | Poisson CDF with response-scale mean ``mu``. |
| [`quantile`](#poissonlss-quantile) | Method | Poisson quantile for probabilities ``p``. |
| [`simulate`](#poissonlss-simulate) | Method | Draw Poisson samples with response-scale mean ``mu``. |
| [`deviance_increment`](#poissonlss-deviance_increment) | Method |  |
| [`pdf`](#poissonlss-pdf) | Method | Poisson probability mass at the integer counts ``y``. |

(poissonlss-response)=
## `PoissonLSS.response`

*Method* of `PoissonLSS`

```python
PoissonLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1629-1630](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1629-L1630)

(poissonlss-mean)=
## `PoissonLSS.mean`

*Method* of `PoissonLSS`

```python
PoissonLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1632-1634](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1632-L1634)

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

Source: [`python/mboostrs/gamlss.py` lines 1636-1644](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1636-L1644)

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

Source: [`python/mboostrs/gamlss.py` lines 1646-1652](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1646-L1652)

(poissonlss-quantile)=
## `PoissonLSS.quantile`

*Method* of `PoissonLSS`

```python
PoissonLSS.quantile(p, params) -> ndarray
```

Poisson quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1654-1656](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1654-L1656)

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

Source: [`python/mboostrs/gamlss.py` lines 1658-1664](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1658-L1664)

(poissonlss-deviance_increment)=
## `PoissonLSS.deviance_increment`

*Method* of `PoissonLSS`

```python
PoissonLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1666-1668](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1666-L1668)

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

Source: [`python/mboostrs/gamlss.py` lines 1670-1680](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1670-L1680)
