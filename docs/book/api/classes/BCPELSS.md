# `BCPELSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BCPELSS`.


*Class*

```python
BCPELSS(stabilization: str = 'none')
```

Box-Cox power-exponential family with four fitted parameters.

Links and response support match `BCTLSS`. `tau` controls the
power-exponential kernel rather than Student-t tails. The truncated
positive-response density, CDF, quantile, and inverse-CDF simulation match
`gamlss.dist::BCPE`. `mu` is a location parameter; no general response
mean is declared.

Source: [`python/mboostrs/gamlss.py` lines 2593-2694](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2593-L2694)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#bcpelss-response) | Method |  |
| [`mean`](#bcpelss-mean) | Method |  |
| [`pdf`](#bcpelss-pdf) | Method |  |
| [`cdf`](#bcpelss-cdf) | Method |  |
| [`quantile`](#bcpelss-quantile) | Method |  |
| [`simulate`](#bcpelss-simulate) | Method |  |

(bcpelss-response)=
## `BCPELSS.response`

*Method* of `BCPELSS`

```python
BCPELSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2607-2613](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2607-L2613)

(bcpelss-mean)=
## `BCPELSS.mean`

*Method* of `BCPELSS`

```python
BCPELSS.mean(params) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 2615-2619](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2615-L2619)

(bcpelss-pdf)=
## `BCPELSS.pdf`

*Method* of `BCPELSS`

```python
BCPELSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2633-2660](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2633-L2660)

(bcpelss-cdf)=
## `BCPELSS.cdf`

*Method* of `BCPELSS`

```python
BCPELSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2662-2685](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2662-L2685)

(bcpelss-quantile)=
## `BCPELSS.quantile`

*Method* of `BCPELSS`

```python
BCPELSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2687-2688](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2687-L2688)

(bcpelss-simulate)=
## `BCPELSS.simulate`

*Method* of `BCPELSS`

```python
BCPELSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2690-2694](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2690-L2694)
