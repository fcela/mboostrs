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

Source: [`python/mboostrs/gamlss.py` lines 2591-2692](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2591-L2692)

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

Source: [`python/mboostrs/gamlss.py` lines 2605-2611](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2605-L2611)

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

Source: [`python/mboostrs/gamlss.py` lines 2613-2617](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2613-L2617)

(bcpelss-pdf)=
## `BCPELSS.pdf`

*Method* of `BCPELSS`

```python
BCPELSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2631-2658](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2631-L2658)

(bcpelss-cdf)=
## `BCPELSS.cdf`

*Method* of `BCPELSS`

```python
BCPELSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2660-2683](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2660-L2683)

(bcpelss-quantile)=
## `BCPELSS.quantile`

*Method* of `BCPELSS`

```python
BCPELSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2685-2686](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2685-L2686)

(bcpelss-simulate)=
## `BCPELSS.simulate`

*Method* of `BCPELSS`

```python
BCPELSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2688-2692](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2688-L2692)
