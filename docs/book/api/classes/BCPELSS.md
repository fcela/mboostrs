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

Source: [`python/mboostrs/gamlss.py` lines 2645-2746](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2645-L2746)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method |  |
| `pdf` | Method |  |
| `cdf` | Method |  |
| `quantile` | Method |  |
| `simulate` | Method |  |

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

Source: [`python/mboostrs/gamlss.py` lines 2659-2665](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2659-L2665)

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

Source: [`python/mboostrs/gamlss.py` lines 2667-2671](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2667-L2671)

(bcpelss-pdf)=
## `BCPELSS.pdf`

*Method* of `BCPELSS`

```python
BCPELSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2685-2712](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2685-L2712)

(bcpelss-cdf)=
## `BCPELSS.cdf`

*Method* of `BCPELSS`

```python
BCPELSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2714-2737](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2714-L2737)

(bcpelss-quantile)=
## `BCPELSS.quantile`

*Method* of `BCPELSS`

```python
BCPELSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2739-2740](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2739-L2740)

(bcpelss-simulate)=
## `BCPELSS.simulate`

*Method* of `BCPELSS`

```python
BCPELSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2742-2746](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2742-L2746)
