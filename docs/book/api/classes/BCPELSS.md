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

Source: [`python/mboostrs/gamlss.py` lines 2638-2739](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2638-L2739)

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

Source: [`python/mboostrs/gamlss.py` lines 2652-2658](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2652-L2658)

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

Source: [`python/mboostrs/gamlss.py` lines 2660-2664](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2660-L2664)

(bcpelss-pdf)=
## `BCPELSS.pdf`

*Method* of `BCPELSS`

```python
BCPELSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2678-2705](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2678-L2705)

(bcpelss-cdf)=
## `BCPELSS.cdf`

*Method* of `BCPELSS`

```python
BCPELSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2707-2730](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2707-L2730)

(bcpelss-quantile)=
## `BCPELSS.quantile`

*Method* of `BCPELSS`

```python
BCPELSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2732-2733](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2732-L2733)

(bcpelss-simulate)=
## `BCPELSS.simulate`

*Method* of `BCPELSS`

```python
BCPELSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2735-2739](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2735-L2739)
