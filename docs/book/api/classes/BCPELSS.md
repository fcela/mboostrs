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

Source: [`python/mboostrs/gamlss.py` lines 2642-2743](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2642-L2743)

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

Source: [`python/mboostrs/gamlss.py` lines 2656-2662](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2656-L2662)

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

Source: [`python/mboostrs/gamlss.py` lines 2664-2668](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2664-L2668)

(bcpelss-pdf)=
## `BCPELSS.pdf`

*Method* of `BCPELSS`

```python
BCPELSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2682-2709](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2682-L2709)

(bcpelss-cdf)=
## `BCPELSS.cdf`

*Method* of `BCPELSS`

```python
BCPELSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2711-2734](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2711-L2734)

(bcpelss-quantile)=
## `BCPELSS.quantile`

*Method* of `BCPELSS`

```python
BCPELSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2736-2737](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2736-L2737)

(bcpelss-simulate)=
## `BCPELSS.simulate`

*Method* of `BCPELSS`

```python
BCPELSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2739-2743](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2739-L2743)
