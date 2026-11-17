# `BCTLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BCTLSS`.


*Class*

```python
BCTLSS(stabilization: str = 'none')
```

Box-Cox t family with `mu`, `sigma`, `nu`, and `tau`.

`mu` and `nu` use identity links; `sigma` and Student-t degrees of
freedom `tau` use log links. The response must be positive, as must the
response-scale `mu`, `sigma`, and `tau`. The Box-Cox transform is
truncated at the positive-response boundary and the density is normalized
by the corresponding Student-t probability, matching
`gamlss.dist::BCT`.

`mu` is a location parameter, not generally the response mean. Use
quantiles and centile curves for interpretation; `mean` refuses.

Source: [`python/mboostrs/gamlss.py` lines 2493-2590](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2493-L2590)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method |  |
| `pdf` | Method |  |
| `cdf` | Method |  |
| `quantile` | Method |  |
| `simulate` | Method |  |

(bctlss-response)=
## `BCTLSS.response`

*Method* of `BCTLSS`

```python
BCTLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2511-2517](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2511-L2517)

(bctlss-mean)=
## `BCTLSS.mean`

*Method* of `BCTLSS`

```python
BCTLSS.mean(params) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 2519-2523](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2519-L2523)

(bctlss-pdf)=
## `BCTLSS.pdf`

*Method* of `BCTLSS`

```python
BCTLSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2537-2561](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2537-L2561)

(bctlss-cdf)=
## `BCTLSS.cdf`

*Method* of `BCTLSS`

```python
BCTLSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2563-2581](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2563-L2581)

(bctlss-quantile)=
## `BCTLSS.quantile`

*Method* of `BCTLSS`

```python
BCTLSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2583-2584](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2583-L2584)

(bctlss-simulate)=
## `BCTLSS.simulate`

*Method* of `BCTLSS`

```python
BCTLSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2586-2590](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2586-L2590)
