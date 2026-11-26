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

Source: [`python/mboostrs/gamlss.py` lines 2538-2635](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2538-L2635)

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

Source: [`python/mboostrs/gamlss.py` lines 2556-2562](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2556-L2562)

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

Source: [`python/mboostrs/gamlss.py` lines 2564-2568](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2564-L2568)

(bctlss-pdf)=
## `BCTLSS.pdf`

*Method* of `BCTLSS`

```python
BCTLSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2582-2606](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2582-L2606)

(bctlss-cdf)=
## `BCTLSS.cdf`

*Method* of `BCTLSS`

```python
BCTLSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2608-2626](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2608-L2626)

(bctlss-quantile)=
## `BCTLSS.quantile`

*Method* of `BCTLSS`

```python
BCTLSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2628-2629](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2628-L2629)

(bctlss-simulate)=
## `BCTLSS.simulate`

*Method* of `BCTLSS`

```python
BCTLSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2631-2635](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2631-L2635)
