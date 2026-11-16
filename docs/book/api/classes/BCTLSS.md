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

Source: [`python/mboostrs/gamlss.py` lines 2491-2588](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2491-L2588)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#bctlss-response) | Method |  |
| [`mean`](#bctlss-mean) | Method |  |
| [`pdf`](#bctlss-pdf) | Method |  |
| [`cdf`](#bctlss-cdf) | Method |  |
| [`quantile`](#bctlss-quantile) | Method |  |
| [`simulate`](#bctlss-simulate) | Method |  |

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

Source: [`python/mboostrs/gamlss.py` lines 2509-2515](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2509-L2515)

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

Source: [`python/mboostrs/gamlss.py` lines 2517-2521](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2517-L2521)

(bctlss-pdf)=
## `BCTLSS.pdf`

*Method* of `BCTLSS`

```python
BCTLSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2535-2559](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2535-L2559)

(bctlss-cdf)=
## `BCTLSS.cdf`

*Method* of `BCTLSS`

```python
BCTLSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2561-2579](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2561-L2579)

(bctlss-quantile)=
## `BCTLSS.quantile`

*Method* of `BCTLSS`

```python
BCTLSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2581-2582](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2581-L2582)

(bctlss-simulate)=
## `BCTLSS.simulate`

*Method* of `BCTLSS`

```python
BCTLSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2584-2588](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2584-L2588)
