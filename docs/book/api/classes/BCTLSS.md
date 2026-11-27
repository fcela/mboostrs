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

Source: [`python/mboostrs/gamlss.py` lines 2545-2642](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2545-L2642)

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

Source: [`python/mboostrs/gamlss.py` lines 2563-2569](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2563-L2569)

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

Source: [`python/mboostrs/gamlss.py` lines 2571-2575](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2571-L2575)

(bctlss-pdf)=
## `BCTLSS.pdf`

*Method* of `BCTLSS`

```python
BCTLSS.pdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2589-2613](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2589-L2613)

(bctlss-cdf)=
## `BCTLSS.cdf`

*Method* of `BCTLSS`

```python
BCTLSS.cdf(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2615-2633](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2615-L2633)

(bctlss-quantile)=
## `BCTLSS.quantile`

*Method* of `BCTLSS`

```python
BCTLSS.quantile(p, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2635-2636](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2635-L2636)

(bctlss-simulate)=
## `BCTLSS.simulate`

*Method* of `BCTLSS`

```python
BCTLSS.simulate(params, rng=None, size=None) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2638-2642](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2638-L2642)
