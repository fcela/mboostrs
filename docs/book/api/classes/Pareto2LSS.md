# `Pareto2LSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.Pareto2LSS`.


*Class*

```python
Pareto2LSS(stabilization: str = 'none')
```

Pareto type-2 (Lomax) family: `mu` (log link, scale), `sigma` (log
link, tail shape). The density is `sigma * mu**sigma / (y + mu)**(sigma + 1)`
for `y > 0`; a smaller `sigma` is a heavier tail, and the mean
`mu / (sigma - 1)` exists only for `sigma > 1`.

gamlss.dist `PARETO2` uses the reciprocal shape: this family's
`(mu, sigma)` is `PARETO2(mu, 1 / sigma)`, so `pdf` equals
`dPARETO2(y, mu, 1 / sigma)`. Requires `y > 0`.

Source: [`python/mboostrs/gamlss.py` lines 2238-2298](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2238-L2298)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Lomax mean ``mu / (sigma - 1)``, defined only for ``sigma > 1``. |
| `cdf` | Method | Pareto-II (Lomax) CDF, ``1 - (mu / (y + mu))**sigma``. |
| `quantile` | Method | Pareto-II quantile, ``mu * ((1 - p)**(-1 / sigma) - 1)``. |
| `pdf` | Method | Pareto-II (Lomax) density, ``sigma * mu**sigma / (y + mu)**(sigma+1)``. |

(pareto2lss-response)=
## `Pareto2LSS.response`

*Method* of `Pareto2LSS`

```python
Pareto2LSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2253-2257](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2253-L2257)

(pareto2lss-mean)=
## `Pareto2LSS.mean`

*Method* of `Pareto2LSS`

```python
Pareto2LSS.mean(params) -> ndarray
```

Lomax mean `mu / (sigma - 1)`, defined only for `sigma > 1`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2259-2265](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2259-L2265)

(pareto2lss-cdf)=
## `Pareto2LSS.cdf`

*Method* of `Pareto2LSS`

```python
Pareto2LSS.cdf(y, params) -> ndarray
```

Pareto-II (Lomax) CDF, `1 - (mu / (y + mu))**sigma`.

This is mboostrs' own parameterization, in which `sigma` is the shape.
`gamlss.dist::PARETO2` uses `1 / sigma` as the shape, so its CDF at
the same nominal `sigma` differs; see the class docstring.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2267-2280](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2267-L2280)

(pareto2lss-quantile)=
## `Pareto2LSS.quantile`

*Method* of `Pareto2LSS`

```python
Pareto2LSS.quantile(p, params) -> ndarray
```

Pareto-II quantile, `mu * ((1 - p)**(-1 / sigma) - 1)`.

Source: [`python/mboostrs/gamlss.py` lines 2282-2284](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2282-L2284)

(pareto2lss-pdf)=
## `Pareto2LSS.pdf`

*Method* of `Pareto2LSS`

```python
Pareto2LSS.pdf(y, params) -> ndarray
```

Pareto-II (Lomax) density, `sigma * mu**sigma / (y + mu)**(sigma+1)`.

mboostrs' parameterization, in which `sigma` is the shape; see the
class docstring for how it relates to `gamlss.dist::PARETO2`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2286-2298](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2286-L2298)
