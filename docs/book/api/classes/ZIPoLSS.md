# `ZIPoLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.ZIPoLSS`.


*Class*

```python
ZIPoLSS(stabilization: str = 'none')
```

Zero-inflated Poisson family: `mu` (log link, Poisson mean), `sigma`
(logit link, zero-inflation probability). Matches R `gamboostLSS::ZIPoLSS()`
(wrapping gamlss.dist `ZIP`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1484-1567](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1484-L1567)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Mixture mean ``(1 - sigma) * mu``. |
| `pmf` | Method | ZIP mass: P(y=0) = sigma + (1-sigma)*exp(-mu); |
| `cdf` | Method | ZIP CDF. |
| `quantile` | Method | ZIP quantile: smallest k with CDF(k) >= p. |
| `simulate` | Method | Draw ZIP samples: with probability sigma, y=0; else Poisson(mu). |
| `deviance_increment` | Method |  |
| `pdf` | Method | Zero-inflated Poisson mass: ``sigma`` is the extra-zero probability, |

(zipolss-response)=
## `ZIPoLSS.response`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1498-1505](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1498-L1505)

(zipolss-mean)=
## `ZIPoLSS.mean`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.mean(params) -> ndarray
```

Mixture mean `(1 - sigma) * mu`.

Source: [`python/mboostrs/gamlss.py` lines 1507-1509](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1507-L1509)

(zipolss-pmf)=
## `ZIPoLSS.pmf`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.pmf(y, params) -> ndarray
```

ZIP mass: P(y=0) = sigma + (1-sigma)*exp(-mu);
P(y>0) = (1-sigma)*Poisson(y; mu).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1511-1523](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1511-L1523)

(zipolss-cdf)=
## `ZIPoLSS.cdf`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.cdf(y, params) -> ndarray
```

ZIP CDF.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1525-1534](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1525-L1534)

(zipolss-quantile)=
## `ZIPoLSS.quantile`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.quantile(p, params) -> ndarray
```

ZIP quantile: smallest k with CDF(k) >= p.

Source: [`python/mboostrs/gamlss.py` lines 1536-1538](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1536-L1538)

(zipolss-simulate)=
## `ZIPoLSS.simulate`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw ZIP samples: with probability sigma, y=0; else Poisson(mu).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1540-1549](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1540-L1549)

(zipolss-deviance_increment)=
## `ZIPoLSS.deviance_increment`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1551-1553](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1551-L1553)

(zipolss-pdf)=
## `ZIPoLSS.pdf`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.pdf(y, params) -> ndarray
```

Zero-inflated Poisson mass: `sigma` is the extra-zero probability,
so `P(0) = sigma + (1-sigma) exp(-mu)`. Matches `gamlss.dist::dZIP`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1555-1567](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1555-L1567)
