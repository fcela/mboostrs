# `ZIPoLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.ZIPoLSS`.


*Class*

```python
ZIPoLSS(stabilization: str = 'none')
```

Zero-inflated Poisson family: `mu` (log link, Poisson mean), `sigma`
(logit link, zero-inflation probability). Matches R `gamboostLSS::ZIPoLSS()`
(wrapping gamlss.dist `ZIP`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1487-1570](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1487-L1570)

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

Source: [`python/mboostrs/gamlss.py` lines 1501-1508](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1501-L1508)

(zipolss-mean)=
## `ZIPoLSS.mean`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.mean(params) -> ndarray
```

Mixture mean `(1 - sigma) * mu`.

Source: [`python/mboostrs/gamlss.py` lines 1510-1512](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1510-L1512)

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

Source: [`python/mboostrs/gamlss.py` lines 1514-1526](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1514-L1526)

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

Source: [`python/mboostrs/gamlss.py` lines 1528-1537](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1528-L1537)

(zipolss-quantile)=
## `ZIPoLSS.quantile`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.quantile(p, params) -> ndarray
```

ZIP quantile: smallest k with CDF(k) >= p.

Source: [`python/mboostrs/gamlss.py` lines 1539-1541](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1539-L1541)

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

Source: [`python/mboostrs/gamlss.py` lines 1543-1552](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1543-L1552)

(zipolss-deviance_increment)=
## `ZIPoLSS.deviance_increment`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1554-1556](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1554-L1556)

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

Source: [`python/mboostrs/gamlss.py` lines 1558-1570](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1558-L1570)
