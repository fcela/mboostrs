# `ZIPoLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.ZIPoLSS`.


*Class*

```python
ZIPoLSS(stabilization: str = 'none')
```

Zero-inflated Poisson family: `mu` (log link, Poisson mean), `sigma`
(logit link, zero-inflation probability). Matches R `gamboostLSS::ZIPoLSS()`
(wrapping gamlss.dist `ZIP`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1433-1516](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1433-L1516)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#zipolss-response) | Method |  |
| [`mean`](#zipolss-mean) | Method | Mixture mean ``(1 - sigma) * mu``. |
| [`pmf`](#zipolss-pmf) | Method | ZIP mass: P(y=0) = sigma + (1-sigma)*exp(-mu); |
| [`cdf`](#zipolss-cdf) | Method | ZIP CDF. |
| [`quantile`](#zipolss-quantile) | Method | ZIP quantile: smallest k with CDF(k) >= p. |
| [`simulate`](#zipolss-simulate) | Method | Draw ZIP samples: with probability sigma, y=0; else Poisson(mu). |
| [`deviance_increment`](#zipolss-deviance_increment) | Method |  |
| [`pdf`](#zipolss-pdf) | Method | Zero-inflated Poisson mass: ``sigma`` is the extra-zero probability, |

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

Source: [`python/mboostrs/gamlss.py` lines 1447-1454](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1447-L1454)

(zipolss-mean)=
## `ZIPoLSS.mean`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.mean(params) -> ndarray
```

Mixture mean `(1 - sigma) * mu`.

Source: [`python/mboostrs/gamlss.py` lines 1456-1458](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1456-L1458)

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

Source: [`python/mboostrs/gamlss.py` lines 1460-1472](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1460-L1472)

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

Source: [`python/mboostrs/gamlss.py` lines 1474-1483](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1474-L1483)

(zipolss-quantile)=
## `ZIPoLSS.quantile`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.quantile(p, params) -> ndarray
```

ZIP quantile: smallest k with CDF(k) >= p.

Source: [`python/mboostrs/gamlss.py` lines 1485-1487](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1485-L1487)

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

Source: [`python/mboostrs/gamlss.py` lines 1489-1498](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1489-L1498)

(zipolss-deviance_increment)=
## `ZIPoLSS.deviance_increment`

*Method* of `ZIPoLSS`

```python
ZIPoLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1500-1502](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1500-L1502)

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

Source: [`python/mboostrs/gamlss.py` lines 1504-1516](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1504-L1516)
