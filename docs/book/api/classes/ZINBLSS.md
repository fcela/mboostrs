# `ZINBLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.ZINBLSS`.


*Class*

```python
ZINBLSS(stabilization: str = 'none')
```

Zero-inflated negative-binomial (type I) family: `mu` (log link, mean),
`sigma` (log link, dispersion), `nu` (logit link, zero-inflation
probability). Matches R `gamboostLSS::ZINBLSS()` (wrapping gamlss.dist
`ZINBI`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1521-1615](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1521-L1615)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#zinblss-response) | Method |  |
| [`mean`](#zinblss-mean) | Method | Mixture mean ``(1 - nu) * mu``. |
| [`pmf`](#zinblss-pmf) | Method | Return the ZINBI probability mass using the family parameterization. |
| [`cdf`](#zinblss-cdf) | Method | ZINB CDF. |
| [`quantile`](#zinblss-quantile) | Method | ZINBI quantile: smallest k with nu + (1 - nu) * NBI_CDF(k) >= p (size 1/sigma). |
| [`simulate`](#zinblss-simulate) | Method | Draw ZINBI samples: with probability nu, y=0; else NBI with dispersion sigma. |
| [`deviance_increment`](#zinblss-deviance_increment) | Method |  |
| [`pdf`](#zinblss-pdf) | Method | Zero-inflated negative-binomial mass. |

(zinblss-response)=
## `ZINBLSS.response`

*Method* of `ZINBLSS`

```python
ZINBLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1536-1543](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1536-L1543)

(zinblss-mean)=
## `ZINBLSS.mean`

*Method* of `ZINBLSS`

```python
ZINBLSS.mean(params) -> ndarray
```

Mixture mean `(1 - nu) * mu`.

Source: [`python/mboostrs/gamlss.py` lines 1545-1547](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1545-L1547)

(zinblss-pmf)=
## `ZINBLSS.pmf`

*Method* of `ZINBLSS`

```python
ZINBLSS.pmf(y, params) -> ndarray
```

Return the ZINBI probability mass using the family parameterization.

This is the same response-scale distribution used by `pdf`.

Source: [`python/mboostrs/gamlss.py` lines 1549-1554](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1549-L1554)

(zinblss-cdf)=
## `ZINBLSS.cdf`

*Method* of `ZINBLSS`

```python
ZINBLSS.cdf(y, params) -> ndarray
```

ZINB CDF.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1556-1566](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1556-L1566)

(zinblss-quantile)=
## `ZINBLSS.quantile`

*Method* of `ZINBLSS`

```python
ZINBLSS.quantile(p, params) -> ndarray
```

ZINBI quantile: smallest k with nu + (1 - nu) * NBI_CDF(k) >= p (size 1/sigma).

Source: [`python/mboostrs/gamlss.py` lines 1568-1570](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1568-L1570)

(zinblss-simulate)=
## `ZINBLSS.simulate`

*Method* of `ZINBLSS`

```python
ZINBLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw ZINBI samples: with probability nu, y=0; else NBI with dispersion sigma.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1572-1583](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1572-L1583)

(zinblss-deviance_increment)=
## `ZINBLSS.deviance_increment`

*Method* of `ZINBLSS`

```python
ZINBLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1585-1587](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1585-L1587)

(zinblss-pdf)=
## `ZINBLSS.pdf`

*Method* of `ZINBLSS`

```python
ZINBLSS.pdf(y, params) -> ndarray
```

Zero-inflated negative-binomial mass.

`nu` is the extra-zero probability, so `P(0)` mixes it with the
negative binomial's own zero. `sigma` is gamlss' dispersion, as in
`NegativeBinomialLSS`, so the size is its reciprocal.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1589-1615](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1589-L1615)
