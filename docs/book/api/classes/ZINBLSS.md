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

Source: [`python/mboostrs/gamlss.py` lines 1519-1613](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1519-L1613)

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

Source: [`python/mboostrs/gamlss.py` lines 1534-1541](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1534-L1541)

(zinblss-mean)=
## `ZINBLSS.mean`

*Method* of `ZINBLSS`

```python
ZINBLSS.mean(params) -> ndarray
```

Mixture mean `(1 - nu) * mu`.

Source: [`python/mboostrs/gamlss.py` lines 1543-1545](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1543-L1545)

(zinblss-pmf)=
## `ZINBLSS.pmf`

*Method* of `ZINBLSS`

```python
ZINBLSS.pmf(y, params) -> ndarray
```

Return the ZINBI probability mass using the family parameterization.

This is the same response-scale distribution used by `pdf`.

Source: [`python/mboostrs/gamlss.py` lines 1547-1552](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1547-L1552)

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

Source: [`python/mboostrs/gamlss.py` lines 1554-1564](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1554-L1564)

(zinblss-quantile)=
## `ZINBLSS.quantile`

*Method* of `ZINBLSS`

```python
ZINBLSS.quantile(p, params) -> ndarray
```

ZINBI quantile: smallest k with nu + (1 - nu) * NBI_CDF(k) >= p (size 1/sigma).

Source: [`python/mboostrs/gamlss.py` lines 1566-1568](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1566-L1568)

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

Source: [`python/mboostrs/gamlss.py` lines 1570-1581](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1570-L1581)

(zinblss-deviance_increment)=
## `ZINBLSS.deviance_increment`

*Method* of `ZINBLSS`

```python
ZINBLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1583-1585](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1583-L1585)

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

Source: [`python/mboostrs/gamlss.py` lines 1587-1613](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1587-L1613)
