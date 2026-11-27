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

Source: [`python/mboostrs/gamlss.py` lines 1573-1667](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1573-L1667)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Mixture mean ``(1 - nu) * mu``. |
| `pmf` | Method | Return the ZINBI probability mass using the family parameterization. |
| `cdf` | Method | ZINB CDF. |
| `quantile` | Method | ZINBI quantile: smallest k with nu + (1 - nu) * NBI_CDF(k) >= p (size 1/sigma). |
| `simulate` | Method | Draw ZINBI samples: with probability nu, y=0; else NBI with dispersion sigma. |
| `deviance_increment` | Method |  |
| `pdf` | Method | Zero-inflated negative-binomial mass. |

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

Source: [`python/mboostrs/gamlss.py` lines 1588-1595](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1588-L1595)

(zinblss-mean)=
## `ZINBLSS.mean`

*Method* of `ZINBLSS`

```python
ZINBLSS.mean(params) -> ndarray
```

Mixture mean `(1 - nu) * mu`.

Source: [`python/mboostrs/gamlss.py` lines 1597-1599](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1597-L1599)

(zinblss-pmf)=
## `ZINBLSS.pmf`

*Method* of `ZINBLSS`

```python
ZINBLSS.pmf(y, params) -> ndarray
```

Return the ZINBI probability mass using the family parameterization.

This is the same response-scale distribution used by `pdf`.

Source: [`python/mboostrs/gamlss.py` lines 1601-1606](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1601-L1606)

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

Source: [`python/mboostrs/gamlss.py` lines 1608-1618](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1608-L1618)

(zinblss-quantile)=
## `ZINBLSS.quantile`

*Method* of `ZINBLSS`

```python
ZINBLSS.quantile(p, params) -> ndarray
```

ZINBI quantile: smallest k with nu + (1 - nu) * NBI_CDF(k) >= p (size 1/sigma).

Source: [`python/mboostrs/gamlss.py` lines 1620-1622](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1620-L1622)

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

Source: [`python/mboostrs/gamlss.py` lines 1624-1635](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1624-L1635)

(zinblss-deviance_increment)=
## `ZINBLSS.deviance_increment`

*Method* of `ZINBLSS`

```python
ZINBLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1637-1639](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1637-L1639)

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

Source: [`python/mboostrs/gamlss.py` lines 1641-1667](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1641-L1667)
