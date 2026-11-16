# `BCCGLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BCCGLSS`.


*Class*

```python
BCCGLSS(stabilization: str = 'none')
```

Box-Cox-Cole-Green family: `mu` (log link, median), `sigma`
(log link, scale), `nu` (identity link, Box-Cox power).
Matches R `gamboostLSS::BCCGLSS()` (wrapping gamlss.dist `BCCG`).
Requires `y > 0`.

Source: [`python/mboostrs/gamlss.py` lines 2338-2416](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2338-L2416)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#bccglss-response) | Method | Return response-scale parameters: log-linked mu and sigma, identity nu. |
| [`cdf`](#bccglss-cdf) | Method | Box-Cox-Cole-Green CDF. Matches ``gamlss.dist::pBCCG``. |
| [`quantile`](#bccglss-quantile) | Method | Box-Cox-Cole-Green quantile. Matches ``gamlss.dist::qBCCG``. |
| [`pdf`](#bccglss-pdf) | Method | Box-Cox-Cole-Green density. Matches ``gamlss.dist::dBCCG``. |

(bccglss-response)=
## `BCCGLSS.response`

*Method* of `BCCGLSS`

```python
BCCGLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

Return response-scale parameters: log-linked mu and sigma, identity nu.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2349-2356](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2349-L2356)

(bccglss-cdf)=
## `BCCGLSS.cdf`

*Method* of `BCCGLSS`

```python
BCCGLSS.cdf(y, params) -> ndarray
```

Box-Cox-Cole-Green CDF. Matches `gamlss.dist::pBCCG`.

The Box-Cox transform maps `y > 0` to a normal variate truncated
below, so the normal CDF is renormalized by `Phi(1 / (sigma |nu|))`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2358-2379](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2358-L2379)

(bccglss-quantile)=
## `BCCGLSS.quantile`

*Method* of `BCCGLSS`

```python
BCCGLSS.quantile(p, params) -> ndarray
```

Box-Cox-Cole-Green quantile. Matches `gamlss.dist::qBCCG`.

Source: [`python/mboostrs/gamlss.py` lines 2381-2383](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2381-L2383)

(bccglss-pdf)=
## `BCCGLSS.pdf`

*Method* of `BCCGLSS`

```python
BCCGLSS.pdf(y, params) -> ndarray
```

Box-Cox-Cole-Green density. Matches `gamlss.dist::dBCCG`.

Includes the truncation normalizer `Phi(1 / (sigma |nu|))` that makes
it integrate to one over `y > 0`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2385-2416](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2385-L2416)
