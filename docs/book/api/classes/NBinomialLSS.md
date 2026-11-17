# `NBinomialLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.NBinomialLSS`.


*Class*

```python
NBinomialLSS(stabilization: str = 'none')
```

Negative-binomial family: `mu` (log link, mean), `sigma` (log link,
size/dispersion). Matches R `gamboostLSS::NBinomialLSS()`. Requires
non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1350-1432](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1350-L1432)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#nbinomiallss-response) | Method |  |
| [`mean`](#nbinomiallss-mean) | Method | Conditional count mean ``mu``. |
| [`pmf`](#nbinomiallss-pmf) | Method | Negative-binomial mass with response-scale mean ``mu`` and size ``sigma``. |
| [`cdf`](#nbinomiallss-cdf) | Method | Negative-binomial CDF with response-scale mean ``mu`` and size ``sigma``. |
| [`quantile`](#nbinomiallss-quantile) | Method | Negative-binomial quantile for probabilities ``p``. |
| [`simulate`](#nbinomiallss-simulate) | Method | Draw negative-binomial samples with response-scale mean and size. |
| [`deviance_increment`](#nbinomiallss-deviance_increment) | Method |  |
| [`pdf`](#nbinomiallss-pdf) | Method | Negative-binomial probability mass with ``sigma`` naming the size, |

(nbinomiallss-response)=
## `NBinomialLSS.response`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1361-1365](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1361-L1365)

(nbinomiallss-mean)=
## `NBinomialLSS.mean`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1367-1369](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1367-L1369)

(nbinomiallss-pmf)=
## `NBinomialLSS.pmf`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.pmf(y, params) -> ndarray
```

Negative-binomial mass with response-scale mean `mu` and size `sigma`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1371-1383](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1371-L1383)

(nbinomiallss-cdf)=
## `NBinomialLSS.cdf`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.cdf(y, params) -> ndarray
```

Negative-binomial CDF with response-scale mean `mu` and size `sigma`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1385-1392](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1385-L1392)

(nbinomiallss-quantile)=
## `NBinomialLSS.quantile`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.quantile(p, params) -> ndarray
```

Negative-binomial quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1394-1396](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1394-L1396)

(nbinomiallss-simulate)=
## `NBinomialLSS.simulate`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw negative-binomial samples with response-scale mean and size.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1398-1405](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1398-L1405)

(nbinomiallss-deviance_increment)=
## `NBinomialLSS.deviance_increment`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1407-1409](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1407-L1409)

(nbinomiallss-pdf)=
## `NBinomialLSS.pdf`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.pdf(y, params) -> ndarray
```

Negative-binomial probability mass with `sigma` naming the size,
so the variance is `mu + mu**2 / sigma`.

Matches `gamlss.dist::dNBI(y, mu, 1 / sigma)`. See
`NegativeBinomialLSS`, which is the same distribution with
`sigma` naming gamlss' dispersion instead.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1411-1432](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1411-L1432)
