# `NBinomialLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.NBinomialLSS`.


*Class*

```python
NBinomialLSS(stabilization: str = 'none')
```

Negative-binomial family: `mu` (log link, mean), `sigma` (log link,
size/dispersion). Matches R `gamboostLSS::NBinomialLSS()`. Requires
non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1399-1481](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1399-L1481)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional count mean ``mu``. |
| `pmf` | Method | Negative-binomial mass with response-scale mean ``mu`` and size ``sigma``. |
| `cdf` | Method | Negative-binomial CDF with response-scale mean ``mu`` and size ``sigma``. |
| `quantile` | Method | Negative-binomial quantile for probabilities ``p``. |
| `simulate` | Method | Draw negative-binomial samples with response-scale mean and size. |
| `deviance_increment` | Method |  |
| `pdf` | Method | Negative-binomial probability mass with ``sigma`` naming the size, |

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

Source: [`python/mboostrs/gamlss.py` lines 1410-1414](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1410-L1414)

(nbinomiallss-mean)=
## `NBinomialLSS.mean`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1416-1418](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1416-L1418)

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

Source: [`python/mboostrs/gamlss.py` lines 1420-1432](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1420-L1432)

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

Source: [`python/mboostrs/gamlss.py` lines 1434-1441](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1434-L1441)

(nbinomiallss-quantile)=
## `NBinomialLSS.quantile`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.quantile(p, params) -> ndarray
```

Negative-binomial quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1443-1445](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1443-L1445)

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

Source: [`python/mboostrs/gamlss.py` lines 1447-1454](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1447-L1454)

(nbinomiallss-deviance_increment)=
## `NBinomialLSS.deviance_increment`

*Method* of `NBinomialLSS`

```python
NBinomialLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1456-1458](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1456-L1458)

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

Source: [`python/mboostrs/gamlss.py` lines 1460-1481](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1460-L1481)
