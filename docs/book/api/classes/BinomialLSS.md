# `BinomialLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BinomialLSS`.


*Class*

```python
BinomialLSS(stabilization: str = 'none')
```

Binomial location family: `mu` (logit link, probability).
Matches R `gamboostLSS::BinomialLSS()` (wrapping gamlss.dist `BI`).
Requires `y` in `[0, 1]` (proportions).

Source: [`python/mboostrs/gamlss.py` lines 1812-1888](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1812-L1888)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Single-trial response mean, the success probability ``mu``. |
| `pmf` | Method | Binomial mass for ``size`` trials and response-scale probability ``mu``. |
| `cdf` | Method | Binomial CDF with ``size`` trials (R's ``pbinom``). |
| `quantile` | Method | Binomial quantile: smallest ``k`` with ``CDF(k) >= p``. |
| `simulate` | Method | Draw binomial samples with ``n_trials`` and response-scale ``mu``. |
| `deviance_increment` | Method |  |
| `pdf` | Method | Bernoulli mass for a response in ``{0, 1}``. |

(binomiallss-response)=
## `BinomialLSS.response`

*Method* of `BinomialLSS`

```python
BinomialLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1823-1825](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1823-L1825)

(binomiallss-mean)=
## `BinomialLSS.mean`

*Method* of `BinomialLSS`

```python
BinomialLSS.mean(params) -> ndarray
```

Single-trial response mean, the success probability `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1827-1829](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1827-L1829)

(binomiallss-pmf)=
## `BinomialLSS.pmf`

*Method* of `BinomialLSS`

```python
BinomialLSS.pmf(y, params, size=1) -> ndarray
```

Binomial mass for `size` trials and response-scale probability `mu`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1831-1847](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1831-L1847)

(binomiallss-cdf)=
## `BinomialLSS.cdf`

*Method* of `BinomialLSS`

```python
BinomialLSS.cdf(y, params, size=1) -> ndarray
```

Binomial CDF with `size` trials (R's `pbinom`).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1849-1855](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1849-L1855)

(binomiallss-quantile)=
## `BinomialLSS.quantile`

*Method* of `BinomialLSS`

```python
BinomialLSS.quantile(p, params, size=1) -> ndarray
```

Binomial quantile: smallest `k` with `CDF(k) >= p`.

Source: [`python/mboostrs/gamlss.py` lines 1857-1859](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1857-L1859)

(binomiallss-simulate)=
## `BinomialLSS.simulate`

*Method* of `BinomialLSS`

```python
BinomialLSS.simulate(params, rng=None, size=None, n_trials=1) -> ndarray
```

Draw binomial samples with `n_trials` and response-scale `mu`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1861-1868](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1861-L1868)

(binomiallss-deviance_increment)=
## `BinomialLSS.deviance_increment`

*Method* of `BinomialLSS`

```python
BinomialLSS.deviance_increment(y, params, size=1) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1870-1872](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1870-L1872)

(binomiallss-pdf)=
## `BinomialLSS.pdf`

*Method* of `BinomialLSS`

```python
BinomialLSS.pdf(y, params) -> ndarray
```

Bernoulli mass for a response in `{0, 1}`.

`mu` is the success probability, so this is `mu` at 1 and
`1 - mu` at 0. The family models a single trial; a count of
successes out of several needs a `Binomial` standard family instead.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1874-1888](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1874-L1888)
