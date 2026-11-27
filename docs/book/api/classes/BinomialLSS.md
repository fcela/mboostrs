# `BinomialLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BinomialLSS`.


*Class*

```python
BinomialLSS(stabilization: str = 'none')
```

Binomial location family: `mu` (logit link, probability).
Matches R `gamboostLSS::BinomialLSS()` (wrapping gamlss.dist `BI`).
Requires `y` in `[0, 1]` (proportions).

Source: [`python/mboostrs/gamlss.py` lines 1815-1891](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1815-L1891)

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

Source: [`python/mboostrs/gamlss.py` lines 1826-1828](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1826-L1828)

(binomiallss-mean)=
## `BinomialLSS.mean`

*Method* of `BinomialLSS`

```python
BinomialLSS.mean(params) -> ndarray
```

Single-trial response mean, the success probability `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1830-1832](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1830-L1832)

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

Source: [`python/mboostrs/gamlss.py` lines 1834-1850](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1834-L1850)

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

Source: [`python/mboostrs/gamlss.py` lines 1852-1858](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1852-L1858)

(binomiallss-quantile)=
## `BinomialLSS.quantile`

*Method* of `BinomialLSS`

```python
BinomialLSS.quantile(p, params, size=1) -> ndarray
```

Binomial quantile: smallest `k` with `CDF(k) >= p`.

Source: [`python/mboostrs/gamlss.py` lines 1860-1862](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1860-L1862)

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

Source: [`python/mboostrs/gamlss.py` lines 1864-1871](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1864-L1871)

(binomiallss-deviance_increment)=
## `BinomialLSS.deviance_increment`

*Method* of `BinomialLSS`

```python
BinomialLSS.deviance_increment(y, params, size=1) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1873-1875](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1873-L1875)

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

Source: [`python/mboostrs/gamlss.py` lines 1877-1891](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1877-L1891)
