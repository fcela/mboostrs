# `BinomialLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BinomialLSS`.


*Class*

```python
BinomialLSS(stabilization: str = 'none')
```

Binomial location family: `mu` (logit link, probability).
Matches R `gamboostLSS::BinomialLSS()` (wrapping gamlss.dist `BI`).
Requires `y` in `[0, 1]` (proportions).

Source: [`python/mboostrs/gamlss.py` lines 1761-1837](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1761-L1837)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#binomiallss-response) | Method |  |
| [`mean`](#binomiallss-mean) | Method | Single-trial response mean, the success probability ``mu``. |
| [`pmf`](#binomiallss-pmf) | Method | Binomial mass for ``size`` trials and response-scale probability ``mu``. |
| [`cdf`](#binomiallss-cdf) | Method | Binomial CDF with ``size`` trials (R's ``pbinom``). |
| [`quantile`](#binomiallss-quantile) | Method | Binomial quantile: smallest ``k`` with ``CDF(k) >= p``. |
| [`simulate`](#binomiallss-simulate) | Method | Draw binomial samples with ``n_trials`` and response-scale ``mu``. |
| [`deviance_increment`](#binomiallss-deviance_increment) | Method |  |
| [`pdf`](#binomiallss-pdf) | Method | Bernoulli mass for a response in ``{0, 1}``. |

(binomiallss-response)=
## `BinomialLSS.response`

*Method* of `BinomialLSS`

```python
BinomialLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1772-1774](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1772-L1774)

(binomiallss-mean)=
## `BinomialLSS.mean`

*Method* of `BinomialLSS`

```python
BinomialLSS.mean(params) -> ndarray
```

Single-trial response mean, the success probability `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1776-1778](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1776-L1778)

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

Source: [`python/mboostrs/gamlss.py` lines 1780-1796](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1780-L1796)

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

Source: [`python/mboostrs/gamlss.py` lines 1798-1804](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1798-L1804)

(binomiallss-quantile)=
## `BinomialLSS.quantile`

*Method* of `BinomialLSS`

```python
BinomialLSS.quantile(p, params, size=1) -> ndarray
```

Binomial quantile: smallest `k` with `CDF(k) >= p`.

Source: [`python/mboostrs/gamlss.py` lines 1806-1808](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1806-L1808)

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

Source: [`python/mboostrs/gamlss.py` lines 1810-1817](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1810-L1817)

(binomiallss-deviance_increment)=
## `BinomialLSS.deviance_increment`

*Method* of `BinomialLSS`

```python
BinomialLSS.deviance_increment(y, params, size=1) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1819-1821](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1819-L1821)

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

Source: [`python/mboostrs/gamlss.py` lines 1823-1837](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1823-L1837)
