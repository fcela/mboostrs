# `NegativeBinomialLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.NegativeBinomialLSS`.


*Class*

```python
NegativeBinomialLSS(stabilization: str = 'none')
```

Negative-binomial location-scale family: `mu` (log link, mean), `sigma`
(log link, dispersion), variance `mu + sigma * mu**2`. This is
gamlss.dist `NBI`: `pdf` equals `dNBI(y, mu, sigma)`. gamboostLSS
has no family of this name; its `NBinomialLSS` (`NBinomialLSS`
here) is the same distribution with `sigma` naming the size
`1/sigma`. Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 1894-1980](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1894-L1980)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional count mean ``mu``. |
| `pmf` | Method | NBI probability mass with mean ``mu`` and variance ``mu + sigma * mu**2``. |
| `cdf` | Method | NBI CDF with mean ``mu`` and dispersion ``sigma`` (size ``1/sigma``). |
| `quantile` | Method | NBI quantile for probabilities ``p`` (size ``1/sigma``). |
| `simulate` | Method | Draw NBI samples with mean ``mu`` and dispersion ``sigma`` (size ``1/sigma``). |
| `deviance_increment` | Method |  |
| `pdf` | Method | NBI probability mass, variance ``mu + sigma * mu**2``. |

(negativebinomiallss-response)=
## `NegativeBinomialLSS.response`

*Method* of `NegativeBinomialLSS`

```python
NegativeBinomialLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1910-1914](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1910-L1914)

(negativebinomiallss-mean)=
## `NegativeBinomialLSS.mean`

*Method* of `NegativeBinomialLSS`

```python
NegativeBinomialLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 1916-1918](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1916-L1918)

(negativebinomiallss-pmf)=
## `NegativeBinomialLSS.pmf`

*Method* of `NegativeBinomialLSS`

```python
NegativeBinomialLSS.pmf(y, params) -> ndarray
```

NBI probability mass with mean `mu` and variance `mu + sigma * mu**2`.

The same function as `pdf`, which documents the
parameterization. It used the size `sigma` instead of `1/sigma`,
the NBinomialLSS parameterization, so mass, CDF, quantiles and draws
described a different distribution from the one the fit estimates.

Source: [`python/mboostrs/gamlss.py` lines 1920-1928](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1920-L1928)

(negativebinomiallss-cdf)=
## `NegativeBinomialLSS.cdf`

*Method* of `NegativeBinomialLSS`

```python
NegativeBinomialLSS.cdf(y, params) -> ndarray
```

NBI CDF with mean `mu` and dispersion `sigma` (size `1/sigma`).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1930-1937](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1930-L1937)

(negativebinomiallss-quantile)=
## `NegativeBinomialLSS.quantile`

*Method* of `NegativeBinomialLSS`

```python
NegativeBinomialLSS.quantile(p, params) -> ndarray
```

NBI quantile for probabilities `p` (size `1/sigma`).

Source: [`python/mboostrs/gamlss.py` lines 1939-1941](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1939-L1941)

(negativebinomiallss-simulate)=
## `NegativeBinomialLSS.simulate`

*Method* of `NegativeBinomialLSS`

```python
NegativeBinomialLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw NBI samples with mean `mu` and dispersion `sigma` (size `1/sigma`).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1943-1951](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1943-L1951)

(negativebinomiallss-deviance_increment)=
## `NegativeBinomialLSS.deviance_increment`

*Method* of `NegativeBinomialLSS`

```python
NegativeBinomialLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1953-1955](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1953-L1955)

(negativebinomiallss-pdf)=
## `NegativeBinomialLSS.pdf`

*Method* of `NegativeBinomialLSS`

```python
NegativeBinomialLSS.pdf(y, params) -> ndarray
```

NBI probability mass, variance `mu + sigma * mu**2`.

Matches `gamlss.dist::dNBI(y, mu, sigma)` with `sigma` passed
straight through: here `sigma` is gamlss' dispersion, so the size is
its reciprocal. `NBinomialLSS` is the same distribution with
`sigma` naming the size instead — the two are separate families for
that reason, not aliases.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1957-1980](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1957-L1980)
