# `GaussianLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GaussianLSS`.


*Class*

```python
GaussianLSS(stabilization: str = 'none')
```

Normal location-scale family: `mu` (identity link), `sigma` (log link).

Matches R `gamboostLSS::GaussianLSS()`.

Source: [`python/mboostrs/gamlss.py` lines 1083-1140](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1083-L1140)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional response mean. |
| `pdf` | Method | Normal density at ``y`` for response-scale ``mu`` and ``sigma``. |
| `cdf` | Method | Normal CDF at ``y`` for response-scale ``mu`` and ``sigma``. |
| `quantile` | Method | Normal quantile for probabilities ``p``. |
| `simulate` | Method | Draw Normal samples for response-scale ``mu`` and ``sigma``. |
| `deviance_increment` | Method | Per-observation deviance contribution, ``-2 * log(pdf(y))``. |

(gaussianlss-response)=
## `GaussianLSS.response`

*Method* of `GaussianLSS`

```python
GaussianLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1093-1099](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1093-L1099)

(gaussianlss-mean)=
## `GaussianLSS.mean`

*Method* of `GaussianLSS`

```python
GaussianLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1101-1103](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1101-L1103)

(gaussianlss-pdf)=
## `GaussianLSS.pdf`

*Method* of `GaussianLSS`

```python
GaussianLSS.pdf(y, params) -> ndarray
```

Normal density at `y` for response-scale `mu` and `sigma`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1105-1113](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1105-L1113)

(gaussianlss-cdf)=
## `GaussianLSS.cdf`

*Method* of `GaussianLSS`

```python
GaussianLSS.cdf(y, params) -> ndarray
```

Normal CDF at `y` for response-scale `mu` and `sigma`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1115-1122](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1115-L1122)

(gaussianlss-quantile)=
## `GaussianLSS.quantile`

*Method* of `GaussianLSS`

```python
GaussianLSS.quantile(p, params) -> ndarray
```

Normal quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1124-1126](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1124-L1126)

(gaussianlss-simulate)=
## `GaussianLSS.simulate`

*Method* of `GaussianLSS`

```python
GaussianLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw Normal samples for response-scale `mu` and `sigma`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1128-1135](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1128-L1135)

(gaussianlss-deviance_increment)=
## `GaussianLSS.deviance_increment`

*Method* of `GaussianLSS`

```python
GaussianLSS.deviance_increment(y, params) -> ndarray
```

Per-observation deviance contribution, `-2 * log(pdf(y))`.

Source: [`python/mboostrs/gamlss.py` lines 1137-1140](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1137-L1140)
