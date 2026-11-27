# `GaussianLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GaussianLSS`.


*Class*

```python
GaussianLSS(stabilization: str = 'none')
```

Normal location-scale family: `mu` (identity link), `sigma` (log link).

Matches R `gamboostLSS::GaussianLSS()`.

Source: [`python/mboostrs/gamlss.py` lines 1080-1137](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1080-L1137)

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

Source: [`python/mboostrs/gamlss.py` lines 1090-1096](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1090-L1096)

(gaussianlss-mean)=
## `GaussianLSS.mean`

*Method* of `GaussianLSS`

```python
GaussianLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1098-1100](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1098-L1100)

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

Source: [`python/mboostrs/gamlss.py` lines 1102-1110](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1102-L1110)

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

Source: [`python/mboostrs/gamlss.py` lines 1112-1119](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1112-L1119)

(gaussianlss-quantile)=
## `GaussianLSS.quantile`

*Method* of `GaussianLSS`

```python
GaussianLSS.quantile(p, params) -> ndarray
```

Normal quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1121-1123](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1121-L1123)

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

Source: [`python/mboostrs/gamlss.py` lines 1125-1132](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1125-L1132)

(gaussianlss-deviance_increment)=
## `GaussianLSS.deviance_increment`

*Method* of `GaussianLSS`

```python
GaussianLSS.deviance_increment(y, params) -> ndarray
```

Per-observation deviance contribution, `-2 * log(pdf(y))`.

Source: [`python/mboostrs/gamlss.py` lines 1134-1137](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1134-L1137)
