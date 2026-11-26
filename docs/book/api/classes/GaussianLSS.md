# `GaussianLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GaussianLSS`.


*Class*

```python
GaussianLSS(stabilization: str = 'none')
```

Normal location-scale family: `mu` (identity link), `sigma` (log link).

Matches R `gamboostLSS::GaussianLSS()`.

Source: [`python/mboostrs/gamlss.py` lines 1076-1133](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1076-L1133)

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

Source: [`python/mboostrs/gamlss.py` lines 1086-1092](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1086-L1092)

(gaussianlss-mean)=
## `GaussianLSS.mean`

*Method* of `GaussianLSS`

```python
GaussianLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1094-1096](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1094-L1096)

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

Source: [`python/mboostrs/gamlss.py` lines 1098-1106](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1098-L1106)

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

Source: [`python/mboostrs/gamlss.py` lines 1108-1115](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1108-L1115)

(gaussianlss-quantile)=
## `GaussianLSS.quantile`

*Method* of `GaussianLSS`

```python
GaussianLSS.quantile(p, params) -> ndarray
```

Normal quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1117-1119](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1117-L1119)

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

Source: [`python/mboostrs/gamlss.py` lines 1121-1128](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1121-L1128)

(gaussianlss-deviance_increment)=
## `GaussianLSS.deviance_increment`

*Method* of `GaussianLSS`

```python
GaussianLSS.deviance_increment(y, params) -> ndarray
```

Per-observation deviance contribution, `-2 * log(pdf(y))`.

Source: [`python/mboostrs/gamlss.py` lines 1130-1133](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1130-L1133)
