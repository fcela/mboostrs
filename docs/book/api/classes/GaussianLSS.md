# `GaussianLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GaussianLSS`.


*Class*

```python
GaussianLSS(stabilization: str = 'none')
```

Normal location-scale family: `mu` (identity link), `sigma` (log link).

Matches R `gamboostLSS::GaussianLSS()`.

Source: [`python/mboostrs/gamlss.py` lines 1031-1088](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1031-L1088)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#gaussianlss-response) | Method |  |
| [`mean`](#gaussianlss-mean) | Method | Conditional response mean. |
| [`pdf`](#gaussianlss-pdf) | Method | Normal density at ``y`` for response-scale ``mu`` and ``sigma``. |
| [`cdf`](#gaussianlss-cdf) | Method | Normal CDF at ``y`` for response-scale ``mu`` and ``sigma``. |
| [`quantile`](#gaussianlss-quantile) | Method | Normal quantile for probabilities ``p``. |
| [`simulate`](#gaussianlss-simulate) | Method | Draw Normal samples for response-scale ``mu`` and ``sigma``. |
| [`deviance_increment`](#gaussianlss-deviance_increment) | Method | Per-observation deviance contribution, ``-2 * log(pdf(y))``. |

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

Source: [`python/mboostrs/gamlss.py` lines 1041-1047](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1041-L1047)

(gaussianlss-mean)=
## `GaussianLSS.mean`

*Method* of `GaussianLSS`

```python
GaussianLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1049-1051](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1049-L1051)

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

Source: [`python/mboostrs/gamlss.py` lines 1053-1061](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1053-L1061)

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

Source: [`python/mboostrs/gamlss.py` lines 1063-1070](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1063-L1070)

(gaussianlss-quantile)=
## `GaussianLSS.quantile`

*Method* of `GaussianLSS`

```python
GaussianLSS.quantile(p, params) -> ndarray
```

Normal quantile for probabilities `p`.

Source: [`python/mboostrs/gamlss.py` lines 1072-1074](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1072-L1074)

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

Source: [`python/mboostrs/gamlss.py` lines 1076-1083](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1076-L1083)

(gaussianlss-deviance_increment)=
## `GaussianLSS.deviance_increment`

*Method* of `GaussianLSS`

```python
GaussianLSS.deviance_increment(y, params) -> ndarray
```

Per-observation deviance contribution, `-2 * log(pdf(y))`.

Source: [`python/mboostrs/gamlss.py` lines 1085-1088](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1085-L1088)
