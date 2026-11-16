# `GammaLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GammaLSS`.


*Class*

```python
GammaLSS(stabilization: str = 'none')
```

Gamma location-shape family: `mu` (log link, mean), `sigma` (log link,
shape). Matches R `gamboostLSS::GammaLSS()`. Requires `y > 0`.

Source: [`python/mboostrs/gamlss.py` lines 1159-1213](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1159-L1213)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#gammalss-response) | Method |  |
| [`mean`](#gammalss-mean) | Method | Conditional response mean. |
| [`pdf`](#gammalss-pdf) | Method | Gamma density with response-scale ``mu`` (mean) and ``sigma`` (shape). |
| [`cdf`](#gammalss-cdf) | Method | Gamma CDF with shape ``sigma`` and scale ``mu / sigma``. |
| [`quantile`](#gammalss-quantile) | Method | Gamma quantile, obtained by inverting :meth:`cdf`. |
| [`simulate`](#gammalss-simulate) | Method | Draw Gamma samples with response-scale ``mu`` and ``sigma``. |
| [`deviance_increment`](#gammalss-deviance_increment) | Method |  |

(gammalss-response)=
## `GammaLSS.response`

*Method* of `GammaLSS`

```python
GammaLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1168-1172](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1168-L1172)

(gammalss-mean)=
## `GammaLSS.mean`

*Method* of `GammaLSS`

```python
GammaLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1174-1176](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1174-L1176)

(gammalss-pdf)=
## `GammaLSS.pdf`

*Method* of `GammaLSS`

```python
GammaLSS.pdf(y, params) -> ndarray
```

Gamma density with response-scale `mu` (mean) and `sigma` (shape).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1178-1187](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1178-L1187)

(gammalss-cdf)=
## `GammaLSS.cdf`

*Method* of `GammaLSS`

```python
GammaLSS.cdf(y, params) -> ndarray
```

Gamma CDF with shape `sigma` and scale `mu / sigma`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1189-1196](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1189-L1196)

(gammalss-quantile)=
## `GammaLSS.quantile`

*Method* of `GammaLSS`

```python
GammaLSS.quantile(p, params) -> ndarray
```

Gamma quantile, obtained by inverting `cdf`.

Source: [`python/mboostrs/gamlss.py` lines 1198-1200](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1198-L1200)

(gammalss-simulate)=
## `GammaLSS.simulate`

*Method* of `GammaLSS`

```python
GammaLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw Gamma samples with response-scale `mu` and `sigma`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1202-1209](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1202-L1209)

(gammalss-deviance_increment)=
## `GammaLSS.deviance_increment`

*Method* of `GammaLSS`

```python
GammaLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1211-1213](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1211-L1213)
