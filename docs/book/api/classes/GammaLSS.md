# `GammaLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GammaLSS`.


*Class*

```python
GammaLSS(stabilization: str = 'none')
```

Gamma location-shape family: `mu` (log link, mean), `sigma` (log link,
shape). Matches R `gamboostLSS::GammaLSS()`. Requires `y > 0`.

Source: [`python/mboostrs/gamlss.py` lines 1161-1215](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1161-L1215)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional response mean. |
| `pdf` | Method | Gamma density with response-scale ``mu`` (mean) and ``sigma`` (shape). |
| `cdf` | Method | Gamma CDF with shape ``sigma`` and scale ``mu / sigma``. |
| `quantile` | Method | Gamma quantile, obtained by inverting :meth:`cdf`. |
| `simulate` | Method | Draw Gamma samples with response-scale ``mu`` and ``sigma``. |
| `deviance_increment` | Method |  |

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

Source: [`python/mboostrs/gamlss.py` lines 1170-1174](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1170-L1174)

(gammalss-mean)=
## `GammaLSS.mean`

*Method* of `GammaLSS`

```python
GammaLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1176-1178](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1176-L1178)

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

Source: [`python/mboostrs/gamlss.py` lines 1180-1189](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1180-L1189)

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

Source: [`python/mboostrs/gamlss.py` lines 1191-1198](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1191-L1198)

(gammalss-quantile)=
## `GammaLSS.quantile`

*Method* of `GammaLSS`

```python
GammaLSS.quantile(p, params) -> ndarray
```

Gamma quantile, obtained by inverting `cdf`.

Source: [`python/mboostrs/gamlss.py` lines 1200-1202](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1200-L1202)

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

Source: [`python/mboostrs/gamlss.py` lines 1204-1211](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1204-L1211)

(gammalss-deviance_increment)=
## `GammaLSS.deviance_increment`

*Method* of `GammaLSS`

```python
GammaLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1213-1215](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1213-L1215)
