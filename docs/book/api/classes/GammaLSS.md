# `GammaLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GammaLSS`.


*Class*

```python
GammaLSS(stabilization: str = 'none')
```

Gamma location-shape family: `mu` (log link, mean), `sigma` (log link,
shape). Matches R `gamboostLSS::GammaLSS()`. Requires `y > 0`.

Source: [`python/mboostrs/gamlss.py` lines 1206-1260](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1206-L1260)

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

Source: [`python/mboostrs/gamlss.py` lines 1215-1219](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1215-L1219)

(gammalss-mean)=
## `GammaLSS.mean`

*Method* of `GammaLSS`

```python
GammaLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1221-1223](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1221-L1223)

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

Source: [`python/mboostrs/gamlss.py` lines 1225-1234](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1225-L1234)

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

Source: [`python/mboostrs/gamlss.py` lines 1236-1243](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1236-L1243)

(gammalss-quantile)=
## `GammaLSS.quantile`

*Method* of `GammaLSS`

```python
GammaLSS.quantile(p, params) -> ndarray
```

Gamma quantile, obtained by inverting `cdf`.

Source: [`python/mboostrs/gamlss.py` lines 1245-1247](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1245-L1247)

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

Source: [`python/mboostrs/gamlss.py` lines 1249-1256](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1249-L1256)

(gammalss-deviance_increment)=
## `GammaLSS.deviance_increment`

*Method* of `GammaLSS`

```python
GammaLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1258-1260](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1258-L1260)
