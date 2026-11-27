# `StudentTLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.StudentTLSS`.


*Class*

```python
StudentTLSS(stabilization: str = 'none')
```

Student's t family: `mu` (identity link), `sigma` (log link, scale),
`df` (log link, degrees of freedom). Matches R `gamboostLSS::StudentTLSS()`.

Source: [`python/mboostrs/gamlss.py` lines 1270-1332](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1270-L1332)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional response mean, defined only for df > 1. |
| `pdf` | Method | Student-t density: location ``mu``, scale ``sigma``, df ``df``. |
| `cdf` | Method | Student-t CDF for location ``mu``, scale ``sigma``, and ``df``. |
| `quantile` | Method | Student-t quantile, obtained by inverting the standardized CDF. |
| `simulate` | Method | Draw Student-t samples: mu + sigma * standard_t(df). |
| `deviance_increment` | Method |  |

(studenttlss-response)=
## `StudentTLSS.response`

*Method* of `StudentTLSS`

```python
StudentTLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1279-1285](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1279-L1285)

(studenttlss-mean)=
## `StudentTLSS.mean`

*Method* of `StudentTLSS`

```python
StudentTLSS.mean(params) -> ndarray
```

Conditional response mean, defined only for df > 1.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1287-1292](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1287-L1292)

(studenttlss-pdf)=
## `StudentTLSS.pdf`

*Method* of `StudentTLSS`

```python
StudentTLSS.pdf(y, params) -> ndarray
```

Student-t density: location `mu`, scale `sigma`, df `df`.

Source: [`python/mboostrs/gamlss.py` lines 1294-1304](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1294-L1304)

(studenttlss-cdf)=
## `StudentTLSS.cdf`

*Method* of `StudentTLSS`

```python
StudentTLSS.cdf(y, params) -> ndarray
```

Student-t CDF for location `mu`, scale `sigma`, and `df`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1306-1314](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1306-L1314)

(studenttlss-quantile)=
## `StudentTLSS.quantile`

*Method* of `StudentTLSS`

```python
StudentTLSS.quantile(p, params) -> ndarray
```

Student-t quantile, obtained by inverting the standardized CDF.

Source: [`python/mboostrs/gamlss.py` lines 1316-1318](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1316-L1318)

(studenttlss-simulate)=
## `StudentTLSS.simulate`

*Method* of `StudentTLSS`

```python
StudentTLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw Student-t samples: mu + sigma * standard_t(df).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1320-1328](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1320-L1328)

(studenttlss-deviance_increment)=
## `StudentTLSS.deviance_increment`

*Method* of `StudentTLSS`

```python
StudentTLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1330-1332](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1330-L1332)
