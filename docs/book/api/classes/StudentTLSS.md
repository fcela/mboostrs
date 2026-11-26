# `StudentTLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.StudentTLSS`.


*Class*

```python
StudentTLSS(stabilization: str = 'none')
```

Student's t family: `mu` (identity link), `sigma` (log link, scale),
`df` (log link, degrees of freedom). Matches R `gamboostLSS::StudentTLSS()`.

Source: [`python/mboostrs/gamlss.py` lines 1263-1325](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1263-L1325)

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

Source: [`python/mboostrs/gamlss.py` lines 1272-1278](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1272-L1278)

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

Source: [`python/mboostrs/gamlss.py` lines 1280-1285](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1280-L1285)

(studenttlss-pdf)=
## `StudentTLSS.pdf`

*Method* of `StudentTLSS`

```python
StudentTLSS.pdf(y, params) -> ndarray
```

Student-t density: location `mu`, scale `sigma`, df `df`.

Source: [`python/mboostrs/gamlss.py` lines 1287-1297](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1287-L1297)

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

Source: [`python/mboostrs/gamlss.py` lines 1299-1307](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1299-L1307)

(studenttlss-quantile)=
## `StudentTLSS.quantile`

*Method* of `StudentTLSS`

```python
StudentTLSS.quantile(p, params) -> ndarray
```

Student-t quantile, obtained by inverting the standardized CDF.

Source: [`python/mboostrs/gamlss.py` lines 1309-1311](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1309-L1311)

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

Source: [`python/mboostrs/gamlss.py` lines 1313-1321](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1313-L1321)

(studenttlss-deviance_increment)=
## `StudentTLSS.deviance_increment`

*Method* of `StudentTLSS`

```python
StudentTLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1323-1325](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1323-L1325)
