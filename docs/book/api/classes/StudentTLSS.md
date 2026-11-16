# `StudentTLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.StudentTLSS`.


*Class*

```python
StudentTLSS(stabilization: str = 'none')
```

Student's t family: `mu` (identity link), `sigma` (log link, scale),
`df` (log link, degrees of freedom). Matches R `gamboostLSS::StudentTLSS()`.

Source: [`python/mboostrs/gamlss.py` lines 1216-1278](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1216-L1278)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#studenttlss-response) | Method |  |
| [`mean`](#studenttlss-mean) | Method | Conditional response mean, defined only for df > 1. |
| [`pdf`](#studenttlss-pdf) | Method | Student-t density: location ``mu``, scale ``sigma``, df ``df``. |
| [`cdf`](#studenttlss-cdf) | Method | Student-t CDF for location ``mu``, scale ``sigma``, and ``df``. |
| [`quantile`](#studenttlss-quantile) | Method | Student-t quantile, obtained by inverting the standardized CDF. |
| [`simulate`](#studenttlss-simulate) | Method | Draw Student-t samples: mu + sigma * standard_t(df). |
| [`deviance_increment`](#studenttlss-deviance_increment) | Method |  |

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

Source: [`python/mboostrs/gamlss.py` lines 1225-1231](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1225-L1231)

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

Source: [`python/mboostrs/gamlss.py` lines 1233-1238](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1233-L1238)

(studenttlss-pdf)=
## `StudentTLSS.pdf`

*Method* of `StudentTLSS`

```python
StudentTLSS.pdf(y, params) -> ndarray
```

Student-t density: location `mu`, scale `sigma`, df `df`.

Source: [`python/mboostrs/gamlss.py` lines 1240-1250](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1240-L1250)

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

Source: [`python/mboostrs/gamlss.py` lines 1252-1260](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1252-L1260)

(studenttlss-quantile)=
## `StudentTLSS.quantile`

*Method* of `StudentTLSS`

```python
StudentTLSS.quantile(p, params) -> ndarray
```

Student-t quantile, obtained by inverting the standardized CDF.

Source: [`python/mboostrs/gamlss.py` lines 1262-1264](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1262-L1264)

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

Source: [`python/mboostrs/gamlss.py` lines 1266-1274](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1266-L1274)

(studenttlss-deviance_increment)=
## `StudentTLSS.deviance_increment`

*Method* of `StudentTLSS`

```python
StudentTLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1276-1278](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1276-L1278)
