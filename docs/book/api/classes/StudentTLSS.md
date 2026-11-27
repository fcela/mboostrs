# `StudentTLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.StudentTLSS`.


*Class*

```python
StudentTLSS(stabilization: str = 'none')
```

Student's t family: `mu` (identity link), `sigma` (log link, scale),
`df` (log link, degrees of freedom). Matches R `gamboostLSS::StudentTLSS()`.

Source: [`python/mboostrs/gamlss.py` lines 1267-1329](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1267-L1329)

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

Source: [`python/mboostrs/gamlss.py` lines 1276-1282](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1276-L1282)

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

Source: [`python/mboostrs/gamlss.py` lines 1284-1289](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1284-L1289)

(studenttlss-pdf)=
## `StudentTLSS.pdf`

*Method* of `StudentTLSS`

```python
StudentTLSS.pdf(y, params) -> ndarray
```

Student-t density: location `mu`, scale `sigma`, df `df`.

Source: [`python/mboostrs/gamlss.py` lines 1291-1301](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1291-L1301)

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

Source: [`python/mboostrs/gamlss.py` lines 1303-1311](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1303-L1311)

(studenttlss-quantile)=
## `StudentTLSS.quantile`

*Method* of `StudentTLSS`

```python
StudentTLSS.quantile(p, params) -> ndarray
```

Student-t quantile, obtained by inverting the standardized CDF.

Source: [`python/mboostrs/gamlss.py` lines 1313-1315](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1313-L1315)

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

Source: [`python/mboostrs/gamlss.py` lines 1317-1325](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1317-L1325)

(studenttlss-deviance_increment)=
## `StudentTLSS.deviance_increment`

*Method* of `StudentTLSS`

```python
StudentTLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1327-1329](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1327-L1329)
