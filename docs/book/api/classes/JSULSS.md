# `JSULSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.JSULSS`.


*Class*

```python
JSULSS(stabilization: str = 'none')
```

Johnson's SU family: `mu` (identity link, location), `sigma`
(log link, scale), `nu` (identity link, skewness), `tau` (log link,
kurtosis). Matches R `gamboostLSS::JSULSS()` (wrapping gamlss.dist
`JSU`).

Source: [`python/mboostrs/gamlss.py` lines 2421-2490](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2421-L2490)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional mean ``mu`` under the centered JSU parameterization. |
| `cdf` | Method | Johnson SU CDF in the centered parameterization. Matches |
| `quantile` | Method | Johnson SU quantile. Matches ``gamlss.dist::qJSU``. |
| `pdf` | Method | Johnson SU density in the centered parameterization. Matches |

(jsulss-response)=
## `JSULSS.response`

*Method* of `JSULSS`

```python
JSULSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2432-2438](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2432-L2438)

(jsulss-mean)=
## `JSULSS.mean`

*Method* of `JSULSS`

```python
JSULSS.mean(params) -> ndarray
```

Conditional mean `mu` under the centered JSU parameterization.

Source: [`python/mboostrs/gamlss.py` lines 2440-2442](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2440-L2442)

(jsulss-cdf)=
## `JSULSS.cdf`

*Method* of `JSULSS`

```python
JSULSS.cdf(y, params) -> ndarray
```

Johnson SU CDF in the centered parameterization. Matches
`gamlss.dist::pJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2462-2471](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2462-L2471)

(jsulss-quantile)=
## `JSULSS.quantile`

*Method* of `JSULSS`

```python
JSULSS.quantile(p, params) -> ndarray
```

Johnson SU quantile. Matches `gamlss.dist::qJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2473-2475](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2473-L2475)

(jsulss-pdf)=
## `JSULSS.pdf`

*Method* of `JSULSS`

```python
JSULSS.pdf(y, params) -> ndarray
```

Johnson SU density in the centered parameterization. Matches
`gamlss.dist::dJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2477-2490](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2477-L2490)
