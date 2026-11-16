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

Source: [`python/mboostrs/gamlss.py` lines 2419-2488](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2419-L2488)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#jsulss-response) | Method |  |
| [`mean`](#jsulss-mean) | Method | Conditional mean ``mu`` under the centered JSU parameterization. |
| [`cdf`](#jsulss-cdf) | Method | Johnson SU CDF in the centered parameterization. Matches |
| [`quantile`](#jsulss-quantile) | Method | Johnson SU quantile. Matches ``gamlss.dist::qJSU``. |
| [`pdf`](#jsulss-pdf) | Method | Johnson SU density in the centered parameterization. Matches |

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

Source: [`python/mboostrs/gamlss.py` lines 2430-2436](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2430-L2436)

(jsulss-mean)=
## `JSULSS.mean`

*Method* of `JSULSS`

```python
JSULSS.mean(params) -> ndarray
```

Conditional mean `mu` under the centered JSU parameterization.

Source: [`python/mboostrs/gamlss.py` lines 2438-2440](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2438-L2440)

(jsulss-cdf)=
## `JSULSS.cdf`

*Method* of `JSULSS`

```python
JSULSS.cdf(y, params) -> ndarray
```

Johnson SU CDF in the centered parameterization. Matches
`gamlss.dist::pJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2460-2469](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2460-L2469)

(jsulss-quantile)=
## `JSULSS.quantile`

*Method* of `JSULSS`

```python
JSULSS.quantile(p, params) -> ndarray
```

Johnson SU quantile. Matches `gamlss.dist::qJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2471-2473](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2471-L2473)

(jsulss-pdf)=
## `JSULSS.pdf`

*Method* of `JSULSS`

```python
JSULSS.pdf(y, params) -> ndarray
```

Johnson SU density in the centered parameterization. Matches
`gamlss.dist::dJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2475-2488](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2475-L2488)
