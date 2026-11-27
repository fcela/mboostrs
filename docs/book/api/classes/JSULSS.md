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

Source: [`python/mboostrs/gamlss.py` lines 2473-2542](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2473-L2542)

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

Source: [`python/mboostrs/gamlss.py` lines 2484-2490](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2484-L2490)

(jsulss-mean)=
## `JSULSS.mean`

*Method* of `JSULSS`

```python
JSULSS.mean(params) -> ndarray
```

Conditional mean `mu` under the centered JSU parameterization.

Source: [`python/mboostrs/gamlss.py` lines 2492-2494](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2492-L2494)

(jsulss-cdf)=
## `JSULSS.cdf`

*Method* of `JSULSS`

```python
JSULSS.cdf(y, params) -> ndarray
```

Johnson SU CDF in the centered parameterization. Matches
`gamlss.dist::pJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2514-2523](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2514-L2523)

(jsulss-quantile)=
## `JSULSS.quantile`

*Method* of `JSULSS`

```python
JSULSS.quantile(p, params) -> ndarray
```

Johnson SU quantile. Matches `gamlss.dist::qJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2525-2527](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2525-L2527)

(jsulss-pdf)=
## `JSULSS.pdf`

*Method* of `JSULSS`

```python
JSULSS.pdf(y, params) -> ndarray
```

Johnson SU density in the centered parameterization. Matches
`gamlss.dist::dJSU`.

Source: [`python/mboostrs/gamlss.py` lines 2529-2542](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2529-L2542)
