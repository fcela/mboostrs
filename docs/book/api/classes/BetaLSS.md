# `BetaLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BetaLSS`.


*Class*

```python
BetaLSS(stabilization: str = 'none')
```

Beta location-precision family: `mu` (logit link, mean in (0,1)),
`phi` (log link, precision). Matches R `gamboostLSS::BetaLSS()`. Requires
`0 < y < 1`.

Source: [`python/mboostrs/gamlss.py` lines 1332-1396](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1332-L1396)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional response mean. |
| `pdf` | Method | Beta density with response-scale ``mu`` and precision ``phi``. |
| `cdf` | Method | Beta CDF with shapes ``mu * phi`` and ``(1 - mu) * phi``. |
| `quantile` | Method | Beta quantile, obtained by inverting :meth:`cdf` on (0, 1). |
| `simulate` | Method | Draw Beta samples with response-scale ``mu`` and precision ``phi``. |
| `deviance_increment` | Method |  |

(betalss-response)=
## `BetaLSS.response`

*Method* of `BetaLSS`

```python
BetaLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1342-1348](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1342-L1348)

(betalss-mean)=
## `BetaLSS.mean`

*Method* of `BetaLSS`

```python
BetaLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1350-1352](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1350-L1352)

(betalss-pdf)=
## `BetaLSS.pdf`

*Method* of `BetaLSS`

```python
BetaLSS.pdf(y, params) -> ndarray
```

Beta density with response-scale `mu` and precision `phi`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1354-1370](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1354-L1370)

(betalss-cdf)=
## `BetaLSS.cdf`

*Method* of `BetaLSS`

```python
BetaLSS.cdf(y, params) -> ndarray
```

Beta CDF with shapes `mu * phi` and `(1 - mu) * phi`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1372-1379](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1372-L1379)

(betalss-quantile)=
## `BetaLSS.quantile`

*Method* of `BetaLSS`

```python
BetaLSS.quantile(p, params) -> ndarray
```

Beta quantile, obtained by inverting `cdf` on (0, 1).

Source: [`python/mboostrs/gamlss.py` lines 1381-1383](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1381-L1383)

(betalss-simulate)=
## `BetaLSS.simulate`

*Method* of `BetaLSS`

```python
BetaLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw Beta samples with response-scale `mu` and precision `phi`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 1385-1392](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1385-L1392)

(betalss-deviance_increment)=
## `BetaLSS.deviance_increment`

*Method* of `BetaLSS`

```python
BetaLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1394-1396](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1394-L1396)
