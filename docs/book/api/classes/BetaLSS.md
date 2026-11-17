# `BetaLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BetaLSS`.


*Class*

```python
BetaLSS(stabilization: str = 'none')
```

Beta location-precision family: `mu` (logit link, mean in (0,1)),
`phi` (log link, precision). Matches R `gamboostLSS::BetaLSS()`. Requires
`0 < y < 1`.

Source: [`python/mboostrs/gamlss.py` lines 1283-1347](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1283-L1347)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#betalss-response) | Method |  |
| [`mean`](#betalss-mean) | Method | Conditional response mean. |
| [`pdf`](#betalss-pdf) | Method | Beta density with response-scale ``mu`` and precision ``phi``. |
| [`cdf`](#betalss-cdf) | Method | Beta CDF with shapes ``mu * phi`` and ``(1 - mu) * phi``. |
| [`quantile`](#betalss-quantile) | Method | Beta quantile, obtained by inverting :meth:`cdf` on (0, 1). |
| [`simulate`](#betalss-simulate) | Method | Draw Beta samples with response-scale ``mu`` and precision ``phi``. |
| [`deviance_increment`](#betalss-deviance_increment) | Method |  |

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

Source: [`python/mboostrs/gamlss.py` lines 1293-1299](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1293-L1299)

(betalss-mean)=
## `BetaLSS.mean`

*Method* of `BetaLSS`

```python
BetaLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1301-1303](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1301-L1303)

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

Source: [`python/mboostrs/gamlss.py` lines 1305-1321](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1305-L1321)

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

Source: [`python/mboostrs/gamlss.py` lines 1323-1330](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1323-L1330)

(betalss-quantile)=
## `BetaLSS.quantile`

*Method* of `BetaLSS`

```python
BetaLSS.quantile(p, params) -> ndarray
```

Beta quantile, obtained by inverting `cdf` on (0, 1).

Source: [`python/mboostrs/gamlss.py` lines 1332-1334](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1332-L1334)

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

Source: [`python/mboostrs/gamlss.py` lines 1336-1343](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1336-L1343)

(betalss-deviance_increment)=
## `BetaLSS.deviance_increment`

*Method* of `BetaLSS`

```python
BetaLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1345-1347](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1345-L1347)
