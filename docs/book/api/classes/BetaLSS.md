# `BetaLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BetaLSS`.


*Class*

```python
BetaLSS(stabilization: str = 'none')
```

Beta location-precision family: `mu` (logit link, mean in (0,1)),
`phi` (log link, precision). Matches R `gamboostLSS::BetaLSS()`. Requires
`0 < y < 1`.

Source: [`python/mboostrs/gamlss.py` lines 1281-1345](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1281-L1345)

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

Source: [`python/mboostrs/gamlss.py` lines 1291-1297](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1291-L1297)

(betalss-mean)=
## `BetaLSS.mean`

*Method* of `BetaLSS`

```python
BetaLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1299-1301](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1299-L1301)

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

Source: [`python/mboostrs/gamlss.py` lines 1303-1319](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1303-L1319)

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

Source: [`python/mboostrs/gamlss.py` lines 1321-1328](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1321-L1328)

(betalss-quantile)=
## `BetaLSS.quantile`

*Method* of `BetaLSS`

```python
BetaLSS.quantile(p, params) -> ndarray
```

Beta quantile, obtained by inverting `cdf` on (0, 1).

Source: [`python/mboostrs/gamlss.py` lines 1330-1332](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1330-L1332)

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

Source: [`python/mboostrs/gamlss.py` lines 1334-1341](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1334-L1341)

(betalss-deviance_increment)=
## `BetaLSS.deviance_increment`

*Method* of `BetaLSS`

```python
BetaLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1343-1345](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1343-L1345)
