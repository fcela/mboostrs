# `BetaLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.BetaLSS`.


*Class*

```python
BetaLSS(stabilization: str = 'none')
```

Beta location-precision family: `mu` (logit link, mean in (0,1)),
`phi` (log link, precision). Matches R `gamboostLSS::BetaLSS()`. Requires
`0 < y < 1`.

Source: [`python/mboostrs/gamlss.py` lines 1335-1399](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1335-L1399)

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

Source: [`python/mboostrs/gamlss.py` lines 1345-1351](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1345-L1351)

(betalss-mean)=
## `BetaLSS.mean`

*Method* of `BetaLSS`

```python
BetaLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 1353-1355](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1353-L1355)

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

Source: [`python/mboostrs/gamlss.py` lines 1357-1373](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1357-L1373)

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

Source: [`python/mboostrs/gamlss.py` lines 1375-1382](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1375-L1382)

(betalss-quantile)=
## `BetaLSS.quantile`

*Method* of `BetaLSS`

```python
BetaLSS.quantile(p, params) -> ndarray
```

Beta quantile, obtained by inverting `cdf` on (0, 1).

Source: [`python/mboostrs/gamlss.py` lines 1384-1386](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1384-L1386)

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

Source: [`python/mboostrs/gamlss.py` lines 1388-1395](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1388-L1395)

(betalss-deviance_increment)=
## `BetaLSS.deviance_increment`

*Method* of `BetaLSS`

```python
BetaLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1397-1399](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1397-L1399)
