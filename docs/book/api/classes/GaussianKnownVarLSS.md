# `GaussianKnownVarLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.GaussianKnownVarLSS`.


*Class*

```python
GaussianKnownVarLSS(stabilization: str = 'none')
```

Normal location-scale family for outcomes that carry a known sampling
variance: `v_i = s2_i + exp(eta_sigma_i)`, the known sampling variance
plus a boosted *excess* variance component.

Made for regressing published small-area estimates (life expectancy,
county means), whose standard errors vary by orders of magnitude across
areas: a plain `GaussianLSS` fit mistakes that sampling noise for
heterogeneity and its sigma-submodel selects proxies for sample size.

The response is a two-column `cbind` tuple — pass the formulas as
`"cbind(le, s2) ~ ..."` with `s2` the squared standard error —
mirroring how survival families take `Surv(time, status)`.
Both submodels are boosted: `mu` carries its own additive predictor
as in `GaussianLSS`, and `eta_sigma` models only the
*unknown excess* variance on top of the known `s2`. The gradient
toward `eta_sigma` is attenuated by `exp(eta_sigma) / v` wherever
the known component dominates, which is the correction. The
natural-scale `sigma` parameter is the excess standard deviation
`exp(eta_sigma / 2)`; the total standard deviation of a new
observation is `sqrt(s2 + exp(eta_sigma))` and is what
`pdf`, `cdf`, and `quantile` compute when given `s2`.

This family is available in both the Python and R frontends as
`GaussianKnownVarLSS()`. It has no upstream `gamboostLSS`
counterpart: it is an mboostrs addition, so there is no upstream
implementation to match or to check results against.

Source: [`python/mboostrs/gamlss.py` lines 1143-1210](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1143-L1210)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method | Link-scale to natural-scale: ``mu`` identity, ``sigma`` the excess |
| `mean` | Method | Conditional response mean; known and excess variance do not shift it. |
| `total_sd` | Method | Total standard deviation ``sqrt(s2 + excess variance)``. |
| `pdf` | Method | Normal density at ``y`` under the total variance. |
| `cdf` | Method |  |
| `quantile` | Method |  |

(gaussianknownvarlss-response)=
## `GaussianKnownVarLSS.response`

*Method* of `GaussianKnownVarLSS`

```python
GaussianKnownVarLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

Link-scale to natural-scale: `mu` identity, `sigma` the excess
standard deviation `exp(eta / 2)`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 1177-1185](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1177-L1185)

(gaussianknownvarlss-mean)=
## `GaussianKnownVarLSS.mean`

*Method* of `GaussianKnownVarLSS`

```python
GaussianKnownVarLSS.mean(params) -> ndarray
```

Conditional response mean; known and excess variance do not shift it.

Source: [`python/mboostrs/gamlss.py` lines 1187-1189](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1187-L1189)

(gaussianknownvarlss-total_sd)=
## `GaussianKnownVarLSS.total_sd`

*Method* of `GaussianKnownVarLSS`

```python
GaussianKnownVarLSS.total_sd(params, s2) -> ndarray
```

Total standard deviation `sqrt(s2 + excess variance)`.

Source: [`python/mboostrs/gamlss.py` lines 1191-1195](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1191-L1195)

(gaussianknownvarlss-pdf)=
## `GaussianKnownVarLSS.pdf`

*Method* of `GaussianKnownVarLSS`

```python
GaussianKnownVarLSS.pdf(y, params, s2) -> ndarray
```

Normal density at `y` under the total variance.

Source: [`python/mboostrs/gamlss.py` lines 1197-1202](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1197-L1202)

(gaussianknownvarlss-cdf)=
## `GaussianKnownVarLSS.cdf`

*Method* of `GaussianKnownVarLSS`

```python
GaussianKnownVarLSS.cdf(y, params, s2) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1204-1207](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1204-L1207)

(gaussianknownvarlss-quantile)=
## `GaussianKnownVarLSS.quantile`

*Method* of `GaussianKnownVarLSS`

```python
GaussianKnownVarLSS.quantile(p, params, s2) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 1209-1210](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L1209-L1210)
