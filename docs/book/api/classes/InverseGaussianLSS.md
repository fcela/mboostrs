# `InverseGaussianLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.InverseGaussianLSS`.


*Class*

```python
InverseGaussianLSS(stabilization: str = 'none')
```

Inverse Gaussian family: `mu` (log link, mean), `sigma` (log link,
dispersion). Requires `y > 0`. Matches `gamlss.dist::IG()`.

`Var(y) = sigma**2 * mu**3`, so the variance grows faster with the mean
than Gamma's does — the usual choice for right-skewed positive data with
heavier mean-dependence than Gamma can express.

Source: [`python/mboostrs/gamlss.py` lines 2022-2079](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2022-L2079)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#inversegaussianlss-response) | Method |  |
| [`mean`](#inversegaussianlss-mean) | Method | Conditional response mean. |
| [`pdf`](#inversegaussianlss-pdf) | Method |  |
| [`cdf`](#inversegaussianlss-cdf) | Method | Inverse-Gaussian CDF. Matches ``gamlss.dist::pIG``. |
| [`quantile`](#inversegaussianlss-quantile) | Method | Inverse-Gaussian quantile, obtained by inverting :meth:`cdf`. |

(inversegaussianlss-response)=
## `InverseGaussianLSS.response`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2035-2039](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2035-L2039)

(inversegaussianlss-mean)=
## `InverseGaussianLSS.mean`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 2041-2043](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2041-L2043)

(inversegaussianlss-pdf)=
## `InverseGaussianLSS.pdf`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.pdf(y, params)
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2045-2060](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2045-L2060)

(inversegaussianlss-cdf)=
## `InverseGaussianLSS.cdf`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.cdf(y, params) -> ndarray
```

Inverse-Gaussian CDF. Matches `gamlss.dist::pIG`.

The closed form pairs a normal tail with a reflected one:
`Phi(a) + exp(2 / (mu * sigma^2)) * Phi(b)`. The exponential factor
overflows for a small `mu * sigma^2`, so the second term is formed in
logs and exponentiated once.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2062-2075](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2062-L2075)

(inversegaussianlss-quantile)=
## `InverseGaussianLSS.quantile`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.quantile(p, params) -> ndarray
```

Inverse-Gaussian quantile, obtained by inverting `cdf`.

Source: [`python/mboostrs/gamlss.py` lines 2077-2079](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2077-L2079)
