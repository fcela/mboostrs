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

Source: [`python/mboostrs/gamlss.py` lines 2024-2081](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2024-L2081)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional response mean. |
| `pdf` | Method |  |
| `cdf` | Method | Inverse-Gaussian CDF. Matches ``gamlss.dist::pIG``. |
| `quantile` | Method | Inverse-Gaussian quantile, obtained by inverting :meth:`cdf`. |

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

Source: [`python/mboostrs/gamlss.py` lines 2037-2041](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2037-L2041)

(inversegaussianlss-mean)=
## `InverseGaussianLSS.mean`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 2043-2045](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2043-L2045)

(inversegaussianlss-pdf)=
## `InverseGaussianLSS.pdf`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.pdf(y, params)
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2047-2062](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2047-L2062)

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

Source: [`python/mboostrs/gamlss.py` lines 2064-2077](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2064-L2077)

(inversegaussianlss-quantile)=
## `InverseGaussianLSS.quantile`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.quantile(p, params) -> ndarray
```

Inverse-Gaussian quantile, obtained by inverting `cdf`.

Source: [`python/mboostrs/gamlss.py` lines 2079-2081](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2079-L2081)
