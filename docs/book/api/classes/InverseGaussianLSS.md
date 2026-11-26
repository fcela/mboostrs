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

Source: [`python/mboostrs/gamlss.py` lines 2069-2126](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2069-L2126)

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

Source: [`python/mboostrs/gamlss.py` lines 2082-2086](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2082-L2086)

(inversegaussianlss-mean)=
## `InverseGaussianLSS.mean`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.mean(params) -> ndarray
```

Conditional response mean.

Source: [`python/mboostrs/gamlss.py` lines 2088-2090](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2088-L2090)

(inversegaussianlss-pdf)=
## `InverseGaussianLSS.pdf`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.pdf(y, params)
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2092-2107](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2092-L2107)

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

Source: [`python/mboostrs/gamlss.py` lines 2109-2122](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2109-L2122)

(inversegaussianlss-quantile)=
## `InverseGaussianLSS.quantile`

*Method* of `InverseGaussianLSS`

```python
InverseGaussianLSS.quantile(p, params) -> ndarray
```

Inverse-Gaussian quantile, obtained by inverting `cdf`.

Source: [`python/mboostrs/gamlss.py` lines 2124-2126](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2124-L2126)
