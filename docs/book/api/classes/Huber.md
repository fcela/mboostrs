# `Huber`

Defined in `mboostrs.families`; exported as `mboostrs.Huber`.


*Class*

```python
Huber(delta=None)
```

Huber's robust loss: quadratic near zero, linear in the tails.

Behaves like `Gaussian` for residuals within `delta` and like
`Laplace` beyond it, so it targets the mean while limiting the
influence of outliers. The negative gradient is `y - f` clipped to
`[-delta, delta]`.

**Parameters**

- `delta` (float, optional): The changeover point. When omitted it is set to the median absolute
  deviation of the current residuals and re-estimated as the fit
  proceeds, which is R `mboost`'s adaptive default.

Source: [`python/mboostrs/families.py` lines 133-152](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L133-L152)
