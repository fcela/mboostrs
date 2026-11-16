# `Cindex`

Defined in `mboostrs.families`; exported as `mboostrs.Cindex`.


*Class*

```python
Cindex(sigma=0.1)
```

Uno's IPCW-weighted concordance ("C-index") loss (mboost::Cindex()).

Response must be a `Surv(time, event)` pair (see `CoxPH`).
Case weights must be either 0 or 1 (this is the valid domain, not a
limitation -- it matches the subsampling/k-fold `cvrisk()`/`stabsel()`
fold format, both of which are supported for this family; bootstrap-style
integer-count folds are not).

Unlike `Gehan`, the offset is identically 0 in R (verified), so
raw fitted values match R directly -- no centering deviation.

**Parameters**

- `sigma` (float): Width of the sigmoid smoothing of the (otherwise non-differentiable)
  pairwise concordance indicator. Default 0.1, matching R.

Source: [`python/mboostrs/families.py` lines 370-392](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L370-L392)
