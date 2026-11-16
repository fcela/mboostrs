# `Bkernel`

Defined in `mboostrs.learners`; exported as `mboostrs.Bkernel`.


*Class*

```python
Bkernel(
    name,
    bandwidth=1.0,
    df=4,
    kernel='gaussian',
    landmarks=None,
    center=False,
    by=None
)
```

Kernel base-learner (R `bkernel`).

Constructs an n×n kernel matrix K[i,j] = kernel(|x_i - x_j| / h),
then fits a kernel ridge regression on it. Because the design is n×n,
this is O(n²) in memory — avoid for large n.

Pass `landmarks=m` to build the Nyström approximation instead: an n×m
design against `m` evenly-spaced quantile landmarks of x, with the same
ridge as the exact path. Memory is O(n·m), and the approximation is exact
when `m` equals the number of distinct x values. Coefficients then index
the landmarks, so they number `m`, not `n`.

Args:
    name: predictor name in the data frame.
    bandwidth: kernel bandwidth h (default 1.0).
    df: degrees of freedom (default 4). Used to compute lambda.
    kernel: kernel type, one of "gaussian" (default),
        "epanechnikov", or "uniform".
    landmarks: number of Nyström landmarks (default None = exact n×n).
    center: centre the design by spectral projection (default False).
    by: optional varying-coefficient variable name.

Source: [`python/mboostrs/learners.py` lines 746-803](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L746-L803)
