# `Bmrf`

Defined in `mboostrs.learners`; exported as `mboostrs.Bmrf`.


*Class*

```python
Bmrf(name, *, K, df=4, lambda_value=None, center=False, by=None)
```

Markov Random Field base-learner (R `bmrf`).

Fits spatially-smooth effects for categorical/factor variables with
a neighborhood penalty matrix K. At each boosting step, the indicator
design (one column per level) is Ridge-penalized by K.

Args:
    name: predictor name in the data frame (must be a factor/column
          whose integer codes give the region levels, 1-based).
    K: neighbourhood penalty matrix (n_levels x n_levels), in graph
       Laplacian form: `K[i, i]` is the number of neighbours of region
       `i`, `K[i, j] = -1` when `i` and `j` are adjacent, and 0
       otherwise. Every row sums to zero. This is what
       `BayesX::bnd2gra` produces and what `mboost::bmrf` requires.
       Passing an adjacency matrix instead (zero diagonal, `+1` for
       neighbours) is rejected: it has the same sparsity pattern but is
       indefinite, so it would penalise the wrong directions.
    df: degrees of freedom (default 4). Used to compute lambda when
        lambda is not supplied.
    lambda_value: optional smoothing parameter lambda. Takes precedence
                  over df when supplied.
    center: centre the design by projecting out the null space of K
            (default False). When True, the constant effect is
            absorbed into the intercept.
    by: optional varying-coefficient variable name.
    bins: optional quantile-bin count per coordinate axis. Only occupied
        two-dimensional cells are stored.

Source: [`python/mboostrs/learners.py` lines 453-493](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L453-L493)
