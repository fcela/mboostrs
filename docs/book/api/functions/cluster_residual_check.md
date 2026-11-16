# `cluster_residual_check`

Defined in `mboostrs.residual_checks`; exported as `mboostrs.cluster_residual_check`.


*Function*

```python
cluster_residual_check(
    model,
    cluster,
    alpha: float = 0.05,
    permutations: int = 999,
    draws: int = 200,
    seed: int = 0
) -> RevisionFinding
```

Measure the share of response-residual variance between clusters.

The statistic is the one-way ANOVA intra-cluster correlation (ICC) of the
response residuals: the share of residual variance that lies between
clusters, after allowing for unequal cluster sizes. An ICC near zero means
the residuals of one cluster are no more alike than those of different
clusters. A positive ICC means rows in the same cluster share a common
departure from the fitted mean, which a random intercept per cluster
(`brandom(cluster)`) would absorb.

**Parameters**

- `model` (glmboost or gamboost fit)
- `cluster` (str or array): Column name of the model's data, or one cluster label per row.
- `alpha` (float): Significance level of the decision.
- `permutations` (int): Shuffles of the residuals across rows, for the permutation p-value.
  Cluster sizes are kept, so the reference is exact under exchangeable
  residuals.
- `draws, seed`: Null datasets for the false-alarm rate, and the random seed.

**Returns**

RevisionFinding
    `flagged` is true when the ICC is positive and the permutation
    p-value is below `alpha`. `evidence` holds the cluster count, the
    cluster sizes and the cluster mean residuals.

**Notes**

A cluster check needs at least two clusters and some cluster with two or
more rows. With few clusters (under five) a random intercept is weakly
identified, so a flag then argues for grouped evaluation more than for a
particular intercept.

Source: [`python/mboostrs/residual_checks.py` lines 260-376](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L260-L376)
