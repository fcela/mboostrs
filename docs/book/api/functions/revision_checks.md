# `revision_checks`

Defined in `mboostrs.residual_checks`; exported as `mboostrs.revision_checks`.


*Function*

```python
revision_checks(
    model,
    cluster=None,
    time=None,
    alpha: float = 0.05,
    permutations: int = 999,
    draws: int = 200,
    seed: int = 0,
    refit_top: int = 5
) -> RevisionReport
```

Run the cluster, time and influence checks and report their revisions.

`cluster` and `time` are passed to `cluster_residual_check` and
`time_residual_check`; a check whose argument is None is skipped.
The influence check always runs. Every check uses the same `alpha` and
the same seed, so a rerun on the same fit gives the same report.

**Returns**

RevisionReport
    `findings` in the order cluster, time, influence.

Source: [`python/mboostrs/residual_checks.py` lines 706-730](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L706-L730)
