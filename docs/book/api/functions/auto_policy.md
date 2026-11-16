# `auto_policy`

Defined in `mboostrs.policy`; exported as `mboostrs.auto_policy`.


*Function*

```python
auto_policy(
    data,
    response,
    family=None,
    metric='mse',
    complexity='balanced',
    folds=None,
    seed=0,
    mstop=100,
    df=4.0,
    budget=None,
    threshold_se=None,
    cache=None,
    progress=None,
    groups=(),
    na_action='error',
    unseen_levels='average',
    score_parameter=None,
    discovery_fits=100,
    preexploration='native',
    **fit_kwargs
)
```

Choose a specification with measured comparisons and a workflow budget.

`budget=CostBudget(...)` reserves fit charges across discovery,
tuning and evaluation. A simple fitted model is retained before that
search begins; if no CV score was completed, the returned `score` is
None and its evaluation status is `not_evaluated`. Fit charges are
hard admission limits, including failed native batches. Seconds and
process-lifetime high-water RSS (MiB) are cooperative checkpoints and
can overshoot within a call or preparation step. `budget_report`
records every admitted native fit operation and the blocked request.

`metric` scores candidates and selects stopping iterations on one
shared fold design. Pinball/expectile require `score_parameter`.
Without explicit folds, declared groups are held whole (connected
components for multiple grouping columns); otherwise five random folds
are used. The paired-SE rule is a search heuristic, not an inferential test.
A rejected stage retains the incumbent while later allowed stages test
other structure; no rejection forces adoption of its proposed terms.
Without a budget the existing ladder runs without admission limits.

`discovery_fits` is the main-effect probe allowance (default 100);
`0` skips fitted shape discovery. Pairs still screen the accepted
model's residual, and a tree can still be compared under the whole-workflow
`budget`. Discovery begins after the linear candidate has been scored
and retained. `preexploration` may
be `"native"`, `"neural"`, `"dual"` or `"off"`. Discovery
currently ranks shape probes by family loss; the requested `metric`
governs each complete candidate's stopping iteration and final adoption.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

Source: [`python/mboostrs/policy.py` lines 749-869](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L749-L869)
