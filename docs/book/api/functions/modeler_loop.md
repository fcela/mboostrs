# `modeler_loop`

Defined in `mboostrs.modeler`; exported as `mboostrs.modeler_loop`.


*Function*

```python
modeler_loop(
    frame,
    response='y',
    family=None,
    transcript=None,
    judgment=None,
    causal=None,
    domain_context=None,
    locks=(),
    budget=250,
    max_rounds=4,
    mstop=300,
    seed=0,
    tune_start=None,
    preexploration='native',
    workflow_budget=None,
    progress=None
)
```

Run a measured specification search with optional LLM proposals.

`workflow_budget=CostBudget(...)` limits the entire call, including
discovery, diagnostic refits, comparison, final tuning and causal
nuisance fits. A simple linear/group model is fitted first as a usable
fallback. Exhaustion returns the most recent completed incumbent with
`stop_reason='budget_<limit>'` and an operation ledger in
`budget_report`. If no initial fit can be admitted, a structured
`workflow_budget` runtime error reports that no model is available.

Fit charges are hard admission limits. Native CV batches reserve all
their folds before dispatch; failed batches retain that charge because
their actual completed-fit count is unavailable. Time and memory are
checked between native calls, so a running call or its preparation can
overshoot them. Memory is the absolute process-lifetime high-water RSS
in MiB, not the incremental allocation of this search. `progress` is
called with one operation record after each admitted native call.

`budget` remains the autospec probe allowance. Without
`workflow_budget`, the original deterministic/LLM search is used.
`family` fixes the response family throughout discovery, comparison,
final tuning and the budget fallback; when omitted, autospec chooses it.
See `SpecificationJudgment` for edits and `LLMJudge` for
provider budgets; only measured candidates can be adopted.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/modeler.py` lines 1139-1228](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L1139-L1228)
