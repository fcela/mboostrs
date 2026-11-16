# `PolicyResult`

Defined in `mboostrs.policy`; exported as `mboostrs.PolicyResult`.


*Class*

```python
PolicyResult(
    spec: str,
    family: object,
    mstop: int,
    profile: str,
    model: object,
    trail: tuple,
    spent: mboostrs.policy.Spent,
    cache: mboostrs.policy.FitCache,
    score: float | None,
    threshold_se: float,
    complexity: str,
    stopped_early: str | None,
    tuning: object = None,
    response: str = '',
    metric: str = 'mse',
    budget_report: dict | None = None,
    action_evidence: tuple = ()
) -> None
```

Specification selected by the policy, with evidence for each step.

`profile` is linear, additive, tensor, varying (interactions using
`by=` without a tensor), or hybrid (an accepted residual tree).

Source: [`python/mboostrs/policy.py` lines 901-963](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L901-L963)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`adopted`](#policyresult-adopted) | Property | The rungs the ladder climbed, in order. |
| [`fits`](#policyresult-fits) | Property | Model fits the whole procedure spent. |
| [`to_dict`](#policyresult-to_dict) | Method | A JSON-safe record: the specification, the trail and the cost. |

(policyresult-adopted)=
## `PolicyResult.adopted`

*Property* of `PolicyResult`

```python
PolicyResult.adopted
```

The rungs the ladder climbed, in order.

Source: [`python/mboostrs/policy.py` lines 927-930](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L927-L930)

(policyresult-fits)=
## `PolicyResult.fits`

*Property* of `PolicyResult`

```python
PolicyResult.fits
```

Model fits the whole procedure spent.

Source: [`python/mboostrs/policy.py` lines 932-935](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L932-L935)

(policyresult-to_dict)=
## `PolicyResult.to_dict`

*Method* of `PolicyResult`

```python
PolicyResult.to_dict(*, include_arrays=False)
```

A JSON-safe record: the specification, the trail and the cost.

Source: [`python/mboostrs/policy.py` lines 937-957](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L937-L957)
