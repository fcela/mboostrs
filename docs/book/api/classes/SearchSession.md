# `SearchSession`

Defined in `mboostrs.search_session`; exported as `mboostrs.SearchSession`.


*Class*

```python
SearchSession(
    data,
    response,
    family=None,
    score='mse',
    folds=5,
    seed=0,
    weights=None,
    groups=(),
    budget=None,
    na_action='error',
    unseen_levels='average',
    score_parameter=None,
    nu=0.1,
    mstop=100,
    locks=(),
    causal=None,
    forbidden_pairs=(),
    mandatory_terms=(),
    heredity='none'
)
```

A persistent training-only search whose next diagnostic is caller-chosen.

`run` performs exactly the named action. `state` lists the operations
currently legal and their preconditions. The data/fold/metric boundary is
fixed when the session opens. Test rows are accepted only by the one-time
`final_evaluate` action after `freeze`. The Python caller must keep its
own test data out of training and domain-context arguments.

Source: [`python/mboostrs/search_session.py` lines 135-1707](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L135-L1707)

## Members

| Member | Kind | Summary |
|---|---|---|
| `constraints` | Property | The search-wide rule in force: heredity, forbidden pairs, mandatory terms. |
| `available_actions` | Method | Operations available now, with a short statement of their purpose. |
| `state` | Method | Bounded, JSON-safe view for an analyst or an external agent. |
| `checkpoint` | Method | Export a training-only search for replay in another Python process. |
| `export_transcript` | Method | Export a JSON-safe decision record, including after freeze. |
| `resume` | Method | Restore the exact fold/evidence boundary, requiring the same data. |
| `drive` | Method | Let a bounded judge choose successive public actions. |
| `run` | Method | Run one chosen operation and return its evidence and native value. |

(searchsession-constraints)=
## `SearchSession.constraints`

*Property* of `SearchSession`

```python
SearchSession.constraints
```

The search-wide rule in force: heredity, forbidden pairs, mandatory terms.

Source: [`python/mboostrs/search_session.py` lines 245-248](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L245-L248)

(searchsession-available_actions)=
## `SearchSession.available_actions`

*Method* of `SearchSession`

```python
SearchSession.available_actions()
```

Operations available now, with a short statement of their purpose.

Source: [`python/mboostrs/search_session.py` lines 504-566](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L504-L566)

(searchsession-state)=
## `SearchSession.state`

*Method* of `SearchSession`

```python
SearchSession.state()
```

Bounded, JSON-safe view for an analyst or an external agent.

Source: [`python/mboostrs/search_session.py` lines 599-623](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L599-L623)

(searchsession-checkpoint)=
## `SearchSession.checkpoint`

*Method* of `SearchSession`

```python
SearchSession.checkpoint()
```

Export a training-only search for replay in another Python process.

Fitted native objects are not serialized. Resume restores every
comparison and charge, but asks for one explicit `fit` or `tune`
action before screening residuals or freezing a model. The returned
record does not include training or test rows; pass the same training
frame to `resume` and its fingerprint is checked.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/search_session.py` lines 625-662](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L625-L662)

(searchsession-export_transcript)=
## `SearchSession.export_transcript`

*Method* of `SearchSession`

```python
SearchSession.export_transcript()
```

Export a JSON-safe decision record, including after freeze.

This record contains no training rows, fold arrays or fitted model. It
documents what was measured and the final test score, if available;
it cannot be resumed for fitting. Use `checkpoint` before freeze
when a resumable training session is needed.

Source: [`python/mboostrs/search_session.py` lines 664-694](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L664-L694)

(searchsession-resume)=
## `SearchSession.resume`

*Method* of `SearchSession`

```python
SearchSession.resume(data, record, *, family=None)
```

Restore the exact fold/evidence boundary, requiring the same data.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/search_session.py` lines 696-793](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L696-L793)

(searchsession-drive)=
## `SearchSession.drive`

*Method* of `SearchSession`

```python
SearchSession.drive(judgment, *, max_actions=4)
```

Let a bounded judge choose successive public actions.

`judgment` receives only `state()` and a small reply schema. It
may be a human-written callable or `mboostrs.LLMJudge` with a
configured client. This method never passes test rows or runs final
evaluation. A failed or malformed reply is logged and stops the
driver; the caller can continue manually from the retained state.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/search_session.py` lines 816-874](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L816-L874)

(searchsession-run)=
## `SearchSession.run`

*Method* of `SearchSession`

```python
SearchSession.run(action, **options)
```

Run one chosen operation and return its evidence and native value.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/search_session.py` lines 915-964](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L915-L964)
