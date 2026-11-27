# `learner_design`

Defined in `mboostrs.interactions`; exported as `mboostrs.learner_design`.


*Function*

```python
learner_design(term: str, data) -> LearnerDesign
```

Build a reusable design and penalty for one base-learner term.

**Parameters**

- `term` (str): One `bbs(...)`, `bols(...)` or `brandom(...)` term. Its fitted
  basis information is retained for `LearnerDesign.transform`.
- `data` (frame or mapping): Rows used to fit the basis and establish spline boundaries or factor
  levels.

**Returns**

LearnerDesign
    `design` and `penalty` can be passed to `buser`; `transform`
    constructs the conformable design for new rows.

Source: [`python/mboostrs/interactions.py` lines 935-953](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/interactions.py#L935-L953)
