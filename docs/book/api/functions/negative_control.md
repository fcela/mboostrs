# `negative_control`

Defined in `mboostrs.causal`; exported as `mboostrs.negative_control`.


*Function*

```python
negative_control(
    data,
    spec: mboostrs.causal.CausalSpec,
    kind,
    rationale,
    expected_effect=0.0,
    **aipw_options
)
```

Estimate a prespecified negative-control exposure or outcome effect.

`spec` is the complete causal specification for the control analysis,
including its treatment contrast, outcome type, timing and adjustment set.
`rationale` must state the scientific reason its effect should equal
`expected_effect`. For example, a negative-control outcome must share
relevant measurement/confounding pathways while being causally unaffected
by treatment; merely choosing an unrelated variable is uninformative.

An incompatible interval can reveal residual confounding, selection,
measurement error or a failed control assumption. A compatible interval
cannot prove the primary effect is identified because the control may not
probe the relevant bias pathway.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/causal.py` lines 6097-6123](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L6097-L6123)
