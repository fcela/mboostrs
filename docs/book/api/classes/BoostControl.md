# `BoostControl`

Defined in `mboostrs._control`; exported as `mboostrs.BoostControl`.


*Class*

```python
BoostControl(
    mstop: ForwardRef('int'),
    nu: ForwardRef('float'),
    risk: ForwardRef('str'),
    trace: ForwardRef('bool'),
    step: ForwardRef('str') = 'fixed',
    step_iterations: ForwardRef('int') = 4
)
```

How long to boost, and how far each step goes.

**Attributes**

- `mstop` (int): Iterations to run. Each one selects a single base-learner and adds
  `nu` times its fit. Small values may leave the path underfit, while
  large values may continue fitting noise. Choose it with `cvrisk`.
- `nu` (float): The step length, between 0 and 1. Smaller steps make each update
  smaller and usually require more iterations; they do not guarantee a
  better fit. The default is 0.1.
- `risk` ({"inbag", "oobag", "none"}): Which risk the path records. `"inbag"` is the fitted sample's own
  risk, which is what `mboost` stores and what the reported path means.
- `trace` (bool): Print the risk at each iteration to stderr while fitting.
- `step` ({"fixed", "line_search"}, default "fixed"): How the step length is chosen after the winning learner is selected.
  `"fixed"` takes the classic constant step `nu * h` that `mboost` uses.
  `"line_search"` first finds the multiple `rho` of the winner's
  direction that minimizes the training loss, then takes `nu * rho * h`:
  `nu` becomes a fraction of the loss-optimal step, so its meaning no
  longer depends on the response scale. This removes the divergence a
  fixed step suffers on large-count Poisson responses and the crawl it
  suffers at rare binary events. Selection is unchanged, and the
  squared-error families are bit-identical either way (their
  least-squares step is already optimal, `rho = 1`).
- `step_iterations` (int, default 4): Golden-section refinement steps per boosting iteration under
  `step="line_search"`. The grid has already bracketed the minimum, so
  small values suffice.

Source: [`python/mboostrs/_control.py` lines 25-78](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_control.py#L25-L78)

## Members

| Member | Kind | Summary |
|---|---|---|
| `describe` | Method | One-line text description of the control. |

(boostcontrol-describe)=
## `BoostControl.describe`

*Method* of `BoostControl`

```python
BoostControl.describe() -> str
```

One-line text description of the control.

**Returns**

str
    For example `"100 iterations at nu=0.1, risk='inbag'"`; the step
    rule is appended when it is not `"fixed"`.

Source: [`python/mboostrs/_control.py` lines 67-78](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_control.py#L67-L78)
