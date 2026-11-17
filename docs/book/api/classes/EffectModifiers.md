# `EffectModifiers`

Defined in `mboostrs.effect_modifiers`; exported as `mboostrs.EffectModifiers`.


*Class*

```python
EffectModifiers(
    method: str,
    level: float,
    adjust: str,
    bounds: tuple,
    n: int,
    n_inside: int,
    groups: int,
    records: tuple,
    joint: tuple,
    _core: dict,
    diagnostics: dict = <factory>,
    message: str = ''
) -> None
```

Per-covariate tests of effect modification, with a joint projection.

`records` holds one entry per covariate, in the order given:

* `blp_slope`, `blp_se`, `blp_lower`, `blp_upper`: the coefficient
  of the covariate in a regression of the AIPW score on that covariate
  alone, with HC1 standard errors. For a numeric covariate it is the change
  in the effect, in outcome units, per standard deviation of the covariate;
  for a categorical covariate it is `None` and `blp_p` is the joint
  test over the level dummies.
* `gates_contrast`, `gates_se`, `gates_lower`, `gates_upper`: the
  difference between the mean score of the top and bottom groups. For a
  numeric covariate the groups are quantile groups in increasing order, so
  the contrast is top quantile group minus bottom quantile group; for a
  categorical covariate the contrast is the largest group mean minus the
  smallest, and `gates_p` is the chi-squared test that all level means
  are equal.
* `blp_levels`: for a categorical covariate, the BLP coefficient of each
  level relative to the first level (in sorted order), with its standard
  error; empty for a numeric covariate.
* `blp_p`, `gates_p`: two-sided p-values of the two tests, and
  `blp_p_adjusted`, `gates_p_adjusted`: the same p-values adjusted across
  covariates with the method given to `effect_modifiers`.
* `estimate`: the point estimate of the method's test (the BLP slope or
  the GATES contrast); `direction` is its sign (`"increases"` or
  `"decreases"` along a numeric covariate, `"differs by level"` for a
  categorical one). `gates_top` and `gates_bottom` name the groups of
  the contrast.
* `test_decision`: True when the adjusted p-value of the method's test is
  below `1 - level`.
* `decision`: `test_decision` when the gate passed or was not run, and False
  for every covariate when the gate failed.

`diagnostics["gate"]` records the gate: whether it ran, `passed`, the
out-of-fold `gain` over a constant effect and its standard error
`gain_se` (in units of the weighted squared R-loss), the modifiers it
used, the seconds it took and the rule. `message` states the reason when
it failed.

Source: [`python/mboostrs/effect_modifiers.py` lines 328-475](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L328-L475)

## Members

| Member | Kind | Summary |
|---|---|---|
| `table` | Method | One row per covariate; see the ``records`` description. |
| `joint_table` | Method | One row per design column of the joint regression on all covariates. |
| `selected` | Method | The covariates whose decision is True, in the order given. |
| `projection` | Method | The joint projection on ``"all"`` covariates, on the ``"passing"`` ones |
| `to_dict` | Method | A JSON-safe record of the tests and the joint regression. |
| `describe` | Method | Each covariate's point estimate, adjusted p-value and decision. |

(effectmodifiers-table)=
## `EffectModifiers.table`

*Method* of `EffectModifiers`

```python
EffectModifiers.table(output='auto')
```

One row per covariate; see the `records` description.

Source: [`python/mboostrs/effect_modifiers.py` lines 385-387](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L385-L387)

(effectmodifiers-joint_table)=
## `EffectModifiers.joint_table`

*Method* of `EffectModifiers`

```python
EffectModifiers.joint_table(output='auto')
```

One row per design column of the joint regression on all covariates.

`block_p` is the joint test of the covariate's columns: the same as
`p` for a numeric covariate and the test over all level dummies for a
categorical one. A column dropped as collinear has NaN coefficients and
`collinear` set to True.

Source: [`python/mboostrs/effect_modifiers.py` lines 389-396](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L389-L396)

(effectmodifiers-selected)=
## `EffectModifiers.selected`

*Method* of `EffectModifiers`

```python
EffectModifiers.selected()
```

The covariates whose decision is True, in the order given.

Source: [`python/mboostrs/effect_modifiers.py` lines 398-400](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L398-L400)

(effectmodifiers-projection)=
## `EffectModifiers.projection`

*Method* of `EffectModifiers`

```python
EffectModifiers.projection(covariates='all')
```

The joint projection on `"all"` covariates, on the `"passing"` ones
(those with a True decision), or on an explicit list of names.

A projection needs at least one covariate; if none passes, pass
`covariates="all"` or name the covariates explicitly.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/effect_modifiers.py` lines 402-440](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L402-L440)

(effectmodifiers-to_dict)=
## `EffectModifiers.to_dict`

*Method* of `EffectModifiers`

```python
EffectModifiers.to_dict(*, include_arrays=False)
```

A JSON-safe record of the tests and the joint regression.

Source: [`python/mboostrs/effect_modifiers.py` lines 442-445](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L442-L445)

(effectmodifiers-describe)=
## `EffectModifiers.describe`

*Method* of `EffectModifiers`

```python
EffectModifiers.describe() -> str
```

Each covariate's point estimate, adjusted p-value and decision.

Source: [`python/mboostrs/effect_modifiers.py` lines 451-475](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L451-L475)
