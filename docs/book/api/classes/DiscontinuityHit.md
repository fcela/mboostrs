# `DiscontinuityHit`

Defined in `mboostrs.terms`; exported as `mboostrs.DiscontinuityHit`.


*Class*

```python
DiscontinuityHit(
    kind: str,
    columns: tuple,
    direction_deg: float,
    threshold: float,
    threshold_x: float,
    boundary_intercept: float,
    boundary_slope: float,
    line: str,
    n_above: int,
    jump_fit: float,
    t_fit: float,
    jump: float,
    se: float,
    z: float,
    gain: float,
    type_z: float,
    type_ratio: float,
    type_gap_t: float
) -> None
```

One evaluated boundary from `screen_discontinuities`.

`kind` is `"axis"` (a step in one column, `column > threshold_x`)
or `"diagonal"` (a step across a line in two columns,
`right > boundary_intercept + boundary_slope * left`, with
`direction_deg` the direction of the projection the boundary is a
threshold on). `threshold` is that threshold on the standardized
projection; `threshold_x`, `boundary_intercept` and
`boundary_slope` are the same boundary on the original scale, with the
fields a purely vertical or horizontal boundary cannot express left as
`NaN` and named in `line`.

`jump_fit` and `t_fit` are the fit-half step estimate and its
t-statistic; selection ranked candidates on `t_fit` alone, so these are
the in-sample numbers. `jump`, `se` and `z` are the same
coefficient estimated on the other half of the rows, which selection
never looked at, so `z` follows a t distribution with
`n_eval - (q + 1)` degrees of freedom under the null of no
jump anywhere in the candidate set (asymptotically standard normal). `gain` is the held-out
mean-squared-error reduction of the fit-half step model against the
fit-half base model, per held-out observation, on the same scale as the
difference between two out-of-fold risks. `type_z` refits the step
against a background augmented with smooth terms along the projection:
large when the step survives (a jump), near zero when a smooth crossing
explains it. `type_ratio` is the local discriminator: the ratio of the
level gap at two scales (nearest 60 vs nearest 15 evaluation-half points
per side), maximized over a window of thresholds around the fit-half
estimate. `type_gap_t` is the t-statistic of the small-scale gap. A
true jump shows a significant gap that persists across scales (large
`type_gap_t`, ratio ~1); a steep smooth transition or kink shows a gap
that grows with scale (ratio ~2+), and a gentle smooth shows no
significant gap at all -- neither is typed as a jump even though the
global `type_z` may be large. `n_above` counts the rows on the far
side of the boundary.

Source: [`python/mboostrs/terms.py` lines 979-1067](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L979-L1067)

## Members

| Member | Kind | Summary |
|---|---|---|
| `flagged` | Property | The screen's decision rule: ``\|z\| >= 2`` with a positive gain. |
| `typed_jump` | Property | A flagged row with a genuine discontinuity: ``flagged``, a |

(discontinuityhit-flagged)=
## `DiscontinuityHit.flagged`

*Property* of `DiscontinuityHit`

```python
DiscontinuityHit.flagged  # -> bool
```

The screen's decision rule: `|z| >= 2` with a positive gain.

Under the null of no jump anywhere this fires on about 4.6% of
screens; a screen that does not flag is a first-class result that
finds no screened evidence for a jump in these columns.

Source: [`python/mboostrs/terms.py` lines 1036-1043](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L1036-L1043)

(discontinuityhit-typed_jump)=
## `DiscontinuityHit.typed_jump`

*Property* of `DiscontinuityHit`

```python
DiscontinuityHit.typed_jump  # -> bool
```

A flagged row with a genuine discontinuity: `flagged`, a
significant small-scale gap (`type_gap_t`), and a scale-invariant
gap (`type_ratio` below threshold). The gap t-test takes the
largest discontinuity over a window around the fit-half threshold;
a true jump shows a large gap that persists across scales (ratio
~1), while a steep smooth transition or kink shows a gap that grows
with scale (ratio ~2+), and a gentle smooth shows no significant gap
at all. Note the resolution limit: a smooth transition narrower
than the small scale's span is indistinguishable from a jump (see
the `type_ratio` discussion in `screen_discontinuities`).

Source: [`python/mboostrs/terms.py` lines 1045-1060](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L1045-L1060)
