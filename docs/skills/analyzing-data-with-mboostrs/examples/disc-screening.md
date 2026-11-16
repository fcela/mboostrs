# Example 5: a diagonal regime jump no smooth term can represent

**Dataset:** 800 rows of six uniform columns `s0`, `s1`, `x_smooth`, `n1`,
`n2`, `n3`, with a response built from a step of size 2.0 across the
diagonal boundary `s1 = 0.45 + 0.35 s0`, a smooth additive mean
`1.5 s0 + 1.0 s1 + 1.5 sin(pi x_smooth)` and Gaussian noise of sd 0.5.
A smooth fit absorbs most of the diagonal jump, and an axis-aligned step
cannot express its boundary. The example uses the shipped discontinuity
screen to locate and classify the jump.

## Step 5: measure what the fit left, then ask where it lives

The screen on the raw response proposes boundaries from two families --
steps in one column and steps across a line in two columns -- and reports
each with a jump estimate that is honest by construction: candidates are
ranked on one half of the rows and the reported `z` comes from the other
half, which selection never saw.

```python
hits = mb.screen_discontinuities(data, y, response="y", seed=0)
hits[0]
# s1 > 0.462 + 0.333*s0: jump +1.965 (se 0.122), z +16.14,
#     gain +0.2797, type_z +12.35, type_gap_t +4.6, ratio 1.40
```

The top row recovers the boundary (`truth: 0.45 + 0.35 s0`, agreement
99.2% of rows), the jump (truth 2.0) and both statistics. `typed_jump` is
true: the local gap discriminator finds a significant gap (`type_gap_t`
+4.6) that persists across scales (`type_ratio` 1.40 ~ 1) -- this is a
jump, not a steep smooth crossing.

The recommended second pass fits the smooth baseline first and screens
its response-scale residual, so the jump stands alone:

```python
model = mb.gamboost("y ~ bbs(s0) + bbs(s1) + bbs(x_smooth)", data,
                    control=mb.boost_control(mstop=100, nu=0.1))
residual = y - np.asarray(model.predict(type="response")).ravel()
mb.screen_discontinuities(data, residual, response="y", seed=0)[0]
# s1 > 0.375 + 0.4468*s0: jump +1.006 (se 0.109), z +9.19,
#     gain +0.07048, type_z +7.43
```

The residual pass sees the part the smooth fit left: a jump of about 1.0
(the fit absorbed the rest and tilted the boundary estimate). On data with
no step anywhere -- the same panel without the jump, or pure noise -- the
residual pass stays quiet, and `|z| >= 2` with a positive `gain` fires on
about 4.6% of such screens. A screen that does not flag is a result: it
finds no screened evidence for a jump in these columns, and the `reference_gap` belongs
to a smooth or interaction term instead.

One caution the typing statistic carries: a steep smooth rise, such as the
`sin(pi x_smooth)` term alone, *does* flag on the raw response -- a step
beyond `[1, u, u^2]` genuinely reduces error there -- but it does not type
as a jump (its gap grows with scale), and the residual pass after the smooth fit
stays quiet. Flagged and typed are different claims.

## Steps 6-7: score the candidates the screen proposes

The screen's output is a boundary, so it proposes two candidates the
comparison step can score: the joint tree `btree(s0, s1)`, and the named
step indicator the boundary itself defines.

```python
indicator = (data["s1"].to_numpy()
             > hits2[0].boundary_intercept
             + hits2[0].boundary_slope * data["s0"].to_numpy()).astype(float)
data2 = data.with_columns(__jump=indicator)
specs = ["bbs(s0) + bbs(s1) + bbs(x_smooth)",
         "bbs(s0) + bbs(s1) + bbs(x_smooth) + btree(s0, s1)",
         "bbs(s0) + bbs(s1) + bbs(x_smooth) + bols(__jump)"]
mb.compare_specifications(specs, data2, "y", family=mb.Gaussian(),
                          folds=5, mstop="cv", seed=0)
```

| candidate | out-of-fold risk | change | SE |
|---|---:|---:|---:|
| additive baseline | 0.4563 | -- | -- |
| `+ btree(s0, s1)` | 0.4647 | +0.0083 | 0.0037 |
| `+ bols(__jump)` | 0.3300 | **-0.1264** | 0.0241 |

The named indicator wins by more than five SE. The joint tree, the
candidate an analyst reaches for when the boundary is unknown, *loses*
here: a depth-limited tree approximates a diagonal boundary coarsely, while
the indicator is the boundary. That is the division of labour the screen
makes possible -- it names the structure, so the readable term wins where
the black box would have had to be the answer.

## Screen interpretation

- `hits[0].flagged` is the gate (`|z| >= 2` and a positive held-out gain);
  `hits[0].typed_jump` is the jump verdict. Both are first-class results,
  including their negation.
- A `typed_jump` row proposes `bols(indicator)` and `btree(pair)` for the
  comparison step; a flagged row with a large `type_ratio` proposes a smooth or
  tensor term instead; no flag ends the jump branch of the search.
- The screen fits small closed-form OLS systems per candidate in Rust-accelerated scan kernels; on this panel (six
  columns, 800 rows, 15 pairs, 16 directions) both passes take well under a
  second.
