# Single-holdout stopping: study record (#541 follow-on)

## Holdout reuse and stopping

When autospec judges on a single supplied holdout, the final specification's
stopping step is the argmin of the aggregate holdout curve. That curve is a
mean over the held-out rows with no spread attached, so it keeps rewarding
flexibility long after the per-row differences have become noise. The
adoption audits were repaired first (#776, #787: every adoption on a single
holdout now needs the paired per-row bar); with unproven terms blocked, the
surviving formula's stopping point is where the reused holdout is spent
next. This study tests a single-holdout stopping rule against the frozen
battery and synthetic controls, requiring improvement without regressions.

## Method

- Frozen cancer battery (the #541 battery: payload split by a single
  supplied holdout, the frozen autospec call of the earlier studies),
  three draws, run on current main (both sibling-gate fixes merged).
  Post-#787 landscape reproduced exactly: chosen steps 3261 / 3894 / 9989,
  test log-losses 0.1244 / 0.0688 / 0.1604, battery mean 0.1179.
- Synthetic controls from the #541 work (exp6 design): C1 correlated
  linear, C2 nonlinear with a product term, C3 null; 364 development rows,
  91 held out, four draws each, judged on 20,000 fresh rows.
- Candidate rules are functions of the winning specification's fold path
  only, read on a fixed step grid (5–1000 by 5, 1025–10000 by 25). A rule
  may only pull the stop earlier than the sweep's argmin, never later.
- Acceptance bar, fixed in advance: tie or beat the current argmin rule on
  every battery draw and improve the battery mean, and leave every
  synthetic-control draw unchanged-or-better.

## Pathology (battery, per draw)

| draw | argmin stop (holdout risk) | test loss at stop | test-curve bottom |
|------|---------------------------|-------------------|-------------------|
| 0 | 3261 (0.0707) | 0.1244 | ≈ step 425, 0.1049 |
| 1 | 3894 | 0.0688 | broad, near the stop |
| 2 | 9989, at the 10000 cap (0.0276) | 0.1604 | ≈ step 995, 0.1246 |

Draw 2 shows the discrepancy: the 91-row holdout curve falls monotonically
to the cap (and reads 0.0276, wildly optimistic) while the same fit's loss
on disjoint rows bottoms near step 995 and rises for the remaining 9000
steps. The holdout curve does not identify the test-loss minimum.

## Candidates and verdicts

Deltas are judged loss minus the current rule's judged loss on the same
draw (test rows for the battery, 20k fresh rows for the controls);
negative is better. A rule fails if any single delta is positive.

| rule | battery deltas (draws 0/1/2) | battery mean | control failures | verdict |
|------|------------------------------|--------------|------------------|---------|
| paired 1-SE (earliest step within one paired SE of the path minimum) | −0.0180 / −0.0025 / −0.0349 | 0.1179 → 0.0994 | C1 d0 +0.0320, C1 d3 +0.0294, C2 d3 +0.00002 | fail |
| plateau 0.5% (earliest step within 0.5% of the curve minimum) | −0.0048 / −0.0010 / −0.0049 | → 0.1143 | C1 d3 +0.0073 | fail |
| plateau 1% | −0.0065 / −0.0013 / −0.0084 | → 0.1125 | C1 d0 +0.0044, C1 d3 +0.0112 | fail |
| plateau 2% | −0.0088 / −0.0017 / −0.0141 | → 0.1097 | C1 d0 +0.0106, C1 d3 +0.0205 | fail |
| unpaired 1-SE (SE of the minimum's own losses) | −0.0195 / +0.0040 / −0.0351 | → 0.1010 | (battery already fails) | fail |
| per-row grid argmin | −0.0008 / +0.0001 / −0.0001 | → 0.1176 | C1 d0 +0.0016, C2 d0 +0.0204, others | fail |
| smoothed argmin (moving average, w=51) | +0.0037 / +0.0008 / +0.0000 | → 0.1194 | several | fail |
| split-half argmin (stop on one half of the holdout) | +0.0332 / −0.0003 / −0.0354 | → 0.1171 | C1 d1 +0.0923, others | fail |

Every pull-back rule that helps the battery worsens the correlated-linear
control C1 on at least one draw. On C1 the holdout curve is already flat near
its minimum (steps 100–185 move the mean by 0.7%), while the fresh-sample
optimum is at or after the argmin. The additional loss reduction is real
signal on those rows, not a noise tail. On C1 draw 0 the paired band at step 45 is ±0.021 per row and the
cumulative gain from step 45 to the argmin is 0.019 per row on the holdout
but 0.032 per row on fresh rows — at 91 held-out rows the two situations
(a still-descending curve, a noise tail) are not separable by any reading
of the holdout alone. The battery's overruns are thousands of steps; the
controls' optima are tens of steps from the argmin; no threshold on the
fold path separates "far past the turn" from "just before it" at this
holdout size.

## Conclusion

No candidate clears the pre-fixed bar, so no stopping rule is shipped and
the argmin behaviour is unchanged. The paired 1-SE rule is the strongest
candidate on the battery (mean 0.0994, every draw better). A larger holdout
would narrow the per-row uncertainty band (approximately 1/√n); a reserve
split held out specifically for stopping would measure a different procedure;
or a policy could state an accepted trade on C1-type panels. Until then, the 91-row holdout has already influenced formula selection and does not provide a reliable stopping signal as well.

## Reproduction

Study scripts and raw records (per-row holdout losses on the step grid,
fresh-sample judging curves, the evaluator that produced the table above)
are kept with this study's working files. The battery runs reproduce the
post-#787 records to all printed digits, and the judge path was verified
faithful: a development fit sliced at the returned step reproduces the
recommendation's own judged loss exactly on every draw.
