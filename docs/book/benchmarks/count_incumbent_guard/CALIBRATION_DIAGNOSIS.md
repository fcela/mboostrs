# Count-set calibration diagnosis: the nonlinear zero-inflated efficiency gate (#679)

This document records the diagnosis of the one failing gate of confirmation 2
and the search for a calibration-side remedy. It was prepared on branch
`fix/679-count-calibration` at main `81479457` (the merge of PR #785). All
remedy numbers below are offline replays on the retained confirmation-2
records (`production_confirmation2_count_nonlinear_zero_inflated.jsonl.xz`,
50 seeds); the replay machinery reproduced the recorded gate exactly before
any variant was priced (candidate count-size excess +1.674%, one-sided 95%
upper +2.734%, coverage 0.9008, in three independent implementations).

Outcome, stated plainly: no admissible change to the count-set construction
or calibration was found that brings the gate within its bound, so this
record ships without a code change, and confirmation 3 was not run. The
reserved confirmation-3 ranges — count seeds 6799500–6799549, Gaussian seeds
6799600–6799649, test seed + 100000 — remain unused.

## The failing gate

On the nonlinear zero-inflated panel, the discovered fit's prediction sets
are oversized against the prespecified LightGBM CQR comparison: count-size
excess +1.674% on average, one-sided 95% upper +2.734%, against the frozen
bound of at most +1%. The interval-width gate passes (−2.70% upper) and
coverage is nominal (0.9008). The tuned supplied track passes the same gate
(count size −1.43% upper in the confirmation-2 summary; −1.99% mean in the
replay). The development phase failed the same gate at +3.26%, so the excess
is a stable property of the shipped configuration, not seed noise.

## Mechanism

The construction is the count-rank conformal set: a zero-inflated Poisson
reference at the fit's mean, the shortest probability window of the nominal
content, randomized ranks against that reference, and a single conformal
quantile `q` widening the window on both sides. On this panel every seed
resolves to the count-rank score. The measured quantities:

| Quantity | Candidate (discovered) | Supplied, tuned |
|---|---|---|
| Conformal quantile `q` | 0.0391 (sd 0.0251) | 0.0141 (sd 0.0161) |
| Reference zero mass `pi` | 0.3112 | 0.2943 |
| Dispersion diagnostic (ratio) | 1.112 | 0.992 |
| Share of calibration rows outside the nominal window | 0.246 | 0.179 |

The calibration ranks of the discovered fit are over-dispersed at every
fitted-mean quartile (rank variance 0.094–0.107 against the uniform value
1/12 ≈ 0.083), so the single quantile inflates to cover them. The bands
follow the fitted mean, which is shrunk on this panel: on calibration rows
the mean residual by fitted-mean quartile is −0.74, +0.14, −0.30, +1.93
counts. Coverage by true-mean quintile is 0.984, 0.946, 0.914, 0.912, 0.748 —
over-covering where the mean runs high, under-covering at the top — while
marginal coverage stays at 0.9008.

Two substitutions isolate the cause. Replacing the fitted mean by the true
mean inside the identical construction (reference shape and fitted `pi`
unchanged) collapses `q` to −0.003 and the count-size excess to −3.02%
(upper −2.54%, coverage 0.8991): with exact centres the same construction
beats the comparison. And the reference's variance formula is right at the
true means: the dispersion ratio computed at true means with the true zero
mass is 1.019. The excess is therefore the conformal price of the discovered
mean's estimation error — latent-mean MSE 12.01 on this panel — not a defect
of the reference's shape, its zero mass, or the window placement. The tuned
supplied track passes because its prespecified mean is more accurate, and
the comparison method fits its band edges directly.

## Remedy search

Each candidate remedy was priced in the replay at the panel level. The bar,
fixed before pricing: candidate count-size excess upper at most +1% with
coverage in [0.88, 0.92], and the supplied track not degraded past the same
bound. Results (candidate excess, mean / upper; coverage):

| Remedy | Excess | Coverage | Verdict |
|---|---|---|---|
| Stratified (Mondrian) quantile by fitted-mean quartile | +3.49% / +4.66% | — | worse: per-stratum order-statistic inflation at n≈125 |
| Zero-inflated negative binomial reference, moment size | +2.40% / +3.43% | 0.90 | worse: `q` collapses but the reference's own window widens as much |
| ZINB reference, zero mass refit by likelihood | +2.08% / +3.14% | 0.90 | worse, same reason |
| ZINB with dispersion profile by fitted-mean quartile, fit on the fitting rows | +2.32% / +3.42% | 0.8999 | worse |
| Same, profile fit on the calibration rows (oracle ceiling) | +2.93% / +4.01% | 0.9014 | worse even with oracle knowledge |
| Count-unit (CQR-style) score on the reference's shortest window | +1.84% / +2.82% | 0.9022 | worse than the rank score |
| Count-unit score on central reference quantiles | +5.51% / +6.26% | 0.9158 | worse |
| Asymmetric expansion weights, splits from (0.75, 1.25) to (0.1, 1.9) | +1.63% to +1.67% | 0.9008–0.9010 | neutral: the max-score already adapts per row |
| Zero mass from other sources (calibration rows, half-sample, fixed 0.3) | +1.61% to +1.86% | ≈0.90 | neutral: `pi` is not the lever |
| Window-placement grid refined from 17 to 161 points | +0.000 change | identical | the placement grid is not binding |
| Sibling scores `poisson_normalized` / `poisson_quantile` | +16.97% / +30.93% | ≈0.907 | far worse |
| Centre recalibration `exp(a + b·log mu)` fit on the fitting rows | +2.47% / +3.63% | — | worse: the in-sample centre relation does not transfer |

Two families were excluded before pricing. Any remedy that fits the
reference — its zero mass, centre, or dispersion — on calibration responses
violates the tested design invariant that the reference and the
randomization do not depend on calibration responses
(`tests/test_count_rank_calibration.py::test_calibration_responses_cannot_change_reference_or_randomization`);
a calibration sub-split is that violation in two steps. For completeness
the sub-split variants were still priced in the replay: they are neutral
(+1.86% / +3.08% for the strongest variant). An earlier replay reading of
−2.07% for that variant did not survive checking: two independent
implementations, one of them a scalar-exact recomputation, agree on the
neutral value, and the favourable reading is discarded.

The pattern across the table is consistent. Widening the reference in any
direction trades the conformal correction for window width one-for-one or
worse, because the correction is a tail price and a global widening pays it
on every row. Repricing the correction by stratum, by side, or in count
units redistributes the same total. The total is set by the dispersion of
the response around the *fitted* mean, and no reparametrization of the set
construction goes below it.

## Decision and what would close the gate

No remedy met the development bar, so no confirmation 3 was run: a fresh
cohort spent on the unchanged procedure would only re-measure a failure
already recorded at +3.26% (development) and +2.73% upper (confirmation 2),
and would spend the reserved ranges that a future remedy needs. The ranges
above stay untouched.

The gate is reachable from the mean side, not the calibration side: the
oracle substitution shows roughly four points of headroom between the
discovered fit's excess (+1.67%) and the same construction at the true mean
(−3.02%). A mean-side improvement of the discovered fit on curved
zero-inflated panels — outside the scope frozen for this branch, which held
the caps, discovery score, guard, stopping tie policy and mean behaviour
fixed — is the change that can close it. If such a change lands, the
calibration question should be re-confirmed on the reserved ranges above
under the confirmation-2 gate structure.
