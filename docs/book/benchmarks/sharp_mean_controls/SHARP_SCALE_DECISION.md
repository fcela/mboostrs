# Sharp varying-scale detection: documented limitation (#815)

## Study objective

In the confirmation battery frozen in CONFIRMATION_DECISION.md and run on
the reserved seeds 6294100-6294139, the candidate (discovery with spread
on) adopted a varying scale on 0 of 40 draws of each sharp varying-scale
process, axis_varying and shifted_varying. That recall was reported
without a gate in the frozen decision (gate G3), because the development
inventory already showed 0 of 10 on both processes. The development inventory left a policy choice: require the default to
recover scale under a sharp mean, or document that it does not. This record
accepts the limitation and places the measured recall beside the scale claims
in the book.

## Verification of the recall figure

The figure was verified against the confirmation record on the current
main tree, not re-measured. The record is CONFIRMATION_RESULT.md and
confirmation_summary.json in this directory. In
confirmation_summary.json, the `discovery_spread` row for each panel
carries `draws`, `ok` and `selected_varying_scale`:

| process | draws | ok | varying scale selected |
|---|---|---|---|
| axis_varying | 40 | 40 | 0 |
| shifted_varying | 40 | 40 | 0 |
| smooth_absolute | 40 | 40 | 40 |
| smooth_bell | 40 | 40 | 40 |
| smooth_monotone | 40 | 40 | 40 |
| axis_constant | 40 | 40 | 0 |
| shifted_constant | 40 | 40 | 0 |
| smooth_constant | 40 | 40 | 0 |
| null_constant | 40 | 40 | 0 |
| heavy_tail | 40 | 40 | 0 |

The pre-change figure also checks: in the development inventory
(summary.json in this directory, ten draws per process, seeds
6293100-6293109), the same two rows read 0 of 10 each, and
CONFIRMATION_DECISION.md froze the same statement before any
confirmation draw was generated ("the frozen baseline adopts 0/10 there
both before and after the candidate"). The miss therefore predates the
mean-adequacy spread guard; that guard did not cause it.

## Mechanism evidence

The spread stage of discovery (`_spread` in python/mboostrs/discover.py)
works in three steps, each of which can stop the scale term:

1. Nomination. `heteroscedasticity` is computed on the residuals of a
   screening mean fit at 200 updates (`_fit_mean` at the screening
   stopping count). A column is nominated when the slope of
   log(residual^2) on the column has |t| >= 2, or when its held-out
   eight-bin curve gain (`curve_removed`) is positive.
2. Screening comparison. The nominated log-scale term, a smooth
   `bbs(x1, df=4)` for a many-valued column, must beat the constant
   scale in a held-out Gaussian location-scale comparison at the same
   200-update stopping count, by more than two change standard errors.
3. Mean-adequacy confirmation. Only a term that passes screening is
   re-scored with each specification's stopping tuned inside the folds
   (`_confirm_spread`). This is the step the #629 guard added.

A small diagnostic run shows at which step the sharp processes stop.
The diagnostic used the frozen generator and partition of
benches/study_sharp_mean_controls.py and the same discover call as the
battery, on three fresh seeds (6295100-6295102) that belong to neither
the development nor the confirmation battery. It is a mechanism
diagnostic only, not a recall measurement. On the six sharp
varying-scale draws:

- In four draws the true scale driver x1 was nominated, on curve gain
  (0.13 to 0.41); its signed slope t was small (-1.74 to 0.56), which is
  expected because the generating scale 0.3 + 1.5|x1| rises on both
  sides of zero, so a signed slope cancels. The nominated
  `bbs(x1, df=4)` term then lost the screening comparison outright:
  risk changes of +1.23 (SE 0.44), +1.03 (SE 0.44), +1.59 (SE 0.83) and
  +0.08 (SE 0.04), where adoption requires a change of at most -2 SE.
  The confirmation step was never reached.
- In the other two draws x1 was not nominated at all (curve gain
  -0.11 and -0.03); another column was nominated instead and its term
  was likewise rejected at screening (+0.26, SE 0.10; +0.01, SE 0.04).

On the same three seeds, the smooth controls behaved as in the battery:
smooth_absolute (which has the identical generating scale
0.3 + 1.5|x1|, under a smooth mean) adopted `bbs(x1, df=4)` on 3 of 3
draws, with screening changes of -0.17 (SE 0.01 to 0.03), and
smooth_bell adopted on 3 of 3. Two conclusions follow from the code and
the diagnostic together. First, the failure sits at nomination and
screening, under the residuals of the 200-update screening mean fit;
the #629 confirmation step is downstream and is not involved. Second,
the scale term's form is not the obstacle: the same `bbs(x1, df=4)`
term is adopted whenever the mean is smooth. What differs on the sharp
processes is the mean fit the residuals come from. In the confirmation
record the candidate's latent-mean MSE on the sharp varying-scale
processes is 0.648 (axis) and 0.402 (shifted), against 0.068 on
smooth_absolute, so the residuals the spread stage reads carry an order
of magnitude more mean approximation error than on the smooth panels.
The conditional-scale prototypes recorded on the Gaussian mean and
scale controls page point the same way: the strongest of them, which
cross-fits a tuned mean before scoring scale, reached only 5 of 10
adoptions on the axis-varying development control, below its frozen
90% requirement.

## Consequences of the limitation

- Marginal coverage is not the cost. On the confirmation battery the
  candidate's coverage on axis_varying and shifted_varying is 0.901 on
  both, inside the [0.88, 0.92] eligibility band, and its intervals are
  narrower than the prespecified LightGBM CQR reference on both
  processes (paired width excess -15.7% and -16.4%). With no scale
  adopted, the candidate's fit on these processes is the constant-scale
  fit, so latent-mean error is unchanged from discovery with spread
  off.
- The cost is attribution and conditional coverage. No varying-scale
  term is reported for data of this shape, and coverage split by the
  true-scale terciles is unbalanced: on axis_varying, 0.963 in the low
  tercile and 0.805 in the high tercile; on shifted_varying, 0.978 and
  0.793, against a nominal 0.90. Intervals are too wide where the true
  scale is small and too narrow where it is large.
- A supplied specification is not limited this way. A GaussianLSS fit
  with an explicit sigma formula (the battery's supplied_varying
  control uses `bbs(x1, df=4)` on every draw), or a mandatory spread
  column named in discover, fits a varying scale on such data without
  relying on the discovery test.

## Decision

A documented limitation is accepted. Sharp-scale detection is not
required of the default, and no library code or default changes in this
change. The grounds are the three measured facts above: the gated
behaviour the default is held to (no false scale adoption on any
constant-variance process, 40 of 40 recall on every smooth
varying-scale process, eligible marginal coverage on the sharp
processes) holds in the same confirmation record that shows the miss;
the miss is a mean-versus-scale attribution failure at the screening
stage, present before the #629 guard, and no candidate on record comes
close to a defensible bar on the sharp panels (the strongest recorded
prototype reaches 5 of 10 on one of them, on development draws); and
the user-visible harm is bounded and stated, namely missed attribution
and unbalanced tercile coverage, not loss of marginal coverage or of
mean accuracy. Requiring detection now would either hold the default
to a standard no recorded method meets, or invite an ungated change to
the adoption rule of exactly the kind the #629 guard exists to prevent.

## Reopening conditions

This decision is revisited if a candidate is proposed for sharp-scale
detection. Such a proposal must freeze its design on development draws,
and it must be confirmed on a fresh cohort: seeds 6294100-6294139 are
spent and cannot serve as fresh evidence for a mean-versus-scale
candidate. The confirmation bars should mirror the frozen gates of the
#629 battery, stated in advance: recall of at least 90% on each sharp
varying-scale process, no false adoption beyond a line frozen in
advance on the constant-variance processes (the #629 gate was zero of
40), and no mean harm by the paired latent-mean-MSE form of gate G4.
Until a candidate passes such a battery, scale claims for the default
carry the limitation stated in the book.

## Book statement

The limitation and the recall figure (0 of 40 draws per process, seeds
6294100-6294139) are stated next to the scale-adoption discussion on
the Gaussian mean and scale controls page
(../gaussian_scale_development.ipynb), and noted on the mean fitting
and interval efficiency page (../poisson_tuning_consistency.ipynb),
which presents the sharp-mean control inventory.
