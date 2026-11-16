# Cancer battery, draw 0: comparison of the tuned linear and searched models

## Study objective

On the frozen cancer battery, the best recorded score belongs to a
plain tuned linear model — mean test log loss 0.0901 over the three
draws — and the automatic search's models score 0.1179. A score says
how well a model predicted. It does not say what the model claims
about the measurements, or whether those claims match the data the
model was fitted on. This record reads both fitted models from
draw 0 (seed 0) term by term and checks each claim against the
fitted rows. Draws 1 and 2 follow at the end: their main terms
briefly, their interaction surfaces in full.

The two models, as fitted on draw 0:

- **Model A, the search's model.** Seven terms: curves on log of
  area error, worst radius, worst perimeter, worst smoothness,
  worst concave points and worst symmetry, and one interaction
  surface, mean concave points × worst texture. Fitted by boosting
  at 3,261 steps. Test log loss 0.1244.
- **Model B, the tuned linear model.** All thirty measurements,
  each standardized, one weight each, ridge-penalized logistic
  regression. Test log loss 0.1010.

Per draw, Model B scored better on all three draws (test log loss
0.1010 / 0.0578 / 0.1115 against Model A's 0.1244 / 0.0688 /
0.1604). This record does not relitigate that. It asks what each
model says.

## Setup and reproduction

The cancer task is the Wisconsin diagnostic breast cancer data
(569 rows, thirty measurements per tumour, outcome malignant or
benign) as the competitive shootout defines it
(`benches/shootout_data.py`): the shootout protocol for seed 0
gives 273 training rows, 91 validation rows, 91 calibration rows
and 114 test rows. Both models are fitted on the 364 training +
validation rows; of those, 37.4% are malignant. Every summary of
the data in this record is computed on those 364 rows. Test rows
are used only for the recorded scores quoted above.

Model A was reproduced by rerunning the battery's own call —
`autospec` on the 364 rows with the single 273/91 holdout as
folds, budget 32, step length 0.1, seed 0 — and refitting its
recommendation. The rerun returned the recorded formula and the
recorded step count, and its validation, calibration and test log
losses match the retained record (`exp_b_postfix_0.json`) with a
difference of 0.0 on all three (0.0303 / 0.0751 / 0.1244). A direct
refit of the recorded formula at the same control gives identical
test predictions (largest difference 0.0).

Model B was reproduced through the runner's own search
(`benches/shootout_models.py`, tier "tuned"): standardize on the
fitted rows, logistic regression at the four prespecified
penalties, selection by validation loss, refit on training +
validation. Its test log loss matches the recorded 0.1010 with a
difference of 0.0. The winning penalty on this draw, which the
existing summaries do not record, is **1.0** (C = 1). The four
candidates' validation losses: penalty 0.1 → 0.1337, penalty 1 →
0.0756, penalty 10 → 0.1187, penalty 100 → 0.2386.

The two search models of draws 1 and 2 were reproduced the same
way for their surface readings below: each rerun returned the
recorded formula and step count (3,894 and 9,989), and every
validation, calibration and test score difference was again 0.0.
Every reading in this record is a reading of a refitted model,
with the recorded-versus-refit gap stated in plain numbers; on
the current code all gaps are 0.0, so each refit is its recorded
model to every printed digit.

One scale note. The library's Binomial predictor is the half
log-odds (the ±1 coding): the logit of the predicted probability
is twice the fitted link, verified to 1.7e-08 on the fitted rows.
Every effect below is stated on the log-odds scale — twice the
fitted contribution — so the numbers read as ordinary logistic
log-odds. Term contributions are centred: a contribution of 0 is
the term's average over the fitted rows, not the absence of the
measurement.

## Model B: thirty weights and their limits

The thirty weights on the standardized scale, ranked by size,
with each measurement's own association alongside (AUC of the
single measurement, and the malignancy rate in each third of its
fitted-row range):

| measurement | weight | own AUC | malignancy rate by tertile (low / mid / high) |
|---|---|---|---|
| radius error (x10) | +1.319 | 0.861 | 0.08 / 0.24 / 0.79 |
| worst radius (x20) | +0.975 | 0.968 | 0.02 / 0.17 / 0.93 |
| mean concave points (x7) | +0.937 | 0.968 | 0.01 / 0.20 / 0.91 |
| worst symmetry (x28) | +0.873 | 0.738 | 0.20 / 0.32 / 0.60 |
| worst concave points (x27) | +0.873 | 0.967 | 0.02 / 0.17 / 0.92 |
| perimeter error (x12) | +0.859 | 0.874 | 0.05 / 0.29 / 0.78 |
| worst texture (x21) | +0.847 | 0.782 | 0.09 / 0.38 / 0.65 |
| worst area (x23) | +0.833 | 0.967 | 0.01 / 0.17 / 0.93 |
| area error (x13) | +0.806 | 0.919 | 0.06 / 0.21 / 0.85 |
| compactness error (x15) | -0.803 | 0.747 | 0.13 / 0.38 / 0.61 |
| worst perimeter (x22) | +0.797 | 0.973 | 0.01 / 0.18 / 0.93 |
| worst concavity (x26) | +0.712 | 0.924 | 0.02 / 0.24 / 0.86 |
| mean smoothness (x4) | +0.656 | 0.753 | 0.12 / 0.40 / 0.61 |
| mean texture (x1) | +0.596 | 0.779 | 0.12 / 0.36 / 0.65 |
| fractal dimension error (x19) | -0.548 | 0.633 | 0.23 / 0.40 / 0.49 |
| mean concavity (x6) | +0.514 | 0.939 | 0.02 / 0.22 / 0.87 |
| worst smoothness (x24) | +0.496 | 0.756 | 0.16 / 0.33 / 0.62 |
| mean radius (x0) | +0.459 | 0.930 | 0.04 / 0.21 / 0.86 |
| worst fractal dimension (x29) | +0.447 | 0.690 | 0.20 / 0.35 / 0.57 |
| concave points error (x17) | +0.442 | 0.803 | 0.07 / 0.37 / 0.68 |
| mean area (x3) | +0.437 | 0.931 | 0.04 / 0.21 / 0.86 |
| mean perimeter (x2) | +0.424 | 0.941 | 0.03 / 0.21 / 0.88 |
| symmetry error (x18) | -0.296 | 0.468 | 0.42 / 0.36 / 0.34 |
| mean symmetry (x8) | -0.243 | 0.694 | 0.21 / 0.34 / 0.57 |
| worst compactness (x25) | +0.205 | 0.874 | 0.08 / 0.26 / 0.78 |
| concavity error (x16) | -0.180 | 0.790 | 0.07 / 0.42 / 0.63 |
| mean compactness (x5) | -0.155 | 0.880 | 0.07 / 0.26 / 0.79 |
| smoothness error (x14) | +0.141 | 0.464 | 0.37 / 0.41 / 0.34 |
| mean fractal dimension (x9) | -0.125 | 0.509 | 0.38 / 0.34 / 0.41 |
| texture error (x11) | -0.111 | 0.507 | 0.34 / 0.42 / 0.36 |

Three readings.

**The size signal is real, and it is one signal.** Radius,
perimeter and area measure the same tumour three ways. On the
fitted rows, mean radius and mean perimeter correlate at 0.998,
mean radius and mean area at 0.986; the worst versions correlate
at 0.979–0.994; the error versions at 0.944–0.973, and area error
also correlates 0.805 with mean area. Twenty-nine pairs of
measurements correlate at 0.85 or more. Model B spreads the size
signal over all nine of these weights: radius error +1.32, worst
radius +0.98, perimeter error +0.86, worst area +0.83, area error
+0.81, worst perimeter +0.80, mean radius +0.46, mean area +0.44,
mean perimeter +0.42. They sum to +6.91. Read together, they say
"bigger tumours, and tumours whose size varies more across the
sampled cells, are malignant" — one claim with nine carriers.
Read singly, no one weight says anything defensible: the weight
on mean radius is small not because radius is weak (its own AUC
is 0.930) but because its near-copies hold most of the shared
signal. Which copy holds how much is an accident of the fit, not
a fact about the measurements.

**Eight weights are negative, and four of them contradict the
measurement's own direction.** Mean compactness (−0.16),
compactness error (−0.80), concavity error (−0.18) and mean
symmetry (−0.24) all carry negative weights, while each rises
with malignancy on its own (own AUCs 0.880, 0.747, 0.790, 0.694;
tertile rates rising, e.g. compactness error 0.13 / 0.38 / 0.61).
The negative weights are properties of the joint fit, not of the
measurements. Compactness shows why: compactness is a ratio built
from perimeter and area, so once the size block is in the model,
the ratio's leftover variation can take either sign. A reader who
took the −0.80 on compactness error as "irregular outline
protects" would be reading a bookkeeping residue. The remaining
four negative weights (mean fractal dimension, texture error,
symmetry error, fractal dimension error) sit on measurements
whose own association is near zero (AUCs 0.46–0.63); neither sign
is supported by the measurement alone.

**Nothing is switched off.** The smallest weight is 0.11 in
absolute value. The winning penalty is mild, so the model is the
full thirty-measurement fit: every measurement, including the
four with no signal of their own, contributes to every
prediction.

## Model A: seven terms, read one at a time

The recorded formula, quoted verbatim (the x-names are the
shootout's column order; the real names are used everywhere
else):

```
y ~ bbs(log(x13), df=4.0) + bbs(x20, df=4.0) + bbs(x22, df=4.0) + bbs(x24, df=4.0) + bbs(x27, df=4.0) + bbs(x28, df=4.0) + bbs(x7, df=2.5, knots=5) %X% bbs(x21, df=2.5, knots=5)
```

Over its 3,261 steps the boosting updated the terms unevenly:
log of area error 710 times, worst radius 578, worst smoothness
543, the surface 531, worst concave points 468, worst symmetry
426, worst perimeter 5. Each curve below is the term's
contribution at six points of its measured range, from the fitted
model.

**area error** (log of area error), log-odds contribution:
| measured value | 6.802 | 16.74 | 21.24 | 30.11 | 53.36 | 542.2 |
|---|---|---|---|---|---|---|
| contribution | -1.70 | -1.57 | -1.38 | -0.45 | +2.15 | +4.06 |

**worst radius** (x20), log-odds contribution:
| measured value | 8.964 | 12.41 | 13.91 | 16.07 | 20.14 | 36.04 |
|---|---|---|---|---|---|---|
| contribution | -3.72 | -2.03 | -1.16 | +0.50 | +3.04 | -0.23 |

**worst perimeter** (x22), log-odds contribution:
| measured value | 57.17 | 80.84 | 90.87 | 105.2 | 132.7 | 251.2 |
|---|---|---|---|---|---|---|
| contribution | -0.93 | -0.53 | -0.33 | +0.02 | +0.64 | +0.77 |

**worst smoothness** (x24), log-odds contribution:
| measured value | 0.07117 | 0.1124 | 0.1255 | 0.1377 | 0.1494 | 0.2098 |
|---|---|---|---|---|---|---|
| contribution | -5.60 | -1.02 | -0.04 | +0.64 | +1.18 | +4.27 |

**worst concave points** (x27), log-odds contribution:
| measured value | 0 | 0.0577 | 0.08309 | 0.1218 | 0.1773 | 0.291 |
|---|---|---|---|---|---|---|
| contribution | +1.08 | -0.81 | -1.34 | -0.63 | +1.21 | +2.27 |

**worst symmetry** (x28), log-odds contribution:
| measured value | 0.1565 | 0.2434 | 0.2701 | 0.2947 | 0.3232 | 0.6638 |
|---|---|---|---|---|---|---|
| contribution | +2.20 | -0.60 | -0.72 | -0.51 | +0.03 | +6.16 |


**Log of area error: a clean rise.** The contribution climbs
monotonically from −1.70 at the smallest area errors to +4.06 at
the largest, a swing of 5.8 log-odds, with most of the climb
above the median. The rows agree: malignancy rates by tertile of
area error are 0.06 / 0.21 / 0.85, and the measurement's own AUC
is 0.919. (Area error is the spread of the area across the
sampled cells of the tumour; its log is the search's chosen
scale.)

**Worst radius: a rise, then a fall the rows contradict.** The
curve rises from −3.72 to +3.04 as worst radius grows to about
23, then falls to −0.23 at the largest observed value, 36.0. The
fitted rows contain no such fall: all 76 rows with worst radius
above 20 are malignant (39 of 39 between 20 and 23.35, 37 of 37
above 23.35). In the region where the data are most one-sided,
the curve turns around. Worst perimeter — correlated 0.994 with
worst radius — rises gently over the same ground (swing 1.9,
ending +0.77), so the two size curves together still turn down
at the top: the fall is not compensated inside the model.

**Worst perimeter: a small, steady rise.** Swing 1.9 log-odds
across the full range, the gentlest term in the model, and the
least updated (5 of 3,261 steps). It reads as the size block's
second carrier: present, positive, and minor next to worst
radius.

**Worst smoothness: the largest curve, carried by its ends.**
The curve is strictly increasing, with a swing of 9.9 log-odds —
the largest of any single term. Through the middle of the range
it moves modestly (−1.0 at the 20th percentile to +1.5 at the
90th). The swing is made at the extremes: −5.6 at the minimum
(0.071) and +4.3 at the maximum (0.210). The rows support the
direction at both ends, on few rows: the 8 rows at or below 0.09
are all benign, and the 37 rows above 0.159 are malignant at a
rate of 0.81, against 0.36 in the middle band. The claim "very
smooth worst cells argue strongly against malignancy, very rough
ones for it" is in the data; the size of the endpoint effects
rests on 8 and 37 rows.

**Worst concave points: a U whose left arm the rows
contradict.** The curve starts at +1.08 log-odds where worst
concave points is 0, dips to −1.34 near the middle of the range
(0.083), and rises to +2.27 at the maximum (0.291). The rising
right arm matches the rows (tertile rates 0.02 / 0.17 / 0.92).
The raised left arm does not: the 15 fitted rows with worst
concave points at or below 0.02 — including all 8 rows at exactly
0 — are uniformly benign. The curve assigns its joint-highest
low-end value to a group of rows that contains no malignant case
at all.

**Worst symmetry: nothing, then a cliff.** Through most of the
range the curve is flat to slightly negative (−0.72 to −0.29
between the 20th and 70th percentiles). It reaches +0.96 at the
90th percentile (0.361) and +6.16 at the maximum (0.664): a jump
of 5.2 log-odds across the top tenth of the range. The tail rise
is real in the rows — the 37 rows above 0.361 are malignant at a
rate of 0.95. But the variable's own rates rise steadily across
its whole range (tertiles 0.20 / 0.32 / 0.60), where the curve
assigns almost nothing. The model's account of worst symmetry is
"irrelevant until the extreme tail, then decisive"; the rows
show a graded rise with a sharp tail on top of it. (The curve is
what remains after the other terms take their share, so the two
are not the same quantity — but the model's claim, read on its
own terms, concentrates the whole effect in the top tenth.)

## The interaction surface: mean concave points × worst texture

The surface is the model's only term for these two measurements.
There is no separate curve for either, so the surface carries
both measurements' main effects and their interaction together;
it cannot be read as a pure interaction. Its size: 81 basis
columns, a nominal penalized df of 6.25, 531 of the model's
3,261 boosting updates (16%), and the largest fitted-row effect
range of any term in the model, 11.6 log-odds. Its fitted values
(log-odds contribution) over the pair's range:

| mean concave points \ worst texture | 12.5 | 17.7 | 21.1 | 25.4 | 30.1 | 34.0 | 49.5 |
|---|---|---|---|---|---|---|---|
| 0.0000 | -4.70 | -3.53 | -2.93 | -2.39 | -2.14 | -2.10 | -3.47 |
| 0.0102 | -4.87 | -3.40 | -2.58 | -1.75 | -1.27 | -1.11 | -2.56 |
| 0.0195 | -5.02 | -3.29 | -2.26 | -1.17 | -0.48 | -0.21 | -1.73 |
| 0.0332 | -5.20 | -3.11 | -1.81 | -0.34 | +0.67 | +1.09 | -0.52 |
| 0.0737 | -5.13 | -2.34 | -0.55 | +1.62 | +3.30 | +4.03 | +2.94 |
| 0.1036 | -4.56 | -1.63 | +0.16 | +2.25 | +3.92 | +4.79 | +5.26 |
| 0.2012 | -2.97 | -0.92 | +0.33 | +1.95 | +3.70 | +5.20 | +11.51 |

In plain words, the surface claims the two measurements amplify
each other:

- Where worst texture is at its lowest (12.5, first column),
  moving mean concave points across its whole range changes the
  contribution by only 2.2 log-odds, and not steadily — the
  column dips from −4.70 to −5.20 before rising to −2.97.
- Where worst texture is high (34.0), the same move in mean
  concave points is worth 7.3 log-odds, rising steadily from
  −2.10 to +5.20.
- At mean concave points of 0, worst texture changes the contribution
  by 2.6 log-odds across its range. At high mean concave points,
  that range is larger; the largest changes occur in the far corner below.

The fitted rows show the same pattern in the middle of the
range. Malignancy rates by thirds of both measurements:

| mean concave points \ worst texture | low | mid | high |
|---|---|---|---|
| low | 0.00 (n=52) | 0.02 (n=42) | 0.00 (n=27) |
| mid | 0.04 (n=51) | 0.16 (n=38) | 0.50 (n=32) |
| high | 0.50 (n=18) | 0.95 (n=41) | 1.00 (n=63) |

Where mean concave points are low, worst texture separates
nothing (rates 0.00 / 0.02 / 0.00). Where they are middling,
texture separates clearly (0.04 / 0.16 / 0.50). That conditional
pattern — the heart of what the surface claims — is visible in
the raw counts, not only in the fit. Two qualifications. First,
the counts also show mean concave points separating at every
texture level (0.00 / 0.04 / 0.50 down the low-texture column),
steadily, where the surface's low-texture column dips before it
rises; the net direction agrees, the dip does not appear in the
counts. Second, the corner: at the largest value of both
measurements the surface adds +11.51 log-odds, and the effect
sizes quoted at the highest texture (49.5) rest on almost no
rows — 7 fitted rows have worst texture above 40, and 4 rows sit
in the corner region where both measurements are above their
90th percentiles. The corner is the spline extending a real
pattern past the rows that constrain it.

## Effective parameter count

**Model B.** Nominally 30 weights and an intercept. At the
winning penalty (λ = 1.0), the ridge effective degrees of
freedom — the trace of the hat matrix, Σ d²/(d² + λ) over the
singular values of the standardized 364 × 30 design — is 26.08,
plus 1 for the unpenalized intercept: **27.08 effective
parameters of 31 nominal**. The mild penalty removes about four.

**Model A.** Nominally far larger: under the library's own
construction each of the six curves carries 24 basis columns and
the surface carries 81 (a 9 × 9 tensor), for **225 coefficient
columns** plus the offset. (Each curve's own penalized degrees of
freedom, the `df` it was specified with, is 4.0; the surface's is
6.25.) The fitted model's effective degrees of freedom at its
3,261 steps, by the library's own hat-matrix trace — the quantity
its `aic(method="classical")` reports, the conditional hat trace,
which the library documents as excluding model-search degrees of
freedom — is **18.49**. The trace of the fitted hat values agrees
(18.4928 by both routes).

So the count of knobs and the amount of flexibility point in
opposite directions: 225 nominal columns shrink to about 18.5
effective parameters, while 31 nominal weights keep about 27.
The boosted model is not the more flexible fit at its stopping
point; it is the more concentrated one — its flexibility sits in
seven named terms, the linear model's is spread over all thirty
measurements at nearly full strength. One asymmetry belongs on
the record: Model A's 18.49 is conditional on the seven terms
the search selected, and does not price the search that selected
them; Model B's 27.08 prices everything it uses, because it uses
everything.

## Model comparison

They agree on the centre of the problem. Both put their largest
effects on the size measurements, both read size upward (larger
tumour, higher malignancy log-odds), both use concave points
upward, area error upward, worst texture upward, and both give
worst symmetry a strong positive role in its tail. A reader who
asked only "what drives this prediction" gets the same answer
from either model.

They disagree in five places. For each, the fitted rows' say:

1. **Worst radius at the top of its range.** Model A's curve
   turns down above ≈23, ending below where it stood at 20.
   Model B is monotone. The rows: all 76 fitted rows above 20
   are malignant. The data side with the monotone reading; the
   downturn is the model's, not the data's.
2. **Worst concave points at zero.** Model A's curve is at a
   local high (+1.08) at 0. Model B is monotone in it (+0.87).
   The rows at 0 are 8 of 8 benign. Again the data side against
   the curve's feature.
3. **Compactness.** Model B's two compactness weights are
   negative (−0.16, −0.80) while both measurements rise with
   malignancy on their own. Model A omits compactness entirely.
   Neither number is a claim about compactness itself: B's is a
   residue of fitting a perimeter–area ratio alongside perimeter
   and area; A's silence is a selection outcome, not a finding
   of no effect.
4. **Worst smoothness.** Model A gives it the largest single
   curve (swing 9.9, mostly at the extremes). Model B gives it a
   middling weight (+0.50, 14th of 30). The rows show a real but
   unspectacular marginal association (own AUC 0.756) with
   support at both extremes on 8 and 37 rows. A's emphasis is
   larger than the measurement's own showing; B's is in line
   with it.
5. **Worst symmetry in the bulk of its range.** Model A assigns
   it almost nothing until the top tenth, then +6.16 at the
   maximum. Model B spreads +0.87 linearly. The rows rise
   steadily (0.20 / 0.32 / 0.60 by tertile) and jump in the tail
   (0.95 above the 90th percentile). Neither shape matches that
   exactly; the graded part supports B's spread, the tail
   supports A's concentration.

A further difference is in what the models' own judging scores
said about them. Model A's validation log loss was 0.0303 and
its test log loss 0.1244 — the score it was chosen by understated
its test loss fourfold. Model B's selection loss was 0.0756 and
its test loss 0.1010. The search's model was fitted much closer
to its judging rows than the linear model was to its.

## Draw 1: thirteen terms, one surface

Draw 1's model (3,894 steps, test log loss 0.0688) spreads over
thirteen terms. Their sizes, briefly — the surface is read in
full below:

| term | boosting updates | effect range on fitted rows (log-odds) |
|---|---|---|
| mean texture | 128 | 1.66 |
| mean concavity | 154 | 1.54 |
| mean concave points | 4 | 1.42 |
| log of area error | 572 | 6.07 |
| concave points error (small tree) | 795 | 1.93 |
| log of symmetry error | 179 | 4.17 |
| worst texture | 607 | 8.11 |
| worst perimeter | 12 | 1.90 |
| log of worst area | 84 | 3.98 |
| worst concavity | 344 | 2.93 |
| worst concave points | 53 | 2.47 |
| worst symmetry | 167 | 3.92 |

The curves concentrate where draw 0's did: worst texture and log
of area error carry the largest ranges. No whole-fit hat-trace
df exists for this model: the library's `aic` refuses, because
the model contains a tree term (concave points error), whose
prediction is not a linear map of the response, so there is no
hat matrix to take a trace of. The sizes above are what the fit
exposes.

### Draw 1's surface: mean smoothness × log of fractal dimension error

The model's joint-largest term by updates (795 of 3,894, 20%)
and its largest by fitted-row range (11.34 log-odds), on 81
basis columns with a nominal df of 6.25. It is the model's only
term for either measurement. Fitted values:

| mean smoothness \ fractal dimension error | 0.0008948 | 0.001751 | 0.002233 | 0.003195 | 0.004559 | 0.006186 | 0.02984 |
|---|---|---|---|---|---|---|---|
| 0.05263 | +2.41 | -1.29 | -2.73 | -4.93 | -7.13 | -9.01 | -18.65 |
| 0.07948 | +0.40 | +0.29 | -0.17 | -1.41 | -3.09 | -4.52 | -10.88 |
| 0.08517 | -0.12 | +0.53 | +0.36 | -0.56 | -2.14 | -3.52 | -9.23 |
| 0.09618 | -1.24 | +0.59 | +0.99 | +0.76 | -0.48 | -1.68 | -6.08 |
| 0.1052 | -2.24 | +0.09 | +0.76 | +0.98 | +0.21 | -0.63 | -3.52 |
| 0.1142 | -3.23 | -0.68 | +0.10 | +0.67 | +0.47 | +0.12 | -0.96 |
| 0.1425 | -6.12 | -2.85 | -1.74 | -0.20 | +1.18 | +2.27 | +7.24 |

The surface claims a crossover, and a large one. At the lowest
mean smoothness, increasing fractal dimension error drives the
contribution steeply down, from +2.41 to −18.65 — a 21.1
log-odds fall, the largest single effect anywhere in the three
draws' models. At the highest mean smoothness the same move
reverses: the contribution rises from −6.12 to +7.24. Read as a
claim: fractal dimension error is strongly protective for
smooth tumours and strongly incriminating for rough ones.

The fitted rows do not show it. Malignancy rates by thirds of
both measurements, on draw 1's 364 fitted rows:

| mean smoothness \ fractal dimension error | low | mid | high |
|---|---|---|---|
| low | 0.11 (n=56) | 0.23 (n=40) | 0.00 (n=25) |
| mid | 0.41 (n=49) | 0.38 (n=39) | 0.50 (n=32) |
| high | 0.25 (n=16) | 0.55 (n=42) | 0.66 (n=65) |

Within every smoothness third, the high-fractal-dimension-error
rows are not the low-risk rows — the rates run 0.11 / 0.23 /
0.00, 0.41 / 0.38 / 0.50 and 0.25 / 0.55 / 0.66, and the
measurement's own rates rise weakly across its range (0.25 /
0.39 / 0.48 by tertile). Neither arm of the claimed crossover
appears in the counts; the one cell pointing the surface's way
(low smoothness, high error: rate 0.00 on 25 rows) is one cell
among nine, contradicted by its neighbours. The scale of the
claim also lives outside the data's bulk: the grid's last column
is the measurement's maximum (0.030), five times its 90th
percentile (0.006), and most of the −18.65 accumulates across
that stretch. The model's largest surface is built on one of
the weakest measurements in the set, and its central claim is
not in the rows it was fitted on.

## Draw 2: eight terms, three surfaces

Draw 2's model (9,989 steps, test log loss 0.1604) has five
curves and three surfaces. The curves, briefly:

| term | boosting updates | effect range on fitted rows (log-odds) |
|---|---|---|
| mean texture | 2345 | 8.45 |
| log of symmetry error | 544 | 6.84 |
| worst area | 1302 | 7.88 |
| worst smoothness | 2322 | 6.61 |
| worst symmetry | 877 | 9.34 |

The whole fit's effective size is 24.67 (the same conditional
hat trace as draw 0) from 363 basis columns — 120 for the five
curves, 243 for the three surfaces. Between them the surfaces
take 2,599 of the 9,989 updates (26%) and carry the model's
largest ranges. Each is the model's only term for its pair.

### Worst radius × worst compactness

The model's most-updated surface (1,503 updates, 15%) and its
largest range (11.86 log-odds). Fitted values:

| worst radius \ worst compactness | 0.02729 | 0.09746 | 0.1502 | 0.2168 | 0.3455 | 0.4392 | 0.9327 |
|---|---|---|---|---|---|---|---|
| 7.93 | -1.72 | -3.19 | -4.25 | -5.47 | -7.18 | -7.72 | -6.16 |
| 11.08 | +0.15 | -1.35 | -2.44 | -3.72 | -5.42 | -5.77 | -4.49 |
| 12.98 | +1.27 | -0.13 | -1.16 | -2.38 | -4.02 | -4.32 | -3.57 |
| 14.75 | +2.31 | +1.08 | +0.16 | -0.94 | -2.45 | -2.79 | -2.79 |
| 18.58 | +4.29 | +3.48 | +2.89 | +2.15 | +0.93 | +0.34 | -1.38 |
| 23.69 | +5.89 | +5.33 | +4.91 | +4.37 | +3.31 | +2.56 | -0.02 |
| 33.13 | +6.83 | +6.20 | +5.73 | +5.17 | +4.19 | +3.60 | +1.74 |

Most of what this surface says is two main effects, not an
interaction. Worst radius reads strongly upward at every level
of worst compactness (its effect varies only between 7.9 and
11.4 log-odds across the compactness range). Worst compactness
reads strongly downward at every level of worst radius, by a
near-constant 5–6 log-odds. The interaction proper — how either
effect changes with the other measurement — is small next to
the two gradients the surface carries.

The rows support one gradient and contradict the other.
Malignancy rates by thirds, on draw 2's fitted rows:

| worst radius \ worst compactness | low | mid | high |
|---|---|---|---|
| low | 0.00 (n=67) | 0.00 (n=42) | 0.00 (n=11) |
| mid | 0.06 (n=47) | 0.07 (n=42) | 0.48 (n=33) |
| high | 0.71 (n=7) | 0.89 (n=37) | 0.97 (n=78) |

Worst compactness's own rates rise (0.07 / 0.30 / 0.75 by
tertile), and within each radius third its association is flat
or rising (0.00 / 0.00 / 0.00; 0.06 / 0.07 / 0.48; 0.71 / 0.89 /
0.97) — nowhere negative. The surface's downward compactness
claim is the same artefact seen in the linear model's negative
compactness weights: compactness is a ratio of perimeter and
area, and once size is otherwise accounted, the ratio's leftover
variation takes the opposite sign. It is a property of fitting
the ratio alongside its own components, not of the measurement.

### Worst perimeter × worst concave points

The least-updated surface (319 updates, 3%), range 8.16
log-odds. Fitted values:

| worst perimeter \ worst concave points | 0 | 0.0394 | 0.06387 | 0.0986 | 0.1617 | 0.209 | 0.291 |
|---|---|---|---|---|---|---|---|
| 50.41 | -3.24 | -3.59 | -3.71 | -3.63 | -2.24 | -0.34 | +3.30 |
| 71.58 | -2.27 | -2.72 | -2.89 | -2.79 | -1.17 | +0.74 | +3.68 |
| 83.59 | -1.70 | -2.17 | -2.33 | -2.17 | -0.45 | +1.38 | +3.84 |
| 96.37 | -1.06 | -1.49 | -1.60 | -1.35 | +0.41 | +2.04 | +3.93 |
| 124.9 | +0.63 | +0.44 | +0.49 | +0.88 | +2.26 | +3.14 | +3.86 |
| 158.1 | +2.95 | +2.96 | +3.05 | +3.28 | +3.63 | +3.61 | +3.36 |
| 229.3 | +8.24 | +7.61 | +7.17 | +6.48 | +5.02 | +3.87 | +1.93 |

The claim is saturation. At the lowest worst concave points,
worst perimeter's effect spans 11.5 log-odds (−3.24 to +8.24);
at the highest, it spans 2.0, and the top row of the grid falls
at the end (3.93 at the middle, 1.93 at the largest perimeter).
Read in plain words: each measurement does the other's work —
once concave points are extreme, perimeter adds almost nothing,
and the largest tumours are not further incriminated by it.

The counts agree where they can speak:

| worst perimeter \ worst concave points | low | mid | high |
|---|---|---|---|
| low | 0.00 (n=85) | 0.00 (n=34) | 0.00 (n=2) |
| mid | 0.03 (n=36) | 0.09 (n=67) | 0.67 (n=18) |
| high | — (n=0) | 0.75 (n=20) | 1.00 (n=102) |

Rates rise with both measurements and compress towards 1 in the
high-high corner, which is populated (21 rows above both 90th
percentiles). But the two measurements are strongly correlated,
so the off-diagonal corners where a saturation claim is tested
hardest are nearly empty: low perimeter with high concave
points holds 2 rows, high perimeter with low concave points
holds none. The rows are consistent with the surface's claim in
every populated cell; the cells that would test it most sharply
do not exist in the data.

### Mean concave points × worst texture, again

The same pair as draw 0's surface (777 updates, 8%; range 8.22
log-odds). Fitted values:

| mean concave points \ worst texture | 12.02 | 17.73 | 21.29 | 25.3 | 29.89 | 33.8 | 49.54 |
|---|---|---|---|---|---|---|---|
| 0 | -4.91 | -3.63 | -2.84 | -1.91 | -0.79 | +0.13 | +3.24 |
| 0.01106 | -4.81 | -3.47 | -2.61 | -1.60 | -0.42 | +0.53 | +3.32 |
| 0.01924 | -4.72 | -3.33 | -2.42 | -1.35 | -0.12 | +0.83 | +3.38 |
| 0.0327 | -4.51 | -3.03 | -2.02 | -0.85 | +0.42 | +1.35 | +3.45 |
| 0.07626 | -3.14 | -1.32 | -0.09 | +1.20 | +2.31 | +2.89 | +3.50 |
| 0.1017 | -1.95 | -0.17 | +0.93 | +1.99 | +2.81 | +3.19 | +3.39 |
| 0.2012 | +2.39 | +2.47 | +2.49 | +2.47 | +2.45 | +2.43 | +2.44 |

This surface claims the opposite of draw 0's. There, the two
measurements amplified each other: texture was worth 2.6
log-odds at the lowest concave points and 14.5 at the highest.
Here they substitute: worst texture's effect is largest at the
lowest mean concave points (8.15 log-odds) and shrinks steadily
as concave points grow, to 0.10 at the maximum — the grid's top
row is flat (2.39 to 2.44 across the whole texture range), as if
texture stopped mattering once concave points are large.

Draw 2's own rows side with draw 0's surface, not with this
one:

| mean concave points \ worst texture | low | mid | high |
|---|---|---|---|
| low | 0.00 (n=57) | 0.03 (n=38) | 0.04 (n=26) |
| mid | 0.02 (n=50) | 0.16 (n=38) | 0.42 (n=33) |
| high | 0.57 (n=14) | 0.93 (n=45) | 1.00 (n=63) |

Where mean concave points are low, worst texture separates
nothing in these rows either (0.00 / 0.03 / 0.04); it separates
in the middle (0.02 / 0.16 / 0.42) and the top is high
throughout (0.57 / 0.93 / 1.00) — the same conditional pattern
as draw 0's rows. The two draws' counts agree with each other.
Their surfaces do not: given the same pair of measurements and
two overlapping samples of the same data, the search returned
an amplification in one draw and a substitution in the other,
and the substitution's fitted rows show the amplification
pattern. The interaction direction is not a stable property of
the search's output on this battery.

Across the three draws the surfaces divide by their support.
Draw 0's amplification claim for this pair is visible in its
rows' counts. Draw 2's saturation claim for worst perimeter ×
worst concave points is consistent with its rows where rows
exist. Draw 2's compactness gradient and draw 1's crossover are
contradicted by their rows, and draw 2's version of this pair's
surface is contradicted by its own rows in direction. Every
surface is also its model's only account of two measurements,
so each of these claims — supported, untestable or contradicted
— is the whole of what the model says about that pair.

## Records and reproducibility

- Model A: rerun of the battery's autospec call for seed 0 in
  this study; formula, step count (3,261) and all three scores
  match `docs/book/benchmarks/competitive_refresh/`'s retained
  run files (`exp_b_postfix_0.json`) to every printed digit.
- Model B: the shootout runner's tuned linear search for seed 0;
  test log loss matches the refresh scorecard's recorded value
  for the draw. The winning penalty (1.0) and the four validation
  losses are recorded here for the first time.
- Curves, the surface grid, weights, correlations and rates: all
  computed in this study from the two refitted models and the 364
  fitted rows, via the library's per-term prediction
  contributions (verified to sum, with the model offset, exactly
  to the fitted link) and the runner's fitted pipeline.
- Draws 1 and 2: reproduced like draw 0 (autospec rerun per
  seed in this study); formulas, step counts (3,894 / 9,989) and
  all scores match the retained run files
  (`exp_b_postfix_1.json`, `exp_b_postfix_2.json`) to every
  printed digit, and direct refits at the recorded controls
  reproduce the same scores. Their term ranges, surface grids
  and rate tables were computed from those refits on each draw's
  own 364 fitted rows.
