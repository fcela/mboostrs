# `aipw`

Defined in `mboostrs.causal`; exported as `mboostrs.aipw`.


*Function*

```python
aipw(
    data,
    spec: mboostrs.causal.CausalSpec,
    folds=5,
    repeats=1,
    seed=42,
    smooth=False,
    df=4,
    mstop=200,
    max_mstop=None,
    tune=True,
    inner_folds=5,
    level=0.95,
    propensity_clip=from_spec,
    outcome_missing=from_spec,
    weights=None,
    outcome_model='cv',
    outcome_interactions=False,
    overlap_bounds=from_spec,
    target='all',
    propensity_terms=None,
    outcome_terms=None,
    extrapolation='warn',
    flat_tolerance=0.01,
    repeat_aggregation='average',
    propensity_learner='smooth',
    interval='influence',
    bootstrap_draws=200
)
```

Cross-fitted augmented inverse-probability estimate for binary treatment.

The supported outcome types are a continuous mean difference and a binary
risk difference, both with numeric baseline covariates and ATE/ATT.
Nuisance models are fitted outside each observation's fold; with
`tune=True` their stopping iterations are chosen by an inner
cross-validation using outer-training rows only. Ordinary folds use IID
influence-function uncertainty. A grouped fold scheme uses the retained
group labels for cluster-robust uncertainty. Ordered dependence needs an
explicit time-series variance model and therefore has no interval.

`interval` chooses how the standard error and interval are computed.
`"influence"` (the default) uses the influence function of the final
estimate, treating the cross-fitted nuisance predictions as fixed. That
variance leaves out the error of the nuisance fits where the propensity is
near 0 or 1 and the data-dependent choice of the outcome model, and on a
thin-overlap design its interval can cover less than `level`.
`"bootstrap"` resamples the whole procedure. Each of `bootstrap_draws`
draws (default 200) resamples rows with replacement, or whole clusters when
the fold scheme is grouped, then reruns the cross-fitting on the resample,
including the `outcome_model="cv"` choice and the tuning of every
nuisance, and records the ATE. The reported interval is the percentile
interval of the draws at `level` and `standard_error` is their standard
deviation. Rows are drawn by the core's `group_bootstrap_counts` stream
seeded by `seed`, so a seed fixes the draws. In a draw, the copies of one
original row (or cluster) are kept in one cross-fitting fold, so no copy is
predicted by a model trained on another copy; the folds of a draw are
therefore grouped folds with the number of folds of the call. A draw that
fails (for example a fold without a treated row) is dropped: the number of
failed draws is in `diagnostics["bootstrap"]["n_failed"]` and in a
warning, and the interval uses the draws that finished. The influence
standard error and interval remain in `diagnostics["bootstrap"]`. The
bootstrap needs `repeats=1` and an unordered fold scheme. The cost is
`bootstrap_draws` times the cost of one call;
`estimate_cost` counts it.
The draws run one after another. Each draw's nuisance fits use the core's
thread pool, and the warning bookkeeping of the fits is not safe to share
between Python threads, so there is no worker-thread option.

`level` is the coverage of the reported interval. The default is 0.95, a
95% interval; pass `level=0.9` for a 90% interval.

`repeats>1` draws independent stratified partitions and averages each
observation's out-of-fold nuisance predictions before constructing its
score. This reduces dependence on one arbitrary partition. The reported
influence interval is conditional on that repeated-fitting procedure; it
does not add a separate variance component for split randomness.

`repeat_aggregation` says how `repeats>1` combines the partitions. The
default `"average"` is the averaged-nuisance estimator described above:
`estimate`, `scores` and `influence` all come from the averaged
nuisances, so the mean of `scores` equals `estimate`. `"median"` is
the median rule of Chernozhukov et al. (2018). Each repeat computes its own
estimate `theta_r` and influence-function standard error `se_r` from its
own nuisances. The reported estimate is the median of the `theta_r`, and
the variance is the median of `se_r^2 + (theta_r - estimate)^2`. The
second term carries the spread of the partitions into the interval, so the
interval includes split-to-split variation. `estimate` and the interval
then differ from `mean(scores)`, and `scores` and `influence` remain
those of the averaged nuisances. `diagnostics["repeat_estimates"]` and
`diagnostics["repeat_standard_errors"]` list each repeat's estimate and
standard error whenever `repeats>1`. The median rule needs an integer
`folds`, as `repeats>1` does.

`smooth=False` (the default) fits each nuisance with linear terms only.
`smooth=True` fits a penalized spline (`bbs`) for every numeric baseline
that has at least ten distinct values, and a linear `bols` term for the
others: a two-valued indicator, a low-cardinality count, or a factor. This
is the rule `effect_model` uses. On 10 simulated draws at n=12,000 with
90% intervals, the median and largest absolute ATE error, the coverage and
the median seconds were: `smooth=False` with `outcome_model="cv"` 434 and
2,322, 0.9, 2.7 s; `smooth=True` with `"cv"` 550 and 1,706, 0.9, 8.0 s;
`smooth=False` with `"auto"` 790 and 2,322, 0.6, 1.5 s; `smooth=True`
with `"auto"` 1,137 and 8,205, 0.6, 5.9 s. Baselines
may be numeric, boolean, string or categorical; a factor is coded with
treatment contrasts, as `mboostrs.gamboost` codes a string column,
and `diagnostics["balance"]` reports one row per factor level.
`smooth` also takes a dict with keys `"propensity"`, `"outcome"` and
`"response"`, for example `smooth={"propensity": False, "outcome": True}`.
Each named nuisance gets its own flag; a nuisance the dict does not name
takes `False`.

`propensity_terms` and `outcome_terms` replace the formula that
`smooth` builds for the propensity model and for the outcome model. Each is
a right-hand side as a string, for example
`"bbs(age) + btree(prior_earnings, maxdepth=1)"`; the response and `~`
are not included. A step in the propensity at an eligibility limit is
fitted by a tree term, which the default formula does not contain. A
custom right-hand side is fitted by `gamboost`, whatever `smooth` says.
The response model keeps the formula that `smooth` builds. With
`outcome_model="pooled"` the treatment is added to `outcome_terms` as a
linear term, and `outcome_interactions=True` is refused with
`outcome_terms`. An interaction is written inside the string with `by=`
set to the treatment column: `"bbs(age, df=4, by=treated)"` when
`treated` is the treatment column. A custom term that names the treatment
as a bare term is refused. The formulas used are in `nuisance["terms"]`.

`propensity_learner` chooses the propensity formula when
`propensity_terms` is not given. `"smooth"` is the formula that the
`smooth` argument builds: `bbs` terms when it is on, and
linear terms when it is off. `"smooth+trees"` adds
`btree(name, maxdepth=1)` for each numeric baseline with more than two
distinct values, and is fitted by `gamboost`. A depth-one tree makes one
split, so it can place a step such as an eligibility cutoff, which a
spline spreads over several basis functions. `"smooth"` (the default)
keeps the formula that `smooth` builds.
`"auto"` fits both formulas on the same folds and keeps the one with the
lower out-of-fold binomial deviance; it changes the propensities, and so the
estimate, when the tree formula wins. The two deviances and the choice are in
`diagnostics["propensity_learner"]`, and `AIPWResult.describe`
states them. With `propensity_terms` the learner is not used, and an
explicit value other than `"smooth"` is refused.

`mstop` is the first grid of each nuisance's step count. With `tune=True`
the grid doubles while the inner cross-validated risk still falls, up to
`max_mstop` (default 1600, and never below `mstop`). A nuisance whose
risk is still falling at `max_mstop` triggers a warning that names this
argument. `diagnostics["nuisance_mstop"]` records, for each nuisance, the
step counts of its folds and whether any fold reached the cap.
`diagnostics["nuisance_mstop_summary"]` gives one step count per nuisance,
the median of its fold counts. With `outcome_model="cv"` it also holds the
nuisances of the candidates that were not chosen (a nuisance shared by two
candidates takes the chosen candidate's count), so it can be passed back to
another `"cv"` call. Passing that mapping as `mstop` with
`tune=False` refits every nuisance at its own fixed step count, without a
second tuned pass. With `tune=False` every nuisance runs exactly `mstop`
steps.

Warnings from the nuisance fits are issued once per call, each naming its
nuisance model (propensity, response, pooled outcome, control-arm outcome or
treated-arm outcome) and the argument that controls it. `extrapolation`
is the policy for nuisance predictions outside a spline's training range,
as in `mboostrs.gamboost.GAMBoostModel.predict`: `"warn"` (the
default) predicts and warns, `"clip"` holds the prediction flat at the
range edge, `"allow"` is silent and `"error"` refuses. The two-part
outcome model is the exception: its covariates are always held to the
training range, because its mean takes an exponential of the log model's
prediction. `flat_tolerance`
is the relative gap at which a tuned nuisance's cross-validated curve counts
as flat, as in `mboostrs.explore.tune_mstop`: a flat curve means the
nuisance is close to a constant. `None` turns that check off.

`outcome_model` says how the outcome models are fitted. `"cv"` (the
default) is described below. `"auto"` fits `"pooled"` without treatment interactions when the data
have at most 1,500 rows. That threshold is the 1,500 rows of the small
simulated draws in `docs/book/benchmarks/causal_discovery`, the largest
row count the measurement covers: over 25 such draws the mean absolute ATE
error was 1,404 for the pooled model without interactions and 2,487 for the
rule that applies above the threshold. With
more rows `"auto"` runs the sparse-arm check described below on the
cross-fitted propensity. If the check flags a bin, the fit uses `"pooled"`
with `outcome_interactions=True` (with `outcome_terms`, `"pooled"`
without interactions); if it flags no bin, the fit uses `"separate"`.
`outcome_interactions=True` with `"auto"` always gives the pooled model
with interactions. `"auto"`
never chooses `"two_part"`, which must be requested by name.
`outcome_model="cv"` fits every candidate outcome model on the
cross-fitting folds the call already uses and keeps the one with the lowest
out-of-fold loss on the observed outcomes: the weighted mean binomial
deviance for a binary outcome, the weighted mean squared error (the Gaussian
family loss) for a continuous one, each computed from the
cross-fitted `mu0` and `mu1` at the arm a row received. The candidates
are `"pooled"`, `"separate"`, `"two_part"` when the outcome is
continuous, zero or positive with at least one zero, and `"pooled"` with
interactions (labelled `"pooled_interactions"`) when the `"auto"` rule
would pick it. `diagnostics["outcome_model_cv"]` lists the candidates, the
loss of each and the chosen label. The outcome models are fitted once per
candidate, so the outcome step costs that many times as much. The chosen
model and the reason are in
`diagnostics["outcome_model_choice"]`, and `AIPWResult.describe`
states them. `"separate"` fits one outcome model per arm, each on that
arm's rows, so the treated model is fitted only where treated rows are.
`"pooled"` fits one outcome model on all observed rows with the treatment
as a linear term. The treated arm then borrows the control arm's baseline
shape and adds the estimated shift. With `outcome_interactions=True` every
baseline term is also multiplied by the treatment, which lets the shape
differ by arm again; that option needs the rows of the thinner arm to fit
its own shape. `outcome_interactions=True` with `"separate"` is refused.
With `"auto"` and `outcome_interactions=True` the fit uses the pooled
model with interactions. A covariate set refitted by
`confounding_benchmark` uses the model chosen for the full set.

`outcome_model="two_part"` is for an outcome that is zero or positive with
some zeros, such as earnings with unemployed people at zero, and a continuous
outcome. The employment model is a Bernoulli model for P(Y > 0 | x, a) on every
observed row. The log model is a Gaussian model for log(Y) on the positive rows.
Both have the treatment as a linear term, as the pooled model does. The mean is
`P(Y > 0 | x, a) * exp(m(x, a) + s2 / 2)`, where `m` is the log model's mean and
`s2` the residual variance of log Y on the training rows. A treatment effect
that scales with the baseline level of a positive outcome is a constant shift on
the log scale, so the log model carries it to baseline levels that have few treated
rows. `outcome_interactions=True` and `outcome_model="separate"` are refused
with it. The log model enters each strictly positive baseline whose sample
skewness exceeds 1 on the log scale (for example prior earnings in dollars
enters as its logarithm), because a linear term in dollars gives a log mean
that grows with the raw value and the exponential then amplifies it. The
employment model keeps the raw baselines. Both models hold every numeric
covariate to its training range at prediction time, whatever `extrapolation`
says.
`nuisance["terms"]["outcome"]` gives the formula of both parts, and a custom
`outcome_terms` is used as written, with no log-scale copy. The model has
no treatment interactions. `"auto"` does not select this model; on a
flagged sparse arm it uses the pooled model with interactions instead.

`diagnostics["outcome_range"]` checks the fitted means `mu0` and `mu1`.
A mean lying more than one observed outcome range beyond the range of the
observed outcome on either side is counted there and warned about, naming the
remedy. The check reports a model that extrapolates; it does not change the
estimate.

`diagnostics["sparse_arms"]` counts treated and control rows in each
propensity bin of the cross-fitted propensity, with the propensity range and
the covariate span of the rows in the bin. The bins hold roughly equal total
weight: one bin per 200 units of weight (the row count when unweighted),
between 4 and 10 bins. Arm counts and shares are weighted sums, so a row with
weight 3 counts as three rows. A bin is flagged for an arm when it holds more
than 5% of the target weight (all rows for the ATE, treated rows for the ATT)
and fewer than 20 weighted rows of that arm. A flagged bin warns
whenever the fit uses separate models or pooled interactions, because those
fit an outcome curve for the thin arm from its own rows. The warning names
the model that was fitted. A pooled model without interactions records the
flag without warning.

`outcome_missing` says what a missing outcome (NaN) means. When the call
does not pass it, the `outcome_missing` of the `CausalSpec` is used.
`"error"` (the default of the spec) requires an observed outcome in every
row. `"ipw"` treats
the outcome as missing at random given the baseline covariates and the
treatment: a response model for P(R = 1 | x, a) is cross-fitted like the
propensity, outcome means are fitted on observed rows, and the augmented
score divides the observed-row terms by the response propensity pi. The
estimand keeps its target population: the ATE is over all rows, and the
ATT is over all treated rows, observed or not. The ATT numerator of row i
is `A (mu1 - mu0) + R A (Y - mu1)/pi - R (1 - A) e/(1 - e) (Y - mu0)/pi`,
divided by the treated share. Its influence-function variance treats the
cross-fitted nuisances as fixed, as the ATE variance does. The result
reports the range of pi and its effective sample size.

`weights` are observation weights read as frequencies: a row with weight
2 counts as two identical rows. They enter the nuisance fits and the
average of the scores, and the uncertainty is the one for the weighted
sample. The weights do not describe a survey design with clustering or
stratification; a grouped `FoldScheme` supplies cluster-robust variance
for dependence among rows. The learners centre their design by the
unweighted mean, so a weighted nuisance fit follows a different boosting
path from the fit on replicated rows. The two agree once the boosting
converges: on a 400-row example the estimates differed by 1e-3 at 100
steps and by 1e-8 at 1000 steps.

`propensity_clip` bounds the propensity and, under `outcome_missing="ipw"`,
the response propensity. When the call does not pass it, the value is the
`propensity_clip` of the `CausalSpec`, which defaults to
`(0.01, 0.99)`, the same default as `effect_model`, so the two
estimators weight rows alike. An argument that is passed always wins. Pass
`None` to disable clipping. Clipping the propensity changes the mean of the score in the rows
it moves: a treated row with propensity 0.005 gets weight 100 instead of
200, so the augmentation no longer removes the outcome-model error in those
rows. Where the propensity is below the bound for many rows, use
`target="overlap"`, which does not divide by the propensity.

`target` says which population the estimate describes. `"all"` (the
default) is the ATE over every row, as described above. `"overlap"` is the
overlap-weighted ATE: the effect averaged over all rows with each row
weighted by `e (1 - e)`, where `e` is the cross-fitted propensity. Its
weights are at most 1/4 and vanish where the propensity is near 0 or 1, so
the rows that have almost no treated (or almost no control) counterparts
contribute little instead of dominating the estimate. It describes a
population whose weights depend on the propensity, not the ATE over all
rows: it is the effect for the covariate region where both arms are likely,
and it equals the ATE when the propensity is 1/2 everywhere. Only
`estimand="ate"` supports it. The propensity clip does not apply to it,
since no propensity divides a residual. The result, its `describe()` and
`diagnostics["target"]` state the target used. The default `"all"` is
unchanged, so existing estimates do not move.

`diagnostics["overlap"]` is a plain dict, the `overlap_report` of the
result after `OverlapReport.to_dict()`. Its keys are:

- `bounds`: the `(lower, upper)` propensity bounds used.
- `n`: the number of rows; `n_below` and `n_above`: rows whose
  propensity is below `lower` or above `upper`; `n_outside` is their sum
  and `share_outside` is `n_outside / n`.
- `regions`: one entry per covariate, giving where most rows lie outside
  the bounds (numeric covariates) or the share outside for each level
  (categorical covariates).
- `estimand`, `estimate`, `standard_error`: the estimand and the
  full-population estimate of this result.
- `n_inside`, `estimate_inside`, `standard_error_inside`: the row count
  and the same estimand restricted to rows whose propensity lies inside the
  bounds.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/causal.py` lines 1889-2856](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L1889-L2856)
