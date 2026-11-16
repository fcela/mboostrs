# Competitive modelling study: protocol

This study measures the predictive and explanatory work a model can deliver
under a small, reproducible search budget. It is an exploratory gap study,
not an attempt to establish an overall winner. Prediction, model complexity,
explanation accuracy, causal-effect recovery and computational costs remain
separate outcomes. A model does not earn a causal interpretation from an
accurate predictive explanation.

## Tasks and populations

The real-data panels are California housing (random rows and new geographic
cells), hourly bike rentals (random rows and future days), diamonds prices,
protein structure, superconductivity, concrete strength, German credit risk
and breast-cancer diagnosis. These cover spatial and temporal transfer,
categorical inputs, many correlated predictors, nonlinear physical systems,
small-sample classification and probability calibration. The supplied housing
and bike CSVs and named OpenML datasets are identified by byte/data digests.
Bike predictors exclude `casual` and `registered`, which sum to the response,
and row identifiers. Future-day prediction conditions on the day's observed
weather; it is not a weather-forecast experiment. Other response columns in
multi-response datasets are excluded.

House values, bike counts and diamond prices use a prespecified log1p response
for every method; the score measures prediction on that scale. Other regression
responses use their recorded units. Binary tasks use log loss as the primary
score, with Brier score and AUROC as additional outcomes. Scores from different
tasks are never added in their original units.

Three independently drawn outer splits are shared by every method. Temporal
panels use three increasing origins and disjoint future test blocks; spatial
panels hold out complete 0.5-degree latitude/longitude cells. Calibration and
validation follow the outer task's split unit. On random-row panels,
classification splits are stratified. Training plus validation is capped at
4,000 rows, calibration at 1,000 and test at 2,000; the same rows reach every
method. The temporal training cap retains the most recent rows. These caps
define a workstation-sized study, not full-data population performance.

The outer test receives approximately 20% of eligible rows/groups; 20% of the
remaining pool is reserved for calibration and 25% of the rest for validation.
Training statistics, vocabularies, imputation and scaling are learned without
calibration or test rows. Configurations are selected only by the validation
loss. The selected model is refitted on training plus validation, with
calibration remaining separate. Missing numerical values receive the training
median; missing categories receive a reserved token. Methods retain native
categorical input where supported, and the linear baseline uses training-only
one-hot encoding. Unseen-category handling is recorded.

## Models and resource controls

The model comparison includes linear ridge/logistic regression, mboostrs
additive splines, mboostrs splines with a joint CART term, EBM, CatBoost,
LightGBM and XGBoost. Each receives four configurations fixed in the runner
before outcomes are inspected. Early stopping uses the same validation rows;
mboostrs uses its native held-out loss path. EBM's supplied bags mark those
same rows as validation and its final refit retains native bagging and
internal early stopping. The first configuration gives a small-search result;
the best of four gives a tuned result. These are matched candidate-count
ceilings, not claims that the algorithms perform equal numerical work.

A separate mboostrs autospec system track receives 32 and 128 probe-fit units
on the same validation design. Its autonomous search is distinguished from
the model search. The search can change shapes and add interactions or trees;
its chosen family is held to the task's Gaussian/Binomial response semantics.
The fixed mboostrs model searches explicitly use `preprocessing="refit"`,
learning spline recipes within the fitting partition. Autospec currently uses
its default fixed-recipe inner CV, and EBM builds its bins from the supplied
development inputs. Their validation covariates can therefore inform a
recipe, but calibration/test covariates and outcomes cannot. Record this
native workflow difference in the interpretation of the system track.

Jobs run sequentially in fresh processes, with one thread for Rayon, BLAS,
OpenMP and each model. A 240-second process ceiling applies to every job;
timeouts, crashes and fit failures remain in the results. Each search tier has
its own job, so a tuned timeout cannot erase a completed small-search result.
Imports and dataset
loading are outside the timed workflow but inside the process ceiling.
Process peak resident memory includes imports. Workflow time includes input
conversion, fitting, selection, final refitting, calibration and production of
the common explanation workload. Training, explanations and prediction are
also timed separately. Prediction latency uses one warmup and seven identical
batch repeats. Workflow times over different splits are descriptive elapsed
costs, not controlled implementation speed ratios or timing confidence intervals.
Method order is rotated deterministically across task/split blocks. No other
benchmarks or builds run while measurements are collected.

## Predictive quality and uncertainty

For every task and search tier, report mean primary test loss and the spread
across splits, paired excess loss against the best non-mboostrs competitor,
and complete failure counts. A prespecified 5% excess-loss threshold marks
predictive competitiveness; 10% marks a material gap. These are engineering
acceptance thresholds, not statistical significance tests. With three splits,
and overlapping training sets, real-data uncertainty is descriptive; it does
not support claims of equivalence or superiority in a wider population.

Every regressor also receives the same 90% absolute-residual split-conformal
prediction interval, calibrated on reserved rows. Report empirical coverage
and interval width, scaled by the training response standard deviation.
Exchangeability is not assumed for future days or new regions, so their
coverage is an empirical transfer measurement. Classifiers additionally
receive the same logistic probability calibration fitted on calibration rows;
report calibrated and original log loss separately. Calibration cost counts.

## Parsimony and explanations

Record the number of raw inputs used, retained native terms, model storage
bytes, and method-specific structural sizes (spline coefficients, EBM cells,
tree leaves). Different structural units are not placed on one numerical
ranking. Storage includes retained training/preprocessing state and is an
engineering cost, not a count of mathematical parameters. Explain the actual
stored model; a joint tree term does not become an interpretable one-variable
curve merely because it is embedded in an additive model.

Every model produces the same training-background marginal-response curves
and permutation-importance workload. Report its cost. Native term models
also report exact term decomposition; boosted-tree competitors retain their
available native feature-attribution capabilities in the capability table.
Post-hoc curves describe predictions, not identified causal responses. For
correlated inputs, rank and recovery results acknowledge substitute predictors
rather than treating every proxy as an independent discovery.

Controlled effect-recovery panels contain sparse smooth additive effects,
sharp multi-variable boundaries, and correlated proxies with irrelevant
columns. Use five independent training/test draws. Grade predictive loss,
the common curves against the true functions on the same background, and
driver-group recovery. True functions and driver labels are evaluator-only
and never select configurations. Intrinsic transparency and post-hoc
explanation fidelity are reported separately. No subjective point score is
assigned to a library for offering more plotting functions.

## Causal-effect panels

Use seven independent draws for smooth heterogeneous effects, joint threshold
effects, weak overlap, and an omitted-confounder negative control. Compare
mboostrs's public R-learner, a CatBoost T-learner and an honest generalized
random forest, alongside an unadjusted difference-in-means baseline. No method
sees true effects during fitting or tuning. Measure PEHE (root mean squared
conditional-effect error), average-effect error, runtime and clipped propensity
counts where available. Public mboostrs AIPW adds an average-effect estimate
and interval-coverage measurement; unsupported conditional-effect intervals
are marked unavailable.

The identifiable panels explicitly provide consistency, measured confounders,
overlap and no interference. The omitted-confounder panel deliberately breaks
exchangeability and tests how misleading conclusions can become. It cannot
establish that any estimator identifies effects under missing confounders.
Seven replications give screening evidence and Monte Carlo variation, not a
precise coverage guarantee. Real-data predictor rankings receive no causal
truth score because these datasets do not supply identified ground truth.

## Analysis and follow-up rules

Use per-task tables and quality/runtime/complexity Pareto fronts; do not build
an arbitrary weighted aggregate that lets interpretability compensate for a
large prediction loss. Separate representation gaps (the best fixed mboostrs
candidate remains weak) from search gaps (the capacity candidate is good but
autospec misses it), inference gaps, reporting failures and computational
costs. Report every attempted task, including failures. Any ablation prompted
by the results is labelled exploratory and judged on development rows; a fix
requires a separately frozen confirmation study before a general improvement
claim. The deliverable is a reproduced evidence artifact, an annotated report,
and prioritized gaps with measured triggers and acceptance tests.

The protocol follows the model/system distinction and refitting principles in
[TabArena](https://github.com/autogluon/tabarena), uses dependency-aware splits
as described in the [scikit-learn evaluation guide](https://scikit-learn.org/stable/modules/cross_validation.html),
and separates causal estimation from prediction following the
[R-learner paper](https://arxiv.org/abs/1712.04912). EBM's native decomposition
is documented in its [public API](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html);
CatBoost provides [native SHAP values](https://catboost.ai/docs/en/concepts/shap-values).
