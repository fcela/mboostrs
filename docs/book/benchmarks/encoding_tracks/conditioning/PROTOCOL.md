# Shifted coding: stopping-range diagnostic

Exploratory instrument `encoding-conditioning/1`, frozen before outcomes.
The full encoding baseline shows a known-specification gap of +10.99% (SE 1.33
percentage points) for shifted coding against CatBoost. All eight stopping
optima are at 1,599 of the 1,600-update cap. The supplied design contains the
main effect and a `by=` term; when the `by` values are coded 2 or 3, their
design columns are highly correlated. Recode the same latent rows as 0/1 and
the known-track gap is -15.49% (SE 1.20 percentage points). This diagnostic
tests a longer stopping search, not a default-policy change or independent
confirmation.

Use all eight original known-specification draws for indicator and shifted encoding.
Keep the data, formula, five folds, seed, `nu=0.1` and `score="mse"` unchanged.
Call `tune_mstop(start=100, max_mstop=25600)`. Read its full CV path and report
both its positive minimum within the first 1,600 positions (or shorter explored
path) and its chosen minimum from the longer search. Select each by validation risk;
score outer test only after selection. Keep every draw, warning, risk path,
setting, grid boundary, input/latent/prediction and source/native identity.

The prefix model is a public truncation of tuned.model at its prefix optimum:
the optimum of a prefix cannot be later than the global selected optimum.
Verify exact prefix prediction/MSE replay against the frozen known baseline
on all 16 fits before interpreting differences. No private-state access or
new numerical library routine is needed. Run jobs serially, with one thread and a 240-second timeout per job.

Report paired changes and SE over8draws per encoding, all failures and costs.
The strongest complete recorded competitor is a retrospective reference with
a different search budget. Do not choose a default cap from these known test
outcomes, pool16rows as independent, or upgrade a scorecard grade. A changed
policy needs its own fresh controls and separate MR; the unchanged original
encoding benchmark remains authoritative for its1600budget.
