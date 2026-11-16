# Friction log: SLID + LSS-simulation benchmark

API traps, errors, missing defaults and confusing return types met while
building `run.py` against the `bench4` site build (`mboostrs 0.1.3`). Each
item is a candidate issue. File:line references are to the repository source in
`python/mboostrs/`.

## mboostrs API

1. **There is no high-level `mb.fit` / `mb.fit() discovery` path.** The brief
   allowed for one, but `mb.fit` does not exist and `mb.report` is a *module*,
   not a method. The documented LSS path had to be assembled by hand:
   `mb.gamboostLSS(formula_dict, data=pl_df, families=mb.GaussianLSS(),
   control=mb.boost_control(mstop=..., nu=0.1), mstop=[mu, sg])`, then
   `m.predict(newdata=..., parameter="mu"/"sigma", type="response")`. Discovery
   is a separate entry point, `mb.discover`. The lack of a single front door is
   the first thing an agent hits.

2. **`mb.discover` returns a `DiscoveryResult`, not a frame.** Predicted values
   are on `result.predictions` (a Polars frame with columns `row`, `mean`,
   `lower`, `upper`); everything else is on `result.to_dict()`. There is no
   fitted `sigma` accessor: the only scale information is the `level` interval
   band, so sigma must be reconstructed as `(upper - lower) / (2 * z_level)`.
   This is a conversion step every caller has to reinvent.

3. **`mb.discover` treats every non-response column as a predictor.** Passing
   the full SLID frame (with `wages` and `rownames`) lets discovery select
   `bbs(log(wages))` - the response's own transform - as a mean effect. The
   caller must pass covariates plus response only. Easy to get wrong, silent
   when wrong.

4. **`mb.discover` did not raise or warn on an intercept-only result.** On SLID
   it returned `formula: 1` ten times with no error; the only signal was a
   boosting-risk warning and a log note (`discover.py:876-877`). See
   `findings.md` #2.

5. **The family probe is off by default in the discovery path.**
   `discover.py:636-637` calls `autospec(frame, response)` with `budget=0`, and
   `_autospec.py:892` turns validation on only when `budget >= 6 * n_folds`.
   The Gamma-vs-Gaussian squared-error guard therefore never runs, even though
   `_autospec.py:1429-1438` documents exactly the Gamma overshoot it protects
   against. See `findings.md` #1.

6. **`cvrisk_gamboostLSS(..., folds=5)` raises `IndexError: tuple index out of
   range`.** Passing an `int` fold count fails; folds must be something else
   (weights from `mb.cv_weights(n, type='kfold', B=5, seed=s)`, or the
   `.cvrisk()` method on a fitted model). The int form is the obvious call.

7. **`with_mstop(k)` predictions disagreed with a direct refit at `mstop=k`.**
   An early probe re-stepped a fitted LSS model with `with_mstop` and got
   predictions that did not match a fresh `gamboostLSS(..., mstop=k)`; all
   reported numbers refit from scratch at the chosen `mstop` instead. If
   `with_mstop` is meant to be equivalent, the discrepancy is a defect.

8. **`build_info()["git_revision"]` is the checkout HEAD, not the build's
   declared SHA.** It reported `65f71c76...` while the build was declared
   `0ae6f81d`. A caller cannot use it to confirm the build identity.

9. **Discovery's `brandom(language)` fold warning is confusing.** The run
   emitted "brandom(language) is scored on folds that split its groups" even
   though no `brandom` term was requested (the draft used `bols(language)`).
   The warning names a learner that is not in the model.

10. **Discover's interval `level` is the only uncertainty knob.** There is no
    `alpha=`; and `predict(parameter=..., type=...)` string arguments have to be
    discovered by reading the signature.

## Competitors / environment

11. **CatBoost writes `catboost_info/` to disk by default.** The first probe
    left a `catboost_info` directory in the study folder; both CatBoost models
    now pass `allow_writing_files=False`.

12. **Patsy `bs()` rejected the test set: "some data points fall outside the
    outermost knots".** The competitor splines needed explicit
    `lower_bound`/`upper_bound` per column. This is a competitor-side trap, not
    mboostrs, but it dominates the two-stage baseline if missed.

13. **EBM's defaults are unusably slow at this size.** `interpret`'s
    `ExplainableBoostingRegressor` defaults (`outer_bags=14`, `interactions='5x'`)
    made a single what-if refit take ~3200 s. The reported runs pin
    `interactions=10, outer_bags=4, max_bins=256`, a documented deviation from
    defaults.

14. **One `catboost_uncertainty` fit recorded 3404 s against a 0.76 s median.**
    The whole run's wall clock was likewise inflated (10340 s). These are
    wall-clock artifacts of the shared machine (suspend/resume), not compute;
    D5 uses medians. Any timing claim from this box needs the caveat.

15. **`interpret` install bumped pandas to 3.0.6.** The study uses Polars
    throughout, so this did not leak into the analysis, but the environment is
    not the one the guidance assumed.
