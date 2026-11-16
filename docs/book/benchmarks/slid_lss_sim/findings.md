# Findings: SLID + LSS-simulation benchmark

Study of `mboostrs` against classical and ML competitors on two arms: the real
`carData::SLID` wage data and a simulated location-scale-shift generator with
planted truth (`answer_key.md`). Ten seeds per arm, ten competitor methods plus
two mboostrs tracks (known specification and discovery). Scorecard
`mboostrs-scorecard/2.1`. Numbers come from `results.parquet` via
`evidence.json`; the profile is in `scores.json`. The library was the
`bench4` site build (`mboostrs 0.1.3`, declared `0ae6f81d`; see
`provenance.json`).

Primary loss is the Gaussian CRPS. The best competitor is a retrospective
per-seed best, chosen on the test loss; comparator settings are untuned. The
gap is mboostrs minus the best competitor, in percent, paired across the ten
seeds (negative is better).

## Headline

| arm | track | CRPS gap vs best competitor | RMSE gap |
|---|---|---|---|
| sim | known spec | **-4.56%** (SE 0.16) | -1.02% (0.14) |
| sim | discovery | **-4.44%** (SE 0.16) | -0.99% (0.14) |
| SLID | known spec | +0.58% (SE 0.13) | +0.75% (0.13) |
| SLID | discovery | **+34.07%** (SE 0.47) | +28.76% (0.42) |

On the simulated arm mboostrs beats the best competitor in both tracks. On the
real arm the known-specification model is at par (inside the 1% goal), but the
discovery path collapses.

## Ranked weaknesses

### 1. Discovery returns an empty model on the real arm (D1b, D11, D9)

On SLID, `mb.discover` - given only the data and the response name - returns
the intercept-only model `y ~ 1` on **10 of 10 seeds**. It predicts at
**+34.07% CRPS (SE 0.47)** and **+28.76% RMSE (SE 0.42)** relative to the best
competitor (APLR on 9 seeds, EBM on 1) and undercovers the nominal 80%
interval at **0.716** (gap -8.4 pp). The known-spec model on the same rows is
at par (+0.58% CRPS, SE 0.13), so the data is learnable; the discovery path
loses it.

**Suspected cause.** Three linked code facts.

- `discover.py:636-637` infers the family by calling `autospec(frame, response)`
  and taking element `[2]`. This call uses `autospec`'s default `budget=0`.
- `_autospec.py:892` computes the family probe's validation flag as
  `validate=budget >= 6 * n_folds`. With `budget=0` this is `False`, so the
  probe at `_autospec.py:1474-1611` recommends `GammaReg` (the spread of the
  positive `logwage` response tracks its level) and the squared-error guard
  never runs. That guard - `_validate_gamma` (`_autospec.py:1645`) and
  `_validate_against_gaussian` (`_autospec.py:1655`) - is exactly the path that
  replaces `GammaReg` when it predicts worse. Its own docstring
  (`_autospec.py:1429-1438`) names the failure mode: a boosted `GammaReg`'s
  recalibrated shape makes the fixed step `nu=0.1` overshoot, so its held-out
  risk stops descending.
- The run's warnings confirm the overshoot: *"the training risk rose on 101 of
  200 boosting steps ... the step nu=0.1 is too large for this loss's
  curvature"*. With a non-descending path the leave-one-out `change` estimates
  in `_mean_effects` (`discover.py:833-877`) carry no signal, every term meets
  the removal rule `change <= SE_RULE * change se` (`discover.py:857`), and the
  loop empties `current`, leaving `y ~ 1` (`discover.py:876-877`).

The fix is confirmed by overriding the family: calling
`mb.discover(..., family=mb.Gaussian())` on the same rows recovers
`bbs(education) + bbs(age) + bols(sex) + bbs(age, by=sex)` and test RMSE
**0.4015 / 0.3880 / 0.3972** on seeds 0/1/2, against the auto-family RMSE
**0.5088 / 0.4998 / 0.5097** - matching the known-spec model (0.4022 on seed 0).

**Proposed fix.** Have discovery run the family probe with validation on
(pass a non-trivial budget through `discover.py:637`, or call `autospec` with
the quadratic-error check enabled), so `GammaReg` is replaced by `Gaussian`
when it predicts worse. A smaller `nu`, or `step='line_search'`, for the
discovery family fit would also steady the path. **Effort: S** for the
validation flag; **M** to retune the step and re-validate.

### 2. Discovery removes every term without flagging the empty model (D1b)

The empty model in finding 1 is recorded only as a log note
(`discover.py:876-877`); the `DiscoveryResult.warnings` returned to the caller
do not say "no term survived; this is an intercept-only model". A caller that
does not read the log gets a plausible object and predicts mean wages for
everyone. `silence must not win` (scorecard D1b) is hard to honour when the
silence is not labelled.

**Proposed fix.** When elimination drops the last term, add a warning to
`result.warnings` (and, with a strict option, raise) naming the empty model
and the family that produced it. **Effort: S**.

### 3. Known-spec SLID is at par, not ahead (D11)

The known-specification model reaches **+0.58% CRPS (SE 0.13)** to the
retrospective best (APLR). It meets the 1% goal but does not lead. The location
spec is `bbs(age, df=4) * bols(sex) + bbs(education) + bols(language)` and the
scale spec `bbs(age, df=4) + bols(sex)`. A flexible spline model (APLR) edges
ahead on the test set, most likely through a more flexible scale surface than
the linear-in-age, sex-only `sigma`.

**Proposed fix.** Widen the SLID scale spec (e.g. `bbs(education)` in the scale
or a `bbs(age, df=4)` interaction with sex) and re-check; this is a protocol
choice recorded for a rerun, not a library defect. **Effort: S** (spec change
plus rerun).

### 4. Discovery is about ten times slower per fit than the known path (D5)

Known-spec LSS: **0.279 s** median per fit (sim, 3000x8) and **0.119 s** (SLID,
2007x4), with what-if refits at 0.246 s and 0.100 s. Discovery:
**3.34 s** and **1.84 s** per fit, with what-if refits at 2.49 s and 1.11 s.
The absolute cost is small, but discovery runs several cross-validated screens
and an LSS tuning stage, so it is an order of magnitude heavier. This is
seconds per fit (scorecard anchor 3), not a defect; it is recorded because D5
is measured.

## Results within scope

- **Estimation under a known specification (D1a).** On sim, RMSE(mu) 0.056
  (SE 0.003) and RMSE(log sigma) 0.014 (0.002), both about half the classical
  GAMLSS-style competitor's error (0.123 and 0.043). The varying x3 slopes are
  recovered within sampling noise.
- **Discovery on planted truth (D1b, sim).** Location precision 0.980, recall
  1.000, the x3-by-g interaction found in 10/10 seeds; scale precision and
  recall 1.000; zero false discoveries among the nuisances x6/x7/x8; zero traps
  (x5 in location, x1/x2/x3/g in scale). Gaussian is inferred in 10/10.
- **Calibration of the known-spec intervals (D9).** 95% coverage 0.9481 (sim)
  and 0.9548 (SLID); both within about two standard errors of nominal.

## Caveats

- Comparator settings are defaults, untuned. The best competitor is a
  retrospective best on the test loss, not a prespecified baseline.
- One `catboost_uncertainty` fit recorded 3404 s against a 0.76 s median; that
  is a wall-clock artifact of the shared machine, so D5 uses medians.
- The whole run reported 10340 s wall-clock, inflated by the same machine
  suspension; per-fit medians are the meaningful timings.
- D1a, D1b, D5, D9 and D11 are measured. D2, D3, D4, D6, D7, D8 and D10 are
  `unmeasured`; no aggregate is formed.
- The per-seed `loc_precision`, `loc_fp`, `loc_fn` and `scale_*` columns written
  into `results.parquet` used an early term-name match that missed the
  two-way interaction (the discovered term is normalised `g:x3`, not `x3:g`).
  `run.py` is fixed at the source and `analyze.py` recomputes D1b from the raw
  `loc_sel`/`scale_sel` term strings, so the D1b numbers reported here (and in
  `evidence.json` and the notebook) are the corrected ones; the stale derived
  columns in `results.parquet` are superseded.
