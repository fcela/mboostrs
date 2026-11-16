# Per-term linear mean forms study for issue #814

## Study objective

On the nonlinear zero-inflated count panel, the discovered mean's
estimation error leaves the count prediction sets too wide: the
count-set excess against the prespecified LightGBM CQR control was
+1.67% (one-sided 95% upper +2.73%) at confirmation 2, against a +1%
efficiency gate. The remedies priced before this study were rejected on
development gates: a rate-scaled mean budget alone (binding-panel
count-size upper +2.13%), a blanket rate-scaled promotion, and a
starved-fit rule (both failed other-panel do-no-harm gates). The next
candidate named in advance is a per-term linear alternative for the
curved term in the count-mean-forms stage, under the rate-scaled budget.
The study tests whether comparing each drafted smooth term with its
linear alternative separately, rather than comparing the whole
specification with the all-linear specification, brings the binding
panel under the +1% gate without harming the other panels.

## Candidate

In the count mean forms stage of `discover()`, each drafted smooth term
that has a linear alternative is tested separately, in draft order. For
each such term the comparison is the current specification against the
specification with only that term replaced by its `bols(...)` linear
alternative, on the same paired folds, with screening fits at the
stage's screening step count (1000 steps for the count families under
the rate-scaled budget), adopting the linear form when its risk is
within two paired fold standard errors of the smooth form (the shipped
prefer-simpler rule, `SE_RULE = 2`). An adoption updates the current
specification before the next term is tested. Terms naming mandatory
predictors are excluded, as in the shipped stage. Candidate
configuration: `discover(count_mean_forms="per_term",
mean_budget="rate_scaled")`; every other option at its production
default.

Routes in the production recorder, pinned explicitly:

- candidate: the candidate configuration above;
- guard_only: the same configuration with `mean_step="fixed"` pinned as
  well (identity control; its rows must equal candidate's on every
  panel);
- baseline: the pre-guard ablation (`baseline_interactions`) with
  `mean_step="fixed"`, `mean_budget="fixed"`, `count_mean_forms="joint"`
  — the configuration confirmation 2 evaluated as its candidate.

## Protocol

Frozen before any development draw of this study was read. Panels:
all thirteen count panels of the production recorder, analysis
1000 / test 1000, the recorder's 500/500 fitting/calibration partition,
independent test seed +100000; and all ten sharp-mean Gaussian controls,
analysis 1000 / test 2000, same test offset. Fresh development seeds,
disjoint from every range used by the earlier rounds: count
6607200-6607209 and Gaussian 6293200-6293209, ten draws per panel.
Analysis conventions of the production summarizer: paired per-seed
percent excess, one-sided 95% upper bound as mean + t(0.95, n-1) * SE,
coverage eligibility [0.88, 0.92].

Development gates (the same gates that rejected the earlier remedies):

1. Deterministic incumbent: every adopted tensor with an incumbent
   comparison strictly improves the actual incumbent; zero violations.
2. Binding panel (`nonlinear_zero_inflated`) efficiency: candidate
   coverage eligible, LightGBM CQR control eligible, and both width and
   count-size one-sided 95% upper excess at or below +1% against
   LightGBM CQR.
3. Binding panel mean error: paired latent-mean-MSE upper 95% at or
   below 0% against baseline.
4. Supplied tuned efficiency: the tuned supplied track passes the same
   +1% width and count-size bounds with eligible coverage.
5. Additive representation: a curve term (`bbs(x1)`) on at least 9 of
   10 draws and a selected interaction on at most 1 of 10 draws, on
   each of `nonlinear_poisson` and `nonlinear_zero_inflated`.
6. Other count panels (`low`, `original`, `high`, `overdispersed`,
   `zero_inflated`, `null_poisson`, `null_zero_inflated`,
   `noise_zero_inflated`): complete finite candidate outcomes,
   null-panel coverage at least 0.88, and paired mean-error upper 95%
   at or below +5% against baseline (the do-no-harm bound applied to
   the earlier remedies; the summarizer's +10% reading is also
   reported).
7. Genuine interaction: candidate interaction selections at least the
   baseline's on `interaction_zero_inflated` and on
   `nonlinear_interaction_zero_inflated`, with paired mean-error upper
   95% at or below +10%; family-supplied interaction representation at
   least 9 of 10 draws.
8. False inclusion: paired exact one-sided McNemar at 0.05 on
   `null_poisson`, `null_zero_inflated` and `noise_zero_inflated` — no
   significant added inclusion against baseline.
9. Gaussian controls: complete outcomes; paired mean-error upper 95%
   at or below +10% on every panel; no significant paired increase in
   false scale adoption on the constant-scale panels; no decrease in
   genuine scale adoption on the varying-scale panels; noise inclusion
   unchanged by the same McNemar check.

Extension rule (as designed in the earlier records): if every gate
except gate 2 passes and gate 2's count-size upper bound at n = 10 is
the sole failure, the 50-draw development extension on the binding
panel (count seeds 6607200-6607249; the battery range is its first ten
draws) is run and the verdict follows that reading of gate 2. If any
other gate fails, the extension is not run and no efficiency reading
can change the verdict.

Confirmation and ship rule: confirmation is drawn only if every
development gate passes (with the extension rule applied to gate 2),
on the reserved ranges count 6799500-6799549 and Gaussian
6799600-6799649, with the headline requirement that the binding panel's
count-size one-sided 95% upper excess is at or below +1% and the other
panels within the bounds above. Library code ships only if
confirmation passes in full. If development fails, the reserved ranges
are not touched at all, and the delivery is record-only: no default
changes, and this record states the verdict against the gates with
every panel result.

## Development results

All 1500 development records were read back and every metric below was
recomputed from the retained predictions by the production summarizer
(1496 ok, 4 failed; the failures are two `supplied_fixed` draws on each
of `interaction_zero_inflated` and `interaction_supplied_family` — the
candidate's outcomes are complete on every panel). The identity control
holds: candidate and guard_only rows are identical on every panel, on
every reported metric.

Binding panel, `nonlinear_zero_inflated` (10 draws):

| route | coverage | mean MSE | count size | width |
|---|---|---|---|---|
| baseline | 0.8882 | 16.6325 | 30.4882 | 29.9866 |
| candidate (= guard_only) | 0.8995 | 9.9215 | 30.0866 | 29.5846 |
| supplied_tuned | 0.8938 | 1.6619 | 28.7821 | 28.2969 |

Against LightGBM CQR (coverage eligible), candidate paired excess:
width mean -5.06%, upper 95% +0.64%; count-size mean +0.37%, upper 95%
+5.31%. The baseline's own count-size upper bound on the same draws is
+4.87%: the candidate lowers the mean error substantially (paired
mean-error excess mean -41.78%) but its count-size upper bound is
slightly worse than the baseline's, because the per-draw excess varies
more. Candidate curve terms on 9 of 10 draws; interactions on 0 of 10
(baseline: 10 and 5).

Per-panel candidate results, count (paired mean-error excess against
baseline is the mean of the per-seed percent excesses):

| panel | coverage | candidate MSE | baseline MSE | paired excess mean | upper 95% |
|---|---|---|---|---|---|
| low | 0.8963 | 0.0104 | 0.0094 | +21.14% | +45.09% |
| original | 0.9062 | 0.0384 | 0.0346 | +26.31% | +71.26% |
| high | 0.9055 | 0.1120 | 0.1120 | +0.00% | +0.00% |
| overdispersed | 0.8948 | 0.3841 | 0.3841 | -0.00% | +0.00% |
| zero_inflated | 0.8969 | 2.0049 | 2.3459 | +100.29% | +353.61% |
| nonlinear_poisson | 0.9038 | 0.3714 | 0.4517 | -19.01% | -6.52% |
| nonlinear_zero_inflated | 0.8995 | 9.9215 | 16.6325 | -41.78% | +4.58% |
| null_poisson | 0.9058 | 0.0372 | 0.0372 | +0.00% | +0.00% |
| null_zero_inflated | 0.9059 | 0.2407 | 0.2407 | +0.00% | +0.00% |
| noise_zero_inflated | 0.8977 | 3.3497 | 3.7915 | -21.37% | +7.72% |
| interaction_zero_inflated | 0.8907 | 17.5860 | 17.5860 | +0.00% | +0.00% |
| interaction_supplied_family | 0.8915 | 1.9796 | 1.9214 | +2.13% | +4.83% |
| nonlinear_interaction_zero_inflated | 0.8981 | 48.1635 | 50.2494 | -7.56% | +6.30% |

The large positive paired uppers on `low`, `original` and
`zero_inflated` come with small absolute mean-error differences (for
example candidate 0.0104 against baseline 0.0094 on `low`); the paired
percent reading is what the frozen gate applies, and it is the reading
reported here.

Gaussian controls: the candidate is a count-stage change and its rows
are identical to the baseline's on all ten panels; every paired
mean-error upper 95% is 0.00%, and the scale-adoption and
noise-inclusion McNemar checks show no changes (added 0, removed 0,
p = 1.0 throughout).

## Gate verdicts

1. Deterministic incumbent: **pass** — 56 adopted tensors with an
   incumbent comparison, 0 violations.
2. Binding panel efficiency: **fail** — coverage eligible for both the
   candidate (0.8995) and the control; width upper 95% +0.64% is within
   +1%; count-size upper 95% +5.31% is above the +1% gate.
3. Binding panel mean error: **fail** — paired upper 95% +4.58% is
   above the 0% bound, although the paired mean is -41.78% and the
   candidate's mean MSE (9.9215) is well below the baseline's
   (16.6325). The bound is an upper confidence bound, and ten draws do
   not separate the improvement from zero at that level.
4. Supplied tuned efficiency: **pass** — width upper 95% -6.84%,
   count-size upper 95% -2.50%, coverage eligible.
5. Additive representation: **pass** — curve terms 10 of 10 on
   `nonlinear_poisson` and 9 of 10 on `nonlinear_zero_inflated`;
   interactions 0 of 10 on both.
6. Other count panels: **fail** at the frozen +5% do-no-harm bound —
   `low` upper 95% +45.09%, `original` +71.26%, `zero_inflated`
   +353.61%, `noise_zero_inflated` +7.72% (the last passes the
   summarizer's +10% reading; the other three fail at +10% as well).
   `high`, `overdispersed`, `null_poisson` and `null_zero_inflated`
   pass at +5%. Candidate outcomes are complete and finite on every
   panel, and null-panel coverage is 0.9058 and 0.9059, both at least
   0.88.
7. Genuine interaction: **pass** — candidate interaction selections
   equal the baseline's (9 of 10) on both genuine-interaction panels,
   with paired mean-error upper 95% of 0.00% and +6.30% against the
   +10% bound; family-supplied interaction representation 10 of 10.
8. False inclusion: **pass** — no added inclusions on any of the three
   panels (added 0, removed 0, one-sided p = 1.0).
9. Gaussian controls: **pass** — complete outcomes; every paired
   mean-error upper 95% is 0.00%; no scale-adoption or noise-inclusion
   changes.

## Extension, confirmation, and conclusion

The extension rule requires gate 2 to be the sole failure. Gates 3 and
6 also fail, so the 50-draw development extension was **not run**, and
no efficiency reading at larger n can change this verdict. Development
failed, so the confirmation ranges (count 6799500-6799549, Gaussian
6799600-6799649) were **not drawn** and remain reserved and untouched.

The per-term linear alternative does not pass the development gates.
It improves the binding panel's mean error on average and keeps the
representation, interaction, false-inclusion and Gaussian gates, but
its count-size efficiency bound at n = 10 (+5.31%) does not reach the
+1% gate, its binding-panel mean-error upper bound (+4.58%) does not
reach the 0% gate, and it breaches the +5% do-no-harm bound on three
other count panels. Under the frozen protocol the delivery is
record-only: the candidate is not shipped, no default changes, and the
issue's acceptance — the +1% bound confirmed on the reserved ranges —
is not met.
