# Job-training causal battery: pre-registered protocol

Written 2026-10-10 before any analyst started. The grader, the evaluation
harness and the key are fixed at this point; the sha256 of this file,
`prepare.py`, `grade.py` and `evaluate.py` are recorded in the report.

## Comparison set

mboostrs 0.1.3, EconML 0.17.0, DoubleML 0.11.4, statsmodels 0.15.0 and
LightGBM 4.7.0. One analyst per library: a Claude Haiku agent given BRIEF.md
verbatim plus a one-paragraph assignment naming the library, its run
directory and its documentation source. No assists after launch. EconML and
DoubleML analysts may use scikit-learn estimators as nuisance learners,
because both libraries take them as inputs; every other analyst uses only the
assigned library for modelling.

The battery is run twice for mboostrs. The first run ("before") measures the
library as it is on 2026-10-10. Defects found in that run are fixed in the
library, and a fresh analyst, given the same brief and assignment text and
no access to the first run's files, runs the battery again ("after").

## Battery

Applicants: the 28,155 men in the March 1988 Current Population Survey
(`AER::CPS1988`, sha256 recorded in `prepare.py`), with prior earnings
derived from the real weekly wage. Simulated pre-enrolment columns: months
unemployed (Poisson, falls with prior earnings and education), distance to a
training centre (log-normal, shorter in metropolitan areas), intake-note
length (pure noise).

Enrolment: logistic in log prior earnings (−1.0 per log unit), education
below 12 (+0.35), √months unemployed (+0.30), log distance (−0.55),
ethnicity (+0.30 for `afam`), south (+0.20), experience (−0.012 per year),
and −3.0 above 40,000 dollars of prior earnings (thin overlap). About 30% enrol.

Outcome, per applicant: certificate (only if enrolled, probability rising
with education); employment, logistic in prior earnings, education, months
unemployed and experience, plus an enrolment effect of
0.15 + 0.85·sigmoid((15,000 − prior earnings)/2,500) on the logit; hours
(lower when part time, +80 if enrolled); log earnings of the employed,
linear in log prior earnings, education, metropolitan, south and log hours,
plus 0.05 + 0.10·[education < 12] + 0.06·clip(30 − age, 0, 15)/10 if
enrolled and 0.05 with a certificate; residual sd 0.32 (0.44 part time).
The follow-up response depends on enrolment, education and prior earnings
(missing at random given those). Null replicates set every enrolment effect
to zero and keep the enrolment model, so the raw enrolled-minus-not
difference stays large and negative.

Truth: per-applicant expected potential outcomes by Monte Carlo with common
random numbers (400 draws per training row, 4,000 per test row).

Replicates: dev (given), H1–H6 (new draws of applicants, enrolment, noise),
N1–N2 (no effect), S1–S2 (1,500 training rows). Test rows: 4,000 new
applicants drawn with probability proportional to 1/√wage (a lower-earning
intake).

Key: roles as in `prepare.py::KEY` (confounders: prior earnings, education,
months unemployed, experience/age, region, metropolitan; affects enrolment
only: distance, ethnicity; outcome only: part time; mediators: certificate,
hours; post-enrolment selection: responded; irrelevant: intake-note length).
Material effect modifiers on the dollar scale: prior earnings, education,
months unemployed, part time. Both channels are active.

## Execution

The grader runs each delivered `run.py` on dev and on all ten hidden
replicates (seed 1), one library at a time, 4 threads, 10-minute limit per
run. A failed or timed-out run counts as no claims and no intervals.

## Measurements

- Effect queries (H, S): per query, 90% interval coverage of the replicate's
  truth, RMSE, mean width. A missing bound is non-covering.
- Null replicates (N): false claims = queries whose interval excludes zero +
  effect modifiers claimed + channels claimed.
- Per-applicant effects (H, S): PEHE (RMSE of `cate` against the true
  expected effect), 90% coverage, and the policy regret of the top-20%
  ranking: (oracle top-20% mean effect − chosen top-20% mean effect) /
  (oracle − random), so 0 is the oracle and 1 is a random choice.
- Roles: share of the 16 columns given the key's role (distance and
  ethnicity count as correct under `instrument` or `treatment_only`);
  whether the adjustment set contains every confounder group and no
  post-enrolment column.
- Effect modifiers: TPR over the four material modifiers; FDR counting
  claims of `caseworker_note_len` or of columns outside the key's material
  and neutral lists.
- Channels and overlap: channels correct; whether `limited_overlap` names
  prior earnings.
- Follow-up session (D5): `refit.fit_ate` on dev, dropping each of the ten
  pre-enrolment columns other than prior earnings in turn (median and p90
  seconds).
- Large n (D6): one `fit_ate` on 200,000 rows resampled from dev with
  jittered prior earnings and distance; pass within 300 s.
- Reproducibility (D10): dev seed 1 run twice by the grader and compared
  with the analyst's `dev_out`.

## Rubric mapping (scorecard.md anchors, operationalised)

D1 (relationship discovery, here causal structure): 1 = no data-driven
claims; 2 = role accuracy < 0.6 or an invalid adjustment set in most
replicates; 3 = valid adjustment set in every non-null replicate and role
accuracy ≥ 0.6; 4 = 3 plus modifier TPR ≥ 0.5 with FDR ≤ 0.25 and null
false claims ≤ 1; 5 = 4 plus both channels correct and overlap named.
D2: judged from REPORT.md against the anchors; 4 needs a stated sensitivity
analysis for unmeasured confounding, 5 also names a design (e.g. the distance
instrument or a randomized rollout) that would test the main claim.
D4: 1 = no intervals; 2 = mean query coverage < 0.70; 3 = 0.70–0.82 or
> 0.98; 4 = 0.82–0.98; 5 = 4 plus the report states dollar magnitudes with
intervals.
D5: median follow-up `fit_ate` > 60 s = 1; 10–60 s = 2; 1–10 s = 3; < 1 s = 4.
D6: 1 = a run failed; 2 = all ran but small-n ATE coverage 0 or PEHE more
than 2× the full-data PEHE; 3 = all ran and small-n degrades within those
limits; 4 = 3 plus large-n pass; 5 = 4 plus a shift or overlap warning for
the test applicants in `warnings`.
D9: 1 = no per-applicant intervals; 2 = per-applicant coverage outside
[0.80, 0.97]; 3 = inside; 4 = 3 plus ATE coverage ≥ 0.8 on S; 5 = 4 plus a
warning naming where intervals are unreliable (thin overlap).
D10: as in the claim-cost protocol: 1 = repeated runs differ; 2 = identical,
no tuning log; 3 = identical with a complete log; 4 = 3 plus the grader's
rerun matches `dev_out`; 5 = 4 plus data/code/environment hashes in the
outputs.
D3, D7, D8: judged from REPORT.md, FRICTION.md and run.py, with evidence
quoted. D3 counts key facts stated correctly (positive effect of about
4,000 dollars; raw difference is negative and misleading; employment effect
concentrated at low prior earnings; do not adjust for certificate, hours or
response; distance affects enrolment only; thin overlap above 40,000 dollars;
dollar effect larger for higher prior earnings and education) and false
statements.

No weighted aggregate is reported without the per-dimension profile.
