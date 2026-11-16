# Interaction incumbent comparison

Instrument interaction-incumbent/1 is frozen before new candidate measurement.
Issue #679's archived production-development seed6607100 exposes an adoption
against a rewritten matched-margin baseline, rather than the actual incumbent:
NBinomial incumbent risk3.4960338900606645, adopted tensor risk3.500004456788099.
The tensor beats its weaker matched reference by the existing2.54 paired SE
threshold, but worsens the model it replaces. No new stochastic observation
is used to identify this source-level decision defect.

## Candidate

Keep the existing matched-margin interaction contrast and its strict negative
change plus INTERACTION_SE_RULE threshold. If that reference differs from the
actual selected mean, also evaluate the actual incumbent on the SAME folds,
family, rate, preprocessing and200-update screening budget in the same CV
comparison. Adopt only when the candidate also strictly lowers mean fold risk
against the actual incumbent. Exact equality retains the incumbent, without
an epsilon. Use the existing shared Rust prefer_simpler_cv(incumbent,candidate,0)
for this additional decision, keeping the candidate only when it does NOT
prefer the incumbent. Log all fold scores, specifications and decisions.
Identical reference/incumbent specifications need no redundant fit.
Budget admission includes every compared specification. Refusal or a failed
comparison retains the incumbent. The subsequent joint-tree comparison still
uses the actual incumbent. No family/gradient/rate/stopping/cap/scale/score
rule changes. No suppression based on known generating structure.

## Development

Reuse count mean-backbone development seeds6607100–6607109, explicitly as
DEVELOPMENT, on all twelve original/additional/interaction/family-hinted panels
with their original generator,1000analysis/1000test,Rust500/500 split and
independent test seed+100000. Baseline is currentmaincedc59c; candidate changes
only the interaction guard. Retain both methods, supplied fixed/tuned and all
five fixed competitors for any final study. Pilot diagnostics may compare the
two procedures alone and are labelled as such. Reuse the ten-panel sharp-mean
Gaussian development battery6293100–6293109 as regression controls, preserving
its original analysis1000/test2000 generator and partition. Do not pool panels
or the family-supplied diagnostic with the ordinary genuine-interaction panel.

## Confirmation

Reserve count6799100–6799149 and Gaussian6799300–6799349, independent test
seed+100000, until the final production candidate and complete analysis are
committed. Freeze any development-informed candidate change in a separate
version/decision before fresh draws. Retain all predictions, actual model
choices/rates/CV curves/shape and interaction fold decisions, actual native
calibration calls, competitor prediction/score vectors, costs, warnings,
failures/empty/unbounded sets, source identities and measured package versions.
Never compute failed-method means only from successes.

## Acceptance

The deterministic invariant requires every newly adopted tensor to improve
both its matched contrast and the actual incumbent, with exact ties retained.
The additive nonlinear count target remains real width AND nonnegative-integer
cardinality one-sided95% upper mean paired excess <=1% against prespecified
LightGBM CQR, nominal coverage.90 and eligibility[.88,.92]. Require the candidate
and properly tuned supplied track to pass on nonlinear zero inflation. Keep
original-rate candidate/fixed supplied and loglinearZIcandidate/tuned supplied
1% targets. Report all five controls and separate Bonferroni bounds.

Require smooth x1 on >=90% nonlinear draws, <=10% selected interaction on each
additive nonlinear count control, and an upper paired latent-mean-MSE excess
<=0 against baseline on nonlinearZI. Other count panels require finite eligible
intervals and upper paired mean-error excess <=10%; null coverage>=.88, with
coverage>.92 explicitly ineligible for efficiency claims. The family-supplied
genuine-interaction diagnostic requires >=90% representation; ordinary genuine
interaction requires no representation regression and mean-error upper<=10%.
Keep the paired exact one-sidedMcNemar.05 null/noise-inclusion check.

Gaussian regression controls require complete outcomes, no significant paired
increase in false scale adoption or mean/noise inclusion on constant controls,
no decrease in genuine scale adoption, and paired mean-error upper<=10%.
This is not a claim to resolve #629: its original false/missed scale cases
remain visible. A failed statistical gate leaves #679 open, even if a
separately verified deterministic adoption defect can be repaired.
