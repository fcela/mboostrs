# Confirmation 2: the shipped procedure on fresh draws (#679)

This decision is frozen before any confirmation-2 observation is generated.
It follows PROTOCOL.md and CONFIRMATION_RESULT.md in this directory.

## Confirmation status

PR #721 merged two changes: the actual-incumbent interaction guard (an
interaction is adopted only when it also strictly lowers paired CV risk
against the current mean model, exact ties retaining the incumbent) and the
`mean_step` option on `discover`, with `"fixed"` as the default and the line
search as an opt-in. The reserved confirmation (count 6799100-6799149,
Gaussian 6799300-6799349, both spent) evaluated the guard together with a
line-search default for NBinomial mean fits. Its nonlinear gates passed and
its other-panel mean-error gate failed; the recorded diagnosis attributes
that failure to the line search's early stopping, not to the guard
(CONFIRMATION_RESULT.md; issue #679, 2026-10-20 status comment, sections
4-6). The configuration that actually shipped -- guard plus the fixed
default -- has never been run on fresh draws. That is the gap this
confirmation closes.

## Candidate and baseline

The candidate is the production default of the tree that runs the recorder:
the recorder's `candidate` method calls `mb.discover` with no step or guard
override. `guard_only` (the same configuration with `mean_step="fixed"`
pinned explicitly) is retained as a reported control; any systematic
difference between `candidate` and `guard_only` is itself a finding.
`baseline` remains the pre-guard ablation of `baseline_interactions_679.py`
under the fixed step rule. Caps, score/guard rules, the 2.54 paired-SE
matched-margin threshold, the exact stopping tie policy and all learning
rates are unchanged.

## New control: nonlinear_interaction_zero_inflated

The existing genuine-interaction controls (`interaction_zero_inflated`,
`interaction_supplied_family`) generate a linear log mean. The curved family
of the original failure -- log mean `.3 + .8*sin(pi*x1) - .4*x2`, response
scale rate 10, zero probability 0.3 with positive-component Poisson mean
mu/0.7, analysis 1000 / test 1000, Rust 500/500 fitting/calibration split --
has additive, null and noise controls but no genuine-interaction control.
The new panel keeps that family and adds the interaction of the existing
genuine control at the same strength: log mean
`.3 + .8*sin(pi*x1) - .4*x2 + 1.2*x1*x2`. The prespecified supplied form is
`bbs(x1, df=4) + bols(x2) + bols(x1, by=x2, intercept=False)`. A correct
procedure should select an interaction term on this panel; the additive
controls remain the ones where it must not. The generator is committed with
this decision, before measurement, and is exercised on the development
seeds before any confirmation draw. It is added to
`benches/study_count_mean_backbone.py`, where every other count generator
of this study lives and which the production recorder imports; the change
there is purely additive (one panel name, one generating branch, one
supplied-form branch) and no existing generator, panel or recorded output
of the #660 backbone study changes.

## Reserved cohorts

Count seeds 6799200-6799249 and Gaussian seeds 6799400-6799449, independent
test seed + 100000. These ranges are disjoint from every spent cohort:
development 6607100-6607109 and 6293100-6293109, backbone confirmation
6608100-6608149, reserved confirmation 6799100-6799149 and 6799300-6799349.

## Panels

Count: nonlinear_poisson, nonlinear_zero_inflated, null_poisson,
null_zero_inflated, noise_zero_inflated, zero_inflated,
interaction_zero_inflated, interaction_supplied_family and
nonlinear_interaction_zero_inflated. Gaussian: all ten sharp-mean controls
as regression controls. The four remaining backbone panels (low, original,
high, overdispersed) are not rerun: they carry no interaction question for
this candidate, and their spent-confirmation failures were traced to the
line-search configuration that is not under test here.

## Acceptance

The gates of PROTOCOL.md, applied to this candidate against the pre-guard
baseline, with the analysis conventions of
`benches/summarize_count_mean_backbone.py` (paired per-seed percent excess,
one-sided 95% upper bound as mean + t(0.95, n-1) * SE, coverage eligibility
[.88, .92]):

- every adopted tensor with an incumbent comparison strictly improves the
  actual incumbent (deterministic; zero violations);
- nonlinear zero-inflated width and count-size one-sided 95% upper excess
  <= 1% against prespecified LightGBM CQR for the candidate and for the
  tuned supplied track, both coverage-eligible;
- nonlinear zero-inflated paired latent-mean-MSE upper excess <= 0 against
  baseline;
- smooth x1 on >= 90% of draws and selected interaction on <= 10% of draws
  on each additive nonlinear count control;
- other retained count panels: complete finite outcomes and paired
  mean-error upper excess <= 10%; null coverage >= .88;
- family-supplied interaction representation >= 90%; ordinary and curved
  genuine interaction: no representation regression against baseline and
  paired mean-error upper excess <= 10%;
- paired exact one-sided McNemar .05 null/noise-inclusion check;
- Gaussian controls: complete outcomes, no significant paired increase in
  false scale adoption or noise inclusion on constant controls, no decrease
  in genuine scale adoption, paired mean-error upper excess <= 10%.

A failed gate leaves #679 open. Raw records stay outside the repository
(#724); the recorder outputs, `production_confirmation2_summary.json` and
`production_confirmation2_gate.json` are retained with this study's records,
and the aggregated result is committed as CONFIRMATION2_RESULT.md.
