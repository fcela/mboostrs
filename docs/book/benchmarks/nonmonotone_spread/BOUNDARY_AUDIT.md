# Discovery calibration boundary

## Correction to the frozen protocol

The original protocol's statement that no calibration response nominates a
column is not true of the existing discovery workflow. Source inspection after
confirmation found that discovery selects on all supplied development rows,
then conformal_interval divides that original training frame into proper-training
and calibration subsets. The rows used later for calibration have therefore
already influenced formula, transforms, interactions, spread and stopping.
Refitting the selected specification on the proper-training subset does not
remove that influence.

This is the existing behavior on baseline main 19e2537 and on the proposed curve
screen, not an intentional protocol arm. Both runs leave the independently
generated test rows untouched. The known-specification estimator and fixed
competitor specifications do not use their calibration responses for model
selection in this study. Discovery's separate empirical outcomes cannot be
promoted to their conditional split-conformal guarantee.

The original protocol remains byte-for-byte preserved. The violation is filed
in #631 with exact source routes and acceptance criteria. The notebook and MR
state it prominently. Near-nominal coverage over 20 independent draws is empirical
standard-process evidence only; it does not repair the selection/calibration
boundary or establish a finite-sample guarantee. The test-based retrospective
competitor choice is a separate evaluator-only selection and is also identified.

## Issue status

The curve nomination repairs the demonstrated zero-slope blindness in #562.
The existing held-out two-SE scale adoption gate, candidate rules and cost gate
remain. Full #562 closure additionally requires the calibration exclusion it
explicitly requested; that acceptance condition remains unmet until #631 is
addressed and verified. This MR therefore references #562 without closing it.

The mean-versus-scale ambiguity in #629 and the remaining monotone/bell interval
width gaps in #630 also remain open. A later calibration fix that excludes selection data must use its
own branch/MR, precommitted procedure and independent confirmation. This study
must remain available as the original adaptive-selection baseline.
