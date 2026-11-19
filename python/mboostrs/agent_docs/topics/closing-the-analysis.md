# Closing the analysis: the final report

## Closing the analysis

The final report states: the prediction target and fold claim (what the
out-of-fold numbers measure — rows, new groups, or later times); the
specification and family adopted, with the measured evidence for each choice
that changed the model (a `compare_specifications` row, not a preference);
what was tried and did not clear two SE; the sensitivity and contrast
findings with their units and the support they hold over; the caveats from
`reading-a-fit` that apply to this fit; and the single next check to run if one more
were possible. Preserve refusals, failures and ties — a candidate that
could not fit, a comparison that tied, a screen that found nothing: these
are results, and their absence from the record is how the next agent
re-runs the whole path to learn nothing new.

Stop at the user's requested scope or the agreed compute budget. If the task
is model search, compare promising candidates on shared validation folds,
record ties and failed fits, and state which checks remain unresolved. A
two-SE gain is a practical comparison rule, not a calibrated significance
test; correlated training folds and repeated search affect its uncertainty.
A basic prediction request need not run every screen or distributional test.

Include the package version, formula, family, preprocessing, random seed,
fold design, tuning or convergence status, and paths to saved artifacts so
another agent can reproduce the result. Distinguish tuning/validation scores
from the final independent test score. If final test results prompted further
revision, state that these rows became validation data.

Write external issue comments or publish findings only when the user has
requested that action. Preparing an analysis alone does not authorize posting.
