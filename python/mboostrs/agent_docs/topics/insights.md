# Insights: the optional first pass

## Step 0 — optional first pass: `insights()`

`mboostrs.insights(df, response)` runs screen → fit → diagnose → propose in
one call and returns a ranked list of `Finding`s (`.finding`,
`.effect_size`, `.uncertainty`, `.basis`, `.next_action`). `basis="measured"`
means the effect was fitted and scored on held-out folds; `"detected"` means
structure was found in a residual with no model of it fitted. Treat every
finding as a hypothesis for the five-step spine in SKILL.md to confirm or reject, not as a
result to report: `effect_size` is in the fit's loss units (squared response
units for a Gaussian fit), not a response-scale magnitude. Pass `family=`
explicitly for a binary or count response.
