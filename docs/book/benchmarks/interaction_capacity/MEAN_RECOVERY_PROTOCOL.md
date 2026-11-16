# Conditional main-effect recovery confirmation

Instrument `interaction-capacity/mean-recovery/1`, frozen after observing the
complete eight-draw sharp-process tree-only results, before evaluating the
new recovery policy. Related issue: missing weaker main effects after a
strong interaction. The original primary protocol, baseline and tree-only
phase remain immutable. This is an additional confirmation experiment, not
an independent re-evaluation of the original generator outcomes.

Use the exact six confirmation families and settings in PROTOCOL.md, but
replicate indices1000000..1000005 through the existing runner. RNG seeds are
therefore203710170 +10000*family_index +0..5. All rows, noise, families,
folds3, max_mstop800 and default discover stages are otherwise unchanged.
These draws have not been viewed. There are36 before and36 candidate jobs.
The before package includes the validated tree-form and offset fixes.

Reconsider a dropped draft main effect only after an interaction was adopted.
Compare incumbent versus incumbent plus that term on the same folds, family,
200-step screen, learning rate and training rows. Adopt a positive paired
out-of-fold improvement exceeding the existing two-SE mean-effect margin.
Rejection, failure or denied budget leaves the incumbent intact. Iterate once
in the original draft order; never select a threshold or order from test loss.
No new numerical routine in Python. Keep all comparison evidence and logs.

Require complete finite predictions for all36 paired draws, each family's
mean after/before MSE ratio<=1.01, and no increase in null draws with adopted
interactions. Report all per-draw changes and failures, including regressions.
The finite gate is not a formal false-discovery guarantee. Previously used
first-policy controls are descriptive replay only for this second policy.

After the gate, rerun the unchanged interaction-capacity/1 primary protocol as
phase recovered, including all280 jobs and all named tuned competitors.
Select the best complete tree competitor retrospectively and use the same
paired gaps/SEs, D11 scoring rules and failure denominators. Retain all earlier
phases. A remaining gap>1% is reported and filed even when within two SE.
