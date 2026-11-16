# Ranking and scoring candidate specifications (steps 6 and 7)

## Step 6 — Ranked changes: `suggest_specifications`

What it does: turns a `term_report` (plus the step 5 screen output via
`screend=`) into ranked candidate specifications. Removals are suggested for
terms that earn nothing or less than `noise` (default 2) SE above zero;
shape changes for gains beyond noise; interactions for screen-positive
pairs, typed by the columns (`pair_term_for`: a tensor surface for two
continuous columns, a smooth switched by a two-valued column, a random slope
for a group column — pass `data=` so a string column is typed as labels
rather than gridded as numbers).

What the output looks like: `Suggestion` rows — `action` (`drop`, `refit`,
`add`, `model`), the term, the `reason` in words, the ready-made
`specification` string, and `basis`: `"measured"` (refit evidence from step 5)
or `"detected"` (structure evidence — a screen hit, a spread pattern — that
no particular term has been measured to deliver). Measured sorts first, on
purpose: a detected signal is weaker evidence whatever its magnitude.

What you decide: which candidates are worth one comparison each. Two
warnings travel with the list and you must carry them forward: gains are
measured one change at a time and do not add up; and `detected` is a
hypothesis, not a result. Suggestions to remove several idle terms are
offered together as one candidate because correlated terms can each look
replaceable and still be needed jointly — score that as one change, never as
n separate drops. A group term that step 5 could not measure is never
suggested for removal here.

## Step 7 — Score the candidates: `compare_specifications`

What it does: the loop's closing step. Scores several right-hand sides on one
shared fold partition, so the numbers are comparable to each other and to
the specification they came from. `term_report` said what is worth changing;
`suggest_specifications` ordered the candidates; neither fitted the changed
model. This does.

What the output looks like: one row per specification — the `rhs` itself,
its out-of-fold `risk`, the `change` from the reference (the first entry)
with `change se`, the paired fold standard error of that change. With
`include_evaluations=True`, rows carry their fold evaluations and can be
passed back via `history=` so a later comparison reports the whole search
(`search optimism` — how much the winning score is flattered by being the
best of the compared set — and `search candidates` / `search fits` counts).

The adoption rule: a candidate that lowers risk by more than **two `change
se`** is better on these folds; inside two SE the candidates are tied and you
keep the simpler or more interpretable one — say "tied" in the report, do not
pick the numerically first. Know what the SE understates: fold training sets
overlap, so even one partition's `change se` runs small — calibration
measured by nested-CV simulation says read it as about 15% too small at
n≈200, with the two-SE rule flagging pure-noise candidates 6.3% of the time
(grouped and rolling schemes are not covered by that check). Two SE is a screen for
"worth believing", not a proof.

Branches: `mstop="cv"` tunes each candidate's step count inside every outer
fold — the right choice when candidates differ in how fast they learn, and
the cost is that rows then compare *tuning procedures*. A fixed `mstop`
compares specifications at one depth. `fit_budget=` refuses before any work
when the comparison would exceed a fit budget you name. For grouped data,
pass the grouped scheme and decide `unseen_levels=` (see the pitfalls topic). With a
distributional family, a candidate may be a mapping — `{"mu": "...",
"sigma": "..."}` — so constant-scale and varying-scale models compare
directly under the family's joint proper score.

Adopt at most a few changes per round (they interact), refit, and loop back
to step 5: re-screen the new residuals, re-report terms. Stop when no
screen-positive lead is left uncompared and no suggestion clears two SE —
not when the list happens to be empty once.
