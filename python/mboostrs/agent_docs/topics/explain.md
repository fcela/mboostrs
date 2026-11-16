# Explaining a fit with llm_explain_prompt (step 10)

## Step 10 — Explain: `llm_explain_prompt`

What it does: renders the fitted model as a prompt for a language model —
how the fit was produced, what each retained learner contributes over the
observed covariate range, the risk shares, and which questions the fit
cannot answer. Nothing is sent anywhere: the return value is text for the
caller to pass to a model of their choosing (`llm_describe` posts it if the
analyst wants that).

Give it context — `dataset_description=` (what the data are, where they
came from, what the columns mean) and `response_description=` (what the
response measures, in the analyst's words) — because the fit does not record
this information. Pass the step-4 tuning as `cv=` so the reply can judge the
stopping iteration. `output="turns"` structures the exchange: findings
first, then a hunt for what the first reading misses, then a compression.
The findings reply is the draft of your report; the surprises turn is where
you check yourself against the fit's own numbers; the condensed turn is the
executive summary. Treat the reply as a draft to verify against the numbers
you already have — every effect it names should match a sensitivity row or a
contrast you ran, and any number that does not appear in your evidence did
not come from the fit.
