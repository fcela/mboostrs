# Before the first fit: target, variable roles, missing data

## Before the first fit

Three decisions belong to the analyst, not the library, and every later
number inherits them:

- **The prediction target.** Which column is the response, and what does a
  good prediction mean for it? Reserving evaluation units that reflect
  deployment — whole groups, later time periods, or a held-out sample — is
  the one act of discipline the rest of the path cannot supply. Pass only
  training rows to every step that follows. Do not inspect held-out responses
  through a screen, a marginal plot, or an `autospec` probe.
- **Variable roles.** A group column (subject, district, hospital) is a
  grouping, not a measurement, whatever its dtype says. Record which columns
  could leak the answer (measured after the response, or encoding it). Write
  roles down before the first fit; the classifier in step 2 reads dtypes, and
  a numeric ID column looks exactly like a measurement to it unless you
  intervene.
- **Missing data.** `na_action='omit'` fits on complete cases and every
  residual diagnostic reads the same rows, but cannot predict a row with a
  missing value. `na_action='impute'` keeps every row: a median fill plus a
  missing-value indicator term per incomplete numeric predictor (a
  `"(missing)"` level for a categorical one), replayed when predicting new
  rows. Both work on `gamboost`, `glmboost`, `gamboostLSS` and
  `glmboostLSS`. Decide once, not per call; `agent_guide("missing-data")`
  has the details.

A predictive association does not identify an intervention effect. If the
analyst's question is causal, that is a different contract (see the
`mboostrs-model-search` skill's causal route); this path reports
associations and must say so.
