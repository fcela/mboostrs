# Pitfalls

## Pitfalls

- **The tensor df floor.** A tensor (surface) penalty leaves constants and
  straight lines unpenalized, so each margin has a two-dimensional null
  space and their product a four-dimensional one. The minimum tensor degree
  of freedom is `TENSOR_MINIMUM_DF = 4.0`; at df=4 a tensor term is at
  maximum shrinkage and can only produce a bilinear
  surface `a + bx + cy + dxy`, whatever the data look like — measured on a
  sin·cos surface it explained R² ≈ 0.0001. The default
  `TENSOR_MARGINAL_DF = 2.5` (df = 6.25) sits above the floor deliberately.
  Never hand a tensor term df ≤ 4; if unbiased selection against df=4 main
  effects matters more than curvature, use `kind="tree"` or a varying
  coefficient (`bbs(x, by=g)`) instead.
- **Screening on the wrong residual scale.** Pass response-scale residuals
  (`y - predict(type="response")`) to `screen_interactions`. For a
  non-identity link, subtracting the link-scale predictor from the response
  can make pairs of signal columns appear to interact.
- **Group folds.** Choose the target before the split. Use
  `grouped_folds(...)` to measure prediction for new groups and
  `known_group_folds(...)` for new observations in groups already seen.
  Random row folds can be optimistic when the claim is about new groups;
  holding out whole groups answers a different question from predicting in
  known groups. Apply the chosen scheme to every fold-based comparison and
  state that claim in the report. For forecasting use `rolling_folds(...)`;
  merely blocking rows does not ensure training precedes evaluation.
- **Unseen levels.** What a prediction does with a group level the fit never
  saw is a policy, not a detail: `unseen_levels="error"` refuses;
  `"average"` predicts the row as a member of an average group (for
  `brandom`, the mean level coefficient); `"zero_effect"` gives the level
  the ridge-shrunk zero — the prediction `cvrisk` makes for a group its
  fold holds out. A column whose
  levels differ from the fitted ones only in dtype (fitted as strings
  `'1'`, `'2'`, `'3'`, arriving as Int64 1, 2, 3) is refused under every
  policy: those are the fitted levels in another dtype, not new groups,
  and `"average"` would silently predict every row at the average group;
  cast the column to the fitted dtype before predicting. Declare the
  policy when tuning or comparing on grouped folds; `"error"` on data whose
  folds guarantee unseen groups will make every held-out score fail loudly,
  which is the point of it.
- **Change-SE calibration.** `compare_specifications`' `change se`
  understates how much a change would vary across datasets because the fold
  training sets overlap. In IID simulation at about 200 rows, it was about
  15% too small and the two-SE rule selected pure-noise changes about 6% of
  the time, compared with 4.6% for a standard normal statistic. This check
  does not cover grouped or rolling folds. Treat candidates within two SE as
  tied and resolve the tie by simplicity or interpretability.
- **Unequal degrees of freedom.** Comparing `bbs` at df=4 with unpenalized
  `bols` gives the spline greater flexibility. Use `term_report` to compare
  shapes on shared held-out folds.
