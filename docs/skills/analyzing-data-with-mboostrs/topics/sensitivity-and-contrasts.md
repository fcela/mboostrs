# Sensitivity and contrasts (step 9)

## Step 9 — Fitted effects and contrasts

**`sensitivity(model, data)`** evaluates each predictor's fitted partial
effect over a grid (others held at their means; categoricals gridded over
their actual levels) and reports the total `range of the effect`, the
`slope` between the 10% and 90% quantiles, and whether the effect ever
`turns` (changes direction). The turn is the judgment call: a monotone
effect can be summarized by one number and reported as that number plus its
range; an effect that turns cannot — report the turn itself (where, and what
the local extremes are), because a single slope across it would average away
exactly the structure worth knowing. An unordered categorical gets NaN slope
and turns — there is no scale to take a slope along, and the effect is
reported level by level, ordered by size. An effect whose range is a few
percent of the response's scale is a flatline: report "no practical
sensitivity", not the range's digits.

For a distributional model (`gamboostLSS()` or `glmboostLSS()`), pass
`parameter=` to measure one distribution parameter's predictors:
`sensitivity(model, data, parameter="sigma")`. Only the columns in that
parameter's terms are gridded, and the effect is the parameter's linear
predictor on its link scale (the log scale for `sigma`). The default is the
family's first parameter (`model.param_names[0]`, `mu` for `GaussianLSS`),
and each row names the parameter it was measured on. Run it once per parameter: a predictor that
moves the mean can have no effect on the spread, and the reverse. `parameter=`
is refused for a `gamboost()`/`glmboost()` model.

**`prediction_contrast(model, data, variable, reference, alternative)`**
standardizes a full-model prediction change over the supplied rows: two
copies of the data, `variable` pinned at each value, average the row-wise
differences. On a non-identity link it transforms the complete predictor
before subtracting — never exponentiate a component and subtract that.
Support matters: values outside the term's training range are refused by
default (`extrapolation="error"`), because the spline extrapolates by
construction there. This is standardization, not causation, and the report
must not drift into intervention language.

**`distributional_contrast(model, data, variable, reference, alternative,
quantity=..., value=...)`** is the LSS analogue: it contrasts a fitted mean,
`quantity="quantile"` at `value=` probability, or
`quantity="exceedance"` at `value=` threshold (strict `P(Y > value)`),
after predicting all response-scale parameters under both settings. A
quantile can change with scale while the mean remains constant. Report an
exceedance shift when it informs a decision: name the threshold's meaning,
the reference and alternative
probabilities, and both the absolute and relative shift ("P(rent > 15 €/m²)
moves from 0.08 under reference to 0.19 under alternative — 2.4×") ; a
0.52 → 0.54 move on a threshold nobody acts on is noise, whatever its
t-statistic would be. Thresholds outside the fitted support are refused for
the same reason as the support check in `prediction_contrast`.
