# `check_declarations`

Defined in `mboostrs.declarations`; exported as `mboostrs.check_declarations`.


*Function*

```python
check_declarations(model_or_recommendation, data, *, folds=5, seed=0)
```

Test each declaration of a specification against the data.

**Parameters**

- `model_or_recommendation` (SpecRecommendation or GAMBoostModel): The specification. A recommendation from `autospec` carries its
  declarations; a fitted `gamboost` model carries the `bmono` terms of
  its formula.
- `data` (dict, Polars frame or pandas frame): The rows the specification was built on. Rows with a missing response
  or a missing formula column are left out.
- `folds` (int, default 5): Held-out folds for the out-of-fold measurements. Rows are assigned to
  folds at random, or by group when the recommendation declares
  `groups=`, so that whole groups are held out.
- `seed` (int, default 0): Seed for the fold assignment.

**Returns**

list of DeclarationCheck
    One record per declaration, in the order the specification lists them.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

A sign declaration (`positive`, `negative`) is tested on the contrast of
the variable between its 10th and 90th percentile, computed on the link scale.
The full-data contrast is compared with its 95% percentile interval over 200
row-bootstrap refits of the unconstrained model (rows resampled with
replacement, each refit evaluated on the original rows). The verdict is
`agrees` when the interval excludes zero in the declared direction and at
least 80% of the five held-out folds (four of five) share the sign of the
full-data contrast; `contradicts` when the interval excludes zero in the
opposite direction with the same fold agreement; otherwise `inconclusive`.

On 30 simulated samples of 300 rows with no association, `positive` gave a
decided verdict (agrees or contradicts) in 3 samples and was inconclusive in
27. The bootstrap costs 200 refits per sign declaration. The earlier rule,
which used the Nadeau-Bengio corrected standard error of the fold contrasts,
decided 8 of the 30 null samples and was replaced.

A shape declaration (`increasing`, `decreasing`, `convex`,
`concave`) is tested on the unconstrained curve over a grid spanning the
1st to the 99th percentile, and on the held-out risk of the model with and
without the constraint. The verdict is `agrees` when at least 90% of the
steps go the declared way and removing the constraint does not lower
held-out risk by more than 1%; `contradicts` when at most 10% go the
declared way and removing the constraint lowers held-out risk by more than
1%; otherwise `inconclusive`.

Held-out rows whose other covariates lie outside the training range of a
fold are clipped to that range, as `predict(extrapolation="clip")` does.

The measurements are associations in the data. They do not estimate what
would happen under an intervention on the variable.

Source: [`python/mboostrs/declarations.py` lines 158-218](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/declarations.py#L158-L218)
