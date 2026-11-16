# Screening columns and drafting a specification (steps 1 and 2)

## Step 1 — Screen the columns: `column_screen`

What it does: for each numeric, boolean, or categorical column, the Rust core
measures how much response variance a straight line (`linear_r2`), a curve
(`curve_r2`, binned means adjusted for df) and the best single step
(`step_r2`) explain marginally, plus Pearson/Spearman correlation, skewness
and distinct counts.

What the output looks like: a frame sorted by `curve_r2`, one row per column,
with `curvature = curve_r2 - linear_r2`, `step_share` (what the step adds
relative to the curvature), and a `hint` — `"none"` (curve explains under
1%), `"levels"`, `"line"`, `"step"` or `"curve"`.

What you decide: which columns enter the draft specification and which shape
to try first. A `"step"` hint wants `btree`; a `"curve"` hint wants `bbs`;
`"line"` wants `bols`; `"levels"` wants a factor treatment. Columns with
`hint="none"` do not go in the first draft, but do not declare them useless:
the screen is marginal, so a column whose effect is masked by a correlated
one scores near zero here and can still earn a place in step 5. Re-read this
table after the first fit, not only before it.

Branches: a wide frame (many columns) still costs little here — the screen
fits no model — so run it on everything. A whole-number column with many
levels is the known trap (an ID, not a measurement); step 2's label logic
uses the response to separate labels from measurements, so defer that call to
step 2 rather than dropping the column now.

## Step 2 — Draft a specification: `classify_columns` / `suggest_terms` / `autospec`

What it does: `classify_columns` reads every predictor and returns one
`ColumnKind` per column — a `kind` (numeric, categorical, datetime,
boolean), the suggested learner `term`, the `reason` for it, its
`alternatives`, and the evidence (level counts, missingness, order).
`suggest_terms` joins the terms into one right-hand-side string. The choices:
numeric gets `bbs` (P-spline) by default, or `btree` / `"menu"` via
`smooth=`; a whole-number column that reads as a count — non-negative
whole numbers from 0 or 1, at least three of them, covering at least half
the range they span or named like a count (`kid5`, `visits`, `age`) — gets
`bols`/`bbs` (#313): one effect that extends to a count the fit never saw,
where `brandom` would have no coefficient for it and `predict()` would
raise; categorical with few levels gets `bols`/`brandom` by level
structure; a many-levelled whole-number column that is not a count is read
as labels when its level means vary more than sampling noise explains
(`label_evidence` is the function that decides; it uses the response, which
is why step 1 could not); datetime gets a smooth over the time axis; a
`brandom` group column is a group term.

What the output looks like: per column, a term and a reason you can quote in
the final report; `suggest_terms` returns something like
`bbs(age, df=4.0) + bols(sex)`.

What you decide: accept the draft, edit it, or delegate it. Edit when the
reasons contradict the analyst's roles (an ID classified as a measurement
because nobody told the classifier it was an ID; a coded 0/1 column that is
really a group). One rule of the library does the rest of the editing for
you: **selection between learners is fair only when they carry equal degrees
of freedom**, and nothing enforces that — `bbs` defaults to df=4 while
`bols` is unpenalized and carries the df its design implies. When the
comparison between terms is the point (and here it always is), write the
`df` on every term. `btree` has no `df`
(`btree(x, df=4)` raises "unknown btree parameter 'df'"); its complexity
knob is `maxdepth`, so compare a tree against a smooth on held-out risk
rather than by matching df.

**Location.** When two columns are coordinates (`x` and `y`, `lat` and `lon`,
`easting` and `northing`), `suggest_terms` fits each one alone and says so in
its reason. When the location effect is patchy (the response changes between
regions with sharp edges, such as flood zones or river bends) or has
boundaries, compare `btree(x, y, maxdepth=3)` with `bspatial(x, y)` on
held-out risk: the tree places cut points on both coordinates, and the surface
smooths across the edges. A default `btree` is a stump, which splits on one
coordinate per boosting step, so the depth is set here. `autospec` with a
positive budget runs this comparison and records it in `rec.report()`.
`insights()` reports when a coordinate tree beats a fitted `bspatial` by more
than its paired fold standard error. To see the regions a fitted joint tree
found, `GAMBoostModel.plot_btree_partition()` draws them as a heatmap and
`btree_partition_table()` lists each leaf with its bounds and contribution;
`btree_splits()` gives the raw cutpoints.

The alternative is `autospec(data, response, ...)` — one call that drafts
the specification *and* measures the choices. `budget=0` (default) costs no
fits and returns the dtype-driven draft plus hygiene findings; a positive
budget buys measured decisions (family probe, shape probe, mstop tuning)
recorded so `rec.report()` distinguishes what was measured from what fell
back to a default. Use `autospec` with a budget when the frame is unfamiliar
and the analyst has not committed to a family; use `classify_columns` +
hand-editing when the analyst has. Watch its family inference: whole-number
responses are read as counts, including floats that happen to be whole; a
rounded continuous measurement needs `family=Gaussian()` declared by you.
With `budget=0` a Poisson recommendation carries `step="line_search"`, not the
fixed step: no fit runs to catch an overflow, and the fixed Poisson step grows
with the mean. Read `rec.evidence["step"]` for the reason.
