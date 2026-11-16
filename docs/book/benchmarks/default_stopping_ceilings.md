# Discovery stopping ceilings

Ordinary discovery uses different default ceilings for its mean-only and
Gaussian location-scale searches. With `max_mstop=None`, the mean search starts
at 100 updates and can double to 400. An adopted GaussianLSS model starts at
400 updates per parameter and can double to 1,600 per parameter. For its cyclic
two-parameter fit, that ceiling permits 3,200 parameter updates in total.
These are search ceilings; the selected stopping count can be smaller.

The implementation is in `python/mboostrs/discover.py`: `_tune_mean` uses
`DEFAULT_MAX_MSTOP`, while `_tune_lss` uses
`LSS_GRID_MSTOP * 2**MAX_GRID_EXTENSIONS` when no ceiling was supplied. An
explicit `discover(..., max_mstop=400)` caps both searches at 400 updates per
parameter, including 400 for the single-parameter mean fit.

Some earlier benchmark protocols shortened the public default to “400 cap.”
For measurements calling `discover` without `max_mstop`, that describes the
mean-only route and does not cap an adopted LSS route at 400. The source
revisions, calls, recorded families, stopping counts and logs determine what
was measured. The frozen protocols and raw records are retained unchanged;
this note corrects their budget interpretation rather than relabelling their
measurements as explicitly capped LSS fits.

The Gaussian sharp-mean controls can select LSS, so their default ceilings are
400 for the mean-only route and 1,600 per parameter for an adopted LSS route.
The count study's standard Poisson/negative-binomial fits use the mean route.
The original ten-column boundary retest selected Gaussian mean-only models in
all 10 development and 50 confirmation draws, so its recorded discovery
results use the 400-update ceiling and are unaffected by this distinction.
The latter family check uses the complete records in PR #668, not a new sample.

Future comparisons state both ceilings or supply an explicit common ceiling.
Elapsed workflow costs remain descriptive when they include different searches;
a common numerical ceiling alone does not equate their computation costs.
