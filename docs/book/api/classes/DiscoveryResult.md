# `DiscoveryResult`

Defined in `mboostrs.discover`; exported as `mboostrs.DiscoveryResult`.


*Class*

```python
DiscoveryResult(
    model: Any,
    refit: Any,
    formula: str,
    spread_formula: str | None,
    family: str,
    mstop: int,
    folds: int,
    seed: int,
    level: float,
    mean_effects: list[dict],
    interactions: list[dict],
    spread_drivers: list[dict],
    predictions: polars.dataframe.frame.DataFrame | None,
    extrapolation: polars.dataframe.frame.DataFrame | None,
    warnings: list[str],
    log: list[dict],
    provenance: dict,
    _frame: polars.dataframe.frame.DataFrame,
    _variables: frozenset = frozenset(),
    constraints: dict = <factory>,
    _dropped_refit: Any = None,
    _dropped_variables: frozenset = frozenset(),
    interval: Any = None,
    stage_seconds: dict = <factory>
) -> None
```

What `discover` found, with the decisions that produced it.

**Attributes**

- `model` (GAMBoostModel or LSSModel): The final boosted model, fitted on the adopted terms at the tuned
  stopping iteration. When prediction bands are requested, this model
  uses only the selection rows; calibration rows never enter it.
- `refit` (GamRefit, GamlssRefit or None): The same terms refitted by REML, with effect intervals and term tests.
  `None` when the refit was skipped or refused; the log gives the reason.
- `formula` (str): Right-hand side of the adopted mean model (`"1"` for an offset-only
  model).
- `spread_formula` (str or None): Right-hand side of the adopted log-scale model, or `None`.
- `mean_effects` (list of dict): One record per adopted mean term: `term`, `variables`, `shape`,
  `range` and `slope` (from `mboostrs.explore.sensitivity`),
  `transform` (`"log"` when the smooth is on `log(x)`, else
  `None`; `range` and `slope` are on the original scale of `x`),
  and `change`, `change se` (the out-of-fold evidence for keeping it).
- `interactions` (list of dict): One record per adopted interaction: `term`, `variables`,
  `change`, `change se`.
- `spread_drivers` (list of dict): One record per adopted log-scale term: `term`, `variables`,
  `parameter`, `shape`, `direction` (`"increasing"` when the
  log scale rises across the range of the variable), `change`,
  `change se`.
- `predictions` (polars.DataFrame or None): One row per `newdata` row, with columns `row` (its position, from
  0), `mean`, `lower` and `upper`. `mean` is the fitted mean on
  the response scale (the `mu` of a location-scale fit); `lower` and
  `upper` bound it at `level`. A fit with a spread model
  (`spread_formula` is not `None`, the Gaussian location-scale
  model) also carries `sigma`, the fitted scale of that row on the
  response scale, so a score (CRPS, log score) reads `mean` and
  `sigma` directly instead of rebuilding the scale from the interval.
  The column is absent for a constant-scale fit and for a family with no
  distributional counterpart, so `spread_formula is not None` says
  whether it is there. `None` when no `newdata` was given.
- `extrapolation` (polars.DataFrame or None): `extrapolation_report` of the model for `newdata`.
- `warnings` (list of str): Every warning raised while the analysis ran, as text.
- `log` (list of dict): Every decision, with keys `stage`, `candidate`, `score` (the
  out-of-fold risk, or `None`), `change`, `se`, `adopted` and
  `reason`.
- `provenance` (dict): `mboostrs.provenance` of the data and `seed`.
  `partition` records selection/calibration row positions and their
  input namespaces. Positions refer to the original supplied frames,
  before missing-data handling or predictor hygiene.
- `interval` (ConformalInterval or None): The calibration result behind `predictions`, including its marginal
  coverage bound, assumptions and calibration source. `None` when no
  band was requested or budget/error prevented calibration.
- `stage_seconds` (dict): Measured wall-clock seconds of each stage, keyed by stage name
  (`"mean effects"`, `"interactions"`, `"spread"`, `"tune"`,
  `"refit"`, and so on, as in `log`), with `"predict"` for the
  prediction and calibration step. A stage that did not run is absent.
  `mboostrs.estimate_cost("discover", ...)` gives the first-pass fit
  count of each stage before a run; this is what the stages took.
- `constraints` (dict): The rule in force: `heredity` and `forbidden_pairs` (for the mean
  model), and `mandatory_terms` and `candidates` keyed by parameter.
  A `None` candidate list means every predictor was allowed.

Source: [`python/mboostrs/discover.py` lines 312-543](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/discover.py#L312-L543)

## Members

| Member | Kind | Summary |
|---|---|---|
| `contrast` | Method | The average change in the prediction when ``variable`` moves. |
| `to_dict` | Method | The result as JSON-ready data (``json.dumps`` accepts it). |

(discoveryresult-contrast)=
## `DiscoveryResult.contrast`

*Method* of `DiscoveryResult`

```python
DiscoveryResult.contrast(
    variable: str,
    reference,
    alternative,
    parameter: str | None = None
) -> DiscoveryContrast
```

The average change in the prediction when `variable` moves.

The rows are the training rows with every predictor present, and
`variable` is set to `reference` in one copy and to `alternative`
in another. The estimate is the mean of the row-wise difference.

With a Gaussian, Binomial, Poisson or other family the refit is a
REML GAM, which keeps no posterior draws. Its interval is the sum of
the pointwise half-widths of the affected effects at each row, a
conservative bound under the refit's Gaussian coefficient
approximation. It is conditional on the adopted terms and does not
account for the selection that chose them. With a distributional
refit, the interval is the quantiles of the average over
`CONTRAST_DRAWS`
posterior draws of the `parameter` (default `"mu"`).

When an ordinary refit holds unsupported terms such as trees at their
boosted contributions, contrasts use the boosted estimate without an
interval. The refit estimates no uncertainty for the held terms.

A variable whose draft term backward elimination dropped is marked
"not selected" in the note. Its estimate is read from the REML refit of
the model before elimination, and its interval is that refit's interval
widened to include 0. This applies to a GAM refit. When the adopted model
is distributional, or the variable had no draft term, the contrast is
exactly 0 with a note that it was not selected. The scale is `"link"`
for a GAM refit and `"response"` for a distributional one.

Source: [`python/mboostrs/discover.py` lines 420-480](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/discover.py#L420-L480)

(discoveryresult-to_dict)=
## `DiscoveryResult.to_dict`

*Method* of `DiscoveryResult`

```python
DiscoveryResult.to_dict(*, include_arrays: bool = False) -> dict
```

The result as JSON-ready data (`json.dumps` accepts it).

Models and refits are not included; the fitted objects stay on
`model` and `refit`. Records keep the `provenance` entry.
`include_arrays=True` adds `"fitted"`: the final model's mean on
the training rows that have every predictor present, on the response
scale.

Source: [`python/mboostrs/discover.py` lines 503-536](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/discover.py#L503-L536)
