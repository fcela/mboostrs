"""Plotting functions for mboostrs models.

Requires Altair. Install with: pip install altair vl-convert-python

Typography: the package bundles DM Sans (SIL OFL 1.1, Google) for
sans-serif and CommitMono (SIL OFL 1.1, Eigil Nikolajsen) for monospace;
see ``mboostrs/fonts/OFL-LICENSE.txt``.  For static export (``chart.save`` to
PNG/SVG via vl-convert) the bundled fonts are registered automatically, so
saved figures look identical everywhere.  Interactive renderers (notebooks,
browsers) use the same family if installed on the viewing system and otherwise
fall back through the CSS font stack.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

try:
    import altair as alt
except ImportError:
    raise ImportError(
        "mboostrs.plot needs Altair, which is not a base dependency: "
        "importing it cost 509 ms of a 559 ms `import mboostrs`, which every "
        "caller paid whether or not they drew a chart. Install the extra with "
        "`pip install 'mboostrs[plot]'` (Altair plus vl-convert-python, which "
        "registers the SVG renderer and the bundled fonts). Nothing else in "
        "the package needs it -- fitting, prediction, cross-validation and "
        "every table still work without it."
    ) from None

alt.data_transformers.enable("default", max_rows=None)


def use_svg_renderer() -> bool:
    """Make Altair render every chart in this process as SVG.

    SVG text is styled by CSS at paint time, so a page that provides DM Sans
    or Commit Mono via `@font-face` -- the documentation book does -- re-draws
    correctly when the webfont arrives, where canvas paints once with whatever
    font was loaded and never updates. It also makes
    `chart._repr_mimebundle_()` emit `image/svg+xml` instead of the vega-embed
    JavaScript, which is what lets MyST embed the chart inline rather than
    stripping it.

    **This is a process-wide setting**, which is why it is a function and not
    something importing this module does. `alt.renderers.enable` changes how
    *every* Altair chart in the process renders, including ones from libraries
    that have nothing to do with mboostrs; a library import that silently
    reconfigures a shared dependency is a surprise its caller cannot see. The
    book calls this once; a notebook that wants it calls it too, and
    `alt.renderers.enable("default")` puts it back.

    Returns whether the renderer was actually enabled -- `False` when no SVG
    renderer is registered, which happens without vl-convert. Charts then
    render as vega-embed HTML, which is the default anyway.
    """
    try:
        alt.renderers.enable("svg")
    except (ValueError, ImportError):
        return False
    return True


#: Set `MBOOSTRS_SVG_RENDERER=1` to have importing this module call
#: :func:`use_svg_renderer`. It exists for a build that executes notebooks and
#: wants SVG output from all of them -- the documentation book sets it in
#: `build-html.sh`, next to the two Polars formatting variables already there,
#: and no notebook has to say anything. Reading it here is still a
#: process-wide effect, but one the operator asked for by name, which is the
#: difference from doing it unconditionally on import.
if os.environ.get("MBOOSTRS_SVG_RENDERER", "").strip() not in ("", "0", "false", "False"):
    use_svg_renderer()

# ---------------------------------------------------------------------------
# Fonts + Altair theme — applied once on module import.
# ---------------------------------------------------------------------------

_FONT_DIR = Path(__file__).parent / "fonts"

# CSS-style stacks: vl-convert picks the first registered family; browser
# renderers fall through to a system face if DM Sans / CommitMono is not installed.
FONT_SANS = "DM Sans, Helvetica Neue, Helvetica, Arial, sans-serif"
FONT_MONO = "CommitMono, SF Mono, Menlo, Consolas, monospace"

# The categorical order used by the package theme. A tuple keeps the shared
# definition immutable; pass ``list(PALETTE)`` to an explicit Altair scale.
PALETTE = (
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ac",
)


def _register_fonts() -> None:
    """Register the bundled fonts with vl-convert.

    Best-effort: interactive renderers don't go through vl-convert, and a
    missing fonts directory (e.g. a stripped-down install) must not break
    plotting — everything falls back to the CSS stacks above.
    """
    if not _FONT_DIR.is_dir():
        return
    try:
        import vl_convert as vlc

        vlc.register_font_directory(str(_FONT_DIR))
    except (ImportError, OSError):
        # vl-convert is optional, and the directory may be unreadable in a
        # packaged install. Charts fall back to the system font stack.
        pass


_register_fonts()


def theme() -> dict:
    """Return a fresh Altair theme configuration used by mboostrs charts.

    The function can be registered as a project-specific Altair theme factory;
    its returned mapping can also be inspected or adapted when matching chart
    typography, axes, legends, and categorical colours. Use :data:`PALETTE`
    directly for a one-off categorical scale.
    """
    return {
        "config": {
            "font": FONT_SANS,
            "title": {
                "font": FONT_SANS,
                "fontSize": 14,
                "fontWeight": 700,  # DM Sans Bold
                "color": "#161616",
                "subtitleFont": FONT_SANS,
                "subtitleFontSize": 11,
                "subtitleColor": "#525252",
                "anchor": "start",
                "offset": 10,
            },
            "axisX": {
                "titleFont": FONT_SANS,
                "titleFontSize": 12,
                "titleFontWeight": 400,  # DM Sans Regular
                "titleColor": "#393939",
                "labelFont": FONT_MONO,  # tabular tick numerals
                "labelFontSize": 10,
                "labelColor": "#525252",
                "titlePadding": 8,
                # A hint, not a hard count: Vega-Lite still chooses round
                # values. Keeps a 0-300 iteration axis at ~8 labels instead of
                # the 16 it defaults to at this width.
                "tickCount": 8,
                "grid": True,
                "gridColor": "#e8e8e8",
                "gridOpacity": 1.0,
                "ticks": True,
                "tickSize": 4,
                "tickColor": "#c6c6c6",
                "domain": True,
                "domainWidth": 1,
                "domainColor": "#c6c6c6",
            },
            "axisY": {
                "titleFont": FONT_SANS,
                "titleFontSize": 12,
                "titleFontWeight": 500,
                "titleColor": "#393939",
                "labelFont": FONT_MONO,
                "labelFontSize": 10,
                "labelColor": "#525252",
                "titlePadding": 8,
                "tickCount": 6,
                "grid": True,
                "gridColor": "#e8e8e8",
                "gridOpacity": 1.0,
                "ticks": True,
                "tickSize": 4,
                "tickColor": "#c6c6c6",
                "domain": True,
                "domainWidth": 1,
                "domainColor": "#c6c6c6",
            },
            "legend": {
                "labelFont": FONT_SANS,
                "labelFontSize": 10,
                "labelColor": "#393939",
                "titleFont": FONT_SANS,
                "titleFontSize": 11,
                "titleFontWeight": 700,
                "titleColor": "#161616",
                "symbolSize": 120,
                "symbolStrokeWidth": 2,
                "padding": 5,
            },
            "bar": {
                # Square corners, mitred joins. Rounded bar ends read as
                # decoration and misstate where a value actually falls.
                "cornerRadius": 0,
                "strokeJoin": "miter",
                "continuousBandSize": 18,
                "discreteBandSize": {"band": 0.62},
            },
            "line": {"strokeJoin": "miter", "strokeCap": "butt"},
            "rule": {"strokeCap": "butt"},
            "text": {"font": FONT_SANS, "fontSize": 10},
            "header": {
                "labelFont": FONT_SANS,
                "titleFont": FONT_SANS,
                "titleFontWeight": 700,
            },
            "range": {"category": list(PALETTE)},
            "view": {"strokeWidth": 0},
        }
    }


def _mboost_theme():
    """Backward-compatible private alias for the registered theme factory."""
    return theme()


# Altair >= 5.5 uses alt.theme.register; keep the legacy registry as a
# fallback so the module imports on older 5.x installs.
try:
    alt.theme.register("mboostrs", enable=True)(_mboost_theme)
except AttributeError:
    alt.themes.register("mboostrs", _mboost_theme)
    alt.themes.enable("mboostrs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WIDTH = 600
HEIGHT = 300


RUG_MAX_TICKS = 220


def _rug(x_values, panel_height):
    """A density rug pinned to the bottom of a panel.

    Thinned above `RUG_MAX_TICKS`. At a few hundred observations over a narrow
    range the ticks abut and the rug reads as one grey band, which shows the
    extent of the data but not its density. Thinning takes every k-th value of
    the *sorted* input, so the retained ticks are evenly spaced in quantile and
    the marginal distribution is preserved while individual ticks stay legible.
    """
    values = np.asarray(x_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size > RUG_MAX_TICKS:
        order = np.sort(values)
        step = values.size / RUG_MAX_TICKS
        idx = (np.arange(RUG_MAX_TICKS) * step).astype(int)
        values = order[np.clip(idx, 0, values.size - 1)]
    return alt.Chart(pl.DataFrame({"x": values})).mark_tick(
        color="#a8a8a8", thickness=1, size=7, opacity=0.85,
    ).encode(
        x=alt.X("x:Q"),
        # Pixel position, so the ticks do not travel through the shared y scale.
        y=alt.value(panel_height - 3),
    )


#: A stretch of the x range holding fewer than this share of the rows it
#: would hold under a uniform spread is shaded as sparse. A tenth is loose
#: enough that an ordinary bell-shaped covariate shades only its tails.
SPARSE_DENSITY_SHARE = 0.10

#: Cells the x range is divided into when looking for sparse stretches.
SPARSE_CELLS = 24


def _sparse_bands(x_values, line_x):
    """Stretches of the plotted range where the data is thin.

    A fitted curve is drawn at the same weight everywhere, so a stretch
    supported by three observations looks exactly like one supported by three
    hundred. The rug shows the density if the reader counts ticks; this shades
    the places worth counting.

    Returns a frame of ``(start, end)`` bands, empty when the data is spread
    evenly enough that nothing stands out.
    """
    values = np.asarray(x_values, dtype=float)
    values = values[np.isfinite(values)]
    grid = np.asarray(line_x, dtype=float)
    grid = grid[np.isfinite(grid)]
    if values.size < SPARSE_CELLS or grid.size == 0:
        return None
    low, high = float(grid.min()), float(grid.max())
    if not (high > low):
        return None
    edges = np.linspace(low, high, SPARSE_CELLS + 1)
    counts, _ = np.histogram(values, bins=edges)
    expected = values.size / SPARSE_CELLS
    thin = counts < SPARSE_DENSITY_SHARE * expected
    if not thin.any() or thin.all():
        # All-thin means the covariate is concentrated outside the plotted
        # grid entirely, which the rug already shows; shading the whole panel
        # says nothing.
        return None

    # Close single-cell gaps and drop single-cell bands. Cell counts near the
    # threshold flip on one observation, which without this drew a normal
    # covariate's tail as four disconnected slivers instead of one stretch.
    for index in range(1, thin.size - 1):
        if thin[index - 1] and thin[index + 1]:
            thin[index] = True
    runs, start = [], None
    for index, is_thin in enumerate(thin):
        if is_thin and start is None:
            start = index
        elif not is_thin and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, thin.size))
    bands = [(edges[a], edges[b]) for a, b in runs if b - a >= 2]
    if not bands:
        return None
    return pl.DataFrame({"start": [b[0] for b in bands],
                         "end": [b[1] for b in bands]})


def _sparse_layer(x_values, line_x):
    """The shading for :func:`_sparse_bands`, or `None`."""
    bands = _sparse_bands(x_values, line_x)
    if bands is None:
        return None
    return alt.Chart(bands).mark_rect(
        opacity=0.10, color="#b07aa1",
    ).encode(
        x=alt.X("start:Q"), x2=alt.X2("end:Q"),
        tooltip=[alt.Tooltip("start:Q", format=".4g", title="sparse from"),
                 alt.Tooltip("end:Q", format=".4g", title="to")],
    )


def _zero_rule(field="y"):
    """The one zero-reference style used by every panel.

    Previously each caller invented its own: partial effects used a 0.6-wide
    rule at 0.5 opacity, residuals an opaque 1.5-wide black dash that competed
    with the data it was annotating.
    """
    return alt.Chart(pl.DataFrame({field: [0]})).mark_rule(
        color="#8d8d8d", strokeWidth=1, strokeDash=[3, 3], opacity=0.9,
    ).encode(**{field: f"{field}:Q"})


def _default_rule(iteration):
    return alt.Chart(
        pl.DataFrame({"x": [iteration]})
    ).mark_rule(
        color="#e15759", strokeDash=[4, 3], strokeWidth=1.5
    ).encode(x="x:Q")


def _rule_label(iteration, label, color="#e15759", dy=-6):
    return alt.Chart(
        pl.DataFrame({"x": [iteration], "label": [label]})
    ).mark_text(
        color=color, fontSize=10, dx=4, dy=dy, y=14
    ).encode(
        x="x:Q", text="label:N",
    )


# ---------------------------------------------------------------------------
# Risk path
# ---------------------------------------------------------------------------

def plot_risk(model):
    """Plot the risk path over boosting iterations.

    Parameters
    ----------
    model : GLMBoostModel or GAMBoostModel
        Fitted model.

    Returns
    -------
    alt.Chart
    """
    risk = model.risk_
    df = pl.DataFrame({"iteration": np.arange(len(risk)), "risk": risk})
    return alt.Chart(df).mark_line(
        strokeWidth=1.8, color="#4e79a7"
    ).encode(
        x=alt.X("iteration:Q", title="Iteration"),
        y=alt.Y("risk:Q", title="Risk", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("iteration:Q", format="d"),
                 alt.Tooltip("risk:Q", format=".6f")],
    ).properties(
        title="Risk path", width=WIDTH, height=HEIGHT
    ).interactive()


# ---------------------------------------------------------------------------
# Coefficient paths (GLM only)
# ---------------------------------------------------------------------------

def plot_coef_path(model, which=None):
    """Plot coefficient paths for a GLMBoostModel.

    Parameters
    ----------
    model : GLMBoostModel
        Fitted GLM boost model.
    which : list of str or str, optional
        Which predictors to plot. If None, plots all non-zero predictors.

    Returns
    -------
    alt.Chart
    """
    mstop = model.mstop
    n_learners = len(model._learner_names)

    selected = np.asarray(model._selected, dtype=np.intp)
    widths = {np.asarray(c).size for c in model._coefficients}
    if widths - {1}:
        raise ValueError(
            "plot_coef_path needs one coefficient per base learner, but this "
            f"model has learners with {sorted(widths)} coefficients. A spline "
            "or other multi-column learner has no single coefficient to trace; "
            "use plot_partial_effects to see how its fitted shape develops, or "
            "plot_selected for which learners were chosen when."
        )
    step_coef = model._nu * np.fromiter(
        (np.asarray(c).item() for c in model._coefficients),
        dtype=np.float64,
        count=mstop,
    )
    contrib = np.zeros((mstop + 1, n_learners))
    contrib[np.arange(1, mstop + 1), selected] = step_coef
    paths = np.cumsum(contrib, axis=0)

    if which is None:
        active = [i for i in range(n_learners) if np.any(np.abs(paths[:, i]) > 1e-15)]
    elif isinstance(which, str):
        active = [model._learner_names.index(which)]
    else:
        active = [model._learner_names.index(w) for w in which]

    if not active:
        return alt.Chart().mark_text(text="No active predictors").properties(
            title="Coefficient paths (no active predictors)",
            width=WIDTH, height=HEIGHT,
        )

    rows = []
    for idx in active:
        name = model._learner_names[idx]
        for t in range(mstop + 1):
            rows.append({"iteration": t, "coefficient": paths[t, idx], "predictor": name})
    df = pl.DataFrame(rows)

    return alt.Chart(df).mark_line(strokeWidth=1.5).encode(
        x=alt.X("iteration:Q", title="Iteration"),
        y=alt.Y("coefficient:Q", title="Coefficient"),
        color=alt.Color("predictor:N", title="Predictor"),
        strokeDash=alt.StrokeDash("predictor:N"),
        tooltip=[alt.Tooltip("iteration:Q", format="d"),
                 alt.Tooltip("coefficient:Q", format=".6f"),
                 alt.Tooltip("predictor:N", title="Predictor")],
    ).properties(
        title="Coefficient paths", width=WIDTH, height=HEIGHT
    ).interactive()


# ---------------------------------------------------------------------------
# Helpers for partial-effect plots
# ---------------------------------------------------------------------------

def _variable_name(model, learner_name):
    """Get the x-variable name for a learner."""
    idx = model._learner_names.index(learner_name)
    spec = model._learners_spec[idx]
    kind = spec.__class__.__name__
    if kind == "Bspatial":
        return spec.name1
    return spec.name


def _partial_effects_with_points(model, effects, which, n_points):
    """Return (line_data, point_data, rug_data) for each learner.

    Parameters
    ----------
    model : GAMBoostModel
    effects : dict
        Output of ``model.partial_effects(which, n_points=n_points)``.
    which : list of str or None
    n_points : int

    Returns
    -------
    dict
        Mapping learner name -> (line_df, point_df, rug_x).
    """
    result = {}

    # A partial residual is the fitted contribution plus the loss's current
    # working residual (negative gradient). The old plot used the contribution
    # alone for both the line and the points, so every point lay on the line and
    # the advertised residual scatter was invisible.
    working_residual = np.asarray(
        model.residuals(type="working"), dtype=np.float64
    ).ravel()

    # Per-observation effects for scatter points
    train_effects = {}
    if which is not None:
        train_effects = model.partial_effects(which=which, n_points=None)
    else:
        train_effects = model.partial_effects(n_points=None)

    for name in effects:
        x_vals, y_vals = effects[name]
        sort_idx = np.argsort(x_vals)
        line_df = pl.DataFrame({
            "x": x_vals[sort_idx],
            "y": y_vals[sort_idx],
        })

        tx, ty = train_effects[name]
        point_df = pl.DataFrame({"x": tx, "y": ty + working_residual})

        try:
            var = _variable_name(model, name)
            rug_x = np.asarray(model._data_array_for(var), dtype=np.float64).ravel()
        except (KeyError, ValueError):
            # A learner over something that is not one stored column -- a
            # tensor product, a user design -- has no single rug variable, so
            # the effect's own x values stand in.
            rug_x = tx
        result[name] = (line_df, point_df, rug_x)

    return result


#: The link scale a learner's contribution is measured on, by family, for
#: the effect axis. A Binomial effect is half a log-odds (mboost's -1/+1
#: coding), which a reader comparing with a log-odds plot must be told.
_EFFECT_SCALE = {
    "Binomial": "half log-odds", "AdaExp": "half log-odds",
    "Poisson": "log expected count", "NBinomial": "log expected count",
    "GammaReg": "log expected response", "CoxPH": "log relative risk",
    "Weibull": "log-time location", "Loglog": "log-time location",
    "Lognormal": "log-time location",
}


def effect_axis_title(model):
    """``"Effect"``, with the link scale named when it is not the response's own."""
    family = getattr(model, "_family", None)
    scale = _EFFECT_SCALE.get(type(family).__name__)
    return f"Effect ({scale})" if scale else "Effect"


def _build_partial_chart(line_df, point_df, rug_x, title, single=False, y_title="Effect"):
    """Build a layered partial-effect chart.

    Layers (bottom to top)
    ---
    1. Shading over stretches where the data is thin
    2. Partial residuals at training observations (semi-transparent circles)
    3. Rug plot showing predictor density along the x-axis
    4. Fitted partial-effect line
    5. Zero reference rule

    The curve is drawn at the same weight everywhere, so a stretch supported
    by three observations looks exactly like one supported by three hundred.
    The rug carries the density for a reader who counts ticks; the shading
    says where counting is worth the trouble.
    """
    w = WIDTH if single else 240
    h = HEIGHT if single else 200

    sparse = _sparse_layer(rug_x, line_df["x"] if "x" in line_df.columns else [])

    # Layer 1 — per-observation partial residuals
    points = alt.Chart(point_df).mark_circle(
        opacity=0.13, size=11, color="#8a6a8a"
    ).encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q"),
        tooltip=[alt.Tooltip("x:Q", format=".4f"),
                 alt.Tooltip("y:Q", format=".4f")],
    )

    # Layer 2 — rug ticks showing data density.
    # `y=alt.value(...)` positions in pixels rather than through the shared y
    # scale. Without it the ticks carry no y encoding and Vega-Lite centres them
    # vertically, which drew the rug across the middle of the panel.
    rug = _rug(rug_x, h)

    # Layer 3 — fitted smooth line
    line = alt.Chart(line_df).mark_line(
        strokeWidth=2.0, color="#4e79a7"
    ).encode(
        x=alt.X("x:Q", title=title if single else ""),
        y=alt.Y("y:Q", title=y_title),
        tooltip=[alt.Tooltip("x:Q", format=".4f"),
                 alt.Tooltip("y:Q", format=".4f")],
    )

    # Layer 4 — zero reference
    zero = _zero_rule("y")

    layers = [layer for layer in (sparse, points, rug, line, zero)
              if layer is not None]
    c = alt.layer(*layers).resolve_scale(x="shared", y="shared").properties(
        width=w, height=h,
    )

    if single:
        c = c.interactive()
        return c.properties(title=f"Partial effect: {title}")
    return c.properties(title=title)


# ---------------------------------------------------------------------------
# Partial effects (GAM only)
# ---------------------------------------------------------------------------

def plot_partial_effects(model, which=None, n_points=100):
    """Plot partial effects for a GAMBoostModel.

    Each selected base-learner is shown as a separate faceted plot
    with the fitted partial-effect line, per-observation partial
    residuals (points), and a rug plot of predictor density.

    Parameters
    ----------
    model : GAMBoostModel
        Fitted GAM boost model.
    which : list of str or str, optional
        Which base-learners to plot. If None, plots all selected learners.
    n_points : int
        Number of evaluation points for smooth (bbs) effects on a grid.

    Returns
    -------
    alt.Chart | alt.VConcatChart
    """
    effects = model.partial_effects(which=which, n_points=n_points)

    if not effects:
        return alt.Chart().mark_text(text="No learners").properties(
            title="Partial effects (no learners)", width=WIDTH, height=HEIGHT
        )

    names = list(effects.keys())

    # Build per-learner data
    parts = _partial_effects_with_points(model, effects, which, n_points)

    if len(names) == 1:
        line_df, point_df, rug_x = parts[names[0]]
        return _build_partial_chart(
            line_df, point_df, rug_x, names[0], single=True,
            y_title=effect_axis_title(model),
        )

    n = len(names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    charts = []
    for name in names:
        line_df, point_df, rug_x = parts[name]
        charts.append(_build_partial_chart(line_df, point_df, rug_x, name,
                                           y_title=effect_axis_title(model)))

    rows_concat = []
    for r in range(nrows):
        row_charts = charts[r * ncols:(r + 1) * ncols]
        if len(row_charts) < ncols:
            blank = alt.Chart().mark_point().properties(width=240, height=200)
            row_charts.append(blank)
        rows_concat.append(alt.hconcat(*row_charts))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return alt.vconcat(*rows_concat).properties(title="Partial effects")


# ---------------------------------------------------------------------------
# Cross-validated risk
# ---------------------------------------------------------------------------

def plot_cvrisk(cvrisk_result):
    """Plot cross-validated risk.

    Parameters
    ----------
    cvrisk_result : CvRiskResult
        Result from model.cvrisk().

    Returns
    -------
    alt.LayerChart
    """
    grid = np.asarray(cvrisk_result.grid)
    aggregate = np.asarray(cvrisk_result.aggregate)

    risk_matrix = cvrisk_result.risk_matrix
    if not isinstance(risk_matrix, np.ndarray):
        risk_matrix = np.array(risk_matrix)

    # Fold traces
    fold_rows = []
    for i in range(risk_matrix.shape[0]):
        for j in range(len(grid)):
            fold_rows.append({"iteration": float(grid[j]), "risk": float(risk_matrix[i, j]), "fold": f"fold {i + 1}"})
    fold_df = pl.DataFrame(fold_rows)

    # Aggregate, with the across-fold spread. Drawing only the mean says
    # nothing about variation across datasets. The one-SE point uses the
    # fold-level variation as a heuristic for selecting an earlier stop.
    standard_error = np.asarray(cvrisk_result.standard_error)
    agg_df = pl.DataFrame({
        "iteration": grid,
        "risk": aggregate,
        "lower": aggregate - standard_error,
        "upper": aggregate + standard_error,
    })

    band = alt.Chart(agg_df).mark_area(opacity=0.18, color="#4e79a7").encode(
        x=alt.X("iteration:Q", title="Iteration"),
        y=alt.Y("lower:Q", title="Risk", scale=alt.Scale(zero=False)),
        y2=alt.Y2("upper:Q"),
    )

    fold_lines = alt.Chart(fold_df).mark_line(
        strokeWidth=0.5, color="gray"
    ).encode(
        x=alt.X("iteration:Q", title="Iteration"),
        y=alt.Y("risk:Q", title="Risk", scale=alt.Scale(zero=False)),
        detail="fold:N",
        opacity=alt.value(0.25),
    )

    agg_line = alt.Chart(agg_df).mark_line(
        strokeWidth=2.5, color="black"
    ).encode(
        x="iteration:Q", y=alt.Y("risk:Q", scale=alt.Scale(zero=False)),
    )

    opt = cvrisk_result.optimal_mstop
    rule = _default_rule(opt)
    label = _rule_label(opt, f"mstop = {opt}")

    # The one-standard-error stop, drawn only when it differs from the
    # minimiser -- when it does not, a second rule on the same line is noise.
    one_se = int(cvrisk_result.mstop_1se)
    se_layers = []
    if one_se != int(opt):
        se_rule = alt.Chart(pl.DataFrame({"iteration": [float(one_se)]})).mark_rule(
            strokeDash=[2, 3], strokeWidth=1.5, color="#59a14f",
        ).encode(x="iteration:Q")
        se_layers = [se_rule, _rule_label(one_se, f"1-SE = {one_se}",
                                          color="#59a14f", dy=14)]

    fold_sel = alt.selection_point(fields=["fold"], bind="legend")

    fold_lines = fold_lines.add_params(fold_sel).encode(
        opacity=alt.condition(fold_sel, alt.value(0.6), alt.value(0.08)),
        tooltip=[alt.Tooltip("iteration:Q", format="d"),
                 alt.Tooltip("risk:Q", format=".6f"),
                 alt.Tooltip("fold:N")],
    )

    agg_line = agg_line.encode(
        tooltip=[alt.Tooltip("iteration:Q", format="d"),
                 alt.Tooltip("risk:Q", format=".6f")],
    )

    return alt.layer(
        band, fold_lines, agg_line, rule, label, *se_layers
    ).resolve_scale(y="shared").properties(
        title=f"CV risk ({cvrisk_result.fold_type})",
        width=WIDTH, height=HEIGHT,
    )


# ---------------------------------------------------------------------------
# AIC
# ---------------------------------------------------------------------------

def plot_aic(aic_result):
    """Plot AIC path over boosting iterations.

    Parameters
    ----------
    aic_result : AicResult
        Result from model.aic().

    Returns
    -------
    alt.LayerChart
    """
    aic_vals = np.asarray(aic_result.aic)
    df = pl.DataFrame({"iteration": np.arange(1, len(aic_vals) + 1), "aic": aic_vals})
    criterion = getattr(aic_result, "criterion", "unspecified")
    normalization = getattr(aic_result, "normalization", "unspecified")
    label_text = "AIC" if criterion == "unspecified" else f"{criterion} ({normalization})"

    line = alt.Chart(df).mark_line(strokeWidth=1.8, color="#4e79a7").encode(
        x=alt.X("iteration:Q", title="Iteration"),
        y=alt.Y("aic:Q", title=label_text, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("iteration:Q", format="d"),
                 alt.Tooltip("aic:Q", format=".4f")],
    )

    opt = aic_result.mstop
    rule = _default_rule(opt)
    label = _rule_label(opt, f"mstop = {opt}")

    return alt.layer(line, rule, label).properties(
        title=f"{label_text} path", width=WIDTH, height=HEIGHT,
    ).interactive()


# ---------------------------------------------------------------------------
# Variable importance
# ---------------------------------------------------------------------------

def plot_varimp(model, sort=True):
    """Plot variable importance as a horizontal bar chart.

    Parameters
    ----------
    model : GLMBoostModel or GAMBoostModel
        Fitted model.
    sort : bool
        If True (default), sort by importance descending.

    Returns
    -------
    alt.Chart
    """
    vi = model.varimp()
    names = list(vi["reduction"].keys())
    vals = [vi["reduction"][n] for n in names]

    if sort:
        pairs = sorted(zip(vals, names, strict=False), reverse=True)
        vals = [p[0] for p in pairs]
        names = [p[1] for p in pairs]

    df = pl.DataFrame({"learner": names, "importance": vals})

    bar = alt.Chart(df).mark_bar().encode(
        # varimp() returns shares that sum to 1, so format as a percentage
        # rather than labelling a 0-1 fraction "(%)".
        x=alt.X("importance:Q", title="Share of risk reduction",
                axis=alt.Axis(format=".0%")),
        y=alt.Y("learner:N", title=None, sort="-x",
                scale=alt.Scale(paddingInner=0.38, paddingOuter=0.24)),
        color=alt.condition(
            alt.datum.importance >= 0,
            alt.value("#4e79a7"),
            alt.value("#e15759"),
        ),
        tooltip=[alt.Tooltip("learner:N", title="Learner"),
                 alt.Tooltip("importance:Q", format=".1%", title="Share")],
    ).properties(
        title="Variable importance", width=WIDTH, height=min(HEIGHT, len(names) * 30 + 30),
    )
    return bar


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def plot_residuals(model, type="response", parameter=None):
    """Plot residuals vs fitted values.

    For binary response families (Binomial, AdaExp), returns a two-panel
    figure with deviance residuals (left) and a binned calibration curve
    (right).  Standard response residuals for {0,1} data form two disconnected
    clusters and are uninformative, so the binary path uses deviance residuals
    instead.

    Parameters
    ----------
    model : GLMBoostModel, GAMBoostModel or LSSModel
        Fitted model.
    type : str
        "response" (default) or "pearson".  Ignored for binary families.
    parameter : str, optional
        For a distributional (LSS) model, which distribution parameter's
        residuals to plot; defaults to the first. Not accepted for other
        models, which have a single residual vector.

    Returns
    -------
    alt.Chart | alt.HConcatChart
    """
    from mboostrs.families import AdaExp as _AdaExp
    from mboostrs.families import Binomial as _Binomial

    if isinstance(getattr(model, "_family", None), (_Binomial, _AdaExp)):
        return _plot_binary_diagnostics(model)

    resid = model.residuals(type=type)
    if isinstance(resid, dict):
        # A distributional model has one residual vector per parameter, and
        # `parameter` selects which. Passing the dict on to np.asarray built a
        # 0-d object array and failed inside float().
        if parameter is None:
            parameter = next(iter(resid))
        if parameter not in resid:
            raise ValueError(
                f"no residuals for parameter {parameter!r}; this model has "
                f"{sorted(resid)}"
            )
        resid = resid[parameter]
        yhat = np.asarray(model.predict(parameter=parameter, type="response"),
                          dtype=np.float64)
    elif parameter is not None:
        raise ValueError(
            "parameter= applies to distributional (LSS) models; this model has "
            "a single residual vector"
        )
    else:
        yhat = np.asarray(model.predict(type="response"), dtype=np.float64)

    resid = np.asarray(resid, dtype=np.float64)
    df = pl.DataFrame({"fitted": yhat, "residual": resid})

    points = alt.Chart(df).mark_circle(opacity=0.35, size=24).encode(
        x=alt.X("fitted:Q", title="Fitted values"),
        y=alt.Y("residual:Q", title="Residuals" if type != "pearson" else "Pearson residuals"),
        tooltip=[alt.Tooltip("fitted:Q", format=".4f"),
                 alt.Tooltip("residual:Q", format=".4f")],
    )

    zero = _zero_rule("y")

    ylabel = "Pearson residuals" if type == "pearson" else "Residuals"
    return alt.layer(points, zero).properties(
        title=f"{ylabel} vs fitted", width=WIDTH, height=HEIGHT,
    ).interactive()


def _plot_binary_diagnostics(model, n_bins=15):
    yhat = np.asarray(model.predict(type="response"), dtype=np.float64)
    resid = np.asarray(model.residuals(type="response"), dtype=np.float64)
    y = resid + yhat

    eps = 1e-15
    dev = np.where(
        y == 1,
        np.sqrt(np.maximum(-2.0 * np.log(np.maximum(yhat, eps)), 0.0)),
        -np.sqrt(np.maximum(-2.0 * np.log(np.maximum(1.0 - yhat, eps)), 0.0)),
    )

    df = pl.DataFrame({"fitted": yhat, "deviance": dev})

    brush_left = alt.selection_interval(name="left")
    brush_right = alt.selection_interval(name="right")

    left = alt.Chart(df).mark_circle(opacity=0.35, size=24).add_params(brush_left).encode(
        x=alt.X("fitted:Q", title="Predicted probability"),
        y=alt.Y("deviance:Q", title="Deviance residuals"),
        tooltip=[alt.Tooltip("fitted:Q", format=".4f"),
                 alt.Tooltip("deviance:Q", format=".4f")],
    )
    zero = _zero_rule("y")
    left_chart = alt.layer(left, zero).properties(
        title="Deviance residuals", width=WIDTH // 2 - 10, height=HEIGHT,
    )

    n = len(yhat)
    n_bins = min(n_bins, n // 5) if n >= 10 else 1
    if n_bins < 1:
        n_bins = 1

    idx = np.argsort(yhat)
    yhat_sorted = yhat[idx]
    y_sorted = y[idx]

    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    bin_pred = []
    bin_obs = []
    for i in range(n_bins):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        if start >= end:
            continue
        bin_pred.append(np.mean(yhat_sorted[start:end]))
        bin_obs.append(np.mean(y_sorted[start:end]))

    df_cal = pl.DataFrame({"predicted": bin_pred, "observed": bin_obs})

    cal = alt.Chart(df_cal).mark_circle(size=50).add_params(brush_right).encode(
        x=alt.X("predicted:Q", title="Predicted probability",
                scale=alt.Scale(domain=[0, 1], zero=True)),
        y=alt.Y("observed:Q", title="Observed rate",
                scale=alt.Scale(domain=[0, 1], zero=True)),
        tooltip=[alt.Tooltip("predicted:Q", format=".3f"),
                 alt.Tooltip("observed:Q", format=".3f")],
    )
    # The 45-degree line is a reference, styled like every other reference.
    diag = alt.Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#8d8d8d", strokeWidth=1, strokeDash=[3, 3], opacity=0.9,
    ).encode(x="x:Q", y="y:Q")
    right_chart = alt.layer(cal, diag).properties(
        title="Calibration curve", width=WIDTH // 2 - 10, height=HEIGHT,
    ).interactive()

    return alt.hconcat(left_chart, right_chart)


def plot_causal_overlap(result):
    """Plot propensity overlap and estimand-specific analysis weights.

    The left panel compares the empirical propensity distributions of the two
    treatment arms. The right panel shows which observations receive large
    weights for the requested ATE or ATT. Separation or isolated large weights
    indicate that the estimate relies on extrapolation or few observations;
    neither panel proves positivity outside the observed sample.

    Parameters
    ----------
    result : AIPWResult
        A fitted causal result from :func:`mboostrs.aipw`.

    Returns
    -------
    alt.HConcatChart
    """
    from mboostrs.causal import AIPWResult

    if not isinstance(result, AIPWResult):
        raise TypeError("result must be an AIPWResult")
    frame = pl.DataFrame(result.overlap_frame(output="records"))
    color = alt.Color(
        "treatment_label:N", title="Treatment arm",
        scale=alt.Scale(domain=["control", "treated"],
                        range=[PALETTE[0], PALETTE[1]]),
    )
    density = alt.Chart(frame).transform_density(
        "propensity", groupby=["treatment_label"], extent=[0, 1],
        as_=["propensity", "density"],
    ).mark_area(opacity=.35).encode(
        x=alt.X("propensity:Q", title="Estimated treatment propensity",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("density:Q", title="Density"),
        color=color,
        tooltip=["treatment_label:N", alt.Tooltip("propensity:Q", format=".3f"),
                 alt.Tooltip("density:Q", format=".3f")],
    ).properties(title="Observed propensity overlap", width=WIDTH // 2,
                 height=HEIGHT)
    weights = alt.Chart(frame).mark_circle(opacity=.35, size=24).encode(
        x=alt.X("propensity_used:Q", title="Propensity used by estimator",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("analysis_weight:Q", title=f"{result.spec.estimand.upper()} weight"),
        color=color,
        tooltip=[alt.Tooltip("row:Q"), "treatment_label:N",
                 alt.Tooltip("propensity_used:Q", format=".4f"),
                 alt.Tooltip("analysis_weight:Q", format=".3f")],
    ).properties(title="Influential support points", width=WIDTH // 2,
                 height=HEIGHT)
    return alt.hconcat(density, weights).resolve_scale(color="shared")


def plot_causal_balance(result):
    """Plot covariate standardized mean differences before and after weighting.

    Zero is exact mean balance. The chart does not impose a universal cutoff:
    an acceptable residual imbalance depends on the outcome relationship,
    sample size, overlap and scientific context.
    """
    from mboostrs.causal import AIPWResult

    if not isinstance(result, AIPWResult):
        raise TypeError("result must be an AIPWResult")
    rows = []
    for item in result.balance_frame(output="records"):
        rows.extend([
            {"covariate": item["covariate"], "stage": "before",
             "smd": item["smd_before"]},
            {"covariate": item["covariate"], "stage": "weighted",
             "smd": item["smd_after"]},
        ])
    frame = pl.DataFrame(rows)
    encodings = {
        "x": alt.X("smd:Q", title="Standardized mean difference"),
        "y": alt.Y("covariate:N", title=None, sort="-x"),
    }
    paths = alt.Chart(frame).mark_line(color="#c6c6c6").encode(
        **encodings, detail="covariate:N",
    )
    points = alt.Chart(frame).mark_point(filled=True, size=65).encode(
        **encodings,
        color=alt.Color("stage:N", title="Adjustment",
                        scale=alt.Scale(domain=["before", "weighted"],
                                        range=[PALETTE[1], PALETTE[0]])),
        tooltip=["covariate:N", "stage:N", alt.Tooltip("smd:Q", format=".3f")],
    )
    zero = alt.Chart(pl.DataFrame({"smd": [0.0]})).mark_rule(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(x="smd:Q")
    return alt.layer(paths, points, zero).properties(
        title="Covariate balance", width=WIDTH, height=max(120, 24 * len(rows) // 2),
    )


def plot_evaluation_residuals(result):
    """Plot held-out response residuals against held-out predictions."""
    from mboostrs.evaluation import EvaluationResult

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    frame = pl.DataFrame(result.residual_frame(output="records"))
    points = alt.Chart(frame).mark_circle(opacity=.35, size=24).encode(
        x=alt.X("prediction:Q", title="Held-out response prediction"),
        y=alt.Y("residual:Q", title="Observed − predicted"),
        color=alt.Color("fold:N", title="Fold"),
        tooltip=["record_id:N", "fold:N",
                 alt.Tooltip("prediction:Q", format=".4g"),
                 alt.Tooltip("residual:Q", format=".4g")],
    )
    zero = alt.Chart(pl.DataFrame({"residual": [0.0]})).mark_rule(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(y="residual:Q")
    return alt.layer(points, zero).properties(
        title=f"Held-out residuals: {result.evaluation_task.replace('_', ' ')}",
        width=WIDTH, height=HEIGHT,
    )


def plot_evaluation_calibration(result, n_bins=10, level=.95):
    """Plot reliability bins from an evaluation result's held-out probabilities.

    Vertical intervals appear only when the result's declared evaluation task
    supports the IID-row approximation used by
    :meth:`EvaluationResult.calibration_frame`.
    """
    from mboostrs.evaluation import EvaluationResult

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    rows = result.calibration_frame(n_bins=n_bins, level=level, output="records")
    frame = pl.DataFrame(rows)
    points = alt.Chart(frame).mark_circle(size=60, color=PALETTE[0]).encode(
        x=alt.X("predicted_mean:Q", title="Mean held-out probability",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed_rate:Q", title="Observed event rate",
                scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("effective_n:Q", title="Effective n"),
        tooltip=["bin:O", "n:Q", alt.Tooltip("weight_total:Q", format=".3g"),
                 alt.Tooltip("predicted_mean:Q", format=".3f"),
                 alt.Tooltip("observed_rate:Q", format=".3f"),
                 "uncertainty:N"],
    )
    layers = []
    interval_rows = [row for row in rows if row["lower"] is not None]
    if interval_rows:
        intervals = alt.Chart(pl.DataFrame(interval_rows)).mark_rule(
            color=PALETTE[0], strokeWidth=1,
        ).encode(
            x="predicted_mean:Q", y="lower:Q", y2="upper:Q",
        )
        layers.append(intervals)
    diagonal = alt.Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(x="x:Q", y="y:Q")
    return alt.layer(diagonal, *layers, points).properties(
        title="Held-out probability calibration", width=WIDTH, height=HEIGHT,
    )


def plot_classification_evaluation(result, *, thresholds=101,
                                   utility=(0.0, -1.0, -1.0, 0.0)):
    """Plot held-out ROC, precision–recall, and decision utility curves."""
    from mboostrs.evaluation import EvaluationResult

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    roc_rows = result.roc_frame(output="records")
    pr_rows = result.precision_recall_frame(output="records")
    utility_rows = result.threshold_frame(
        thresholds, utility=utility, output="records"
    )
    auc = roc_rows[0]["auc"]
    average_precision = pr_rows[0]["average_precision"]
    prevalence = result.classification_metrics()["prevalence"]

    roc = alt.Chart(pl.DataFrame(roc_rows)).mark_line(
        color=PALETTE[0], strokeWidth=2,
    ).encode(
        x=alt.X("false_positive_rate:Q", title="False-positive rate",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("sensitivity:Q", title="Sensitivity",
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("threshold:Q", format=".3g"),
                 alt.Tooltip("false_positive_rate:Q", format=".3f"),
                 alt.Tooltip("sensitivity:Q", format=".3f")],
    )
    diagonal = alt.Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(x="x:Q", y="y:Q")
    roc_panel = alt.layer(diagonal, roc).properties(
        title=f"ROC · AUC {auc:.3f}", width=WIDTH // 3, height=HEIGHT,
    )

    precision_recall = alt.Chart(pl.DataFrame(pr_rows)).mark_line(
        color=PALETTE[1], strokeWidth=2,
    ).encode(
        x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("threshold:Q", format=".3g"),
                 alt.Tooltip("recall:Q", format=".3f"),
                 alt.Tooltip("precision:Q", format=".3f")],
    )
    prevalence_rule = alt.Chart(pl.DataFrame({"prevalence": [prevalence]})).mark_rule(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(y="prevalence:Q")
    pr_panel = alt.layer(prevalence_rule, precision_recall).properties(
        title=f"Precision–recall · AP {average_precision:.3f}",
        width=WIDTH // 3, height=HEIGHT,
    )

    utility_chart = alt.Chart(pl.DataFrame(utility_rows)).mark_line(
        color=PALETTE[2], strokeWidth=2,
    ).encode(
        x=alt.X("threshold:Q", title="Decision threshold",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("average_utility:Q", title="Average declared utility",
                scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("threshold:Q", format=".3f"),
                 alt.Tooltip("average_utility:Q", format=".4g"),
                 "utility_tn:Q", "utility_fp:Q", "utility_fn:Q", "utility_tp:Q"],
    ).properties(title="Threshold utility", width=WIDTH // 3, height=HEIGHT)
    return alt.hconcat(roc_panel, pr_panel, utility_chart)


def plot_multiclass_evaluation(result, *, n_bins=8):
    """Plot weighted confusion cells and one-vs-rest held-out calibration."""
    from mboostrs.evaluation import EvaluationResult

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    confusion_rows = result.multiclass_confusion_frame(output="records")
    calibration_rows = result.multiclass_calibration_frame(
        n_bins=n_bins, output="records"
    )
    confusion_frame = pl.DataFrame(confusion_rows)
    heatmap = alt.Chart(confusion_frame).mark_rect().encode(
        x=alt.X("predicted_label:N", title="Predicted class"),
        y=alt.Y("observed_label:N", title="Observed class"),
        color=alt.Color("weight:Q", title="Evaluation weight",
                        scale=alt.Scale(scheme="blues")),
        tooltip=["observed_label:N", "predicted_label:N",
                 alt.Tooltip("weight:Q", format=".4g")],
    ).properties(title="Weighted confusion matrix", width=WIDTH // 2,
                 height=HEIGHT)
    labels = alt.Chart(confusion_frame).mark_text().encode(
        x="predicted_label:N", y="observed_label:N",
        text=alt.Text("weight:Q", format=".3g"),
        color=alt.condition(alt.datum.weight > confusion_frame["weight"].max() / 2,
                            alt.value("white"), alt.value("#161616")),
    )
    confusion = alt.layer(heatmap, labels)

    calibration_frame = pl.DataFrame(calibration_rows)
    curves = alt.Chart(calibration_frame).mark_line(point=True).encode(
        x=alt.X("predicted_mean:Q", title="Mean held-out class probability",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed_rate:Q", title="Observed one-vs-rest rate",
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("class_label:N", title="Class"),
        tooltip=["class_label:N", "bin:O", "n:Q",
                 alt.Tooltip("predicted_mean:Q", format=".3f"),
                 alt.Tooltip("observed_rate:Q", format=".3f")],
    )
    diagonal = alt.Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(x="x:Q", y="y:Q")
    calibration = alt.layer(diagonal, curves).properties(
        title="One-vs-rest calibration", width=WIDTH // 2, height=HEIGHT,
    )
    return alt.hconcat(confusion, calibration)


def plot_distribution_evaluation(result):
    """Plot PIT, row log scores, and interval calibration for an LSS evaluation."""
    from mboostrs.distribution_evaluation import DistributionEvaluationResult

    if not isinstance(result, DistributionEvaluationResult):
        raise TypeError("result must be a DistributionEvaluationResult")
    rows = pl.DataFrame(result.to_frame(output="records"))
    pit = alt.Chart(rows).transform_density(
        "pit", extent=[0, 1], as_=["pit", "density"]
    ).mark_area(color=PALETTE[0], opacity=.55).encode(
        x=alt.X("pit:Q", title="Randomized PIT", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("density:Q", title="Density"),
        tooltip=[alt.Tooltip("pit:Q", format=".3f"),
                 alt.Tooltip("density:Q", format=".3f")],
    )
    uniform = alt.Chart(pl.DataFrame({"density": [1.0]})).mark_rule(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(y="density:Q")
    pit_panel = alt.layer(uniform, pit).properties(
        title="PIT against uniform density", width=WIDTH // 3, height=HEIGHT,
    )

    log_score = alt.Chart(rows).mark_circle(opacity=.35, size=20).encode(
        x=alt.X("row:Q", title="Evaluation row"),
        y=alt.Y("negative_log_score:Q", title="Negative log score"),
        tooltip=["row:Q", alt.Tooltip("observed:Q", format=".4g"),
                 alt.Tooltip("negative_log_score:Q", format=".4g")],
    ).properties(title="Row predictive log score", width=WIDTH // 3,
                 height=HEIGHT)

    interval_rows = result.interval_frame(output="records")
    interval_frame = pl.DataFrame(interval_rows)
    coverage = alt.Chart(interval_frame).mark_line(
        point=True, color=PALETTE[1], strokeWidth=2,
    ).encode(
        x=alt.X("nominal_coverage:Q", title="Nominal central coverage",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("empirical_coverage:Q", title="Empirical coverage",
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("nominal_coverage:Q", format=".1%"),
                 alt.Tooltip("empirical_coverage:Q", format=".1%"),
                 alt.Tooltip("mean_width:Q", format=".4g"),
                 alt.Tooltip("mean_interval_score:Q", format=".4g")],
    )
    diagonal = alt.Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(x="x:Q", y="y:Q")
    coverage_panel = alt.layer(diagonal, coverage).properties(
        title=f"Interval calibration · {result.data_role}",
        width=WIDTH // 3, height=HEIGHT,
    )
    return alt.hconcat(pit_panel, log_score, coverage_panel)


def plot_prediction_contrast(result):
    """Plot the distribution of a contrast's row-specific prediction changes."""
    from mboostrs.effects import ContrastResult

    if not isinstance(result, ContrastResult):
        raise TypeError("result must be a ContrastResult")
    frame = pl.DataFrame(result.to_frame(output="records"))
    histogram = alt.Chart(frame).mark_bar(color=PALETTE[0], opacity=.8).encode(
        x=alt.X("contrast:Q", bin=alt.Bin(maxbins=30),
                title=f"Row {result.scale}-scale contrast"),
        y=alt.Y("count():Q", title="Rows"),
        tooltip=[alt.Tooltip("count():Q", title="rows")],
    )
    average = alt.Chart(pl.DataFrame({"estimate": [result.estimate]})).mark_rule(
        color=PALETTE[2], strokeWidth=2,
    ).encode(x="estimate:Q")
    return alt.layer(histogram, average).properties(
        title=f"Prediction contrast: {result.variable}", width=WIDTH, height=HEIGHT,
    )


def plot_marginal_effect(result):
    """Plot row-specific finite derivatives against their predictor values."""
    from mboostrs.effects import MarginalEffectResult

    if not isinstance(result, MarginalEffectResult):
        raise TypeError("result must be a MarginalEffectResult")
    rows = result.to_frame(output="records")
    for row in rows:
        row["variable_value"] = (row["lower_value"] + row["upper_value"]) / 2
    frame = pl.DataFrame(rows)
    points = alt.Chart(frame).mark_circle(opacity=.35, size=24).encode(
        x=alt.X("variable_value:Q", title=result.variable),
        y=alt.Y("marginal_effect:Q",
                title=f"Derivative of {result.scale}-scale prediction"),
        tooltip=["record_id:N", alt.Tooltip("variable_value:Q", format=".4g"),
                 alt.Tooltip("marginal_effect:Q", format=".4g")],
    )
    average = alt.Chart(pl.DataFrame({"estimate": [result.estimate]})).mark_rule(
        color=PALETTE[2], strokeWidth=2,
    ).encode(y="estimate:Q")
    return alt.layer(points, average).properties(
        title=f"Marginal effect: {result.variable}", width=WIDTH, height=HEIGHT,
    )


def plot_accumulated_local_effect(result):
    """Plot an ALE curve with interval support encoded by point size."""
    from mboostrs.effects import ALE1DResult

    if not isinstance(result, ALE1DResult):
        raise TypeError("result must be an ALE1DResult")
    frame = pl.DataFrame(result.to_frame(output="records"))
    line = alt.Chart(frame).mark_line(color=PALETTE[0], strokeWidth=2).encode(
        x=alt.X("grid:Q", title=result.variable),
        y=alt.Y("effect:Q", title=f"ALE on {result.scale} scale"),
    )
    points = alt.Chart(frame).mark_circle(color=PALETTE[0], opacity=.85).encode(
        x="grid:Q", y="effect:Q",
        size=alt.Size("weight_total:Q", title="Weight in interval"),
        tooltip=["bin:O", alt.Tooltip("lower:Q", format=".4g"),
                 alt.Tooltip("upper:Q", format=".4g"), "n:Q",
                 alt.Tooltip("weight_total:Q", format=".4g"),
                 alt.Tooltip("effect:Q", format=".4g")],
    )
    zero = alt.Chart(pl.DataFrame({"effect": [0.0]})).mark_rule(
        color="#8d8d8d", strokeDash=[3, 3], strokeWidth=1,
    ).encode(y="effect:Q")
    return alt.layer(zero, line, points).properties(
        title=f"Accumulated local effect: {result.variable}",
        width=WIDTH, height=HEIGHT,
    )


def plot_prediction_profile(result, *, max_ice=50):
    """Plot weighted partial dependence, ICE, and joint-support diagnostics."""
    from mboostrs._validate import check_int_at_least
    from mboostrs.effects import PredictionProfileResult

    if not isinstance(result, PredictionProfileResult):
        raise TypeError("result must be a PredictionProfileResult")
    max_ice = check_int_at_least("max_ice", max_ice, 0, allow_none=False)
    average_frame = pl.DataFrame(result.to_frame(output="records"))
    layers = []
    if result.ice is not None and max_ice > 0:
        ids = result.observation_ids[:max_ice]
        selected_ids = set(ids.tolist())
        records = [row for row in result.ice_frame(output="records")
                   if row["record_id"] in selected_ids]
        ice = alt.Chart(pl.DataFrame(records)).mark_line(
            color="#a8a8a8", opacity=.18, strokeWidth=1,
        ).encode(
            x=alt.X("grid", title=result.variable),
            y=alt.Y("prediction:Q", title=f"Prediction ({result.scale})"),
            detail="record_id:N",
        )
        layers.append(ice)
    average = alt.Chart(average_frame).mark_line(
        color=PALETTE[0], strokeWidth=2.5,
    ).encode(
        x=alt.X("grid", title=result.variable),
        y=alt.Y("average_prediction:Q", title=f"Prediction ({result.scale})"),
        tooltip=["grid", alt.Tooltip("average_prediction:Q", format=".4g")],
    )
    layers.append(average)
    main = alt.layer(*layers).properties(
        title=f"Prediction profile: {result.variable}",
        width=WIDTH, height=HEIGHT,
    )
    if np.all(np.isnan(result.support_fraction_far)):
        return main
    support = alt.Chart(average_frame).mark_bar(color=PALETTE[2], opacity=.7).encode(
        x=alt.X("grid", title=result.variable),
        y=alt.Y("support_fraction_far:Q", title="Fraction far from joint support",
                scale=alt.Scale(domain=[0, 1])),
        tooltip=["grid", alt.Tooltip("support_fraction_far:Q", format=".1%")],
    ).properties(width=WIDTH, height=120)
    return alt.vconcat(main, support).resolve_scale(x="shared")


def plot_reliability(predicted, observed, n_bins=8, title=None):
    """Reliability diagram: binned predicted probability against observed rate.

    The 45-degree line marks perfect calibration. Points above it mean the
    predictions are under-confident (the observed rate exceeds the predicted
    probability); points below it mean over-confident predictions. Each point
    is one equal-sized bin of the predictions, ordered by predicted value.

    Parameters
    ----------
    predicted : array-like
        Predicted probabilities in [0, 1].
    observed : array-like
        Binary outcomes aligned with ``predicted``.
    n_bins : int, optional
        Number of equal-sized bins (default 8).
    title : str, optional
        Chart title.

    Returns
    -------
    alt.Chart
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    n = len(predicted)
    n_bins = max(1, min(n_bins, n // 5))

    idx = np.argsort(predicted)
    p = predicted[idx]
    o = observed[idx]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if lo >= hi:
            continue
        rows.append({
            "predicted": float(p[lo:hi].mean()),
            "observed": float(o[lo:hi].mean()),
            "n": int(hi - lo),
        })
    df = pl.DataFrame(rows)

    pts = alt.Chart(df).mark_circle(size=50).encode(
        x=alt.X("predicted:Q", title="Predicted probability",
                scale=alt.Scale(domain=[0, 1], zero=True)),
        y=alt.Y("observed:Q", title="Observed rate",
                scale=alt.Scale(domain=[0, 1], zero=True)),
        tooltip=[alt.Tooltip("predicted:Q", format=".3f"),
                 alt.Tooltip("observed:Q", format=".3f"),
                 alt.Tooltip("n:Q", title="plants")],
    )
    diag = alt.Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#8d8d8d", strokeWidth=1, strokeDash=[3, 3], opacity=0.9,
    ).encode(x="x:Q", y="y:Q")
    chart = alt.layer(pts, diag).properties(
        width=WIDTH, height=HEIGHT,
    ).interactive()
    if title is not None:
        chart = chart.properties(title=title)
    return chart


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def plot_confint(model, parm=None, level=0.95, n_points=100, B=200, seed=None,
                 interval=None):
    """Plot confidence intervals for model effects or coefficients.

    For GLMBoostModel: coefficient forest plot with CI bars.
    For GAMBoostModel: partial effects with CI bands.

    Parameters
    ----------
    model : GAMBoostModel or GLMBoostModel
        Fitted model.
    parm : str or list of str, optional
        Which learner(s)/predictor(s) to plot. If None, all selected.
    level : float
        Confidence level (default 0.95).
    n_points : int
        Number of evaluation points for smooth effects (GAMBoost only).
    B : int
        Number of bootstrap replications.
    seed : int, optional
        Random seed for bootstrap reproducibility.
    interval : dict, optional
        An interval `model.confint` already computed, plotted instead of
        bootstrapping again. Pass it when the interval was computed with
        settings this helper does not expose (``B_mstop``, ``n_jobs``) or
        after inspecting its tally -- ``n_used``, ``n_failed``,
        ``n_untuned`` -- so the replicates are bootstrapped once and the
        numbers behind the band stay visible. For a `GAMBoostModel` the
        interval pairs with the effect on the rows it was evaluated on: an
        interval from `confint`'s default is evaluated on the training
        rows, and the helper then plots the effect on those rows rather
        than on its own grid.

    Returns
    -------
    alt.Chart | alt.ConcatChart
    """
    from mboostrs.glmboost import GLMBoostModel

    if interval is not None and not (
            isinstance(interval, dict) and "lower" in interval
            and "upper" in interval):
        raise ValueError(
            "interval must be the dict model.confint() returns, with "
            "'lower' and 'upper' entries; got "
            f"{type(interval).__name__}"
        )

    if isinstance(model, GLMBoostModel):
        return _plot_confint_glm(model, parm=parm, level=level, B=B, seed=seed,
                                 interval=interval)

    return _plot_confint_gam(model, parm=parm, level=level, n_points=n_points,
                             B=B, seed=seed, interval=interval)


def _plot_confint_glm(model, parm, level, B, seed, interval=None):
    """Coefficient forest plot for GLMBoostModel."""
    if interval is not None:
        ci = interval
        level = float(interval.get("level", level))
    else:
        ci = model.confint(parm=parm, level=level, B=B, seed=seed)
    names = list(ci["lower"].keys())
    lowers = [ci["lower"][n] for n in names]
    uppers = [ci["upper"][n] for n in names]
    coefs_dict = model.coef()
    points = [coefs_dict.get(n, 0.0) for n in names]

    df = pl.DataFrame({
        "learner": names,
        "coef": points,
        "lower": lowers,
        "upper": uppers,
    })

    base = alt.Chart(df).encode(
        y=alt.Y("learner:N", title=None, sort="-x"),
    )

    ci_bars = base.mark_errorbar(extent="ci", thickness=1.5).encode(
        x=alt.X("lower:Q", title="Coefficient"),
        x2="upper:Q",
        tooltip=[alt.Tooltip("learner:N", title="Learner"),
                 alt.Tooltip("lower:Q", format=".4f", title="Lower"),
                 alt.Tooltip("upper:Q", format=".4f", title="Upper")],
    )

    points_chart = base.mark_circle(size=80, color="#4e79a7").encode(
        x="coef:Q",
        tooltip=[alt.Tooltip("learner:N", title="Learner"),
                 alt.Tooltip("coef:Q", format=".4f", title="Coefficient")],
    )

    zero = _zero_rule("x")

    return alt.layer(ci_bars, points_chart, zero).properties(
        title=f"Coefficients with {level:.0%} CI",
        width=WIDTH, height=min(HEIGHT, len(names) * 35 + 30),
    ).interactive()


def _effect_grid_frame(model, n_points):
    """Build an `n_points`-row frame on which every partial effect can be
    evaluated at once.

    Each predictor column is independently spread over its own observed range.
    The rows are therefore not observations — row *i* combines the *i*-th
    quantile-spaced value of every predictor — but a partial effect reads only
    the column belonging to its own learner, so each curve is evaluated on a
    proper grid for its own predictor. Building one frame rather than one per
    learner keeps `confint` to a single bootstrap pass.
    """
    data = model._data
    columns = data.columns if hasattr(data, "columns") else list(data.keys())

    grid = {}
    for column in columns:
        values = np.asarray(
            data[column].to_numpy() if hasattr(data[column], "to_numpy")
            else data[column]
        )
        if values.dtype.kind in "fiu" and np.isfinite(values).any():
            finite = values[np.isfinite(values)]
            grid[column] = np.linspace(finite.min(), finite.max(), n_points)
        else:
            # Non-numeric or all-missing: repeat the first value so the column
            # exists with the right length. Learners on such columns are not
            # smooth effects, so their grid is not meaningful anyway.
            grid[column] = np.repeat(values[:1], n_points)
    return pl.DataFrame(grid)


def _plot_confint_gam(model, parm, level, n_points, B, seed, interval=None):
    """Partial effects with confidence bands for GAMBoostModel.

    The effect and its interval are evaluated on the same frame. Evaluating
    the effect on an `n_points` grid while taking the interval from
    `confint`'s default (one value per training row) pairs each bound with an
    unrelated x, which raises IndexError when `n_points` exceeds the sample
    size and silently mislocates the band when it does not.

    A caller-supplied `interval` was evaluated on its own rows -- for
    `confint`'s default, the training rows -- so the effect is plotted on
    those rows rather than on the grid, keeping each bound beside the x it
    was computed for.
    """
    if interval is not None:
        grid = None
        level = float(interval.get("level", level))
        effects = model.partial_effects(which=parm, newdata=grid)
    else:
        grid = _effect_grid_frame(model, n_points)
        effects = model.partial_effects(which=parm, newdata=grid)

    if not effects:
        return alt.Chart().mark_text(text="No learners").properties(
            title="Partial effects (no learners)", width=WIDTH, height=HEIGHT
        )

    if interval is not None:
        ci = interval
    else:
        ci = model.confint(parm=parm, level=level, B=B, seed=seed, newdata=grid)

    def _one_ci(name):
        x_vals, y_vals = effects[name]
        sort_idx = np.argsort(x_vals)
        x = x_vals[sort_idx]
        y = y_vals[sort_idx]
        lower, upper = ci["lower"][name], ci["upper"][name]
        if len(lower) != len(x):
            raise ValueError(
                f"confidence bounds for {name!r} have length {len(lower)} but "
                f"the effect has {len(x)} points; they must be evaluated on "
                "the same data"
            )
        lo = lower[sort_idx]
        hi = upper[sort_idx]

        df = pl.DataFrame({"x": x, "y": y, "lower": lo, "upper": hi})

        band = alt.Chart(df).mark_area(opacity=0.12, color="#4e79a7").encode(
            x="x:Q", y="lower:Q", y2="upper:Q",
        )
        line = alt.Chart(df).mark_line(strokeWidth=1.8, color="#4e79a7").encode(
            x=alt.X("x:Q", title=name),
            y=alt.Y("y:Q", title=effect_axis_title(model)),
            tooltip=[alt.Tooltip("x:Q", format=".4f"),
                     alt.Tooltip("y:Q", format=".4f")],
        )
        c = (band + line).properties(
            title=f"Partial effect: {name}",
            width=200 if len(effects) > 1 else WIDTH,
            height=180 if len(effects) > 1 else HEIGHT,
        )
        if len(effects) == 1:
            c = c.interactive()
        return c

    names = list(effects.keys())
    if len(names) == 1:
        return _one_ci(names[0]).properties(
            title=f"Partial effects with {level:.0%} CI"
        )

    n = len(names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    charts = [_one_ci(n) for n in names]
    rows_concat = []
    for r in range(nrows):
        row_charts = charts[r * ncols:(r + 1) * ncols]
        if len(row_charts) < ncols:
            blank = alt.Chart().mark_point().properties(width=200, height=180)
            row_charts.append(blank)
        rows_concat.append(alt.hconcat(*row_charts))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return alt.vconcat(*rows_concat).properties(
            title=f"Partial effects with {level:.0%} CI"
        )


# ---------------------------------------------------------------------------
# Selection frequency
# ---------------------------------------------------------------------------

def plot_selected(model):
    """Plot selection frequency per base-learner as a bar chart.

    Parameters
    ----------
    model : GLMBoostModel or GAMBoostModel
        Fitted model.

    Returns
    -------
    alt.Chart
    """
    selected = model._selected
    n_learners = len(model._learner_names)
    names = model._learner_names
    counts = [selected.count(i) for i in range(n_learners)]
    total = len(selected)
    freqs = [c / total if total > 0 else 0.0 for c in counts]

    df = pl.DataFrame({"learner": names, "frequency": freqs})

    return alt.Chart(df).mark_bar(color="#4e79a7").encode(
        x=alt.X("frequency:Q", title="Selection frequency"),
        y=alt.Y("learner:N", title=None, sort="-x",
                scale=alt.Scale(paddingInner=0.38, paddingOuter=0.24)),
        tooltip=[alt.Tooltip("learner:N", title="Learner"),
                 alt.Tooltip("frequency:Q", format=".3f", title="Frequency")],
    ).properties(
        title="Selection frequency per base-learner",
        width=WIDTH, height=min(HEIGHT, n_learners * 30 + 30),
    )


# ---------------------------------------------------------------------------
# Btree partitions
# ---------------------------------------------------------------------------

def _btree_points_frame(data, xcol, ycol):
    """The training rows to overlay, as a Polars frame with the two columns."""
    if hasattr(data, "columns"):
        columns = list(data.columns)
        missing = [name for name in (xcol, ycol) if name not in columns]
        if missing:
            raise ValueError(
                f"data is missing the btree column(s) {missing}; it has "
                f"{columns}"
            )
        return pl.DataFrame({
            xcol: np.asarray(data[xcol], dtype=np.float64),
            ycol: np.asarray(data[ycol], dtype=np.float64),
        })
    if isinstance(data, dict):
        missing = [name for name in (xcol, ycol) if name not in data]
        if missing:
            raise ValueError(
                f"data is missing the btree column(s) {missing}; it has "
                f"{sorted(data)}"
            )
        return pl.DataFrame({
            xcol: np.asarray(data[xcol], dtype=np.float64),
            ycol: np.asarray(data[ycol], dtype=np.float64),
        })
    raise TypeError(
        "data must be a Polars/pandas DataFrame or a dict of columns, got "
        f"{type(data).__name__}"
    )


def _btree_split_rule(column, cutpoint, axis):
    """A dashed rule at one btree cutpoint on ``axis`` (``"x"`` or ``"y"``)."""
    frame = pl.DataFrame({column: [float(cutpoint)]})
    if axis == "x":
        return alt.Chart(frame).mark_rule(
            color="#333333", strokeWidth=1, strokeDash=[4, 3], opacity=0.8,
        ).encode(x=alt.X(f"{column}:Q"))
    return alt.Chart(frame).mark_rule(
        color="#333333", strokeWidth=1, strokeDash=[4, 3], opacity=0.8,
    ).encode(y=alt.Y(f"{column}:Q"))


def _plot_btree_partition_1d(model, index, name, column, grid):
    """A step plot of a one-column btree term's fitted contribution."""
    from mboostrs.gamboost import _btree_partition_cutpoints, _btree_term_effect

    training = np.asarray(model._data_array_for(column), dtype=np.float64)
    low, high = float(np.min(training)), float(np.max(training))
    values = np.linspace(low, high, int(grid))
    effect = _btree_term_effect(model, index, {column: values})
    frame = pl.DataFrame({column: values, "value": effect})
    line = alt.Chart(frame).mark_line(
        interpolate="step-after", strokeWidth=2.0, color="#4e79a7",
    ).encode(
        x=alt.X(f"{column}:Q", title=column),
        y=alt.Y("value:Q", title="Effect"),
        tooltip=[alt.Tooltip(f"{column}:Q", format=".4g", title=column),
                 alt.Tooltip("value:Q", format=".4g", title="Effect")],
    )
    layers = [line, _zero_rule("y")]
    for cutpoint in _btree_partition_cutpoints(model, name).get(column, ()):
        layers.append(_btree_split_rule(column, cutpoint, "x"))
    return alt.layer(*layers).properties(
        title=f"Btree partition: {column}", width=WIDTH, height=HEIGHT,
    ).interactive()


def plot_btree_partition(model, term=None, grid=60, data=None, at=None):
    """Plot the fitted partition of a btree term.

    A heatmap of the term's own fitted contribution over the two columns it
    splits on -- rectangle marks coloured by the contribution, with the
    ``btree_splits()`` cutpoints drawn as dashed rules. When ``data`` is given
    the training rows are overlaid as points. A one-column term falls back to a
    step plot of its contribution. A term over more than two columns needs
    ``at`` to fix all but two of them, which are then plotted.

    The value plotted is the term's partial prediction (its fitted
    contribution), not the full model's. :func:`btree_partition_table` returns
    the same partition as a table.

    Parameters
    ----------
    model : GLMBoostModel or GAMBoostModel
        A fit whose ``btree_splits()`` records a tree.
    term : str or int, optional
        Which btree term to plot; the only btree term is picked when there is
        exactly one.
    grid : int
        Number of grid points per axis (at least 2). The btree cutpoints are
        added to the grid so the rectangles align with the splits.
    data : DataFrame or dict, optional
        Training rows to overlay on a two-column heatmap.
    at : dict, optional
        Required for a term over more than two columns: ``{column: value}``
        fixing every column but the two to plot.

    Returns
    -------
    alt.Chart
    """
    if not isinstance(grid, (int, np.integer)) or int(grid) < 2:
        raise ValueError(f"grid must be an integer of at least 2, got {grid!r}")
    grid = int(grid)

    from mboostrs.gamboost import (
        _btree_partition_cutpoints,
        _btree_term_effect,
        _resolve_btree_learner,
    )

    index, name, spec = _resolve_btree_learner(model, term)
    columns = list(spec.names)
    if not columns:
        raise ValueError(f"btree term {name!r} has no columns to plot")

    if len(columns) == 1:
        return _plot_btree_partition_1d(model, index, name, columns[0], grid)

    if len(columns) == 2:
        xcol, ycol = columns
        fixed = {}
    else:
        if at is None:
            raise ValueError(
                f"btree term {name!r} splits on {len(columns)} columns "
                f"{columns}; pass at={{other_column: value, ...}} fixing all "
                "but two of them, which are then plotted"
            )
        if not isinstance(at, dict):
            raise ValueError(
                "at must be a mapping of column name to value, got "
                f"{type(at).__name__}"
            )
        unknown = [key for key in at if key not in columns]
        if unknown:
            raise ValueError(
                f"at names column(s) {unknown}, which btree term {name!r} does "
                f"not split on; its columns are {columns}"
            )
        plotted = [column for column in columns if column not in at]
        if len(plotted) != 2:
            raise ValueError(
                f"at must fix all but two of {name!r}'s {len(columns)} columns "
                f"{columns}; it leaves {plotted} to plot"
            )
        xcol, ycol = plotted
        fixed = {column: float(value) for column, value in at.items()}

    cutpoints = _btree_partition_cutpoints(model, name)
    x_train = np.asarray(model._data_array_for(xcol), dtype=np.float64)
    y_train = np.asarray(model._data_array_for(ycol), dtype=np.float64)
    xlow, xhigh = float(np.min(x_train)), float(np.max(x_train))
    ylow, yhigh = float(np.min(y_train)), float(np.max(y_train))
    xedges = np.unique(np.concatenate([
        np.linspace(xlow, xhigh, grid + 1),
        [point for point in cutpoints.get(xcol, ()) if xlow < point < xhigh],
    ]))
    yedges = np.unique(np.concatenate([
        np.linspace(ylow, yhigh, grid + 1),
        [point for point in cutpoints.get(ycol, ()) if ylow < point < yhigh],
    ]))
    xcentres = 0.5 * (xedges[:-1] + xedges[1:])
    ycentres = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xcentres, ycentres)
    varying = {xcol: xx.ravel(), ycol: yy.ravel()}
    for column, value in fixed.items():
        varying[column] = np.full(xx.size, value)
    effect = _btree_term_effect(model, index, varying)

    x_lower, x_upper, y_lower, y_upper, values = [], [], [], [], []
    position = 0
    for j in range(len(ycentres)):
        for i in range(len(xcentres)):
            x_lower.append(float(xedges[i]))
            x_upper.append(float(xedges[i + 1]))
            y_lower.append(float(yedges[j]))
            y_upper.append(float(yedges[j + 1]))
            values.append(float(effect[position]))
            position += 1
    x_end = f"{xcol}_end"
    y_end = f"{ycol}_end"
    frame = pl.DataFrame({
        xcol: x_lower, x_end: x_upper,
        ycol: y_lower, y_end: y_upper,
        "value": values,
    })

    heat = alt.Chart(frame).mark_rect().encode(
        x=alt.X(f"{xcol}:Q", title=xcol),
        x2=alt.X2(f"{x_end}:Q"),
        y=alt.Y(f"{ycol}:Q", title=ycol),
        y2=alt.Y2(f"{y_end}:Q"),
        color=alt.Color("value:Q", title="Effect"),
        tooltip=[alt.Tooltip(f"{xcol}:Q", format=".4g", title=xcol),
                 alt.Tooltip(f"{x_end}:Q", format=".4g", title=f"{xcol} to"),
                 alt.Tooltip(f"{ycol}:Q", format=".4g", title=ycol),
                 alt.Tooltip(f"{y_end}:Q", format=".4g", title=f"{ycol} to"),
                 alt.Tooltip("value:Q", format=".4g", title="Effect")],
    )
    layers = [heat]
    for cutpoint in cutpoints.get(xcol, ()):
        layers.append(_btree_split_rule(xcol, cutpoint, "x"))
    for cutpoint in cutpoints.get(ycol, ()):
        layers.append(_btree_split_rule(ycol, cutpoint, "y"))
    if data is not None:
        points = _btree_points_frame(data, xcol, ycol)
        layers.append(alt.Chart(points).mark_circle(
            size=9, opacity=0.4, color="#222222",
        ).encode(
            x=alt.X(f"{xcol}:Q"), y=alt.Y(f"{ycol}:Q"),
        ))
    return alt.layer(*layers).properties(
        title=f"Btree partition: {name}", width=WIDTH, height=HEIGHT,
    ).interactive()


def btree_partition_table(model, term=None):
    """The fitted btree partition as a Polars frame, one row per region.

    The non-plotting companion to :func:`plot_btree_partition`. The
    implementation lives in :mod:`mboostrs.gamboost` so it does not need
    Altair; this wrapper keeps it importable from either module.
    """
    from mboostrs.gamboost import btree_partition_table as _table

    return _table(model, term=term)


# ---------------------------------------------------------------------------
# LSS
# ---------------------------------------------------------------------------

def plot_lss(model, parameter=None, n_points=100):
    """Plot partial effects for each distribution parameter of an LSS model.

    Parameters
    ----------
    model : LSSModel
        Fitted gamboostLSS or glmboostLSS model.
    parameter : str or int, optional
        Which parameter to plot. If None, all parameters are shown.
    n_points : int
        Number of grid points for smooth effects.

    Returns
    -------
    alt.Chart | alt.ConcatChart
    """
    param_names = model.param_names
    if parameter is not None:
        k = model._param_index(parameter)
        param_names = [param_names[k]]

    if len(param_names) == 1:
        return _plot_lss_parameter(model, param_names[0], n_points).interactive()

    charts = [_plot_lss_parameter(model, p, n_points) for p in param_names]
    n = len(param_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    rows_concat = []
    for r in range(nrows):
        row_charts = charts[r * ncols:(r + 1) * ncols]
        if len(row_charts) < ncols:
            blank = alt.Chart().mark_point().properties(width=280, height=200)
            row_charts.append(blank)
        rows_concat.append(alt.hconcat(*row_charts))
    return alt.vconcat(*rows_concat)


def _plot_lss_parameter(model, pname, n_points):
    k = model._param_index(pname)
    p = model._params[k]

    if "coefs" in p and "coefs_flat" not in p:
        return _plot_lss_gam(model, k, p)
    return _plot_lss_linear(model, k, p)


def _plot_lss_gam(model, k, p):
    """Partial effects for a gamboostLSS parameter.

    gamboostLSS parameters store one coefficient vector per boosting step
    ("coefs"), shaped by the selected learner's design, so the scalar-coefficient
    arithmetic of `_plot_lss_linear` does not apply. `partial_effects` replays
    the per-learner updates through the learner store and returns each learner's
    effect on the training observations; adding them to the offset reproduces the
    link predictor exactly.
    """
    pname = model.param_names[k]
    effects = model.partial_effects(pname, newdata=None)
    rows = []
    for pred_name, (x, effect) in effects.items():
        x = np.asarray(x, dtype=np.float64)
        effect = np.asarray(effect, dtype=np.float64)
        order = np.argsort(x)
        for xi, yi in zip(x[order], effect[order], strict=False):
            rows.append({"x": float(xi), "y": float(yi), "predictor": str(pred_name)})
    df = pl.DataFrame(rows)

    if df.height == 0:
        eta = model.predict(parameter=pname, type="link")
        df = pl.DataFrame({"index": np.arange(len(eta)), "eta": eta})
        return alt.Chart(df).mark_circle(opacity=0.3, size=12).encode(
            x=alt.X("index:Q", title="Observation"),
            y=alt.Y("eta:Q", title="Linear predictor"),
        ).properties(width=WIDTH, height=HEIGHT)

    line = alt.Chart(df).mark_line(strokeWidth=1.5).encode(
        x=alt.X("x:Q", title=""),
        y=alt.Y("y:Q", title="Partial effect"),
        color=alt.Color("predictor:N", title="Predictor"),
        tooltip=[alt.Tooltip("x:Q", format=".4f"),
                 alt.Tooltip("y:Q", format=".4f"),
                 alt.Tooltip("predictor:N")],
    )
    return line.properties(width=WIDTH, height=HEIGHT)


def _plot_lss_linear(model, k, p):
    names = p.get("predictor_names", [])

    # Fallback: no named predictors -> one-column predictor (plot raw eta)
    if not names:
        eta = model.predict(parameter=model.param_names[k], type="link")
        df = pl.DataFrame({"index": np.arange(len(eta)), "eta": eta})
        return alt.Chart(df).mark_circle(opacity=0.3, size=12).encode(
            x=alt.X("index:Q", title="Observation"),
            y=alt.Y("eta:Q", title="Linear predictor"),
        ).properties(width=WIDTH, height=HEIGHT)

    # Accumulate per-predictor coefficients across boosting steps
    nu = model._nu
    coefs_by_pred = {}
    for step, sel in enumerate(p["selected"]):
        if sel == 0:
            continue
        pred_name = names[sel - 1]
        coefs_by_pred.setdefault(pred_name, 0.0)
        coefs_by_pred[pred_name] += nu * p["coefs_flat"][step]

    # Build line data and collect raw x values for rug
    data = model._data
    rows = []
    rug_data = {}
    for pred_name, coef_val in coefs_by_pred.items():
        x = np.asarray(getattr(data, pred_name, data[pred_name]), dtype=np.float64)
        order = np.argsort(x)
        for xi in x[order]:
            rows.append({"x": xi, "y": coef_val * xi, "predictor": pred_name})
        rug_data[pred_name] = x
    df = pl.DataFrame(rows)

    line = alt.Chart(df).mark_line(strokeWidth=1.5).encode(
        x=alt.X("x:Q", title=""),
        y=alt.Y("y:Q", title="Partial effect"),
        color=alt.Color("predictor:N", title="Predictor"),
        tooltip=[alt.Tooltip("x:Q", format=".4f"),
                 alt.Tooltip("y:Q", format=".4f"),
                 alt.Tooltip("predictor:N")],
    )

    # Single predictor: layer with rug + zero reference line
    if len(rug_data) == 1 and len(names) == 1:
        rug_x = list(rug_data.values())[0]
        rug = _rug(rug_x, HEIGHT)
        zero = _zero_rule("y")
        return alt.layer(line, rug, zero).resolve_scale(y="shared").properties(
            width=WIDTH, height=HEIGHT
        )

    return line.properties(
        width=WIDTH, height=HEIGHT
    )


def _plot_lss_bbs(model, k, p, n_points):
    learners = p.get("learners", [])
    if not learners:
        eta = model.predict(parameter=model.param_names[k], type="link")
        df = pl.DataFrame({"index": np.arange(len(eta)), "eta": eta})
        return alt.Chart(df).mark_circle(opacity=0.3, size=12).encode(
            x=alt.X("index:Q", title="Observation"),
            y=alt.Y("eta:Q", title="Linear predictor"),
        ).properties(width=WIDTH, height=HEIGHT)

    selected = p["selected"]
    coefs_list = p["coefs"]
    nu = model._nu
    data = model._data
    mmax = len(selected)

    var_names_unique = list(dict.fromkeys(ln.name for ln in learners))

    try:
        from mboostrs._rust import bbs_raw_design
    except ImportError:
        rows = []
        for var in var_names_unique:
            x_raw = np.asarray(getattr(data, var, data[var]), dtype=np.float64)
            order = np.argsort(x_raw)
            for xi in x_raw[order]:
                rows.append({"x": xi, "y": 0.0, "variable": var})
        df = pl.DataFrame(rows)
        return alt.Chart(df).mark_circle(opacity=0.3, size=12).encode(
            x="x:Q", y="y:Q", color=alt.Color("variable:N"),
        ).properties(width=WIDTH, height=HEIGHT)

    # Initialize per-variable effect accumulators
    n_obs = len(np.asarray(
        getattr(data, learners[0].name, data[learners[0].name]), dtype=np.float64
    ))
    per_var = {var: np.zeros(n_obs) for var in var_names_unique}

    # Build B-spline basis matrices (cached per learner index)
    bases_cache: dict[int, np.ndarray] = {}
    for li, ln in enumerate(learners):
        x_train = np.asarray(getattr(data, ln.name, data[ln.name]), dtype=np.float64)
        bk = (float(x_train.min()), float(x_train.max()))
        flat, pcols = bbs_raw_design(
            np.ascontiguousarray(x_train, dtype=np.float64),
            bk, int(ln.knots), int(ln.degree),
        )
        bases_cache[li] = np.asarray(flat, dtype=np.float64).reshape(len(x_train), pcols)

    # Accumulate partial effects across all boosting steps
    for s in range(mmax):
        li = selected[s]
        per_var[learners[li].name] += nu * (bases_cache[li] @ coefs_list[s])

    rows = []
    rug_data = {}
    for var in var_names_unique:
        effect = per_var[var]
        x_raw = np.asarray(getattr(data, var, data[var]), dtype=np.float64)
        order = np.argsort(x_raw)
        for xi, yi in zip(x_raw[order], effect[order], strict=True):
            rows.append({"x": xi, "y": yi, "variable": var})
        rug_data[var] = x_raw
    df = pl.DataFrame(rows)

    line = alt.Chart(df).mark_line(strokeWidth=1.5).encode(
        x=alt.X("x:Q", title=""),
        y=alt.Y("y:Q", title="Smooth effect"),
        color=alt.Color("variable:N", title="Variable"),
        tooltip=[alt.Tooltip("x:Q", format=".4f"),
                 alt.Tooltip("y:Q", format=".4f"),
                 alt.Tooltip("variable:N")],
    )

    if len(var_names_unique) == 1:
        rug_x = rug_data[var_names_unique[0]]
        rug = _rug(rug_x, HEIGHT)
        zero = _zero_rule("y")
        return alt.layer(line, rug, zero).resolve_scale(y="shared").properties(
            width=WIDTH, height=HEIGHT
        )

    return line.properties(
        width=WIDTH, height=HEIGHT
    )


# ---------------------------------------------------------------------------
# LSS risk
# ---------------------------------------------------------------------------

def plot_lss_selected(model):
    """Plot selection frequency per base learner, faceted by distribution
    parameter.

    The standard-model `plot_selected` reads `model._selected` and
    `model._learner_names`, which an `LSSModel` does not have — it carries one
    selection sequence and one name list per parameter. This is the LSS
    counterpart, and is what `LSSModel.plot_selected()` calls.

    Parameters
    ----------
    model : LSSModel
        Fitted distributional model.

    Returns
    -------
    alt.Chart
    """
    rows = []
    for index_q, (name, param) in enumerate(zip(model.param_names, model._params, strict=False)):
        selected = list(param["selected"])
        names = model._learner_labels(index_q)
        # A glmboostLSS parameter (one scalar per step, `coefs_flat`) has an
        # intercept learner at index 0; a gamboostLSS parameter names every
        # learner already. Testing `kind` here once prepended the intercept
        # to gamboostLSS names too, so every bar carried its neighbour's label.
        if "coefs_flat" in param:
            names = ["(Intercept)", *names]
        total = len(selected)
        for index, learner in enumerate(names):
            rows.append({
                "parameter": str(name),
                "learner": str(learner),
                "frequency": selected.count(index) / total if total else 0.0,
            })

    frame = pl.DataFrame(rows)
    n_learners = frame["learner"].n_unique()
    return alt.Chart(frame).mark_bar(color=PALETTE[0]).encode(
        x=alt.X("frequency:Q", title="Selection frequency"),
        y=alt.Y("learner:N", title=None, sort="-x",
                scale=alt.Scale(paddingInner=0.38, paddingOuter=0.24)),
        row=alt.Row("parameter:N", title=None),
        tooltip=[alt.Tooltip("parameter:N", title="Parameter"),
                 alt.Tooltip("learner:N", title="Learner"),
                 alt.Tooltip("frequency:Q", format=".3f", title="Frequency")],
    ).properties(
        title="Selection frequency per base-learner",
        width=WIDTH, height=min(HEIGHT, n_learners * 30 + 30),
    )


def plot_lss_risk(model, parameter=None):
    """Plot risk paths for an LSS model.

    Parameters
    ----------
    model : LSSModel
        Fitted LSS model.
    parameter : str or int, optional
        Not used; reserved for future per-parameter decomposition.

    Returns
    -------
    alt.Chart
    """
    risk = model.risk_
    df = pl.DataFrame({"iteration": np.arange(len(risk)), "risk": risk})
    return alt.Chart(df).mark_line(
        strokeWidth=1.8, color="#4e79a7"
    ).encode(
        x=alt.X("iteration:Q", title="Iteration"),
        y=alt.Y("risk:Q", title="Risk", scale=alt.Scale(zero=False)),
    ).properties(
        title="LSS risk path", width=WIDTH, height=HEIGHT
    )


def plot_distribution_density(model, newdata=None, mstop=None, grid=None,
                              n_grid=200, weights=None):
    """Overlay the observed marginal distribution and fitted conditional mixture.

    For continuous responses the observed line is a kernel-density estimate;
    for discrete responses it is the empirical mass. The fitted line averages
    each row's complete predicted distribution, so varying scale and shape are
    retained. Pass held-out ``newdata`` for a predictive comparison.
    """
    from mboostrs.diagnostics import density_overlay_data  # noqa: PLC0415

    frame = pl.DataFrame(density_overlay_data(
        model, newdata=newdata, mstop=mstop, grid=grid, n_grid=n_grid,
        weights=weights, output="records",
    ))
    kinds = set(frame["kind"].to_list())
    kind = "mass" if "mass" in kinds else "density"
    curve_frame = frame.filter(pl.col("kind") == kind)
    line = alt.Chart(curve_frame).mark_line(strokeWidth=2, point=kind == "mass").encode(
        x=alt.X("x:Q", title="Response"),
        y=alt.Y("value:Q", title="Probability mass" if kind == "mass" else "Density",
                scale=alt.Scale(zero=True)),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["observed", "fitted"],
                                        range=[PALETTE[1], PALETTE[0]])),
        strokeDash=alt.StrokeDash("series:N", title=None,
                                  scale=alt.Scale(domain=["observed", "fitted"],
                                                  range=[[5, 3], [1, 0]])),
        tooltip=[alt.Tooltip("series:N", title="Curve"),
                 alt.Tooltip("x:Q", title="Response", format=".4g"),
                 alt.Tooltip("value:Q", title=kind.title(), format=".4g")],
    )
    line = line.properties(
        title="Observed and average fitted distribution", width=WIDTH, height=HEIGHT
    )
    if "atom" not in kinds:
        return line
    atom_frame = frame.filter(pl.col("kind") == "atom")
    atom = alt.Chart(atom_frame).mark_bar().encode(
        x=alt.X("series:N", title=None),
        y=alt.Y("value:Q", title="Probability mass", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["observed", "fitted"],
                                        range=[PALETTE[1], PALETTE[0]])),
        tooltip=[alt.Tooltip("series:N", title="Mass"),
                 alt.Tooltip("value:Q", title="Probability", format=".4g")],
    ).properties(title=f"Atom at {atom_frame['x'][0]:g}", width=WIDTH // 3, height=HEIGHT)
    return alt.hconcat(line, atom)


def plot_centile_fan(model, xvar, probs=(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
                     split_by=None, newdata=None, mstop=None):
    """Plot nested fitted centile bands against a numeric covariate.

    ``split_by`` facets the same fitted model by a categorical column; it does
    not fit group-specific models. Bands describe the conditional distribution
    estimated for each row and are not confidence intervals for the centile
    curves.
    """
    from mboostrs.diagnostics import centile_fan_data  # noqa: PLC0415
    from mboostrs.gamlss import _prob_col  # noqa: PLC0415

    probabilities = np.unique(np.asarray(probs, dtype=np.float64))
    frame = pl.DataFrame(centile_fan_data(
        model, xvar=xvar, probs=probabilities, split_by=split_by,
        newdata=newdata, mstop=mstop, output="records",
    ))
    lower = probabilities[probabilities < 0.5]
    band_rows = []
    for probability in lower:
        upper = 1.0 - probability
        coverage = upper - probability
        lo_col, hi_col = _prob_col(float(probability)), _prob_col(float(upper))
        for row in frame.select("x", "split", lo_col, hi_col).iter_rows(named=True):
            band_rows.append({
                "x": row["x"], "split": row["split"],
                "lower": row[lo_col], "upper": row[hi_col],
                "coverage": f"{100 * coverage:g}%",
                "coverage_value": coverage,
            })
    bands = pl.DataFrame(band_rows)
    median_column = _prob_col(0.5)

    def panel(panel_frame, panel_bands, title):
        area = alt.Chart(panel_bands).mark_area(color=PALETTE[0]).encode(
            x=alt.X("x:Q", title=str(xvar)),
            y=alt.Y("lower:Q", title="Response", scale=alt.Scale(zero=False)),
            y2="upper:Q",
            detail="coverage:N",
            order=alt.Order("coverage_value:Q", sort="descending"),
            opacity=alt.Opacity("coverage_value:Q", legend=None,
                                scale=alt.Scale(domain=[0, 1], range=[0.42, 0.10])),
            tooltip=[alt.Tooltip("coverage:N", title="Central band"),
                     alt.Tooltip("x:Q", title=str(xvar), format=".4g"),
                     alt.Tooltip("lower:Q", format=".4g"),
                     alt.Tooltip("upper:Q", format=".4g")],
        )
        median = alt.Chart(panel_frame).mark_line(
            color=PALETTE[0], strokeWidth=2
        ).encode(
            x=alt.X("x:Q", title=str(xvar)),
            y=alt.Y(f"{median_column}:Q", title="Response",
                    scale=alt.Scale(zero=False)),
        )
        observed = alt.Chart(panel_frame).mark_circle(
            color="#555555", opacity=0.35, size=20
        ).encode(
            x=alt.X("x:Q", title=str(xvar)),
            y=alt.Y("observed:Q", title="Response", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("x:Q", title=str(xvar), format=".4g"),
                     alt.Tooltip("observed:Q", format=".4g")],
        )
        return alt.layer(area, observed, median).properties(
            title=title, width=WIDTH, height=HEIGHT
        )

    if split_by is None:
        return panel(frame, bands, "Fitted distribution centiles")
    charts = []
    for group in frame["split"].unique(maintain_order=True).to_list():
        charts.append(panel(
            frame.filter(pl.col("split") == group),
            bands.filter(pl.col("split") == group),
            f"{split_by} = {group}",
        ))
    return alt.hconcat(*charts).properties(title="Fitted distribution centiles")


def plot_deviance_profile(model, xvar=None, bins=10, newdata=None, mstop=None,
                          weights=None):
    """Plot binned joint deviance along a covariate or observation order."""
    from mboostrs.diagnostics import deviance_profile  # noqa: PLC0415

    frame = pl.DataFrame(deviance_profile(
        model, xvar=xvar, bins=bins, newdata=newdata, mstop=mstop,
        weights=weights, output="records",
    ))
    variable = "Observation order" if xvar is None else str(xvar)
    line = alt.Chart(frame).mark_line(point=True, color=PALETTE[0], strokeWidth=2).encode(
        x=alt.X("x_mid:Q", title=variable),
        y=alt.Y("mean_deviance:Q", title="Mean joint deviance",
                scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("interval:Q", title="Bin"),
                 alt.Tooltip("x_min:Q", format=".4g"),
                 alt.Tooltip("x_max:Q", format=".4g"),
                 alt.Tooltip("n:Q"),
                 alt.Tooltip("mean_deviance:Q", format=".4g"),
                 alt.Tooltip("q90_deviance:Q", format=".4g")],
    )
    q90 = alt.Chart(frame).mark_line(
        color=PALETTE[1], strokeDash=[5, 3], strokeWidth=1.5
    ).encode(x="x_mid:Q", y="q90_deviance:Q")
    return alt.layer(line, q90).properties(
        title="Joint deviance profile (mean; dashed 90th percentile)",
        width=WIDTH, height=HEIGHT,
    )


# ---------------------------------------------------------------------------
# Multinomial
# ---------------------------------------------------------------------------

def plot_multinomial(model):
    """Plot predicted class probabilities for a Multinomial model.

    Parameters
    ----------
    model : GAMBoostModel
        Fitted model with Multinomial family.

    Returns
    -------
    alt.Chart
    """
    from mboostrs.families import Multinomial as _Multinomial

    if not isinstance(model._family, _Multinomial):
        raise TypeError("plot_multinomial requires a model fitted with Multinomial family")

    prob = model.predict(type="response")
    K = model._family.n_classes
    n_classes = K if K is not None else prob.shape[1]
    class_names = getattr(model._family, "class_names", None)
    if class_names is None:
        response_name = getattr(model, "_response_name", None)
        if response_name is not None and response_name in model._data:
            response = np.asarray(model._data[response_name])
            if response.size == len(prob):
                class_names = [str(c) for c in np.unique(response)]
    if class_names is None:
        class_names = [f"class {k}" for k in range(n_classes)]

    x_data = None
    learner_names = model._learner_names
    if len(learner_names) > 0:
        name = learner_names[0].split(",")[0].strip()
        if name in model._data:
            x_data = np.asarray(model._data[name], dtype=np.float64)

    rows = []
    for k in range(n_classes):
        lab = class_names[k] if k < len(class_names) else f"class {k}"
        if x_data is not None:
            o = np.argsort(x_data)
            for i in range(len(o)):
                rows.append({"x": float(x_data[o[i]]), "prob": float(prob[o[i], k]), "class": lab})
        else:
            o = np.argsort(prob[:, k])
            for i in range(len(o)):
                rows.append({"x": float(i), "prob": float(prob[o[i], k]), "class": lab})

    df = pl.DataFrame(rows)

    return alt.Chart(df).mark_line(strokeWidth=1.5).encode(
        x=alt.X("x:Q", title="x" if x_data is not None else "Observation (sorted)"),
        y=alt.Y("prob:Q", title="P(class | x)"),
        color=alt.Color("class:N", title="Class"),
        tooltip=[alt.Tooltip("x:Q", format=".4f"),
                 alt.Tooltip("prob:Q", format=".4f"),
                 alt.Tooltip("class:N", title="Class")],
    ).properties(
        title="Class probabilities", width=WIDTH, height=HEIGHT,
    ).interactive()


# ---------------------------------------------------------------------------
# LSS CV risk
# ---------------------------------------------------------------------------

def plot_lss_cvrisk(cvrisk_result, model=None):
    """Plot cross-validated risk for an LSS model.

    Delegates to :func:`plot_cvrisk`.

    Parameters
    ----------
    cvrisk_result : CvRiskResult
        Result from cvrisk_glmboostLSS().
    model : LSSModel, optional
        Not used currently.

    Returns
    -------
    alt.LayerChart
    """
    if not hasattr(cvrisk_result, "to_frame"):
        return plot_cvrisk(cvrisk_result)

    frame = cvrisk_result.to_frame()
    if hasattr(frame, "to_dicts"):
        records = frame.to_dicts()
    elif hasattr(frame, "to_dict"):
        records = frame.to_dict(orient="records")
    else:
        records = list(frame)
    if not records or "mstop" not in records[0]:
        return plot_cvrisk(cvrisk_result)
    for row in records:
        row["mstop_label"] = ", ".join(str(int(value)) for value in row["mstop"])

    fold_lines = alt.Chart(alt.Data(values=records)).mark_line(
        strokeWidth=0.5, color="gray"
    ).encode(
        x=alt.X("grid_index:Q", title="Grid point"),
        y=alt.Y("risk:Q", title="Risk", scale=alt.Scale(zero=False)),
        detail="fold:N",
        opacity=alt.value(0.25),
        tooltip=[
            alt.Tooltip("grid_index:Q", format="d", title="Grid"),
            alt.Tooltip("mstop_label:N", title="mstop"),
            alt.Tooltip("risk:Q", format=".6f"),
            alt.Tooltip("fold:N"),
        ],
    )

    aggregate_records = list({row["grid_index"]: row for row in records}.values())
    aggregate_records.sort(key=lambda row: row["grid_index"])
    agg_line = alt.Chart(alt.Data(values=aggregate_records)).mark_line(
        strokeWidth=2.5, color="black"
    ).encode(
        x="grid_index:Q",
        y=alt.Y("aggregate:Q", title="Risk", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("grid_index:Q", format="d", title="Grid"),
            alt.Tooltip("mstop_label:N", title="mstop"),
            alt.Tooltip("aggregate:Q", format=".6f", title="Mean risk"),
        ],
    )

    # The across-fold spread, not only its mean: a grid search that varies by
    # less than one standard error over its whole surface has chosen its
    # optimum from the fold split rather than from the data.
    standard_error = np.asarray(cvrisk_result.standard_error)
    band_layers = []
    if standard_error.size == len(aggregate_records):
        for index, row in enumerate(aggregate_records):
            row["se_lower"] = float(row["aggregate"] - standard_error[index])
            row["se_upper"] = float(row["aggregate"] + standard_error[index])
        band_layers = [
            alt.Chart(alt.Data(values=aggregate_records)).mark_area(
                opacity=0.18, color="#4e79a7",
            ).encode(
                x=alt.X("grid_index:Q", title="Grid point"),
                y=alt.Y("se_lower:Q", title="Risk",
                        scale=alt.Scale(zero=False)),
                y2=alt.Y2("se_upper:Q"),
            )
        ]

    optimal_records = [row for row in aggregate_records if row["optimal"]]
    if not optimal_records:
        optimal_records = [
            aggregate_records[int(np.argmin(np.asarray(cvrisk_result.aggregate)))]
        ]
    opt_index = float(optimal_records[0]["grid_index"])
    opt_label = f"mstop = {optimal_records[0]['mstop_label']}"

    rule = _default_rule(opt_index)
    label = _rule_label(opt_index, opt_label)

    # The one-standard-error point, when it is not the minimiser.
    one_se = list(cvrisk_result.mstop_1se)
    se_layers = []
    matching = [row for row in aggregate_records
                if list(row["mstop"]) == one_se]
    if matching and float(matching[0]["grid_index"]) != opt_index:
        se_index = float(matching[0]["grid_index"])
        se_layers = [
            alt.Chart(pl.DataFrame({"grid_index": [se_index]})).mark_rule(
                strokeDash=[2, 3], strokeWidth=1.5, color="#59a14f",
            ).encode(x="grid_index:Q"),
            _rule_label(se_index, f"1-SE = {matching[0]['mstop_label']}",
                        color="#59a14f", dy=14),
        ]

    return alt.layer(
        *band_layers, fold_lines, agg_line, rule, label, *se_layers
    ).properties(
        title="Cross-validated LSS risk",
        width=WIDTH,
        height=HEIGHT,
    ).interactive()


# ---------------------------------------------------------------------------
# Stability selection
# ---------------------------------------------------------------------------

def plot_stabsel(stabsel_result):
    """Plot stability selection probabilities as a horizontal bar chart.

    Parameters
    ----------
    stabsel_result : StabSelResult
        Result of ``mboostrs.stabsel()``.

    Returns
    -------
    alt.LayerChart
    """
    names = stabsel_result.learner_names
    phat = np.asarray(stabsel_result.phat)
    if phat.ndim == 2:
        phat = phat[:, -1]
    cutoff = stabsel_result.cutoff
    selected = set(stabsel_result.selected)

    order = np.argsort(phat)[::-1]
    sorted_names = [names[int(i)] for i in order]
    sorted_phat = phat[order]
    sorted_selected = [
        int(i) in selected or names[int(i)] in selected
        for i in order
    ]

    df = pl.DataFrame({
        "learner": sorted_names,
        "probability": sorted_phat,
        "stable": sorted_selected,
    })

    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("probability:Q", title="Selection probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("learner:N", title=None, sort="-x",
                scale=alt.Scale(paddingInner=0.38, paddingOuter=0.24)),
        color=alt.condition(
            alt.datum.stable,
            alt.value("#4e79a7"),
            alt.value("#cccccc"),
        ),
        tooltip=[alt.Tooltip("learner:N", title="Learner"),
                 alt.Tooltip("probability:Q", format=".3f", title="Probability"),
                 alt.Tooltip("stable:N", title="Selected")],
    )

    cutoff_line = alt.Chart(pl.DataFrame({"cutoff": [cutoff]})).mark_rule(
        color="#e15759", strokeDash=[4, 3], strokeWidth=1.2,
    ).encode(x="cutoff:Q")

    cutoff_label = alt.Chart(pl.DataFrame({"x": [cutoff], "label": [f"cutoff = {cutoff:.2f}"], "y": [1]})).mark_text(
        color="#e15759", fontSize=10, dx=5, dy=-5
    ).encode(x="x:Q", text="label:N")

    return alt.layer(bars, cutoff_line, cutoff_label).properties(
        title="Stability selection",
        width=WIDTH,
        height=min(HEIGHT, len(names) * 30 + 30),
    ).resolve_scale(y="shared")


def plot_specification_curve(curve):
    """Plot each specification's out-of-fold gap to the selected one, with a paired-SE interval.

    Parameters
    ----------
    curve : SpecificationCurve
        Result of ``mboostrs.specification_curve()``.

    Returns
    -------
    alt.LayerChart
        One row per specification, ranked by out-of-fold score. Each point is
        the gap to the selected specification (its score minus the selected
        one's) and the bar spans two paired fold standard errors either side.
        Colour shows whether the specification was declared before scoring,
        raised by observed results, or is the selected one. The dashed line at
        zero is the selected specification.
    """
    rows = pl.DataFrame([
        {"rank": int(row["rank"]), "specification": row["specification"],
         "kind": "selected" if row["role"] == "selected" else row["origin"],
         "family": row["family"], "rhs": row["rhs"],
         "score": float(row["out-of-fold score"]),
         "gap": float(row["gap vs selected"]),
         "gap_se": None if row["gap se"] is None else float(row["gap se"])}
        for row in curve.records
    ], schema_overrides={"gap_se": pl.Float64})
    rows = rows.with_columns(
        (pl.col("gap") - 2 * pl.col("gap_se")).alias("lower"),
        (pl.col("gap") + 2 * pl.col("gap_se")).alias("upper"))
    tooltip = [alt.Tooltip("specification:N", title="Specification"),
               alt.Tooltip("family:N", title="Family"),
               alt.Tooltip("rhs:N", title="Right-hand side"),
               alt.Tooltip("kind:N", title="Status"),
               alt.Tooltip("score:Q", format=".4g", title="Out-of-fold score"),
               alt.Tooltip("gap:Q", format=".4g", title="Gap to selected"),
               alt.Tooltip("gap_se:Q", format=".4g", title="Paired SE of gap")]
    base = alt.Chart(rows).encode(
        y=alt.Y("rank:O", title="Specification, ranked by out-of-fold score",
                axis=alt.Axis(labels=False, ticks=False)),
    )
    interval = base.mark_rule(color="#8d8d8d", strokeWidth=1.2).encode(
        x=alt.X("lower:Q", title="Out-of-fold score minus selected score"),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=60, opacity=0.9).encode(
        x="gap:Q",
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["declared", "raised", "selected"],
                            range=[PALETTE[0], PALETTE[1], "#161616"]),
            legend=alt.Legend(title="Written down")),
        tooltip=tooltip,
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(
        color="#161616", strokeDash=[4, 3], strokeWidth=1).encode(x="x:Q")
    joint = curve.joint_test
    if joint.get("available"):
        test = f"joint test p = {joint['p_value']:.3f} (sign flips, folds reused)"
    else:
        test = "joint test not available"
    subtitle = f"{curve.label}; {test}"
    return alt.layer(interval, zero, points).properties(
        title={"text": "Specification curve",
               "subtitle": subtitle},
        width=WIDTH,
        height=max(HEIGHT, 16 * len(curve.records) + 60),
    )


# ---------------------------------------------------------------------------
# Diagnostic dashboard
# ---------------------------------------------------------------------------

WORM_MAX_POINTS = 400


def _thin_worm_panel(panel):
    """Thin a worm-plot panel to at most `WORM_MAX_POINTS` order statistics.

    A worm plots one point per observation against its theoretical quantile,
    so the points trace a monotone curve and adjacent ones overlap. Keeping
    every k-th, plus both endpoints, draws the same curve: the tails set the
    verdict and are preserved exactly. At 7294 observations the untinned panel
    is a 2.5 MB SVG that renders as a solid band.
    """
    if panel.height <= WORM_MAX_POINTS:
        return panel
    ordered = panel.sort("expected")
    deviation = ordered["deviation"].to_numpy()
    keep = np.unique(
        np.concatenate([
            np.linspace(0, ordered.height - 1, WORM_MAX_POINTS).round().astype(int),
            # Both ends of the worm, and the largest excursion in each
            # direction — the points a reader judges the fit on.
            [0, ordered.height - 1, int(deviation.argmin()), int(deviation.argmax())],
        ])
    )
    return ordered[keep.tolist()]


def plot_worm(model, xvar=None, n_intervals=4, newdata=None, mstop=None,
              randomized=True, random_state=None):
    """Worm plot(s): detrended QQ-plot of (randomized) quantile residuals.

    A worm plot shows ``deviation = sorted(residual) - theoretical_quantile``
    against ``theoretical_quantile`` within each bin of ``xvar`` (R's
    ``gamlss::wp()``). A flat cloud around zero indicates the fitted
    distribution matches the data in that bin; a curved or off-center cloud
    indicates a specific distributional mismatch (skew, kurtosis, location).

    Parameters
    ----------
    model : LSSModel (or any model with .quantile_residuals())
        Fitted model.
    xvar : str, optional
        Predictor to bin by. If None, one panel covering all observations.
    n_intervals : int
        Number of bins.

    Returns
    -------
    alt.Chart | alt.VConcatChart
    """
    from mboostrs.diagnostics import worm_data  # noqa: PLC0415

    df = worm_data(
        model, xvar=xvar, n_intervals=n_intervals, newdata=newdata,
        mstop=mstop, randomized=randomized, random_state=random_state,
    )
    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)

    bins = sorted(frame["bin"].unique().to_list())
    y_max = (
        max(0.5, float(np.abs(frame["deviation"].to_numpy()).max()) * 1.15)
        if len(frame)
        else 0.5
    )

    def _panel(bin_id):
        panel_df = _thin_worm_panel(frame.filter(pl.col("bin") == bin_id))
        points = alt.Chart(panel_df).mark_circle(opacity=0.6, size=30).encode(
            x=alt.X("expected:Q", title="Theoretical quantile"),
            y=alt.Y("deviation:Q", title="Deviation", scale=alt.Scale(domain=[-y_max, y_max])),
            tooltip=[alt.Tooltip("expected:Q", format=".3f"),
                     alt.Tooltip("deviation:Q", format=".3f")],
        )
        fit = alt.Chart(panel_df).transform_regression(
            "expected", "deviation", order=3,
        ).mark_line(color="#e15759", strokeWidth=1.5).encode(
            x="expected:Q", y="deviation:Q",
        )
        zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(
            color="#999999", strokeWidth=1, strokeDash=[4, 3]
        ).encode(y="y:Q")
        title = f"Bin {bin_id}" if xvar is not None or n_intervals > 1 else "Worm plot"
        return alt.layer(zero, points, fit).properties(
            title=title, width=WIDTH // 2, height=HEIGHT // 2
        )

    if len(bins) <= 1:
        bin_id = bins[0] if bins else 1
        return _panel(bin_id)

    ncols = min(3, len(bins))
    nrows = (len(bins) + ncols - 1) // ncols
    charts = [_panel(b) for b in bins]
    rows_concat = []
    for r in range(nrows):
        row_charts = charts[r * ncols:(r + 1) * ncols]
        if len(row_charts) < ncols:
            blank = alt.Chart().mark_point().properties(width=WIDTH // 2, height=HEIGHT // 2)
            row_charts.append(blank)
        rows_concat.append(alt.hconcat(*row_charts))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return alt.vconcat(*rows_concat).properties(title="Worm plot")


def plot_diagnostics(model):
    """2x2 diagnostic dashboard: risk path, selected frequencies,
    partial effects, and residual plot.

    Parameters
    ----------
    model : GLMBoostModel or GAMBoostModel
        Fitted model.

    Returns
    -------
    alt.VConcatChart
    """
    import warnings

    from mboostrs._dispatch import is_standard

    if not is_standard(model):
        raise TypeError(
            "plot_diagnostics needs a model with a single selection path. A "
            "distributional (LSS) model boosts one path per distribution "
            "parameter; use plot_lss_risk, plot_lss_selected, plot_lss and "
            "plot_worm, which each take the parameter into account."
        )

    risk = plot_risk(model).properties(width=WIDTH // 2, height=HEIGHT // 2)
    sel = plot_selected(model).properties(width=WIDTH // 2, height=HEIGHT // 2)

    fx = model.fitted_
    fx_df = pl.DataFrame({"index": np.arange(len(fx)), "fitted": fx})
    fx_chart = alt.Chart(fx_df).mark_circle(opacity=0.3, size=16).encode(
        x=alt.X("index:Q", title="Observation index"),
        y=alt.Y("fitted:Q", title="Fitted value"),
        tooltip=[alt.Tooltip("index:Q", format="d"),
                 alt.Tooltip("fitted:Q", format=".4f")],
    ).properties(title="Fitted values", width=WIDTH // 2, height=HEIGHT // 2).interactive()

    top = alt.hconcat(risk, sel)

    try:
        resid = model.residuals(type="response")
        res_df = pl.DataFrame({"residual": resid})
        res_hist = alt.Chart(res_df).mark_bar(opacity=0.8).encode(
            x=alt.X("residual:Q", title="Residual", bin=alt.Bin(maxbins=30)),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("residual:Q", bin=alt.Bin(maxbins=30), title="Residual"),
                     alt.Tooltip("count():Q", title="Count")],
        )

        # Kernel-density estimate overlay
        res_density = alt.Chart(res_df).transform_density(
            "residual", as_=["residual", "density"],
        ).mark_line(strokeWidth=1.5, color="#e15759").encode(
            x=alt.X("residual:Q"),
            y=alt.Y("density:Q", title="Density", scale=alt.Scale(zero=False)),
        )

        zero_line = _zero_rule("x")

        # Dual-axis: histogram on left y, density on right y
        res_chart = alt.layer(res_hist, res_density, zero_line).resolve_scale(
            y="independent"
        ).properties(
            title="Residual distribution", width=WIDTH // 2, height=HEIGHT // 2
        )
        bottom = alt.hconcat(fx_chart, res_chart)
    except NotImplementedError:
        bottom = fx_chart

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return alt.vconcat(top, bottom).properties(
            title="Diagnostic dashboard"
        )
