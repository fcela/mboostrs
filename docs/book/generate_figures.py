#!/usr/bin/env python3
"""Generate all methodology diagrams as Altair SVGs for the mboostrs book.

Run:  python3 docs/book/generate_figures.py
Output: docs/book/figures/*.svg

All diagrams use Altair. The script is called automatically by build-html.sh
and build-pdf.sh before the MyST build step.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

alt.data_transformers.disable_max_rows()

# -- palette -----------------------------------------------------------------
#
# Taken from the package's own chart theme rather than restated here. These 22
# figures used to declare six colours of their own and never set a font, so the
# book carried three visual systems: mermaid diagrams in DM Sans (matched on
# purpose, at the cost of base64-embedding two woff2 files as data URIs),
# charts the reader produces from the package in DM Sans with Tableau-10, and
# these figures in whatever generic sans-serif the renderer picked, with a
# different blue, orange, and green. Importing the theme also means a future
# change to it reaches these figures, which it previously could not.
from mboostrs.plot import PALETTE, theme  # noqa: E402

BLUE, ORANGE, RED, _TEAL, GREEN = PALETTE[:5]
GRAY = "#888888"
LGRAY = "#cccccc"

_THEME_CONFIG = theme()["config"]


def finalize(chart, width=480, height=300):
    """Apply the package chart theme to a top-level chart and set its size."""
    return chart.properties(width=width, height=height).configure(**_THEME_CONFIG)


def save(chart, name):
    path = OUT / f"{name}.svg"
    chart.save(str(path))
    print(f"  wrote {path.name}")


# ---------------------------------------------------------------------------
# 1. Shrinkage spectrum (prerequisites)
# ---------------------------------------------------------------------------
def fig_shrinkage_spectrum():
    ell = np.arange(1, 11)
    h = 1.0 / (1.0 + 0.5 * (ell ** 4) / 1000.0)
    df = pd.DataFrame({"ell": ell, "h": h})
    line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2.5, interpolate="monotone")
    pts = alt.Chart(df).mark_circle(color=BLUE, size=80)
    ref = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color=LGRAY, strokeDash=[4, 4])
    layered = (line + pts + ref).encode(
        x=alt.X("ell:Q", title="eigenvalue index ℓ", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("h:Q", title="shrinkage factor h", scale=alt.Scale(domain=[0, 1.05])),
    )
    save(finalize(layered), "fig-shrinkage-spectrum")


# ---------------------------------------------------------------------------
# 2. Gradient descent on convex function (prerequisites)
# ---------------------------------------------------------------------------
def fig_gradient_descent():
    m = np.arange(0, 16)
    F = np.exp(-0.25 * m) + 0.02
    df = pd.DataFrame({"m": m, "F": F})
    base = alt.Chart(df).encode(
        x=alt.X("m:Q", title="iteration m", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("F:Q", title="F(x)", scale=alt.Scale(domain=[0, 1.05])),
    )
    layered = base.mark_line(color=BLUE, strokeWidth=2.5) + base.mark_circle(color=BLUE, size=60)
    save(finalize(layered, 460, 280), "fig-gradient-descent")


# ---------------------------------------------------------------------------
# 3. GLM vs GAM effect (prerequisites)
# ---------------------------------------------------------------------------
def fig_glm_vs_gam():
    x = np.linspace(0, 10, 200)
    df = pd.DataFrame({
        "x": np.concatenate([x, x]),
        "y": np.concatenate([0.3 * x + 1, np.sin(x) * 2 + 3]),
        "model": ["GLM (linear)"] * 200 + ["GAM (smooth)"] * 200,
    })
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("x:Q", title="covariate x"),
        y=alt.Y("y:Q", title="effect on link scale g(μ)"),
        color=alt.Color("model:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[GRAY, BLUE])),
    )
    save(finalize(chart, 460, 250), "fig-glm-vs-gam")


# ---------------------------------------------------------------------------
# 4. Bias-variance tradeoff (prerequisites)
# ---------------------------------------------------------------------------
def fig_bias_variance():
    flex = np.linspace(0.1, 5, 200)
    bias2 = (3 / (1 + flex)) ** 2 * 0.05
    var = flex ** 1.5 * 0.08
    total = bias2 + var
    df = pd.DataFrame({"flexibility": np.tile(flex, 3),
                       "error": np.concatenate([bias2, var, total]),
                       "component": ["Bias²"] * 200 + ["Variance"] * 200 + ["Total error"] * 200})
    df["component"] = pd.Categorical(df["component"], ["Bias²", "Variance", "Total error"])
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("flexibility:Q", title="model flexibility (1/λ)"),
        y=alt.Y("error:Q", title="error"),
        color=alt.Color("component:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[ORANGE, GREEN, BLUE])),
    )
    opt = flex[np.argmin(total)]
    sweet = alt.Chart(pd.DataFrame({"x": [opt], "y": [total.min()]})).mark_circle(color=RED, size=100).encode(x="x:Q", y="y:Q")
    save(finalize(chart + sweet, 460, 280), "fig-bias-variance")


# ---------------------------------------------------------------------------
# 5. Forward stagewise coefficient growth (from_glm_to_boosting)
# ---------------------------------------------------------------------------
def fig_forward_stagewise():
    m = np.arange(0, 60)
    b1 = np.minimum(m * 0.04, 1.96)
    b2 = np.zeros_like(m, dtype=float)
    b2[20:] = np.minimum((m[20:] - 20) * 0.02, 0.5)
    df = pd.DataFrame({"m": np.concatenate([m, m]),
                       "coef": np.concatenate([b1, b2]),
                       "variable": ["x₁"] * 60 + ["x₂"] * 60})
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("m:Q", title="boosting iteration m"),
        y=alt.Y("coef:Q", title="coefficient"),
        color=alt.Color("variable:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[BLUE, ORANGE])),
    )
    save(finalize(chart, 460, 280), "fig-forward-stagewise")


# ---------------------------------------------------------------------------
# 6. Selection path heatmap (from_glm_to_boosting)
# ---------------------------------------------------------------------------
def fig_selection_path():
    rng = np.random.default_rng(42)
    rows = []
    for m in range(50):
        sel = {"x1": 1 if rng.random() < 0.65 else 0,
               "x2": 1 if rng.random() < 0.25 else 0,
               "x3": 1 if (m > 35 and rng.random() < 0.1) else 0,
               "x4": 1 if rng.random() < 0.08 else 0,
               "x5": 0}
        for var, s in sel.items():
            rows.append({"m": m, "var": var, "selected": s})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("m:O", title="boosting iteration m", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("var:O", title="variable", sort=["x1", "x2", "x3", "x4", "x5"]),
        color=alt.Color("selected:Q", scale=alt.Scale(range=["#f5f5f5", BLUE]), legend=None),
    )
    save(finalize(chart, 500, 180), "fig-selection-path")


# ---------------------------------------------------------------------------
# 7. df competition (from_glm_to_boosting)
# ---------------------------------------------------------------------------
def fig_df_competition():
    learners = ["bols(x₁)", "bbs(x₂)", "bbs(x₃)"]
    rows = []
    for lrn, sse_u, sse_f in zip(learners, [8.0, 1.5, 3.0], [5.0, 4.5, 5.2]):
        rows.append({"learner": lrn, "SSE": sse_u, "type": "unfair (different df)"})
        rows.append({"learner": lrn, "SSE": sse_f, "type": "fair (all df=4)"})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("SSE:Q", title="residual sum of squares (lower = better fit)"),
        y=alt.Y("type:N", title=None),
        color=alt.Color("type:N", legend=None, scale=alt.Scale(range=[GRAY, BLUE])),
        row=alt.Row("learner:N", title=None, header=alt.Header(labelAngle=0, labelFontSize=12)),
    )
    save(finalize(chart, 400, 200), "fig-df-competition")


# ---------------------------------------------------------------------------
# 8. Boosting risk U-curve (from_glm_to_boosting)
# ---------------------------------------------------------------------------
def fig_boosting_risk_curve():
    m = np.arange(0, 201)
    train = np.exp(-m / 40) + 0.05
    test = 0.3 * np.exp(-m / 30) + 0.15 + 0.002 * np.maximum(m - 60, 0) ** 1.5 * 0.01
    test = np.minimum(test, 0.9)
    df = pd.DataFrame({"m": np.concatenate([m, m]),
                       "risk": np.concatenate([train, test]),
                       "type": ["Training risk"] * 201 + ["Test risk (CV)"] * 201})
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("m:Q", title="boosting iteration m_stop"),
        y=alt.Y("risk:Q", title="risk", scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("type:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[GRAY, BLUE])),
    )
    opt_m = m[np.argmin(test)]
    sweet = alt.Chart(pd.DataFrame({"x": [opt_m], "y": [test.min()]})).mark_circle(color=RED, size=100).encode(x="x:Q", y="y:Q")
    save(finalize(chart + sweet, 480, 300), "fig-boosting-risk-curve")


# ---------------------------------------------------------------------------
# 9. Monotone risk decrease (theory_properties)
# ---------------------------------------------------------------------------
def fig_risk_decrease():
    m = np.arange(0, 50)
    risk = np.exp(-m / 12) + 0.08
    df = pd.DataFrame({"m": m, "risk": risk})
    base = alt.Chart(df).encode(
        x=alt.X("m:Q", title="iteration m", axis=alt.Axis(tickMinStep=5)),
        y=alt.Y("risk:Q", title="empirical risk", scale=alt.Scale(domain=[0, 1.05])),
    )
    layered = base.mark_line(color=BLUE, strokeWidth=2.5) + base.mark_circle(color=BLUE, size=40)
    save(finalize(layered, 460, 280), "fig-risk-decrease")


# ---------------------------------------------------------------------------
# 10. Convergence speed for different ν (theory_properties)
# ---------------------------------------------------------------------------
def fig_convergence_nu():
    m = np.arange(0, 200)
    rows = []
    for nu in [0.1, 0.25, 0.5]:
        r = np.exp(-nu * m / 5) + 0.02
        for mi, ri in zip(m, r):
            rows.append({"m": mi, "residual": ri, "nu": f"ν = {nu}"})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("m:Q", title="iteration m"),
        y=alt.Y("residual:Q", title="‖residual‖²", scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("nu:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[BLUE, ORANGE, GREEN])),
    )
    save(finalize(chart, 480, 280), "fig-convergence-nu")


# ---------------------------------------------------------------------------
# 11. Consistency rate (theory_properties)
# ---------------------------------------------------------------------------
def fig_consistency_rate():
    n = np.logspace(1.5, 4, 100)
    rate = 50 * np.log(200) / n
    df = pd.DataFrame({"n": n, "error": rate})
    base = alt.Chart(df).encode(
        x=alt.X("n:Q", title="sample size n", scale=alt.Scale(type="log")),
        y=alt.Y("error:Q", title="estimation error", scale=alt.Scale(type="log")),
    )
    save(finalize(base.mark_line(color=BLUE, strokeWidth=2.5), 440, 300), "fig-consistency-rate")


# ---------------------------------------------------------------------------
# 12. Sparse recovery phases (theory_properties)
# ---------------------------------------------------------------------------
def fig_sparse_recovery():
    m = np.arange(0, 120)
    sig = np.minimum(m / 15, 5)
    noise = np.maximum(0, (m - 60) / 20)
    df = pd.DataFrame({"m": np.concatenate([m, m]),
                       "count": np.concatenate([sig, noise]),
                       "type": ["signal vars (in S)"] * 120 + ["noise vars (not in S)"] * 120})
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("m:Q", title="boosting iteration m"),
        y=alt.Y("count:Q", title="cumulative variables selected"),
        color=alt.Color("type:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[BLUE, RED])),
    )
    band = alt.Chart(pd.DataFrame({"x": [60]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(x="x:Q")
    save(finalize(chart + band, 480, 300), "fig-sparse-recovery")


# ---------------------------------------------------------------------------
# 13. Oracle hierarchy bars (theory_properties)
# ---------------------------------------------------------------------------
def fig_oracle_hierarchy():
    methods = ["Oracle (knows S)", "Boosting / Lasso", "OLS on all p vars"]
    rates = [5 / 200, 5 * np.log(200) / 200, 1.0]
    df = pd.DataFrame({"method": methods, "rate": rates})
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("rate:Q", title="estimation error rate", scale=alt.Scale(domain=[0, 1.1])),
        y=alt.Y("method:N", title=None, sort="-x"),
        color=alt.Color("method:N", legend=None, scale=alt.Scale(range=[GREEN, BLUE, RED])),
    )
    save(finalize(chart, 420, 220), "fig-oracle-hierarchy")


# ---------------------------------------------------------------------------
# 14. Coefficient paths: lasso vs boosting (theory_properties)
# ---------------------------------------------------------------------------
def fig_coef_paths():
    m = np.arange(0, 100)
    rows = []
    for var, entry, slope, cap in [("x₁", 0, 0.025, 1.96), ("x₂", 20, 0.02, 1.2), ("x₃", 100, 0.01, 0.0)]:
        vals = np.maximum(0, np.minimum((m - entry) * slope, cap)) if entry < 100 else np.zeros_like(m, dtype=float)
        for mi, v in zip(m, vals):
            rows.append({"m": mi, "coef": v, "variable": var, "method": "L2Boosting (m →)"})
    for var, entry, slope, cap in [("x₁", 0, 0.03, 1.96), ("x₂", 15, 0.022, 1.2), ("x₃", 60, 0.005, 0.2)]:
        vals = np.maximum(0, np.minimum((m - entry) * slope, cap))
        for mi, v in zip(m, vals):
            rows.append({"m": mi, "coef": v, "variable": var, "method": "Lasso (1/λ →)"})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("m:Q", title="regularization path direction →"),
        y=alt.Y("coef:Q", title="coefficient"),
        color=alt.Color("variable:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[BLUE, ORANGE, GRAY])),
        strokeDash=alt.StrokeDash("method:N", legend=alt.Legend(title=None)),
    )
    save(finalize(chart, 500, 280), "fig-coef-paths")


# ---------------------------------------------------------------------------
# 15. Joint risk: cyclic vs non-cyclic (theory_properties)
# ---------------------------------------------------------------------------
def fig_joint_risk():
    m = np.arange(0, 100)
    nc = np.exp(-m / 25) + 0.05
    cyc = np.exp(-m / 22) + 0.05 + 0.015 * np.sin(m * 0.8) * np.exp(-m / 50)
    cyc = np.maximum(cyc, 0.04)
    df = pd.DataFrame({"m": np.concatenate([m, m]),
                       "risk": np.concatenate([nc, cyc]),
                       "type": ["Non-cyclic (monotone)"] * 100 + ["Cyclic (per-cycle)"] * 100})
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("m:Q", title="iteration m"),
        y=alt.Y("risk:Q", title="joint risk", scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("type:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[GREEN, BLUE])),
    )
    save(finalize(chart, 480, 300), "fig-joint-risk")


# ---------------------------------------------------------------------------
# 16. Heteroscedastic + skew examples (gamlss_framework)
# ---------------------------------------------------------------------------
def fig_heteroscedastic():
    rng = np.random.default_rng(99)
    x1 = np.linspace(0, 10, 200)
    y1 = 2 * x1 + rng.normal(0, 0.3 + 0.5 * x1, 200)
    x2 = rng.uniform(0, 10, 200)
    y2 = rng.exponential(0.5 * (1 + x2 / 3), 200)
    df = pd.DataFrame({"x": np.concatenate([x1, x2]),
                       "y": np.concatenate([y1, y2]),
                       "type": ["Heteroscedastic (σ grows with x)"] * 200 + ["Right-skewed (shape varies)"] * 200})
    chart = alt.Chart(df).mark_circle(opacity=0.5, size=25).encode(
        x=alt.X("x:Q", title="x"),
        y=alt.Y("y:Q", title="y"),
        color=alt.Color("type:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[BLUE, ORANGE])),
    )
    save(finalize(chart, 480, 280), "fig-heteroscedastic")


# ---------------------------------------------------------------------------
# 17. GaussianLSS scores (gamlss_framework)
# ---------------------------------------------------------------------------
def fig_gaussian_scores():
    r = np.linspace(-4, 4, 200)
    df = pd.DataFrame({"r": np.concatenate([r, r]),
                       "score": np.concatenate([r, -1 + r ** 2]),
                       "param": ["μ-score: (y−μ)/σ²"] * 200 + ["σ-score: −1+(y−μ)²/σ²"] * 200})
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("r:Q", title="standardized residual (y−μ)/σ"),
        y=alt.Y("score:Q", title="score"),
        color=alt.Color("param:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[BLUE, ORANGE])),
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=LGRAY, strokeDash=[4, 4]).encode(y="y:Q")
    save(finalize(chart + zero, 480, 280), "fig-gaussian-scores")


# ---------------------------------------------------------------------------
# 18. Tuning space: K-dim grid vs 1-D (gamlss_framework)
# ---------------------------------------------------------------------------
def fig_tuning_space():
    g1, g2 = np.meshgrid(range(5), range(5))
    g3 = (g1 + g2) % 5
    df_grid = pd.DataFrame({"λ₁": g1.ravel(), "λ₂": g2.ravel(), "λ₃": g3.ravel()})
    grid_chart = alt.Chart(df_grid).mark_square(opacity=0.35, size=80).encode(
        x=alt.X("λ₁:O", title=None), y=alt.Y("λ₂:O", title=None),
        color=alt.Color("λ₃:O", scale=alt.Scale(scheme="blues"), legend=None),
    ).properties(title="RS: 5³ = 125 evaluations", width=180, height=180)

    m = np.arange(0, 200)
    cv = np.minimum(0.3 * np.exp(-m / 40) + 0.15 + 0.002 * np.maximum(m - 60, 0) ** 1.3 * 0.01, 0.8)
    df_line = pd.DataFrame({"m": m, "CV risk": cv})
    line_chart = alt.Chart(df_line).mark_line(color=BLUE, strokeWidth=2.5).encode(
        x=alt.X("m:Q", title="m_stop"), y=alt.Y("CV risk:Q", title=None),
    ).properties(title="Boosting: 1 CV curve", width=180, height=180)
    concat = (grid_chart | line_chart).configure_axis(
        labelFontSize=10, titleFontSize=10, domainColor=LGRAY, tickColor=LGRAY,
    ).configure_title(fontSize=12).configure_view(stroke=None)
    save(concat, "fig-tuning-space")


# ---------------------------------------------------------------------------
# 19. Q-Q residual diagnostics (gamlss_framework)
# ---------------------------------------------------------------------------
def fig_qq_residuals():
    rng = np.random.default_rng(12)
    theoretical = np.linspace(-3, 3, 50)
    good = theoretical + rng.normal(0, 0.15, 50)
    bad = theoretical + 0.5 * np.sin(theoretical * 0.8) + rng.normal(0, 0.2, 50)
    df = pd.DataFrame({"theoretical": np.concatenate([theoretical, theoretical]),
                       "sample": np.concatenate([good, bad]),
                       "fit": ["good fit (correct family)"] * 50 + ["bad fit (wrong family)"] * 50})
    ref = pd.DataFrame({"x": [-3, 3], "y": [-3, 3]})
    chart = alt.Chart(df).mark_circle(size=40, opacity=0.7).encode(
        x=alt.X("theoretical:Q", title="theoretical quantiles"),
        y=alt.Y("sample:Q", title="sample quantiles"),
        color=alt.Color("fit:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[GREEN, RED])),
    )
    refline = alt.Chart(ref).mark_line(color=GRAY, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    save(finalize(chart + refline, 420, 320), "fig-qq-residuals")


# ---------------------------------------------------------------------------
# 20. Worked example: data + fitted values (functional_gradient_boosting)
# ---------------------------------------------------------------------------
def fig_worked_example():
    x1 = np.array([0.5, 1.0, -0.5, 0.0, -1.0])
    y = np.array([1.1, 1.9, -0.8, 0.1, -2.3])
    f1 = np.array([0.098, 0.196, -0.098, 0, -0.196])
    f10 = np.array([0.78, 1.57, -0.78, 0, -1.57])
    df = pd.DataFrame({"x₁": np.tile(x1, 3),
                       "y": np.concatenate([y, f1, f10]),
                       "iteration": ["original y"] * 5 + ["f̂ after iter 1"] * 5 + ["f̂ after iter 10"] * 5})
    chart = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("x₁:Q", title="x₁", axis=alt.Axis(tickMinStep=0.5)),
        y=alt.Y("y:Q", title="value", scale=alt.Scale(domain=[-2.5, 2.2])),
        color=alt.Color("iteration:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[GRAY, ORANGE, BLUE])),
    )
    save(finalize(chart, 460, 300), "fig-worked-example")


# ---------------------------------------------------------------------------
# 21. Worked example: selection path + coefficient accumulation
# ---------------------------------------------------------------------------
def fig_worked_path():
    m = np.arange(0, 51)
    b1 = np.minimum(m * 0.04, 1.96)
    b2 = np.zeros(51)
    df = pd.DataFrame({"m": np.concatenate([m, m]),
                       "coefficient": np.concatenate([b1, b2]),
                       "variable": ["x₁ (signal)"] * 51 + ["x₂ (noise)"] * 51})
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("m:Q", title="iteration m", axis=alt.Axis(tickMinStep=10)),
        y=alt.Y("coefficient:Q", title="coefficient β̂", scale=alt.Scale(domain=[0, 2.2])),
        color=alt.Color("variable:N", legend=alt.Legend(title=None), scale=alt.Scale(range=[BLUE, RED])),
    )
    save(finalize(chart, 460, 280), "fig-worked-path")


# ---------------------------------------------------------------------------
# 22. B-spline basis functions (base_learners)
# ---------------------------------------------------------------------------
def fig_bspline_basis():
    """Cubic B-spline basis: 23 overlapping bell-shaped functions over [0,1]."""
    def cox_de_boor(x, knots, i, d):
        """Evaluate i-th B-spline basis function of degree d at points x."""
        n = len(x)
        # degree 0: piecewise constant
        left = knots[i]
        right = knots[i + 1]
        if d == 0:
            return np.where((x >= left) & (x < right), 1.0, 0.0)
        # recursion
        left_weight = np.zeros(n)
        right_weight = np.zeros(n)
        denom_left = knots[i + d] - knots[i]
        denom_right = knots[i + d + 1] - knots[i + 1]
        if denom_left > 0:
            left_weight = (x - knots[i]) / denom_left * cox_de_boor(x, knots, i, d - 1)
        if denom_right > 0:
            right_weight = (knots[i + d + 1] - x) / denom_right * cox_de_boor(x, knots, i + 1, d - 1)
        return left_weight + right_weight

    degree = 3
    n_inner = 20
    x = np.linspace(0, 1, 500)
    knots = np.linspace(0, 1, n_inner + 2)
    left_pad = np.full(degree + 1, knots[0])
    right_pad = np.full(degree + 1, knots[-1])
    knots = np.concatenate([left_pad, knots, right_pad])
    n_basis = len(knots) - degree - 1

    rows = []
    for i in range(n_basis):
        y = cox_de_boor(x, knots, i, degree)
        for xi, yi in zip(x, y, strict=False):
            if yi > 0.01:
                rows.append({"x": xi, "y": yi, "basis": f"B_{{{i}}}"})

    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_line(strokeWidth=1, opacity=0.85).encode(
        x=alt.X("x:Q", title="x", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("y:Q", title=None, scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("basis:N", legend=None, scale=alt.Scale(scheme="viridis", rangeMax=23)),
    )
    save(finalize(chart, 520, 280), "fig-bspline-basis")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FIGURES = [
    fig_shrinkage_spectrum, fig_gradient_descent, fig_glm_vs_gam,
    fig_bias_variance, fig_forward_stagewise, fig_selection_path,
    fig_df_competition, fig_boosting_risk_curve, fig_risk_decrease,
    fig_convergence_nu, fig_consistency_rate, fig_sparse_recovery,
    fig_oracle_hierarchy, fig_coef_paths, fig_joint_risk,
    fig_heteroscedastic, fig_gaussian_scores, fig_tuning_space,
    fig_qq_residuals, fig_worked_example, fig_worked_path,
    fig_bspline_basis,
]

if __name__ == "__main__":
    print(f"Generating {len(FIGURES)} Altair SVGs into {OUT}/...")
    for fn in FIGURES:
        fn()
    print(f"Done. {len(list(OUT.glob('*.svg')))} SVGs in {OUT}")
