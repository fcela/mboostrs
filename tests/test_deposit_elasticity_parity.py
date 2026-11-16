"""R parity for the designs used in the deposit-elasticity case study.

Fixtures come from ``tests/fixtures/generate_deposit_elasticity_fixtures.R``
(mboost 2.9.11, R 4.5.1) and are committed, so the test runs without R.

Each design is compared on the whole boosting path, not only at the final
iteration: the selected learner per iteration, the empirical risk per
iteration, the fitted values, and the semi-elasticity read off the fitted
partial effect. A divergence in the selection path compounds, so the failure
message reports the first iteration at which the paths part.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

FIXTURES = Path(__file__).parent / "fixtures"
NU = 0.05


def load(name):
    path = FIXTURES / f"deposit_elasticity_{name}.json"
    if not path.exists():
        pytest.skip(f"fixture {path.name} not generated")
    return json.loads(path.read_text())


def historical_frame():
    macro = mb.datasets.usmacro().filter(pl.col("year") <= 1993)
    n = macro.height
    r = macro["tbilrate"].to_numpy()
    real_m1 = macro["m1"].to_numpy() / (macro["cpi"].to_numpy() / 100.0)
    frame = pl.DataFrame({
        "y": np.log(real_m1 / macro["realgdp"].to_numpy()),
        "r_c": r - r.mean(),
        "t_c": np.arange(1, n + 1, dtype=float) - (n + 1) / 2.0,
        "infl_c": macro["infl"].to_numpy() - macro["infl"].mean(),
    })
    return frame, float(r.mean())


def modern_frame():
    dm = mb.datasets.deposit_market().drop_nulls(
        subset=["fedfunds", "deposits", "personal_income", "natl_mmda", "cpi"])
    m = dm.height
    spread = (dm["fedfunds"] - dm["natl_mmda"]).to_numpy()
    balance = np.log(dm["walcl"].to_numpy() / (dm["personal_income"].to_numpy() * 1000))
    dates = dm["date"].to_numpy()
    frame = pl.DataFrame({
        "y": np.log(dm["deposits"].to_numpy() / dm["personal_income"].to_numpy()),
        "s_c": spread - spread.mean(),
        "bs_c": balance - balance.mean(),
        "t_c": np.arange(1, m + 1, dtype=float) - (m + 1) / 2.0,
        "covid": ((dates >= "2020-03-01") & (dates <= "2020-07-01")).astype(float),
        "svb": ((dates >= "2023-03-01") & (dates <= "2023-05-01")).astype(float),
    })
    return frame, float(spread.mean())


def theta_on_grid(model, points, center, columns, key, h=0.05):
    """-d fitted / d x at `points`, holding every other centered column at 0."""
    def predict_at(values):
        data = {name: np.zeros(len(values)) for name in columns}
        data[key] = values - center
        return np.asarray(model.predict(newdata=pl.DataFrame(data))).ravel()
    points = np.asarray(points, dtype=float)
    return -(predict_at(points + h) - predict_at(points - h)) / (2 * h)


def first_divergence(a, b, tol):
    """Index of the first entry where |a - b| exceeds `tol`, or None."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    bad = np.flatnonzero(np.abs(a - b) > tol)
    return int(bad[0]) if bad.size else None


def compare_path(model, fixture, tol_risk=1e-8, tol_fitted=1e-8):
    selected = model.path_frame()["learner_index"].to_numpy()
    expected = np.asarray(fixture["selected"], dtype=int)
    assert len(selected) == len(expected)
    mismatch = np.flatnonzero(selected != expected)
    assert mismatch.size == 0, (
        f"selection path diverges first at iteration {int(mismatch[0]) + 1}: "
        f"mboostrs {selected[mismatch[0]]}, R {expected[mismatch[0]]} "
        f"({mismatch.size} of {len(expected)} iterations differ)")

    risk = np.asarray(model.risk_, dtype=float)
    r_risk = np.asarray(fixture["risk"], dtype=float)
    # R's risk vector carries the offset-only risk at index 0.
    assert len(risk) == len(r_risk), (len(risk), len(r_risk))
    where = first_divergence(risk, r_risk, tol_risk)
    assert where is None, (
        f"risk path diverges first at iteration {where}: "
        f"mboostrs {risk[where]:.12g}, R {r_risk[where]:.12g}")

    fitted = np.asarray(model.fitted(), dtype=float)
    r_fitted = np.asarray(fixture["fitted"], dtype=float)
    np.testing.assert_allclose(fitted, r_fitted, rtol=0, atol=tol_fitted)


def test_centered_df1_matches_r():
    fixture = load("centered_df1")
    frame, r_mean = historical_frame()
    model = mb.gamboost(
        "y ~ bols(r_c, intercept=False) + bbs(r_c, center=True, df=1)"
        " + bols(t_c, intercept=False) + bols(infl_c, intercept=False)"
        " + bbs(infl_c, center=True, df=1)",
        data=frame, family=mb.Gaussian(),
        control=mb.boost_control(mstop=fixture["mstop"], nu=NU))
    compare_path(model, fixture)
    theta = theta_on_grid(model, fixture["theta_grid"], r_mean,
                          ["y", "r_c", "t_c", "infl_c"], "r_c")
    np.testing.assert_allclose(theta, fixture["theta"], rtol=0, atol=1e-8)


def test_varying_coefficient_matches_r():
    fixture = load("vc")
    macro = mb.datasets.usmacro().filter(pl.col("year") <= 1993)
    r = macro["tbilrate"].to_numpy()
    real_m1 = macro["m1"].to_numpy() / (macro["cpi"].to_numpy() / 100.0)
    y = np.log(real_m1 / macro["realgdp"].to_numpy())
    infl_c = macro["infl"].to_numpy() - macro["infl"].mean()
    dy, dr, dinfl = np.diff(y), np.diff(r), np.diff(infl_c)
    r_lag = r[:-1]
    frame = pl.DataFrame({
        "dy": dy[1:], "dr": dr[1:], "dinfl": dinfl[1:], "dy_lag": dy[:-1],
        "r_lag_c": r_lag[1:] - r_lag[1:].mean(),
    })
    model = mb.gamboost(
        "dy ~ bols(dr, intercept=False) + bols(r_lag_c, by=dr, intercept=False)"
        " + bbs(r_lag_c, by=dr, center=True, df=1)"
        " + bols(dy_lag, intercept=False) + bols(dinfl, intercept=False)",
        data=frame, family=mb.Gaussian(),
        control=mb.boost_control(mstop=fixture["mstop"], nu=NU))
    compare_path(model, fixture)


def test_modern_spread_matches_r():
    fixture = load("modern")
    frame, s_mean = modern_frame()
    model = mb.gamboost(
        "y ~ bols(s_c, intercept=False) + bbs(s_c, center=True, df=1)"
        " + bols(bs_c, intercept=False) + bbs(bs_c, center=True, df=1)"
        " + bols(t_c, intercept=False) + bols(covid, intercept=False)"
        " + bols(svb, intercept=False)",
        data=frame, family=mb.Gaussian(),
        control=mb.boost_control(mstop=fixture["mstop"], nu=NU))
    compare_path(model, fixture)
    theta = theta_on_grid(model, fixture["theta_grid"], s_mean,
                          ["y", "s_c", "bs_c", "t_c", "covid", "svb"], "s_c")
    np.testing.assert_allclose(theta, fixture["theta"], rtol=0, atol=1e-8)


@pytest.mark.xfail(strict=True, reason=(
    "Not the constrained solver, which agrees with R here: this design asks "
    "for df=2 on a second-difference penalty, i.e. df at the rank of the "
    "penalty's null space. lambda behaves like c/(df - rank) there, so the "
    "~1e-11 gap between R's LAPACK spectrum and faer's is amplified without "
    "bound: fitted values differ 9.4e-8 on a single learner, and compounding "
    "that through the path flips the selected learner at iteration 126. The "
    "same design at df=2.5, 3 or 4 agrees with R to 10 decimals. Tiers "
    "measured in crates/mboostrs-core/tests/df_lambda_r_parity.rs."))
def test_bmono_matches_r():
    fixture = load("bmono")
    frame, r_mean = historical_frame()
    model = mb.gamboost(
        "y ~ bols(r_c) + bmono(r_c, constraint='decreasing', df=2)"
        " + bols(t_c) + bols(infl_c) + bbs(infl_c, center=True, df=2)",
        data=frame, family=mb.Gaussian(),
        control=mb.boost_control(mstop=fixture["mstop"], nu=NU))
    compare_path(model, fixture)


def test_constrained_spline_matches_r():
    fixture = load("constrained")
    frame, r_mean = historical_frame()
    model = mb.gamboost(
        "y ~ bols(r_c, intercept=False)"
        " + bbs(r_c, center=True, df=1, constraint='decreasing')"
        " + bols(t_c, intercept=False) + bols(infl_c, intercept=False)"
        " + bbs(infl_c, center=True, df=1)",
        data=frame, family=mb.Gaussian(),
        control=mb.boost_control(mstop=fixture["mstop"], nu=NU))
    compare_path(model, fixture)
    theta = theta_on_grid(model, fixture["curve_grid"], r_mean,
                          ["y", "r_c", "t_c", "infl_c"], "r_c")
    np.testing.assert_allclose(theta, fixture["curve_theta"], rtol=0, atol=1e-6)


def test_constrained_spline_centering_works():
    """Verify that constrained splines respect center=.

    With center=True, the learner operates on a reduced-rank basis and can
    fit curved monotone shapes even at df=1. Without centering, df=1 leaves
    only the penalty's null space and produces a straight line.
    """
    fixture = load("constraint_isolated")
    data = pl.DataFrame({"x": np.asarray(fixture["x"], dtype=float),
                         "y": np.asarray(fixture["y"], dtype=float)})
    fits = {}
    for center, df in ((True, 1), (False, 4), (True, 4)):
        model = mb.gamboost(
            f"y ~ bbs(x, center={center}, df={df}, constraint='decreasing')",
            data=data, family=mb.Gaussian(),
            control=mb.boost_control(mstop=fixture["mstop"], nu=fixture['nu']))
        fits[(center, df)] = np.asarray(model.fitted(), dtype=float)

    def curvature(values):
        return float(np.abs(np.diff(np.diff(values))).max())

    # center=True removes the penalty null space; df=1 no longer forces a line.
    assert curvature(fits[(True, 1)]) > 1e-3, "centered df=1 should be curved"
    # center changes the effective df, so center=True and center=False differ.
    assert not np.allclose(fits[(True, 4)], fits[(False, 4)], atol=1e-6)

    # R's behavior matches.
    r_fits = fixture["fitted"]
    assert curvature(np.asarray(r_fits["centerTRUE_df1"])) > 1e-2
    assert not np.allclose(np.asarray(r_fits["centerTRUE_df4"]),
                           np.asarray(r_fits["centerFALSE_df4"]), atol=1e-6)
