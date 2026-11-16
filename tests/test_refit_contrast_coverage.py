"""Categorical contrasts in the REML refit (gamlss_refit): the level design, the
default penalty and the factor_ridge option, and interval coverage on a planted
factor effect.

The planted effects are not monotone in the level order (B is high, C is low),
so a factor read as one numeric slope on its integer codes (the refit's
earlier behaviour for a factor the fit had encoded) gives a D-A contrast about
0.11 too small.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.selection import _LevelDesign

LEVEL_EFFECT = {"A": 0.0, "B": 0.3, "C": 0.05, "D": 0.35}
TRUE_D_MINUS_A = 0.35


def _planted(seed, n=600):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    g = rng.choice(list(LEVEL_EFFECT), size=n)
    effect = np.array([LEVEL_EFFECT[v] for v in g])
    y = 0.5 * np.sin(2 * np.pi * x) + effect + rng.normal(0, 0.3, n)
    return pl.DataFrame({"x": x, "g": g, "y": y})


def _fit(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboostLSS({"mu": "y ~ bbs(x) + bols(g)", "sigma": "y ~ bbs(x, df=2)"},
                              data=df, families=mb.GaussianLSS(),
                              control=mb.boost_control(mstop=300, nu=0.1))


def _d_minus_a(refit, df):
    a = df.with_columns(pl.lit("A").alias("g"))
    d = df.with_columns(pl.lit("D").alias("g"))
    return (np.asarray(refit.predict(d, parameter="mu")).ravel()
            - np.asarray(refit.predict(a, parameter="mu")).ravel())


def test_level_design_has_one_column_per_level_and_no_default_penalty():
    df = _planted(0)
    design = _LevelDesign("g", df)
    assert design.design.shape == (df.height, 4)
    assert np.array_equal(np.unique(design.design[:, 0]), [1.0])
    assert np.all(design.penalties[0] == 0)


def test_factor_ridge_penalizes_only_the_level_indicators():
    df = _planted(0)
    penalty = _LevelDesign("g", df, ridge=True).penalties[0]
    assert np.array_equal(np.diag(penalty), [0.0, 1.0, 1.0, 1.0])
    assert np.count_nonzero(penalty - np.diag(np.diag(penalty))) == 0


def test_refit_of_a_factor_the_fit_encoded_uses_the_level_design():
    # The LSS training frame holds the factor as integer codes. The refit must
    # still see four level columns, not one numeric code column.
    fit = _fit(_planted(3))
    refit = mb.gamlss_refit(fit, which="all")
    block = {name: design for name, design, _ in refit._blocks["mu"]}
    assert block["bols(g)"].design.shape[1] == 4
    assert refit.edf["mu"]["bols(g)"] == pytest.approx(3.0, abs=0.05)


def test_planted_factor_effect_is_recovered_without_shrinkage_and_covers_at_nominal_rate():
    estimates, covered = [], 0
    for seed in range(20):
        df = _planted(1000 + seed)
        refit = mb.gamlss_refit(_fit(df), which="all")
        estimates.append(float(_d_minus_a(refit, df).mean()))
        a = df.with_columns(pl.lit("A").alias("g"))
        d = df.with_columns(pl.lit("D").alias("g"))
        per_draw = (refit.draws(d, n_draws=400, seed=seed)["mu"]
                    - refit.draws(a, n_draws=400, seed=seed)["mu"]).mean(axis=1)
        lower, upper = np.quantile(per_draw, [0.05, 0.95])
        covered += int(lower <= TRUE_D_MINUS_A <= upper)
    bias = float(np.mean(estimates)) - TRUE_D_MINUS_A
    assert abs(bias) < 0.02, f"mean D-A estimate {np.mean(estimates):.4f}, bias {bias:+.4f}"
    assert covered >= 15, f"90% intervals covered {covered} of 20 seeds"


def test_factor_ridge_shrinks_level_effects_toward_the_reference():
    df = _planted(1000)
    default = _d_minus_a(mb.gamlss_refit(_fit(df), which="all"), df).mean()
    ridge = _d_minus_a(mb.gamlss_refit(_fit(df), which="all", factor_ridge=True), df).mean()
    assert ridge < default - 0.005, (default, ridge)
    assert ridge > 0, ridge
