"""gamlss_refit reports where its time went (claims-discovery benchmark, #357)."""

import warnings

import numpy as np
import polars as pl

import mboostrs as mb


def test_convergence_report_counts_the_work():
    rng = np.random.default_rng(0)
    n = 400
    df = pl.DataFrame({"x": rng.uniform(0, 1, n), "z": rng.uniform(0, 1, n)})
    df = df.with_columns(y=np.sin(3 * df["x"].to_numpy())
                         + rng.normal(0, 0.2 + 0.4 * df["z"].to_numpy(), n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(z)"}, data=df,
                             families=mb.GaussianLSS(), control=mb.boost_control(mstop=200))
        conv = mb.gamlss_refit(fit).convergence
    assert conv["converged"] is True
    assert conv["stop_reason"] in ("gradient", "stationary", "flat")
    assert conv["evaluations"] >= conv["iterations"] + 1
    assert conv["inner_iterations"] >= conv["evaluations"]
    assert isinstance(conv["restarted"], bool)
    assert conv["restart_iterations"] >= 0
    assert conv["seconds"] > 0.0
