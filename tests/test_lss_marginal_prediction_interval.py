"""Marginal prediction intervals for a `brandom` group the fit never saw.

`prediction_interval(unseen_levels="marginal", group_sd=)` gives an unseen
group the zero-effect location and widens its scale from sigma to
sqrt(sigma^2 + group_sd^2). Rows whose group the fit saw keep the interval
of `unseen_levels="zero_effect"`.
"""

from statistics import NormalDist

import numpy as np
import pytest

import mboostrs as mb

CONTROL = mb.boost_control(60, 0.1)
LEVEL = 0.9
GROUP_SD = 1.5


def _grouped_frame(seed=3, n_groups=6, per_group=50):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group).astype(float)
    x = rng.uniform(-2, 2, g.size)
    y = 0.5 * x + 0.3 * g + rng.normal(scale=1.0, size=g.size)
    return {"x": x, "g": g, "y": y}


def _records(table):
    if hasattr(table, "to_dicts"):
        return table.to_dicts()
    return table.to_dict(orient="records")


def _fit_and_newdata():
    frame = _grouped_frame()
    seen = frame["g"] < 4
    train = {name: values[seen] for name, values in frame.items()}
    fit = mb.gamboostLSS({"mu": "y ~ bols(x) + brandom(g)", "sigma": "y ~ 1"},
                         data=train, family=mb.GaussianLSS(), control=CONTROL)
    # One row from a seen group (g=1) and one from an unseen group (g=5).
    newdata = {"x": np.array([0.4, -0.7]), "g": np.array([1.0, 5.0]),
               "y": np.array([0.0, 0.0])}
    return fit, newdata


def test_marginal_interval_runs_and_widens_only_the_unseen_row():
    fit, newdata = _fit_and_newdata()
    marginal = fit.prediction_interval(level=LEVEL, newdata=newdata,
                                       unseen_levels="marginal", group_sd=GROUP_SD)
    zero = fit.prediction_interval(level=LEVEL, newdata=newdata,
                                   unseen_levels="zero_effect")

    marginal_rows = _records(marginal)
    zero_rows = _records(zero)
    seen_marginal, unseen_marginal = marginal_rows[0], marginal_rows[1]
    seen_zero = zero_rows[0]

    # Seen row: identical to the conditional interval under the same policy.
    for key in ("lower", "median", "upper"):
        assert seen_marginal[key] == pytest.approx(seen_zero[key], abs=1e-12)

    # Unseen row: wider, and its half-width is z * sqrt(sigma^2 + group_sd^2).
    unseen_zero = zero_rows[1]
    assert unseen_marginal["upper"] - unseen_marginal["lower"] > (
        unseen_zero["upper"] - unseen_zero["lower"])

    sigma = float(fit.params(newdata=newdata, unseen_levels="zero_effect")["sigma"][1])
    z = NormalDist().inv_cdf((1.0 + LEVEL) / 2.0)
    expected_half_width = z * np.sqrt(sigma ** 2 + GROUP_SD ** 2)
    half_width = (unseen_marginal["upper"] - unseen_marginal["lower"]) / 2.0
    assert half_width == pytest.approx(expected_half_width, rel=0, abs=1e-10)


def test_marginal_interval_refuses_a_missing_group_sd():
    fit, newdata = _fit_and_newdata()
    with pytest.raises(ValueError, match="needs group_sd="):
        fit.prediction_interval(level=LEVEL, newdata=newdata,
                                unseen_levels="marginal", group_sd=None)
