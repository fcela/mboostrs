"""`confint(groups=)`: bootstrap samples that draw whole groups.

Rows that are repeated measurements of one unit (the points of a curve, the
visits of a patient) share that unit's deviation. A bootstrap that draws rows
separately treats them as independent and gives intervals that are too narrow;
drawing whole groups keeps each unit's rows together.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._bootstrap import group_bootstrap_counts


def test_rows_of_a_group_share_their_draw_count():
    groups = np.repeat([0, 1, 2, 3], 3)
    eligible = np.ones(12, dtype=bool)
    eligible[5] = False
    counts = group_bootstrap_counts(groups, eligible, 50, 0)

    for b in range(50):
        column = counts[:, b]
        # Every eligible row of a group carries the group's count.
        for g in range(4):
            rows = np.flatnonzero((groups == g) & eligible)
            assert len(set(column[rows])) == 1
        assert column[5] == 0
        # Four units drawn per sample.
        assert sum(column[np.flatnonzero(groups == g)[0]] for g in range(4)) == 4


def test_too_few_groups_are_refused():
    with pytest.raises(ValueError, match="at least two groups"):
        group_bootstrap_counts(np.zeros(5), np.ones(5, bool), 3, 0)


def _stacked_curves(seed=1, n_curves=60, n_points=30):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_points)
    z = rng.uniform(-1, 1, n_curves)
    # Each curve's own level is shared by all its points.
    level = rng.normal(0, 0.5, n_curves)
    y = (np.sin(np.pi * t)[None, :] + np.outer(z, np.cos(np.pi * t))
         + level[:, None] + rng.normal(0, 0.1, (n_curves, n_points)))
    return {"curve": np.repeat(np.arange(n_curves), n_points),
            "t": np.tile(t, n_curves),
            "z": np.repeat(z - z.mean(), n_points),
            "y": y.ravel()}


def test_grouped_bands_are_wider_than_row_bands_on_stacked_curves():
    data = _stacked_curves()
    term = "bols(z, intercept=False) %X% bbs(t, df=4)"
    fit = mb.gamboost(f"y ~ bbs(t, df=4) + {term}", data=data,
                      control=mb.boost_control(mstop=100, nu=0.1))
    grid = {"t": np.linspace(0, 1, 11), "z": np.ones(11)}

    rows = fit.confint(parm=term, newdata=grid, B=60, B_mstop=0, seed=3)
    curves = fit.confint(parm=term, newdata=grid, B=60, B_mstop=0, seed=3,
                         groups="curve")

    assert rows["resampled"] == "rows" and curves["resampled"] == "groups"
    width_rows = np.mean(rows["upper"][term] - rows["lower"][term])
    width_curves = np.mean(curves["upper"][term] - curves["lower"][term])
    assert width_curves > 1.5 * width_rows


def test_grouped_confint_with_inner_tuning_and_an_array_of_labels():
    data = _stacked_curves(n_curves=30)
    fit = mb.glmboost("y ~ z + t", data=data,
                      control=mb.boost_control(mstop=50, nu=0.1))
    result = fit.confint(B=8, B_mstop=3, seed=0, groups=data["curve"])

    assert result["resampled"] == "groups"
    assert result["n_used"] == 8
    assert result["n_untuned"] == 0


def test_group_labels_must_cover_every_row():
    data = _stacked_curves(n_curves=10)
    fit = mb.glmboost("y ~ z + t", data=data, control=mb.boost_control(mstop=20))
    with pytest.raises(ValueError, match="one label per training row"):
        fit.confint(B=4, groups=np.arange(3))


@pytest.mark.parametrize("kind", ["dict", "pandas", "polars"])
def test_confint_accepts_every_frame_type_as_newdata(kind):
    """`confint(newdata=)` counted rows with `nd.values()`, which is an array
    attribute on a pandas frame, not a method."""
    import pandas as pd
    import polars as pl

    data = _stacked_curves(n_curves=20)
    fit = mb.gamboost("y ~ bbs(t, df=4) + bbs(z, df=4)", data=data,
                      control=mb.boost_control(mstop=30, nu=0.1))
    grid = {"t": np.linspace(0, 1, 7), "z": np.zeros(7)}
    newdata = {"dict": grid, "pandas": pd.DataFrame(grid), "polars": pl.DataFrame(grid)}[kind]
    band = fit.confint(parm="t", newdata=newdata, B=5, B_mstop=0, seed=0)
    assert band["lower"]["t"].shape == (7,)
