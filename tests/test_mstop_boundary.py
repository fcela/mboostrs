"""One boundary rule for tune_mstop and compare_specifications(mstop="cv") (#314).

A search grows its grid while the optimum lies in the boundary band at the end
of the grid (`BOUNDARY_SHARE` of the grid's length). It warns only when the
extension limit is reached, and the warning names the limit that applied.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import explore
from mboostrs.cvrisk import BOUNDARY_SHARE, boundary_tolerance, is_at_boundary


def sine_data(n=300, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return {"x": x, "y": np.sin(3 * x) + rng.normal(scale=0.1, size=n)}


def boundary_messages(caught):
    """The messages of the boundary warnings among `caught`."""
    return [str(w.message) for w in caught
            if "falling" in str(w.message) or "exhausted" in str(w.message)
            or "stopped at max_mstop" in str(w.message)]


@pytest.fixture
def small_inner_cap(monkeypatch):
    """A first inner grid of 8 steps, so the inner search has to grow."""
    monkeypatch.setattr(explore, "_MAX_TUNED_MSTOP", 8)


def test_the_band_is_the_same_for_the_rule_and_the_warning():
    assert BOUNDARY_SHARE == 0.05
    assert boundary_tolerance(360) == 18
    assert boundary_tolerance(10) == 1
    assert is_at_boundary(349, 360)
    assert not is_at_boundary(341, 360)


def test_an_optimum_near_the_end_extends_instead_of_warning():
    data = sine_data()
    # One path to 400 steps gives the risk at every iteration. A grid that
    # ends sooner reads the same prefix of it, so the curves can be compared.
    full = mb.tune_mstop("y ~ bbs(x, df=4)", data, folds=5, start=400,
                         max_mstop=400, extensions=0)
    risk = np.asarray(full.risk)
    best = int(np.argmin(risk[1:]) + 1)
    # The grid ends 3% past the optimum: inside the band, not at the end.
    end = round(best / 0.97)
    assert best < end < 400
    assert is_at_boundary(best, end)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        near = mb.tune_mstop("y ~ bbs(x, df=4)", data, folds=5, start=end,
                             max_mstop=400)

    assert near.extended
    assert near.grid_ends[0] == end
    assert near.largest > end
    assert boundary_messages(caught) == []
    assert near.mstop == best
    np.testing.assert_allclose(np.asarray(near.risk), risk[: len(near.risk)])


def test_the_warning_names_the_max_mstop_that_applied():
    data = sine_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capped = mb.tune_mstop("y ~ bbs(x, df=4)", data, folds=5, start=20,
                               max_mstop=20)
    assert capped.at_boundary and not capped.extended
    messages = boundary_messages(caught)
    assert len(messages) == 1
    assert "max_mstop=20" in messages[0]
    assert "still falling at 20 steps" in messages[0]


def test_the_warning_names_the_extension_count_that_applied():
    data = sine_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        limited = mb.tune_mstop("y ~ bbs(x, df=4)", data, folds=5, start=20,
                                extensions=1)
    assert limited.at_boundary and limited.grid_ends == (20, 40)
    messages = boundary_messages(caught)
    assert len(messages) == 1
    assert "extensions=1" in messages[0]
    assert "max_mstop" not in messages[0]


def test_the_inner_cap_of_compare_is_configurable(small_inner_cap):
    frame = sine_data(n=120)
    rows = mb.compare_specifications({"a": "bbs(x, df=4)"}, frame, "y", folds=3,
                                     mstop="cv", max_mstop=64)
    # The first grid is 8 steps; the search grows past it, not past 64.
    assert rows[0]["inner_mstop_max"] > 8
    assert rows[0]["inner_mstop_max"] <= 64


def test_a_compare_inner_cap_reached_with_the_risk_falling_warns_with_the_cap(small_inner_cap):
    frame = sine_data(n=120)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = mb.compare_specifications({"a": "bbs(x, df=4)"}, frame, "y", folds=3,
                                         mstop="cv", max_mstop=8)
    row = rows[0]
    assert row["inner_mstop_max"] == 8 and row["inner_at_boundary"] is True
    messages = boundary_messages(caught)
    assert len(messages) == 1
    assert "compare_specifications" in messages[0]
    assert "max_mstop=8" in messages[0] and "now 8" in messages[0]
    assert "re-running" not in messages[0] and "cvrisk" not in messages[0]


def test_compare_rows_report_the_inner_mstop_per_fold(small_inner_cap):
    frame = sine_data(n=120)
    tuned = mb.compare_specifications({"a": "bbs(x, df=4)"}, frame, "y", folds=3,
                                      mstop="cv", max_mstop=32)[0]
    assert tuned["inner_mstop_min"] <= tuned["inner_mstop_median"] <= tuned["inner_mstop_max"]
    assert isinstance(tuned["inner_at_boundary"], bool)

    fixed = mb.compare_specifications({"a": "bbs(x, df=4)"}, frame, "y", folds=3,
                                      mstop=20)[0]
    assert fixed["inner_mstop_min"] is None
    assert fixed["inner_mstop_max"] is None
    assert fixed["inner_at_boundary"] is None


def test_cv_risk_reports_each_fold_inner_search_and_takes_the_cap(small_inner_cap):
    frame = sine_data(n=120)
    result = mb.cv_risk("bbs(x, df=4)", frame, "y", folds=3, mstop="cv",
                        max_mstop=32, return_result=True)
    stops = [fold["inner_mstop"] for fold in result.folds]
    assert len(stops) == 3 and all(1 <= stop <= 32 for stop in stops)
    assert all(isinstance(fold["inner_at_boundary"], bool) for fold in result.folds)


def test_a_cap_with_a_fixed_mstop_is_refused():
    frame = sine_data(n=120)
    with pytest.raises(ValueError, match="max_mstop caps the inner search"):
        mb.compare_specifications({"a": "bbs(x, df=4)"}, frame, "y", folds=3,
                                  mstop=20, max_mstop=64)
    with pytest.raises(ValueError, match="max_mstop caps the inner search"):
        mb.cv_risk("bbs(x, df=4)", frame, "y", folds=3, mstop=20, max_mstop=64)


def test_tune_mstop_budget_counts_a_gram_refit_at_every_grid_for_a_response_score():
    data = sine_data(n=120)
    # Gaussian bbs folds take the Gram route and are refitted at each grid:
    # 20 -> 40 -> 80 is three grids, so the worst case is 3 * 3 + 1 = 10 fits.
    with pytest.raises(ValueError, match=r"tune_mstop needs 10 fits.*fit_budget=9"):
        mb.tune_mstop("y ~ bbs(x, df=4)", data, folds=3, start=20, max_mstop=80,
                      score="mse", fit_budget=9)
    result = mb.tune_mstop("y ~ bbs(x, df=4)", data, folds=3, start=20, max_mstop=80,
                           score="mse", fit_budget=10)
    assert result.fits == 3 * len(result.grid_ends)
    assert result.total_fits <= 10


def test_tune_mstop_budget_for_the_family_score_is_not_raised_by_extensions():
    data = sine_data(n=120)
    # The family score resumes the folds, so growing the grid adds no fits.
    result = mb.tune_mstop("y ~ bbs(x, df=4)", data, folds=3, start=20,
                           max_mstop=80, fit_budget=4)
    assert result.fits == 3


def test_compare_budget_counts_inner_grids_under_a_refitting_score(small_inner_cap):
    frame = sine_data(n=120)
    # Outer fold: one fit, plus five inner folds at each of three grids
    # (8 -> 16 -> 32). Three outer folds.
    planned = 3 * (1 + 5 * 3)
    with pytest.raises(ValueError, match=rf"needs {planned} fits.*fit_budget={planned - 1}"):
        mb.compare_specifications({"a": "bbs(x, df=4)"}, frame, "y", folds=3,
                                  mstop="cv", score="mse", max_mstop=32,
                                  fit_budget=planned - 1)
    rows = mb.compare_specifications({"a": "bbs(x, df=4)"}, frame, "y", folds=3,
                                     mstop="cv", score="mse", max_mstop=32,
                                     fit_budget=planned)
    assert rows[0]["planned fit count"] == planned
    assert rows[0]["fit count"] <= planned
