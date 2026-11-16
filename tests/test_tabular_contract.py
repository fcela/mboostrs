"""Every tabular helper answers to the same `output=` contract, on both models.

A caller who asks for records must get records, from every helper, on a
standard fit and on a distributional one. Three things stood between that and
the code: `coef_all` accepted an `output` value it then dropped on the floor,
several method forms did not take the argument at all, and `calibration_frame`
had no distributional implementation.

The two `q_stats` implementations are also checked against each other. They
used to compute different numbers from the same fit -- equal-width bins in one
and equal-count in the other, a population variance in one and a sample
variance in the other -- so a reader comparing `q_stats(model)` with
`model.q_stats()` saw different variances under identical column names.
"""

import numpy as np
import pytest

import mboostrs as mb

#: Every helper that returns a table, with the arguments that keep it cheap.
TABULAR_HELPERS = {
    "coef_all": {},
    "predict_all": {},
    "residual_frame": {},
    "q_stats": {"n_intervals": 4},
    "worm_data": {"n_intervals": 2},
    "calibration_frame": {"bins": 4},
    "outlier_frame": {},
}

#: The helpers a distributional model also has. `outlier_frame` is absent by
#: design: it ranks by a single response residual, which an LSS fit does not
#: have one of.
LSS_HELPERS = {
    name: kwargs for name, kwargs in TABULAR_HELPERS.items()
    if name != "outlier_frame"
}


@pytest.fixture(scope="module")
def standard_model():
    rng = np.random.default_rng(3)
    n = 150
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n)
    return mb.glmboost("y ~ x", data={"x": x, "y": y},
                       control=mb.boost_control(20, 0.1))


@pytest.fixture(scope="module")
def lss_model():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    y = rng.normal(1 + 0.8 * x, np.exp(-0.3 + 0.3 * x))
    return mb.glmboostLSS("y ~ x", data={"x": x, "y": y},
                          families=mb.GaussianLSS(),
                          control=mb.boost_control(40, 0.1))


def _backend(frame):
    return type(frame).__module__.split(".")[0]


@pytest.mark.parametrize("name", sorted(TABULAR_HELPERS))
@pytest.mark.parametrize("output", ["polars", "pandas", "records"])
def test_standard_helper_honours_output(standard_model, name, output):
    frame = getattr(standard_model, name)(output=output, **TABULAR_HELPERS[name])
    expected = {"polars": "polars", "pandas": "pandas", "records": "builtins"}
    assert _backend(frame) == expected[output]
    if output == "records":
        assert isinstance(frame, list)
        assert all(isinstance(row, dict) for row in frame)


@pytest.mark.parametrize("name", sorted(LSS_HELPERS))
@pytest.mark.parametrize("output", ["polars", "pandas", "records"])
def test_lss_helper_honours_output(lss_model, name, output):
    frame = getattr(lss_model, name)(output=output, **LSS_HELPERS[name])
    expected = {"polars": "polars", "pandas": "pandas", "records": "builtins"}
    assert _backend(frame) == expected[output]


@pytest.mark.parametrize("name", sorted(TABULAR_HELPERS))
def test_an_unknown_output_value_is_rejected(standard_model, name):
    with pytest.raises(ValueError, match="output must be"):
        getattr(standard_model, name)(output="frame", **TABULAR_HELPERS[name])


def test_coef_all_still_accepts_its_old_default(standard_model):
    """`"dataframe"` was this helper's own default and is not a policy value.

    It never did anything -- the argument was dropped before the frame was
    built -- so a caller who wrote it got `auto`. That is what it still means,
    rather than an error on code that used to work.
    """
    assert _backend(standard_model.coef_all(output="dataframe")) == "polars"
    assert _backend(standard_model.coef_all()) == "polars"


def test_coef_all_output_actually_changes_the_result(standard_model):
    """The regression: every value produced the same frame."""
    records = standard_model.coef_all(output="records")
    polars_frame = standard_model.coef_all(output="polars")
    assert isinstance(records, list)
    assert _backend(polars_frame) == "polars"
    assert len(records) == polars_frame.height


# ─── q_stats: one implementation ───────────────────────────────────


@pytest.mark.parametrize("fixture", ["standard_model", "lss_model"])
def test_the_two_q_stats_paths_agree_exactly(request, fixture):
    model = request.getfixturevalue(fixture)
    method = model.q_stats(n_intervals=5, output="records")
    module = mb.q_stats(model, n_intervals=5, output="records")
    assert method == module


def test_q_stats_reports_the_binning_variable(standard_model):
    rows = standard_model.q_stats(n_intervals=3, output="records")
    assert {row["xvar"] for row in rows} == {"index"}
    named = standard_model.q_stats(xvar="x", n_intervals=3, output="records")
    assert {row["xvar"] for row in named} == {"x"}


def test_q_stats_bins_hold_near_equal_counts(standard_model):
    """Equal-count bins, so the moments down the column are comparable.

    Equal-width bins on a normal `x` put most of the data in the middle
    intervals and a handful in the tails, and a variance estimated from three
    points is not comparable with one estimated from sixty -- which is exactly
    the comparison a reader scanning the column is making.
    """
    rows = standard_model.q_stats(xvar="x", n_intervals=5, output="records")
    counts = [row["n"] for row in rows]
    assert max(counts) - min(counts) <= 1
    assert sum(counts) == 150


def test_q_stats_variance_is_the_unbiased_estimate(standard_model):
    """The two paths used to differ by exactly the n/(n-1) factor."""
    rows = standard_model.q_stats(n_intervals=1, output="records")
    residual = np.asarray(standard_model.quantile_residuals(random_state=0))
    # The residuals are randomized, so recompute both conventions from one
    # draw rather than comparing against a separately drawn vector.
    rows_seeded = standard_model.q_stats(
        n_intervals=1, random_state=0, output="records"
    )
    assert rows_seeded[0]["variance"] == pytest.approx(
        float(np.var(residual, ddof=1))
    )
    assert rows_seeded[0]["variance"] != pytest.approx(
        float(np.var(residual, ddof=0))
    )
    assert len(rows) == 1


def test_q_stats_kurtosis_is_excess(standard_model):
    """0 for a normal, not 3, so a column of near-zeros reads as 'fine'."""
    rows = standard_model.q_stats(n_intervals=1, random_state=0, output="records")
    residual = np.asarray(standard_model.quantile_residuals(random_state=0))
    centered = residual - residual.mean()
    deviation = np.sqrt(np.var(residual, ddof=1))
    expected = float(np.mean(centered ** 4) / deviation ** 4 - 3.0)
    assert rows[0]["kurtosis"] == pytest.approx(expected)


def test_q_stats_rejects_a_non_positive_interval_count(standard_model):
    with pytest.raises(ValueError, match="n_intervals must be positive"):
        standard_model.q_stats(n_intervals=0)


# ─── calibration_frame ─────────────────────────────────────────────


def test_calibration_frame_uses_the_lss_response_mean(lss_model):
    """Gaussian mu is the response mean, so the common name is exact."""
    rows = lss_model.calibration_frame(bins=4, output="records")
    assert "fitted_mean" in rows[0]
    assert "location_mean" not in rows[0]


def test_calibration_frame_keeps_fitted_mean_on_a_standard_model(standard_model):
    rows = standard_model.calibration_frame(bins=4, output="records")
    assert "fitted_mean" in rows[0]
    assert "location_mean" not in rows[0]


def test_calibration_frame_bins_hold_near_equal_counts(standard_model):
    rows = standard_model.calibration_frame(xvar="x", bins=5, output="records")
    counts = [row["n"] for row in rows]
    assert max(counts) - min(counts) <= 1


def test_calibration_frame_tracks_a_well_calibrated_fit(standard_model):
    """A correct fit puts the two columns close together in every bin."""
    rows = standard_model.calibration_frame(xvar="x", bins=5, output="records")
    for row in rows:
        assert row["observed_mean"] == pytest.approx(row["fitted_mean"], abs=0.5)


def test_calibration_frame_refuses_a_probability_matrix():
    """A multi-class fit predicts a matrix, so there is no column to bin."""
    rng = np.random.default_rng(9)
    n = 150
    x = rng.normal(size=n)
    y = rng.integers(0, 3, size=n).astype(float)
    model = mb.gamboost("y ~ bols(x)", data={"x": x, "y": y},
                        family=mb.Multinomial(),
                        control=mb.boost_control(20, 0.1))
    with pytest.raises(NotImplementedError, match="probability matrix"):
        model.calibration_frame()


def test_calibration_frame_rejects_a_non_positive_bin_count(standard_model):
    with pytest.raises(ValueError, match="bins must be positive"):
        standard_model.calibration_frame(bins=0)
