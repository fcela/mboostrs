"""Predictions at several path positions cost one pass over the basis.

P3.4 item 1: "reuse or batch new-data basis evaluation across parameter grids
and repeated diagnostic calls".

`predict(newdata=, mstop=k)` already builds each learner's basis once per
call — that is what `predict_newdata_grouped` is for. What it does not do is
share that work across a *grid*: comparing six stopping points rebuilt every
B-spline basis six times to change nothing but how many path steps were
summed. Measured on 20,000 rows and eight `bbs` learners, six positions cost
51.0 ms against 8.7 for one.

`predict_over_mstop` walks the path once, carrying the coefficient sums
forward from one position to the next: 20.8 ms for the same six, and the
saving grows with the grid.

The answers must be identical, not close: this is the same arithmetic in a
different order of accumulation, and every test here compares against the
one-call-per-position result exactly.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb

N = 2000


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    frame = {f"x{i}": rng.normal(size=N) for i in range(5)}
    frame["y"] = (frame["x0"] + np.sin(2 * frame["x1"])
                  + rng.normal(scale=0.3, size=N))
    frame["g"] = rng.integers(0, 4, N).astype(float)
    return frame


def fit(formula, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboost(formula, data, control=mb.boost_control(60, 0.1),
                           **kwargs)


GRID = [5, 12, 30, 45, 60]


@pytest.mark.parametrize("formula", [
    "y ~ bbs(x0, df=4) + bbs(x1, df=4)",
    "y ~ bols(x0) + bbs(x1, df=4)",
    "y ~ bols(x0) + brandom(g, df=3)",
])
def test_it_matches_one_call_per_position_exactly(formula, data):
    # Bit-for-bit, not close: summing a learner's coefficients before
    # evaluating its basis once is the same arithmetic in the same order as
    # evaluating it per step and adding, for every learner linear in its
    # coefficients.
    model = fit(formula, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one_by_one = np.vstack([
            np.asarray(model.predict(newdata=data, mstop=position))
            for position in GRID
        ])
        batched = model.predict_over_mstop(GRID, newdata=data, output="numpy")
    assert batched.shape == one_by_one.shape
    assert np.array_equal(batched, one_by_one), (
        f"max difference {np.max(np.abs(batched - one_by_one)):.3e}")


def test_a_tree_term_matches_to_rounding(data):
    """Two fitted trees do not add, so tree steps cannot be summed the way
    linear ones are: they accumulate into their own running total and are
    added to the offset afterwards, where `predict` adds them into an
    offset-seeded vector. The two differ in the order of one addition —
    4.4e-16 at mstop 60 — which is why this one is a tolerance and the others
    are exact."""
    model = fit("y ~ btree(x0) + bbs(x1, df=4)", data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batched = model.predict_over_mstop([1, 2, 3, 60], newdata=data,
                                           output="numpy")
        expected = np.vstack([
            np.asarray(model.predict(newdata=data, mstop=position))
            for position in (1, 2, 3, 60)
        ])
    assert np.allclose(batched, expected, rtol=0, atol=1e-12)


def test_the_response_scale_matches_too(data):
    binary = dict(data)
    binary["y"] = (data["y"] > 0).astype(float)
    model = fit("y ~ bbs(x0, df=4)", binary, family=mb.Binomial())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batched = model.predict_over_mstop(GRID, newdata=binary,
                                           type="response", output="numpy")
        expected = np.vstack([
            np.asarray(model.predict(newdata=binary, mstop=position,
                                     type="response"))
            for position in GRID
        ])
    assert np.array_equal(batched, expected)
    assert np.all((batched >= 0.0) & (batched <= 1.0))


def test_a_descending_grid_is_refused(data):
    """The grid is walked once and each position continues the previous one's
    sums, so an out-of-order position would silently get the wrong prefix."""
    model = fit("y ~ bbs(x0, df=4)", data)
    with pytest.raises(ValueError, match="ascending"):
        model.predict_over_mstop([30, 10], newdata=data)


def test_an_empty_grid_is_refused(data):
    model = fit("y ~ bbs(x0, df=4)", data)
    with pytest.raises(ValueError, match="at least one position"):
        model.predict_over_mstop([], newdata=data)


def test_a_negative_position_is_refused(data):
    model = fit("y ~ bbs(x0, df=4)", data)
    with pytest.raises(ValueError, match="non-negative"):
        model.predict_over_mstop([-1, 5], newdata=data)


def test_a_position_past_the_path_is_the_whole_path(data):
    model = fit("y ~ bbs(x0, df=4)", data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batched = model.predict_over_mstop([60, 500], newdata=data,
                                           output="numpy")
    assert np.array_equal(batched[0], batched[1])


def test_a_restored_model_falls_back_rather_than_failing(data, tmp_path):
    """A model with no learner store cannot batch. It takes the ordinary path
    once per position: correct, and no faster."""
    model = fit("y ~ bbs(x0, df=4)", data)
    path = tmp_path / "model.json"
    model.save(str(path))
    restored = mb.load_model(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batched = restored.predict_over_mstop([10, 30], newdata=data,
                                              output="numpy")
        expected = np.vstack([
            np.asarray(restored.predict(newdata=data, mstop=position))
            for position in (10, 30)
        ])
    assert np.array_equal(batched, expected)


def test_the_frame_names_its_columns_after_the_positions(data):
    model = fit("y ~ bbs(x0, df=4)", data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = model.predict_over_mstop([5, 60], newdata=data)
    assert set(frame.columns) == {"row", "mstop_5", "mstop_60"}
    assert len(frame) == N
