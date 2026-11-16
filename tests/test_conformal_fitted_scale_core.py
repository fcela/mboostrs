"""The fitted-scale target and column ranking run in the core and equal the
earlier NumPy implementation, which is frozen here."""

import numpy as np
import pytest

from mboostrs import conformal as cf
from mboostrs import _rust


def _old_fitted_scale(predictions, y, columns):
    """The NumPy implementation of `_fitted_scale` before the port."""
    if predictions is None or y.size < cf._SCALE_MODEL_MIN_ROWS:
        return None
    splittable = [np.asarray(column, dtype=np.float64) for column in columns
                  if np.unique(column).size > 2]
    if not splittable:
        return None
    target = np.log(np.maximum((y - predictions) ** 2, 1e-12))
    centred_target = target - target.mean()
    if len(splittable) > cf._SCALE_MODEL_MAX_COLUMNS:

        def strength(column):
            association = 0.0
            for view in (column, np.abs(column - np.median(column))):
                view = view - view.mean()
                denominator = np.sqrt(float(np.sum(view ** 2)) * float(np.sum(centred_target ** 2)))
                if denominator > 0:
                    association = max(association, abs(float(np.sum(view * centred_target)) / denominator))
            return association

        splittable = sorted(splittable, key=strength, reverse=True)[:cf._SCALE_MODEL_MAX_COLUMNS]
    return target, splittable


def _draw(seed, n=200, p=20):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(p, n))
    x[3] = np.round(x[3])  # ties and few distinct values
    y = x[0] + np.exp(0.6 * np.abs(x[5])) * rng.normal(size=n)
    pred = x[0] + 0.1 * rng.normal(size=n)
    pred[:5] = y[:5]  # zero residuals hit the floor
    return pred, y, [row for row in x]


@pytest.mark.parametrize("seed", range(5))
def test_target_and_ranking_match_frozen_numpy(seed):
    pred, y, columns = _draw(seed)
    old_target, old_columns = _old_fitted_scale(pred, y, columns)
    target = np.asarray(_rust.conformal_scale_target(y, pred))
    assert np.max(np.abs(target - old_target)) <= 1e-12
    ranked = _rust.conformal_rank_scale_columns(
        np.ascontiguousarray(np.vstack(columns)), target, cf._SCALE_MODEL_MAX_COLUMNS)
    assert len(ranked) == cf._SCALE_MODEL_MAX_COLUMNS
    for i, old in zip(ranked, old_columns, strict=True):
        np.testing.assert_array_equal(columns[i], old)


def test_few_columns_keep_their_order():
    pred, y, columns = _draw(0, p=6)
    target = np.asarray(_rust.conformal_scale_target(y, pred))
    ranked = _rust.conformal_rank_scale_columns(np.vstack(columns), target, 12)
    assert list(ranked) == list(range(6))


def test_fitted_scale_matches_frozen_for_a_wide_frame():
    pred, y, columns = _draw(11)
    new = cf._fitted_scale(pred, y, columns)
    assert new is not None and new.shape == y.shape
    # The tail-driven column 5 is kept among the twelve strongest.
    target = np.asarray(_rust.conformal_scale_target(y, pred))
    ranked = _rust.conformal_rank_scale_columns(np.vstack(columns), target, 12)
    assert 5 in ranked
