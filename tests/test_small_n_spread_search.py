"""A spread search whose mean model holds a factor by= term.

Before this fix, gamboostLSS refused ``bols(f, by=z)`` for a factor ``f`` with
more than two levels, and every candidate of a spread search carries the mean
model, so the whole search raised "no specification could be scored". gamboost
accepted the same term and multiplied the integer codes of ``f`` by ``z``, as
if the levels were a measurement. Both now lower the term to one column per
non-reference level, multiplied by ``z``, which is what mboost does.

Fold construction for compare_specifications keeps each combination of the
factor levels the candidates use in every training fold where a combination
has at least two rows, so a small sample does not drop a level from a fold.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.explore import _level_balanced_folds


def _rows(n=300, seed=3):
    rng = np.random.default_rng(seed)
    f = rng.choice(["A", "B", "C", "D"], n)
    z = (rng.uniform(size=n) > 0.5).astype(float)
    x = rng.uniform(size=n)
    effect = {"A": 0.0, "B": 0.5, "C": -0.3, "D": 1.0}
    y = np.array([effect[v] for v in f]) * z + np.sin(3 * x) + rng.normal(0, 0.3, n)
    return {"y": y, "x": x, "f": f, "z": z}


def test_factor_bols_with_numeric_by_is_a_level_design_times_by():
    # One Gaussian boosting step with nu = 1 and no penalty is the least-squares
    # projection of the centred response onto the columns (level == l) * z.
    d = _rows()
    fit = mb.gamboost("y ~ bols(f, by=z)", data=d, family=mb.Gaussian(), control=mb.boost_control(mstop=1, nu=1.0))
    columns = np.column_stack([(d["f"] == level) * d["z"] for level in "ABCD"])
    mean = d["y"].mean()
    coef, *_ = np.linalg.lstsq(columns, d["y"] - mean, rcond=None)
    np.testing.assert_allclose(np.ravel(fit.fitted()), mean + columns @ coef, atol=1e-8)


def test_factor_bols_with_numeric_by_is_not_read_as_level_codes():
    # Codes 0..3 times z is the silent error this change removes: the fit must
    # differ from the fit on the codes.
    d = _rows()
    codes = np.array([{"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}[v] for v in d["f"]])
    ctrl = mb.boost_control(mstop=30, nu=0.1)
    by_factor = mb.gamboost("y ~ bols(f, by=z)", data=d, family=mb.Gaussian(), control=ctrl)
    on_codes = mb.gamboost("y ~ bols(code, by=z)", data={**d, "code": codes}, family=mb.Gaussian(), control=ctrl)
    assert np.max(np.abs(np.ravel(by_factor.fitted()) - np.ravel(on_codes.fitted()))) > 1e-3


def test_gamboostlss_accepts_factor_bols_with_numeric_by_and_replays_it():
    d = _rows()
    fit = mb.gamboostLSS(
        {"mu": "y ~ bols(f, by=z) + bbs(x, df=4)", "sigma": "y ~ 1"},
        data=d,
        families=mb.GaussianLSS(),
        control=mb.boost_control(mstop=40),
    )
    in_sample = fit.predict(newdata=d, type="link")
    refit_rows = {k: np.asarray(v) for k, v in in_sample.items()}
    assert all(np.all(np.isfinite(v)) for v in refit_rows.values())
    np.testing.assert_allclose(refit_rows["mu"], np.asarray(fit.predict(type="link")["mu"]), atol=1e-10)


def test_factor_bols_with_by_predicts_new_rows_through_the_same_design():
    d = _rows()
    d["g"] = np.where(d["z"] > 0.5, "p", "q")
    fit = mb.gamboost("y ~ bols(f, by=z) + bols(f, by=g)", data=d, family=mb.Gaussian(),
                      control=mb.boost_control(mstop=30, nu=0.1))
    rows = {k: np.asarray(v)[:25] for k, v in d.items()}
    in_sample = np.ravel(fit.fitted())[:25]
    np.testing.assert_allclose(np.ravel(fit.predict(newdata=rows)), in_sample, atol=1e-10)


def test_factor_bols_by_a_factor_is_one_coefficient_per_pair_of_levels():
    # bols(f, by=g) is the cell design of the two factors: one indicator per
    # (level of f, level of g). One unpenalized step with nu = 1 projects the
    # centred response onto those indicators.
    d = _rows()
    d["g"] = np.where(d["z"] > 0.5, "p", "q")
    fit = mb.gamboost("y ~ bols(f, by=g)", data=d, family=mb.Gaussian(), control=mb.boost_control(mstop=1, nu=1.0))
    cells = [(d["f"] == a) & (d["g"] == b) for a in "ABCD" for b in ("p", "q")]
    columns = np.column_stack(cells).astype(float)
    mean = d["y"].mean()
    coef, *_ = np.linalg.lstsq(columns, d["y"] - mean, rcond=None)
    np.testing.assert_allclose(np.ravel(fit.fitted()), mean + columns @ coef, atol=1e-8)


def test_compare_specifications_scores_spread_candidates_on_a_small_sample():
    # The shape of the failing search: a mean model with a factor by= term and
    # sigma candidates that include a categorical one. Every candidate scores.
    d = _rows(n=160, seed=8)
    mu = "bbs(x, df=4) + bols(f, by=z)"
    specs = [
        {"mu": mu, "sigma": "1"},
        {"mu": mu, "sigma": "bbs(x, df=4)"},
        {"mu": mu, "sigma": "bols(f)"},
    ]
    rows = mb.compare_specifications(
        specs, d, "y", family=mb.GaussianLSS(), folds=4, mstop=40, seed=5, na_action="impute"
    )
    assert [row["error"] for row in rows] == [None, None, None]
    assert all(np.isfinite(row["out-of-fold risk"]) for row in rows)


def test_compare_specifications_names_the_cause_when_no_candidate_scores():
    d = _rows(n=120, seed=9)
    specs = [{"mu": "bbs(f, df=4)", "sigma": "1"}, {"mu": "bbs(f, df=4)", "sigma": "bbs(x, df=4)"}]
    with pytest.raises(ValueError, match="no specification could be scored") as info:
        mb.compare_specifications(specs, d, "y", family=mb.GaussianLSS(), folds=3, mstop=10, seed=1)
    assert "categorical column with 4 levels" in str(info.value)


def test_level_balanced_folds_keep_each_level_in_every_training_fold():
    rng = np.random.default_rng(0)
    # 40 rows; level "E" has 5 rows, so five folds can each hold one out.
    f = np.array(["A"] * 15 + ["B"] * 12 + ["C"] * 8 + ["E"] * 5, dtype=object)
    rng.shuffle(f)
    data = {"f": f, "y": np.zeros(f.size)}
    scheme = _level_balanced_folds(5, data, ["y ~ bols(f)"], f.size, seed=2, weights=None)
    held = np.asarray(scheme.held_out, dtype=bool)
    assert held.shape == (40, 5)
    assert np.all(held.sum(axis=1) == 1)  # every row is held out once
    for b in range(5):
        train_levels = set(f[~held[:, b]])
        assert train_levels == {"A", "B", "C", "E"}, b


def _far_row_data():
    # One row sits 1000 away from the curve on a noise scale of 0.01, so its
    # held-out density underflows to zero in double precision.
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 160)
    y = np.sin(2 * x) + rng.normal(0, 0.01, 160)
    y[0] += 1000.0
    return {"x": x, "y": y}


_FAR_ROW_SPECS = [{"mu": "bbs(x, df=4)", "sigma": "1"}, {"mu": "bbs(x, df=4)", "sigma": "bbs(x, df=4)"}]


def test_held_out_density_underflow_is_scored_on_the_log_scale():
    # The core loss is the negative log-likelihood, computed on the log scale,
    # so a row whose density underflows still has a finite loss.
    rows = mb.compare_specifications(
        _FAR_ROW_SPECS, _far_row_data(), "y", family=mb.GaussianLSS(), folds=4, mstop=100,
        nu=0.1, seed=1, na_action="impute", extrapolation="allow",
    )
    assert all(np.isfinite(row["out-of-fold risk"]) for row in rows)


def test_a_non_finite_held_out_loss_names_its_cause(monkeypatch):
    # A row whose core loss is not finite fails the fold, and the comparison
    # reports the cause instead of a bare "fold_scores must be finite".
    from mboostrs import _rust

    real = _rust.lss_row_nll

    def first_row_infinite(*args):
        loss, parameters = real(*args)
        loss = np.array(loss)
        loss[0] = np.inf
        return loss, parameters

    monkeypatch.setattr(_rust, "lss_row_nll", first_row_infinite)
    with pytest.raises(ValueError, match="held-out density of 1 row") as info:
        mb.compare_specifications(
            _FAR_ROW_SPECS, _far_row_data(), "y", family=mb.GaussianLSS(), folds=4,
            mstop=100, nu=0.1, seed=1, na_action="impute", extrapolation="allow",
        )
    assert "fold_scores must be finite" not in str(info.value)


def test_level_balanced_folds_leave_numeric_only_comparisons_unchanged():
    data = {"x": np.arange(30.0), "y": np.zeros(30)}
    assert _level_balanced_folds(5, data, ["y ~ bbs(x)"], 30, seed=1, weights=None) == 5
