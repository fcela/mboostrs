"""Helpers that close named gaps against mboost, gamboostLSS and gamlss.

Each is checked against the R function it corresponds to, on the same inputs,
rather than against its own docstring.
"""

import numpy as np
import pytest
from mboostrs.cvrisk import make_grid, weighted_median
from mboostrs.formula import expand_formula
from mboostrs.model_selection import r_squared

import mboostrs as mb

# ---------------------------------------------------------------------------
# make.grid
# ---------------------------------------------------------------------------


def test_scalar_grid_matches_r():
    # gamboostLSS::make.grid(100, length.out = 6)
    assert make_grid(100, length_out=6) == [1, 3, 6, 16, 40, 100]


def test_product_grid_matches_r():
    # make.grid(c(mu = 50, sigma = 20), length.out = 4, dense_mu_grid = FALSE)
    points = make_grid({"mu": 50, "sigma": 20}, length_out=4, dense_mu_grid=False)
    assert sorted((p["mu"], p["sigma"]) for p in points) == sorted(
        (mu, sigma) for mu in (1, 4, 14, 50) for sigma in (1, 3, 7, 20)
    )


def test_dense_mu_grid_matches_r():
    """The dense fill exists because cross-validation can extend one fitted
    path along the first parameter without refitting, so those points are
    nearly free. R produces 179 points where the sparse grid has 16."""
    dense = make_grid({"mu": 50, "sigma": 20}, length_out=4)
    sparse = make_grid({"mu": 50, "sigma": 20}, length_out=4, dense_mu_grid=False)
    assert len(sparse) == 16
    assert len(dense) == 179
    # Every sparse point survives, and the extra ones only fill in along mu.
    assert {(p["mu"], p["sigma"]) for p in sparse} <= {
        (p["mu"], p["sigma"]) for p in dense
    }


def test_linear_grid_is_not_log_spaced():
    assert make_grid(100, length_out=5, log=False) == [1, 26, 50, 75, 100]


def test_grid_rejects_impossible_bounds():
    with pytest.raises(ValueError, match="min must be below max"):
        make_grid(10, min=10)
    with pytest.raises(ValueError, match="logarithmic grid needs min >= 1"):
        make_grid(10, min=0)


def test_grid_drives_cvrisk():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    data = {"x": x, "y": 1.5 * x + rng.normal(0, np.exp(-0.3 + 0.4 * x), n)}
    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"}, data=data,
        families=mb.GaussianLSS(), control=mb.boost_control(40, 0.1),
    )
    points = make_grid({"mu": 40, "sigma": 20}, length_out=3, dense_mu_grid=False)
    result = fit.cvrisk(
        folds=mb.cv_weights(n, type="kfold", B=4),
        grid=[[p["mu"], p["sigma"]] for p in points],
    )
    assert len(result["mstop"]) == 2


# ---------------------------------------------------------------------------
# weighted.median
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x,w,expected", [
    ([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 3.0),      # odd total weight
    ([1, 2, 3, 4], [1, 1, 1, 1], 2.5),            # even total: average the pair
    ([1, 2, 3, 4, 5], [0.1, 0.1, 0.1, 0.1, 5], 5.0),   # one dominant weight
    ([5, 1, 4, 2, 3], [2, 1, 1, 3, 1], 2.5),      # unsorted input
    ([10, 20, 30], [0, 1, 1], 25.0),              # zero weights are dropped
])
def test_weighted_median_matches_r(x, w, expected):
    assert weighted_median(x, w) == pytest.approx(expected)


def test_weighted_median_handles_missing_values():
    assert np.isnan(weighted_median([1.0, np.nan, 3.0], [1, 1, 1]))
    assert weighted_median([1.0, np.nan, 3.0, 5.0], [1, 1, 1, 1], na_rm=True) == \
        pytest.approx(3.0)


def test_weighted_median_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="w has length"):
        weighted_median([1, 2, 3], [1, 1])


# ---------------------------------------------------------------------------
# `.` inside a learner call
# ---------------------------------------------------------------------------


def _four_predictors():
    rng = np.random.default_rng(1)
    n = 300
    data = {f"x{j}": rng.normal(size=n) for j in range(4)}
    data["y"] = np.sin(data["x0"]) + data["x1"] * data["x2"] + rng.normal(0, 0.3, n)
    return data


def test_dot_inside_a_learner_call_expands_to_every_predictor():
    """`y ~ btree(.)` is how a tree over all predictors is written without
    naming them — the equivalent of R's `blackboost(y ~ ., data)`."""
    data = _four_predictors()
    assert expand_formula("y ~ btree(.)", data) == "y ~ btree(x0, x1, x2, x3)"
    assert expand_formula("y ~ btree(., maxdepth=3)", data) == \
        "y ~ btree(x0, x1, x2, x3, maxdepth=3)"


def test_dot_inside_a_call_skips_predictors_already_named():
    data = _four_predictors()
    assert expand_formula("y ~ btree(.) + bols(x0)", data) == \
        "y ~ btree(x1, x2, x3) + bols(x0)"


def test_a_tree_over_all_predictors_fits():
    data = _four_predictors()
    fit = mb.gamboost("y ~ btree(., maxdepth=3)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(80, 0.1))
    assert len(fit._learners_spec) == 1
    assert fit._learners_spec[0].names == ["x0", "x1", "x2", "x3"]
    assert fit.risk_[-1] < fit.risk_[0] / 2


def test_plain_formulas_are_untouched():
    data = _four_predictors()
    assert expand_formula("y ~ bbs(x0, df=4) + bols(x1)", data) == \
        "y ~ bbs(x0, df=4) + bols(x1)"


# ---------------------------------------------------------------------------
# Rsq
# ---------------------------------------------------------------------------


def _heteroscedastic_fit(mstop):
    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(size=n)
    data = {"x": x, "y": 1.5 * x + rng.normal(0, np.exp(-0.3 + 0.4 * x), n)}
    return mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"}, data=data,
        families=mb.GaussianLSS(), control=mb.boost_control(mstop, 0.1),
    )


def test_r_squared_rises_along_the_path():
    values = [r_squared(_heteroscedastic_fit(m), "cox-snell") for m in (1, 5, 30, 150)]
    assert values == sorted(values)
    assert values[0] < 0.1 and values[-1] > 0.7


def test_r_squared_matches_its_definition():
    fit = _heteroscedastic_fit(150)
    risk = np.asarray(fit.risk_)
    n = len(np.asarray(fit.fitted(parameter="mu")))
    expected = 1.0 - np.exp((2.0 / n) * (-risk[0] + risk[-1]))
    assert r_squared(fit, "cox-snell") == pytest.approx(expected)
    both = r_squared(fit)
    assert both["cragg_uhler"] >= both["cox_snell"]


def test_r_squared_refuses_a_standard_model():
    """A standard model's `risk_` is the family's loss. Using `-risk_` as a
    log-likelihood regardless gave 0.990 on a Gaussian fit whose ordinary
    R-squared is 0.902, which is why this raises instead."""
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=n)
    fit = mb.glmboost("y ~ x", data={"x": x, "y": 1.5 * x + rng.normal(0, 0.5, n)},
                      family=mb.Gaussian(), control=mb.boost_control(200, 0.1))
    with pytest.raises(NotImplementedError, match="distributional"):
        r_squared(fit)


def test_r_squared_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        r_squared(_heteroscedastic_fit(10), kind="adjusted")


# ---------------------------------------------------------------------------
# The fold argument an LSS cvrisk needs
# ---------------------------------------------------------------------------


def test_lss_cvrisk_says_what_folds_must_be():
    """`folds=3` used to fail with `AttributeError: 'int' object has no
    attribute 'shape'` from inside the fold machinery."""
    fit = _heteroscedastic_fit(20)
    with pytest.raises(ValueError, match=r"folds must be an \(n, B\) matrix"):
        fit.cvrisk(folds=3)


def test_lss_cvrisk_needs_no_arguments():
    """It used to require both `folds` and `grid`, arguing that the fold
    scheme has to be shared across the whole grid -- which is a reason to
    build one and share it, not a reason to have none. Reaching a first
    cross-validated result meant assembling two matrices by hand."""
    import numpy as np

    fit = _heteroscedastic_fit(30)
    result = fit.cvrisk()
    assert result.risk_matrix.shape[0] == 5
    assert np.all(np.isfinite(result.aggregate))
    # The default grid is the diagonal, which the engine reads off one fitted
    # path per fold.
    cost = result.grid_cost()
    assert cost["diagonal"] == cost["points"]
    assert cost["fresh_fits"] == 0
