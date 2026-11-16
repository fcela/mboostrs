"""One weight contract, matching upstream's, across every fitting entry point.

A boosting fit uses weights in three places at once: the base-learner normal
equations (``X'WX``), the ``df -> lambda`` calibration that reads the hat
matrix of those same equations, and the empirical risk. All three scale with
the weight vector, so the *scale* of the weights, not only their ratios,
changes the fitted path.

``mboost_fit`` settles that scale before anything else runs::

    rescale_weights <- function(w) {
        if (max(abs(w - floor(w))) < sqrt(.Machine$double.eps)) return(w)
        return(w / sum(w) * sum(w > 0))
    }

Integer weights are frequencies and keep their scale; fractional weights are
analytic and are rescaled to sum to the number of weight-positive rows.
``mboostrs`` sent raw fractional weights to the engine, which moved every
fitted value and every risk value away from R. Integer-weight fixtures cannot
see it, because integers are the case upstream leaves unchanged, so the values
pinned here were taken from R directly.
"""

import importlib

import numpy as np
import pytest
from mboostrs._weights import (
    NormalizedWeights,
    is_integral,
    normalize_weights,
    rescale_fold_weights,
    rescale_weights,
)

import mboostrs as mb

# `mboostrs.stabsel` the module is shadowed by the re-exported function.
stabsel_module = importlib.import_module("mboostrs.stabsel")

# A ten-row Gaussian fixture whose weights sum to 8.4 over 10 positive rows,
# so the rescaling factor is 10 / 8.4 and no value survives it unchanged.
Y = np.array([
    -0.3718140203505656, -0.22288374723536017, -1.0441854737522702,
    -0.008991429626466318, -2.3172496935752873, -0.009077244884482139,
    2.0637411074029646, 1.5778617698893298, -0.9958399352398524,
    -1.4883295726494274,
])
X = np.array([
    0.1257302210933933, -0.1321048632913019, 0.6404226504432821,
    0.10490011715303971, -0.535669373161111, 0.36159505490948474,
    1.3040000451301372, 0.9470809631292422, -0.7037352358069926,
    -1.2654214710460525,
])
W_FRACTIONAL = np.array([0.3, 0.7, 1.2, 0.5, 2.4, 0.1, 0.9, 1.1, 0.4, 0.8])
DATA = {"y": Y, "x": X}

# Taken from R:
#   gamboost(y ~ bols(x, lambda = 1), data = d, weights = w,
#            control = boost_control(mstop = 20, nu = 0.1))
R_GAMBOOST_FITTED_HEAD = [-0.504738, -0.8489102, 0.1823014]
R_GAMBOOST_FINAL_RISK = 8.361211
#   glmboost(y ~ x, data = d, weights = w,
#            control = boost_control(mstop = 30, nu = 0.1))
R_GLMBOOST_COEF = {"(Intercept)": -0.7346396, "x": 1.529009}
R_GLMBOOST_FINAL_RISK = 7.984031


# ─── the policy itself ─────────────────────────────────────────────


def test_integral_detection_uses_upstream_tolerance():
    assert is_integral(np.array([1.0, 2.0, 0.0]))
    assert is_integral(np.array([1.0, 2.0 + 1e-12]))
    assert not is_integral(np.array([1.0, 2.5]))
    assert is_integral(np.array([]))


def test_rescale_leaves_integer_frequencies_alone():
    w = np.array([1.0, 3.0, 0.0, 2.0])
    np.testing.assert_array_equal(rescale_weights(w), w)


def test_rescale_scales_fractional_weights_to_the_positive_count():
    out = rescale_weights(W_FRACTIONAL)
    assert out.sum() == pytest.approx(10.0)
    # Ratios are untouched; only the scale moves.
    np.testing.assert_allclose(out / W_FRACTIONAL, np.full(10, 10.0 / 8.4))


def test_rescale_counts_positive_rows_not_all_rows():
    """A zero-weight row is not one of the rows the scale is set to."""
    w = np.array([0.5, 0.0, 1.5, 0.0])
    out = rescale_weights(w)
    assert out.sum() == pytest.approx(2.0)
    assert out[1] == 0.0 and out[3] == 0.0


def test_normalize_reports_input_and_effective_separately():
    got = normalize_weights(W_FRACTIONAL, 10)
    np.testing.assert_array_equal(got.given, W_FRACTIONAL)
    assert got.fit.sum() == pytest.approx(10.0)
    assert got.policy == "upstream" and got.rescaled
    assert "rescaled to sum to 10" in got.describe()


def test_raw_policy_passes_the_vector_through():
    got = normalize_weights(W_FRACTIONAL, 10, policy="raw")
    np.testing.assert_array_equal(got.fit, W_FRACTIONAL)
    assert not got.rescaled
    assert "no rescaling" in got.describe()


def test_unit_weights_have_no_input_vector():
    got = normalize_weights(None, 5)
    assert got.given is None and got.is_unit
    np.testing.assert_array_equal(got.fit, np.ones(5))
    assert got.describe() == "unit weights"


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.ones(3), "weights length 3 != n=10"),
        (np.array([1.0] * 9 + [np.nan]), "must be finite"),
        (np.array([1.0] * 9 + [-1.0]), "non-negative"),
        (np.zeros(10), "weights are all zero"),
    ],
)
def test_invalid_weights_are_rejected_before_any_arithmetic(weights, message):
    with pytest.raises(ValueError, match=message):
        normalize_weights(weights, 10)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="policy must be one of"):
        normalize_weights(np.ones(4), 4, policy="frequency")


def test_all_zero_can_be_allowed_explicitly():
    got = normalize_weights(np.zeros(4), 4, allow_all_zero=True)
    np.testing.assert_array_equal(got.fit, np.zeros(4))


def test_fold_matrix_rescales_column_by_column():
    """Each fold is its own weight vector, so each is rescaled on its own."""
    folds = np.array([
        [0.3, 0.0],
        [0.0, 0.7],
        [1.2, 0.0],
        [0.0, 0.5],
    ])
    out = rescale_fold_weights(folds)
    assert out[:, 0].sum() == pytest.approx(2.0)
    assert out[:, 1].sum() == pytest.approx(2.0)
    np.testing.assert_array_equal(out == 0, folds == 0)


def test_binary_fold_matrix_is_left_alone():
    folds = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(rescale_fold_weights(folds), folds)
    np.testing.assert_array_equal(rescale_fold_weights(folds, "raw"), folds)


def test_normalized_weights_repr_names_the_policy():
    got = NormalizedWeights(W_FRACTIONAL, np.ones(10), "upstream", True)
    assert "policy='upstream'" in repr(got)


# ─── the fitting entry points ──────────────────────────────────────


def test_gamboost_fractional_weights_match_r():
    model = mb.gamboost("y ~ bols(x, lambda=1)", data=DATA,
                        weights=W_FRACTIONAL,
                        control=mb.boost_control(20, 0.1))
    np.testing.assert_allclose(np.asarray(model.fitted())[:3],
                               R_GAMBOOST_FITTED_HEAD, atol=1e-6)
    assert np.asarray(model._risk)[-1] == pytest.approx(R_GAMBOOST_FINAL_RISK,
                                                        abs=1e-5)


def test_glmboost_fractional_weights_match_r():
    model = mb.glmboost("y ~ x", data=DATA, weights=W_FRACTIONAL,
                        control=mb.boost_control(30, 0.1))
    coefficients = model.coef(off2int=True)
    for name, expected in R_GLMBOOST_COEF.items():
        assert coefficients[name] == pytest.approx(expected, abs=1e-6)
    assert np.asarray(model._risk)[-1] == pytest.approx(R_GLMBOOST_FINAL_RISK,
                                                        abs=1e-5)


def test_raw_policy_differs_from_upstream_on_the_same_weights():
    """The policy is not cosmetic: it moves the fitted path."""
    upstream = mb.gamboost("y ~ bols(x, lambda=1)", data=DATA,
                           weights=W_FRACTIONAL,
                           control=mb.boost_control(20, 0.1))
    raw = mb.gamboost("y ~ bols(x, lambda=1)", data=DATA,
                      weights=W_FRACTIONAL, weight_policy="raw",
                      control=mb.boost_control(20, 0.1))
    assert not np.allclose(upstream.fitted(), raw.fitted(), atol=1e-6)
    np.testing.assert_allclose(raw.weights, W_FRACTIONAL)


@pytest.mark.parametrize("fit", ["glmboost", "gamboost"])
def test_explicit_lambda_is_where_a_weight_scale_error_shows(fit):
    """``df`` calibration can hide a weight-scale mismatch; ``lambda`` cannot.

    ``df -> lambda`` solves for the lambda giving the requested trace of the
    hat matrix, and that solve absorbs a change in the weight scale by moving
    lambda with it. An explicitly given lambda has nothing to absorb it, so a
    scale error lands directly in the fit. Both learners are checked with
    explicit lambda for exactly that reason.
    """
    formula = "y ~ x" if fit == "glmboost" else "y ~ bols(x, lambda=1)"
    entry = mb.glmboost if fit == "glmboost" else mb.gamboost
    scaled = entry(formula, data=DATA, weights=W_FRACTIONAL,
                   control=mb.boost_control(20, 0.1))
    doubled = entry(formula, data=DATA, weights=2.0 * W_FRACTIONAL,
                    control=mb.boost_control(20, 0.1))
    np.testing.assert_allclose(scaled.fitted(), doubled.fitted(), atol=1e-12)
    np.testing.assert_allclose(scaled._risk, doubled._risk, atol=1e-10)


def test_df_calibration_also_agrees_after_a_global_rescaling():
    a = mb.gamboost("y ~ bbs(x, df=3)", data=DATA, weights=W_FRACTIONAL,
                    control=mb.boost_control(20, 0.1))
    b = mb.gamboost("y ~ bbs(x, df=3)", data=DATA, weights=3.0 * W_FRACTIONAL,
                    control=mb.boost_control(20, 0.1))
    # `df -> lambda` is solved iteratively, so the two runs agree to the
    # solver's tolerance rather than bit-for-bit; without the rescaling they
    # differ in the third decimal.
    np.testing.assert_allclose(a.fitted(), b.fitted(), atol=1e-5)


def test_offset_is_computed_on_the_effective_weights():
    """The offset is a weighted summary, so it moves with the weight scale."""
    model = mb.glmboost("y ~ x", data=DATA, weights=W_FRACTIONAL,
                        control=mb.boost_control(1, 0.1))
    effective = model.weights
    assert model._offset == pytest.approx(float(np.average(Y, weights=effective)))
    # The raw and rescaled vectors are proportional, so a weighted *mean* is
    # the same either way -- which is why the offset alone cannot detect the
    # scale error, and the risk and lambda checks above are the ones that do.
    assert model._offset == pytest.approx(float(np.average(Y, weights=W_FRACTIONAL)))


@pytest.mark.parametrize("kind", ["glmboost", "gamboost"])
def test_model_reports_input_and_effective_weights(kind):
    formula = "y ~ x" if kind == "glmboost" else "y ~ bbs(x, df=3)"
    entry = mb.glmboost if kind == "glmboost" else mb.gamboost
    model = entry(formula, data=DATA, weights=W_FRACTIONAL,
                  control=mb.boost_control(10, 0.1))
    np.testing.assert_array_equal(model.input_weights(), W_FRACTIONAL)
    assert model.weights.sum() == pytest.approx(10.0)
    assert model.weight_policy == "upstream"
    assert "rescaled" in model.weights_note()


def test_unweighted_model_reports_no_input_weights():
    model = mb.glmboost("y ~ x", data=DATA, control=mb.boost_control(5, 0.1))
    assert model.input_weights() is None
    np.testing.assert_array_equal(model.weights, np.ones(10))
    assert model.weights_note() == "unit weights"


@pytest.mark.parametrize("kind", ["glmboostLSS", "gamboostLSS"])
def test_lss_applies_the_same_policy(kind):
    rng = np.random.default_rng(4)
    n = 120
    data = {"x1": rng.normal(size=n), "x2": rng.normal(size=n)}
    data["y"] = rng.normal(1.0 + 0.8 * data["x1"], np.exp(0.3 * data["x2"]))
    w = rng.uniform(0.4, 2.5, n)
    formula = "y ~ x1 + x2" if kind == "glmboostLSS" else "y ~ bbs(x1, df=3) + bbs(x2, df=3)"
    entry = mb.glmboostLSS if kind == "glmboostLSS" else mb.gamboostLSS

    scaled = entry(formula, data=data, families=mb.GaussianLSS(), weights=w,
                   control=mb.boost_control(30, 0.1))
    doubled = entry(formula, data=data, families=mb.GaussianLSS(),
                    weights=2.0 * w, control=mb.boost_control(30, 0.1))

    assert scaled.weights.sum() == pytest.approx(float(n))
    np.testing.assert_array_equal(scaled.input_weights(), w)
    np.testing.assert_allclose(scaled._risk, doubled._risk, atol=1e-9)


def test_lss_rejects_all_zero_weights():
    rng = np.random.default_rng(5)
    n = 40
    data = {"x1": rng.normal(size=n)}
    data["y"] = rng.normal(size=n)
    with pytest.raises(ValueError, match="weights are all zero"):
        mb.glmboostLSS("y ~ x1", data=data, families=mb.GaussianLSS(),
                       weights=np.zeros(n), control=mb.boost_control(10, 0.1))


def test_zero_weights_keep_their_rows_in_place():
    """An excluded row keeps its position, so row-aligned objects stay aligned."""
    w = W_FRACTIONAL.copy()
    w[[2, 5]] = 0.0
    model = mb.glmboost("y ~ x", data=DATA, weights=w,
                        control=mb.boost_control(20, 0.1))
    assert len(model.fitted()) == 10
    effective = model.weights
    assert effective[2] == 0.0 and effective[5] == 0.0
    assert effective.sum() == pytest.approx(8.0)  # eight positive rows


def test_binary_zero_one_weights_are_frequencies_and_pass_through():
    w = np.array([1.0, 0.0] * 5)
    model = mb.glmboost("y ~ x", data=DATA, weights=w,
                        control=mb.boost_control(20, 0.1))
    np.testing.assert_array_equal(model.weights, w)
    assert not model._weights_rescaled


def test_survey_like_weights_rescale_to_the_sample_size():
    """Weights summing to a population total are analytic, not frequencies."""
    # A multiplier that leaves the weights fractional: scaled by an integer
    # factor they would land on integers and be read as frequencies instead.
    w = W_FRACTIONAL * 12_345.0 + 0.5
    model = mb.glmboost("y ~ x", data=DATA, weights=w,
                        control=mb.boost_control(20, 0.1))
    assert model.weights.sum() == pytest.approx(10.0)
    raw = mb.glmboost("y ~ x", data=DATA, weights=w, weight_policy="raw",
                      control=mb.boost_control(20, 0.1))
    assert raw.weights.sum() == pytest.approx(w.sum())
    # Same coefficients, different risk scale: the ratios drive the fit and
    # the scale drives the risk.
    np.testing.assert_allclose(model.fitted(), raw.fitted(), atol=1e-10)
    assert not np.allclose(model._risk, raw._risk)


# ─── resampling inherits the policy ────────────────────────────────


def test_cv_folds_are_rescaled_column_by_column():
    """Upstream refits each fold through ``mboost_fit``, which rescales it.

    Asserted through the result rather than the call: a fold matrix whose
    columns have already been rescaled must give the same cross-validated
    risk as the unrescaled one, which is only true if the fold columns go
    through the policy on the way in.
    """
    rng = np.random.default_rng(7)
    n = 80
    data = {"x": rng.normal(size=n)}
    data["y"] = 2.0 * data["x"] + rng.normal(size=n)
    w = rng.uniform(0.4, 2.0, n)
    model = mb.glmboost("y ~ x", data=data, weights=w,
                        control=mb.boost_control(30, 0.1))

    folds = mb.cv_weights(n, type="kfold", B=4, seed=3, weights=model.weights)
    prescaled = rescale_fold_weights(folds)
    assert not np.allclose(folds, prescaled), "fixture must have fractional folds"
    for j in range(prescaled.shape[1]):
        column = prescaled[:, j]
        assert column.sum() == pytest.approx(float(np.count_nonzero(column)))

    np.testing.assert_allclose(
        model.cvrisk(folds=folds).risk_matrix,
        model.cvrisk(folds=prescaled).risk_matrix,
        atol=1e-10,
    )


def test_stabsel_folds_come_from_the_effective_weights():
    rng = np.random.default_rng(8)
    n = 90
    data = {f"x{j}": rng.normal(size=n) for j in range(4)}
    data["y"] = 2.0 * data["x0"] + rng.normal(size=n)
    w = rng.uniform(0.4, 2.0, n)
    model = mb.glmboost("y ~ " + " + ".join(f"x{j}" for j in range(4)),
                        data=data, weights=w,
                        control=mb.boost_control(40, 0.1))

    seen = {}
    original = stabsel_module._selected_paths

    def _capture(model_, folds, n_fits, *args, **kwargs):
        seen["folds"] = folds
        return original(model_, folds, n_fits, *args, **kwargs)

    stabsel_module._selected_paths = _capture
    try:
        result = mb.stabsel(model, q=2, PFER=1.0, B=3, verbose=False)
    finally:
        stabsel_module._selected_paths = original

    folds = seen["folds"]
    effective = model.weights
    for b in range(3):
        np.testing.assert_allclose(folds[:, b] + folds[:, 3 + b], effective)
    assert (result.B, result.n_fits) == (3, 6)


def test_folds_heavier_than_the_model_are_rejected_with_the_fix():
    """Drawing folds from the raw input under a rescaled fit is caught."""
    rng = np.random.default_rng(9)
    n = 60
    data = {"x": rng.normal(size=n)}
    data["y"] = 2.0 * data["x"] + rng.normal(size=n)
    w = np.full(n, 2.5)  # rescales down to 1.0 per row
    model = mb.glmboost("y ~ x", data=data, weights=w,
                        control=mb.boost_control(20, 0.1))
    raw_folds = mb.cv_weights(n, type="subsampling", B=2, seed=4, weights=w)
    with pytest.raises(ValueError, match="more weight than the model does"):
        mb.stabsel(model, q=1, PFER=1.0, folds=raw_folds, verbose=False)


def test_continue_to_keeps_the_policy():
    model = mb.glmboost("y ~ x", data=DATA, weights=W_FRACTIONAL,
                        weight_policy="raw", control=mb.boost_control(10, 0.1))
    grown = model.continue_to(20)
    assert grown.weight_policy == "raw"
    np.testing.assert_allclose(grown.weights, W_FRACTIONAL)


@pytest.mark.parametrize("fit", ["gamboost", "glmboost"])
def test_fold_mask_carries_the_model_weights(fit):
    """A 0/1 fold mask given to a weighted model refits each fold with the
    model's weights, the same curve as folds built from those weights."""
    rng = np.random.default_rng(4)
    n = 120
    x = rng.uniform(-1, 1, n)
    w = rng.uniform(0.2, 3.0, n)
    data = {"y": np.sin(2 * x) + rng.normal(0, 0.3, n), "x": x}
    formula = "y ~ bbs(x, df=4)" if fit == "gamboost" else "y ~ x"
    model = getattr(mb, fit)(formula, data=data, weights=w,
                             control=mb.boost_control(mstop=40))
    mask = mb.cv_weights(n, "kfold", 4, seed=1)
    carried = mb.cv_weights(n, "kfold", 4, seed=1, weights=w)
    from_mask = model.cvrisk(folds=mask)
    from_weights = model.cvrisk(folds=carried)
    np.testing.assert_allclose(np.asarray(from_mask.risk_matrix),
                               np.asarray(from_weights.risk_matrix), rtol=1e-8)
