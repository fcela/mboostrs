"""Binomial fits with a proportion or cbind(success, failure) response (#816).

A row with ``s`` successes in ``t`` trials enters the fit as the proportion
``s / t`` with prior weight ``t``. That fit must equal the fit on the Bernoulli
expansion, one row per trial with a {0, 1} response.
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import r_call


def _grouped(n_groups=60, seed=816):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-2.0, 2.0, n_groups))
    trials = rng.integers(1, 9, n_groups).astype(float)
    prob = 1.0 / (1.0 + np.exp(-(0.3 + 1.1 * x)))
    success = rng.binomial(trials.astype(int), prob).astype(float)
    return x, success, trials - success


def _expanded(x, success, failure):
    rows_x, rows_y = [], []
    for xi, s, f in zip(x, success, failure, strict=True):
        rows_x += [xi] * int(s + f)
        rows_y += [1.0] * int(s) + [0.0] * int(f)
    return np.asarray(rows_x), np.asarray(rows_y)


def test_constant_seven_in_ten_has_mean_probability_point_seven():
    n = 200
    rng = np.random.default_rng(0)
    data = {"s": np.full(n, 7.0), "f": np.full(n, 3.0), "x": rng.normal(size=n)}
    target = 0.5 * np.log(0.7 / 0.3)  # half-logit link: pi = 1 / (1 + exp(-2 f))

    glm = mb.glmboost("cbind(s, f) ~ x", data=data, family=mb.Binomial())
    gam = mb.gamboost("cbind(s, f) ~ bbs(x)", data=data, family=mb.Binomial())
    for model in (glm, gam):
        assert model.offset == pytest.approx(target, abs=1e-12)
        assert np.mean(model.predict(type="response")) == pytest.approx(0.7, abs=1e-6)


# glmboost centres x by its unweighted mean, which differs between the grouped
# rows and the expanded rows, so the comparison turns centring off.
@pytest.mark.parametrize(
    "fitter, formula_rhs, options",
    [("glmboost", "x", {"center": False}), ("gamboost", "bbs(x)", {})],
)
def test_cbind_fit_equals_bernoulli_expansion(fitter, formula_rhs, options):
    x, success, failure = _grouped()
    x_long, y_long = _expanded(x, success, failure)
    fit = getattr(mb, fitter)

    grouped = fit(
        f"cbind(s, f) ~ {formula_rhs}",
        data={"s": success, "f": failure, "x": x},
        family=mb.Binomial(),
        control=mb.boost_control(mstop=80),
        **options,
    )
    expanded = fit(
        f"y ~ {formula_rhs}",
        data={"y": y_long, "x": x_long},
        family=mb.Binomial(),
        control=mb.boost_control(mstop=80),
        **options,
    )

    assert grouped.offset == pytest.approx(expanded.offset, abs=1e-10)
    eta_grouped = np.asarray(grouped.predict(newdata={"x": x}, type="link"))
    eta_expanded = np.asarray(expanded.predict(newdata={"x": x}, type="link"))
    np.testing.assert_allclose(eta_grouped, eta_expanded, rtol=0, atol=1e-10)


def test_proportion_with_trial_weights_equals_cbind():
    x, success, failure = _grouped()
    trials = success + failure
    via_cbind = mb.glmboost(
        "cbind(s, f) ~ x", data={"s": success, "f": failure, "x": x}, family=mb.Binomial()
    )
    via_weights = mb.glmboost(
        "p ~ x", data={"p": success / trials, "x": x}, family=mb.Binomial(), weights=trials
    )
    assert via_cbind.offset == via_weights.offset
    np.testing.assert_array_equal(
        np.asarray(via_cbind.predict(type="link")), np.asarray(via_weights.predict(type="link"))
    )


def test_proportion_cvrisk_equals_expanded_cvrisk():
    # cvrisk encodes the response and then fits on the encoded vector, so the
    # encoding must leave a proportion unchanged the second time. Each trial
    # row inherits its group's fold, so the two risk matrices must agree.
    x, success, failure = _grouped()
    trials = success + failure
    x_long, y_long = _expanded(x, success, failure)
    group_of_row = np.repeat(np.arange(x.size), trials.astype(int))
    fold_id = np.random.default_rng(5).permutation(np.arange(x.size) % 5)
    folds = (fold_id[:, None] != np.arange(5)[None, :]).astype(float)
    control = mb.boost_control(mstop=60)

    grouped = mb.gamboost(
        "p ~ bbs(x)",
        data={"p": success / trials, "x": x},
        family=mb.Binomial(),
        weights=trials,
        control=control,
    )
    expanded = mb.gamboost(
        "y ~ bbs(x)", data={"y": y_long, "x": x_long}, family=mb.Binomial(), control=control
    )
    cv_grouped = grouped.cvrisk(folds=folds)
    cv_expanded = expanded.cvrisk(folds=folds[group_of_row])
    np.testing.assert_allclose(
        cv_grouped.risk_matrix, cv_expanded.risk_matrix, rtol=1e-10, atol=1e-12
    )
    assert cv_grouped.optimal_mstop > 0


def _cbind_and_proportion(fitter, rhs, success, failure, x, control):
    trials = success + failure
    fit = getattr(mb, fitter)
    via_cbind = fit(
        f"cbind(s, f) ~ {rhs}",
        data={"s": success, "f": failure, "x": x},
        family=mb.Binomial(),
        control=control,
    )
    via_weights = fit(
        f"p ~ {rhs}",
        data={"p": success / trials, "x": x},
        family=mb.Binomial(),
        weights=trials,
        control=control,
    )
    return via_cbind, via_weights


# cvrisk on a cbind fit used to read the two response columns as Surv(time,
# event) and stack them (glmboost) or pass the tuple on as a column name
# (gamboost). It must refit on the success proportion with trial weights,
# which is the proportion-plus-weights model's cvrisk (#877).
@pytest.mark.parametrize("fitter, rhs", [("glmboost", "x"), ("gamboost", "bbs(x)")])
@pytest.mark.parametrize("fold_source", ["generated", "matrix", "refit"])
def test_cbind_cvrisk_equals_proportion_with_trial_weights(fitter, rhs, fold_source):
    x, success, failure = _grouped()
    via_cbind, via_weights = _cbind_and_proportion(
        fitter, rhs, success, failure, x, mb.boost_control(mstop=60)
    )
    if fold_source == "generated":
        kwargs = {"type": "kfold", "B": 5, "seed": 3}
    else:
        fold_id = np.random.default_rng(5).permutation(np.arange(x.size) % 5)
        kwargs = {"folds": (fold_id[:, None] != np.arange(5)[None, :]).astype(float)}
        if fold_source == "refit":
            kwargs["preprocessing"] = "refit"

    cv_cbind = via_cbind.cvrisk(**kwargs)
    cv_weights = via_weights.cvrisk(**kwargs)
    np.testing.assert_allclose(
        cv_cbind.risk_matrix, cv_weights.risk_matrix, rtol=1e-10, atol=1e-12
    )
    assert cv_cbind.optimal_mstop == cv_weights.optimal_mstop
    # The digest of the response and weights the folds were fitted to.
    assert cv_cbind.data_fingerprint == cv_weights.data_fingerprint


def _grouped_with_noise():
    x, success, failure = _grouped()
    z = np.random.default_rng(99).normal(size=x.size)
    return x, z, success, failure


def test_cbind_stabsel_equals_proportion_weights_stabsel_glmboost():
    # stabsel rebuilds the response from the model's retained columns. A
    # cbind fit's response is the proportion, with the trial counts in the
    # weights the model keeps, so its selection frequencies must equal the
    # proportion-plus-weights fit's. Before the fix this stacked the two
    # cbind columns as a survival pair and raised
    # "fold matrix has 60 rows but n = 120".
    x, z, success, failure = _grouped_with_noise()
    trials = success + failure
    control = mb.boost_control(mstop=20)

    via_cbind = mb.glmboost(
        "cbind(s, f) ~ x + z",
        data={"s": success, "f": failure, "x": x, "z": z},
        family=mb.Binomial(),
        control=control,
        center=False,
    )
    via_weights = mb.glmboost(
        "p ~ x + z",
        data={"p": success / trials, "x": x, "z": z},
        family=mb.Binomial(),
        weights=trials,
        control=control,
        center=False,
    )

    sel_cbind = mb.stabsel(via_cbind, B=3, q=1, PFER=1, verbose=False)
    sel_weights = mb.stabsel(via_weights, B=3, q=1, PFER=1, verbose=False)
    np.testing.assert_array_equal(sel_cbind.phat, sel_weights.phat)
    assert sel_cbind.selected == sel_weights.selected
    assert sel_cbind.data_fingerprint == sel_weights.data_fingerprint != ""


@pytest.mark.parametrize(
    "formula_rhs, options",
    [("bbs(x) + bbs(z)", {"bins": 20}), ("bbs(x) + bols(z)", {})],
)
def test_cbind_stabsel_equals_proportion_weights_stabsel_gamboost(formula_rhs, options):
    # The binned case reaches stabsel through the compact-spec selection
    # engine, which stacked the two cbind columns the same way; the
    # unbinned case refits per fold and then failed rebuilding the
    # response for the result's data fingerprint, which treated the tuple
    # of column names as a single column name. Both must match the
    # proportion-plus-weights fit.
    x, z, success, failure = _grouped_with_noise()
    trials = success + failure
    control = mb.boost_control(mstop=20)

    via_cbind = mb.gamboost(
        f"cbind(s, f) ~ {formula_rhs}",
        data={"s": success, "f": failure, "x": x, "z": z},
        family=mb.Binomial(),
        control=control,
        **options,
    )
    via_weights = mb.gamboost(
        f"p ~ {formula_rhs}",
        data={"p": success / trials, "x": x, "z": z},
        family=mb.Binomial(),
        weights=trials,
        control=control,
        **options,
    )

    sel_cbind = mb.stabsel(via_cbind, B=3, q=1, PFER=1, verbose=False)
    sel_weights = mb.stabsel(via_weights, B=3, q=1, PFER=1, verbose=False)
    np.testing.assert_array_equal(sel_cbind.phat, sel_weights.phat)
    assert sel_cbind.selected == sel_weights.selected
    assert sel_cbind.data_fingerprint == sel_weights.data_fingerprint != ""


@pytest.mark.parametrize(
    "fitter, formula_rhs, options",
    [("glmboost", "x", {"center": False}), ("gamboost", "bbs(x)", {})],
)
def test_cbind_cvrisk_equals_proportion_weights_cvrisk(fitter, formula_rhs, options):
    # cvrisk rebuilds the response from the model's retained columns. A
    # cbind fit's response is the proportion, with the trial counts in the
    # weights the model keeps, so its risk curve on a fixed fold matrix
    # must equal the proportion-plus-weights fit's curve on the same folds.
    # Before the fix the glmboost route stacked the two cbind columns
    # ("x_matrix has 60 rows but y has 120 elements") and the gamboost
    # route treated the tuple as a single column name
    # ("'tuple' object has no attribute 'lower'").
    x, success, failure = _grouped()
    trials = success + failure
    fold_id = np.random.default_rng(5).permutation(np.arange(x.size) % 5)
    folds = (fold_id[:, None] != np.arange(5)[None, :]).astype(float)
    control = mb.boost_control(mstop=60)
    fit = getattr(mb, fitter)

    via_cbind = fit(
        f"cbind(s, f) ~ {formula_rhs}",
        data={"s": success, "f": failure, "x": x},
        family=mb.Binomial(),
        control=control,
        **options,
    )
    via_weights = fit(
        f"p ~ {formula_rhs}",
        data={"p": success / trials, "x": x},
        family=mb.Binomial(),
        weights=trials,
        control=control,
        **options,
    )

    cv_cbind = via_cbind.cvrisk(folds=folds)
    cv_weights = via_weights.cvrisk(folds=folds)
    np.testing.assert_allclose(
        cv_cbind.risk_matrix, cv_weights.risk_matrix, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        cv_cbind.aggregate, cv_weights.aggregate, rtol=1e-10, atol=1e-12
    )
    assert cv_cbind.optimal_mstop == cv_weights.optimal_mstop
    assert cv_cbind.data_fingerprint == cv_weights.data_fingerprint


@pytest.mark.r_parity
def test_converged_cbind_fit_matches_r_glm():
    x, success, failure = _grouped()
    model = mb.glmboost(
        "cbind(s, f) ~ x",
        data={"s": success, "f": failure, "x": x},
        family=mb.Binomial(),
        control=mb.boost_control(mstop=5000, nu=0.5),
    )

    def r_glm_fitted():
        import rpy2.robjects as ro

        ro.globalenv["s"] = ro.FloatVector(success)
        ro.globalenv["f"] = ro.FloatVector(failure)
        ro.globalenv["x"] = ro.FloatVector(x)
        return np.asarray(ro.r("fitted(glm(cbind(s, f) ~ x, family = binomial()))"))

    expected = r_call(r_glm_fitted, packages=())
    np.testing.assert_allclose(model.predict(type="response"), expected, rtol=0, atol=1e-6)
