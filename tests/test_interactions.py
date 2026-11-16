"""Tensor-product interaction learners (the %X% operator).

An interaction is built as a buser learner over the row-wise tensor of two
marginal designs. The tests check three things: that it captures structure an
additive model cannot, that it equals the explicit buser construction it
compiles to, and that prediction on new data rebuilds the marginals correctly.
"""

import itertools
import warnings

import numpy as np
import pytest
from mboostrs.interactions import (
    build_tensor,
    build_tensor_design,
    expand_interactions,
    learner_design,
)

import mboostrs as mb


def _group_slope_data(n=400, seed=0):
    """y = slope(g) * x — a group-varying linear effect, the canonical case
    an additive model structurally cannot fit."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    g = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    slope = {"a": 1.0, "b": -1.0, "c": 2.0}
    y = np.array([slope[v] for v in g]) * x + rng.normal(scale=0.2, size=n)
    return {"x": x, "g": g, "y": y}


def _surface_data(n=600, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    y = np.sin(x * z) + rng.normal(scale=0.2, size=n)
    return {"x": x, "z": z, "y": y}


# --------------------------------------------------------------------------
# the interaction captures what additivity cannot
# --------------------------------------------------------------------------


def test_interaction_beats_additive_on_group_varying_slope():
    data = _group_slope_data()
    ctrl = mb.boost_control(400, 0.1)

    interaction = mb.gamboost(
        "y ~ bbs(x, df=4) %X% brandom(g)", data=data, family=mb.Gaussian(),
        control=ctrl,
    )
    additive = mb.gamboost(
        "y ~ bbs(x, df=4) + brandom(g)", data=data, family=mb.Gaussian(),
        control=ctrl,
    )

    def r2(model):
        y = data["y"]
        return 1.0 - np.sum((y - model.fitted_) ** 2) / np.sum((y - y.mean()) ** 2)

    assert r2(interaction) > 0.9
    assert r2(interaction) > r2(additive) + 0.3


def test_public_learner_design_replays_the_fitted_spline_basis():
    rng = np.random.default_rng(41)
    data = {"x": np.sort(rng.uniform(0, 10, 120))}
    data["y"] = np.sin(data["x"]) + rng.normal(scale=0.2, size=120)
    basis = learner_design("bbs(x, df=4)", data)

    formula_fit = mb.gamboost(
        "y ~ bbs(x, df=4)", data=data, control=mb.boost_control(80, 0.1)
    )
    custom_fit = mb.gamboost(
        "y ~ buser(x, X=DX, K=DK, df=4)",
        data=data,
        env={"DX": basis.design, "DK": basis.penalty},
        control=mb.boost_control(80, 0.1),
    )
    np.testing.assert_allclose(custom_fit.fitted_, formula_fit.fitted_, atol=1e-12)

    newdata = {"x": np.array([2.0, 4.0, 6.0])}
    replay = basis.transform(newdata)
    assert replay.shape == (3, basis.design.shape[1])
    predicted = custom_fit.predict(newdata=newdata, buser_newdata={"x": replay})
    np.testing.assert_allclose(predicted, formula_fit.predict(newdata=newdata), atol=1e-12)


def test_continuous_interaction_surface_fits():
    data = _surface_data()
    fit = mb.gamboost(
        "y ~ bbs(x, df=4) %X% bbs(z, df=4)", data=data, family=mb.Gaussian(),
        control=mb.boost_control(400, 0.1),
    )
    y = data["y"]
    r2 = 1.0 - np.sum((y - fit.fitted_) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.7, r2


# --------------------------------------------------------------------------
# equals the explicit buser construction it compiles to
# --------------------------------------------------------------------------


def test_interaction_equals_explicit_buser_construction():
    """The %X% sugar must fit exactly what a hand-built buser term fits."""
    data = _group_slope_data()
    ctrl = mb.boost_control(200, 0.1)

    rewritten, _, interactions = expand_interactions(
        "y ~ bbs(x, df=4) %X% brandom(g)", data
    )
    design, penalty = build_tensor(interactions[0], data)
    # Use the exact df the sugar chose, so the two fits are the same model.
    df = float(rewritten.split("df=")[1].rstrip(")"))

    explicit = mb.gamboost(
        f"y ~ buser(t, X=Xd, K=Kp, df={df})",
        data={**data, "t": np.zeros(len(data["y"]))},
        family=mb.Gaussian(), control=ctrl, env={"Xd": design, "Kp": penalty},
    )
    sugar = mb.gamboost(
        "y ~ bbs(x, df=4) %X% brandom(g)", data=data, family=mb.Gaussian(),
        control=ctrl,
    )
    np.testing.assert_allclose(sugar.fitted_, explicit.fitted_, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------
# prediction rebuilds the marginals
# --------------------------------------------------------------------------


def test_prediction_recovers_group_specific_slopes():
    data = _group_slope_data(n=800)
    fit = mb.gamboost(
        "y ~ bbs(x, df=4) %X% brandom(g)", data=data, family=mb.Gaussian(),
        control=mb.boost_control(500, 0.1),
    )
    # slope(a)=1, slope(b)=-1, slope(c)=2 at x=1
    newdata = {"x": np.array([1.0, 1.0, 1.0]), "g": np.array(["a", "b", "c"])}
    pred = np.asarray(fit.predict(newdata=newdata, type="response"))
    np.testing.assert_allclose(pred, [1.0, -1.0, 2.0], atol=0.25)


def test_prediction_matches_training_fit_on_training_rows():
    data = _group_slope_data()
    fit = mb.gamboost(
        "y ~ bbs(x, df=4) %X% brandom(g)", data=data, family=mb.Gaussian(),
        control=mb.boost_control(300, 0.1),
    )
    on_train = np.asarray(fit.predict(newdata=data, type="link"))
    np.testing.assert_allclose(on_train, fit.fitted_, rtol=0, atol=1e-8)


def test_unseen_group_level_in_interaction_is_rejected():
    data = _group_slope_data()
    fit = mb.gamboost(
        "y ~ bbs(x, df=4) %X% brandom(g)", data=data, family=mb.Gaussian(),
        control=mb.boost_control(100, 0.1),
    )
    with pytest.raises(ValueError, match="not seen at fit time"):
        fit.predict(newdata={"x": np.array([0.0]), "g": np.array(["z"])})


# --------------------------------------------------------------------------
# construction details
# --------------------------------------------------------------------------


def test_tensor_design_is_the_row_wise_outer_product():
    data = _group_slope_data(n=50)
    _, _, interactions = expand_interactions(
        "y ~ bols(x) %X% brandom(g)", data
    )
    design = build_tensor_design(interactions[0], data)
    # bols -> (intercept, x) = 2 cols; brandom over 3 levels = 3 cols; 6 total
    assert design.shape == (50, 6)


def test_kron_sum_penalty_has_the_right_shape():
    data = _group_slope_data(n=50)
    _, _, interactions = expand_interactions(
        "y ~ bbs(x, df=4) %X% brandom(g)", data
    )
    _, penalty = build_tensor(interactions[0], data)
    width = (20 + 3 + 1) * 3  # bbs width * n_levels
    assert penalty.shape == (width, width)
    np.testing.assert_allclose(penalty, penalty.T)  # symmetric


def test_formula_without_operator_is_unchanged():
    data = _group_slope_data()
    formula, env, interactions = expand_interactions("y ~ bbs(x, df=4)", data)
    assert formula == "y ~ bbs(x, df=4)"
    assert env == {}
    assert interactions == []


def test_malformed_interaction_operand_is_rejected():
    data = _group_slope_data()
    with pytest.raises(ValueError, match="must be one of"):
        expand_interactions("y ~ btree(x) %X% brandom(g)", data)


def test_removing_an_interaction_term_is_rejected():
    data = _group_slope_data()
    with pytest.raises(ValueError, match="removing an interaction"):
        expand_interactions("y ~ bbs(x, df=4) - bbs(x, df=4) %X% brandom(g)", data)


# --------------------------------------------------------------------------
# the marginal df= reaches the tensor
# --------------------------------------------------------------------------


def _tensor_hyper_of(formula, data):
    """The `df=` or `lambda=` the sugar wrote on the buser term."""
    rewritten, _, _ = expand_interactions(formula, data)
    key, _, value = rewritten.rpartition(", ")[2].rstrip(")").partition("=")
    return key, float(value)


def test_tensor_df_is_the_product_of_the_marginal_dfs():
    """R mboost's `%X%` rule: the tensor carries df1 * df2."""
    data = _surface_data(n=50)
    assert _tensor_hyper_of("y ~ bbs(x, df=4) %X% bbs(z, df=4)", data) == ("df", 16.0)
    assert _tensor_hyper_of("y ~ bbs(x, df=5) %X% bbs(z, df=8)", data) == ("df", 40.0)
    # A marginal with no df of its own — `bols` is unpenalized — counts as 1.
    assert _tensor_hyper_of("y ~ bbs(x, df=6) %X% bols(z)", data) == ("df", 6.0)


def test_marginal_df_controls_the_interaction_flexibility():
    """The df written on the marginals must change the fit.

    It used to be parsed for knot placement and then dropped: every `%X%` term
    was assigned `df = ncol // 4` — 144 degrees of freedom for two default
    marginals — so `df=2` and `df=8` produced bit-identical fits, and the term
    won every selection step against df=4 main effects.
    """
    rng = np.random.default_rng(0)
    n = 200
    data = {"a": rng.normal(size=n), "b": rng.normal(size=n), "y": rng.normal(size=n)}
    ctrl = mb.boost_control(200, 0.1)

    def train_r2(formula):
        fit = mb.gamboost(formula, data=data, family=mb.Gaussian(), control=ctrl)
        y = data["y"]
        return 1.0 - np.sum((y - fit.fitted_) ** 2) / np.sum((y - y.mean()) ** 2)

    modest = train_r2("y ~ bbs(a, df=4) %X% bbs(b, df=4)")
    flexible = train_r2("y ~ bbs(a, df=8) %X% bbs(b, df=8)")
    assert flexible > modest + 0.2, (modest, flexible)
    # y is noise independent of a and b: a df=4x4 tensor must not explain it the
    # way a 144-df one did (0.79 of the variance, against 0.01 for a main effect).
    assert modest < 0.25, modest


def test_two_lambda_marginals_give_the_tensor_a_lambda():
    """With both margins pinned there is no df left to calibrate."""
    data = _surface_data(n=50)
    assert _tensor_hyper_of(
        "y ~ bbs(x, lambda=2) %X% bbs(z, lambda=3)", data
    ) == ("lambda", 1.0)


def test_marginal_lambda_scales_that_margin_of_the_penalty():
    """A margin that fixes its own lambda carries it into the tensor penalty."""
    data = _group_slope_data(n=50)
    _, _, plain = expand_interactions("y ~ bbs(x, df=4) %X% brandom(g)", data)
    _, _, scaled = expand_interactions("y ~ bbs(x, lambda=3) %X% brandom(g)", data)
    _, k_plain = build_tensor(plain[0], data)
    _, k_scaled = build_tensor(scaled[0], data)
    # The x-margin block is tripled and the group block is untouched, so the
    # scaled penalty is neither equal to nor a multiple of the plain one.
    assert not np.allclose(k_plain, k_scaled)
    assert not np.allclose(3.0 * k_plain, k_scaled)


def test_an_unpenalized_bols_margin_contributes_no_penalty():
    """`bols` fixes lambda = 0, so its block of the Kronecker sum is zero."""
    data = _group_slope_data(n=50)
    _, _, interactions = expand_interactions("y ~ bols(x) %X% brandom(g)", data)
    _, penalty = build_tensor(interactions[0], data)
    # kron(0, I3) + kron(I2, I3) = I6 — only the brandom margin penalizes.
    np.testing.assert_allclose(penalty, np.eye(6))


def test_df_at_the_null_space_floor_warns():
    """Two df=2 margins multiply to 4, the dimension the penalty cannot shrink."""
    data = _surface_data(n=50)
    with pytest.warns(UserWarning, match="unpenalized part of the tensor penalty"):
        expand_interactions("y ~ bbs(x, df=2) %X% bbs(z, df=2)", data)


def test_df_above_the_null_space_floor_does_not_warn():
    data = _surface_data(n=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expand_interactions("y ~ bbs(x, df=4) %X% bbs(z, df=4)", data)


# --------------------------------------------------------------------------
# the boosting risk decreases at every step
# --------------------------------------------------------------------------


def _disjoint_tensor_formula(n_terms, df, data):
    """`n_terms` tensor buser terms over disjoint variable pairs."""
    env, terms = {}, []
    for t in range(n_terms):
        a, b = f"f{2 * t}", f"f{2 * t + 1}"
        _, tensor_env, interactions = expand_interactions(
            f"y ~ bbs({a}, df=4, knots=8) %X% bbs({b}, df=4, knots=8)", data
        )
        design, penalty = build_tensor(interactions[0], data)
        env[f"X{t}"], env[f"K{t}"] = design, penalty
        data[f"t{t}"] = np.zeros(len(data["y"]))
        terms.append(f"buser(t{t}, X=X{t}, K=K{t}, df={df})")
    return "y ~ " + " + ".join(terms), env


@pytest.mark.parametrize("n_terms", [2, 8])
def test_tensor_terms_do_not_make_the_training_risk_increase(n_terms):
    """Component-wise boosting at nu < 2 cannot increase the training risk.

    It did: `df=4` is the null-space dimension of a two-margin Kronecker sum,
    no finite lambda attains it, and the lambda the calibration clamped to left
    the Gram beyond what a double can factorize. The hat matrix that came back
    had eigenvalues outside [0, 1] — a step through a negative one moves the fit
    away from the residual — and the risk diverged to 1e34 with eight terms.
    """
    rng = np.random.default_rng(0)
    n = 200
    data = {f"f{j}": rng.normal(size=n) for j in range(2 * n_terms)}
    data["y"] = data["f0"] * data["f1"] + rng.normal(scale=0.5, size=n)
    formula, env = _disjoint_tensor_formula(n_terms, 4.0, data)

    fit = mb.gamboost(formula, data=data, family=mb.Gaussian(), env=env,
                      control=mb.boost_control(100, 0.1))
    risk = np.asarray(fit.risk_)
    assert np.all(np.diff(risk) <= 1e-8), np.where(np.diff(risk) > 1e-8)[0][:5]
    assert risk[-1] < 0.5 * risk[0]


@pytest.mark.parametrize("n_terms", [1, 6, 30])
def test_bspatial_terms_do_not_make_the_training_risk_increase(n_terms):
    """The same defect reached `bspatial`, which builds its own tensor penalty."""
    rng = np.random.default_rng(2)
    n, p = 400, 12
    data = {f"f{j}": rng.normal(size=n) for j in range(p)}
    data["y"] = (data["f0"] * data["f1"] + 0.5 * data["f2"]
                 + rng.normal(scale=0.3, size=n))
    pairs = list(itertools.combinations(range(p), 2))[:n_terms]
    terms = [f"bbs(f{j}, df=4)" for j in range(p)]
    terms += [f"bspatial(f{a}, f{b}, df=4)" for a, b in pairs]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # df=4 is the null-space floor here
        fit = mb.gamboost("y ~ " + " + ".join(terms), data=data,
                          family=mb.Gaussian(), control=mb.boost_control(200, 0.1))
    risk = np.asarray(fit.risk_)
    assert np.all(np.diff(risk) <= 1e-8), np.where(np.diff(risk) > 1e-8)[0][:5]


def test_marginal_argument_the_tensor_would_drop_is_rejected():
    """A keyword a marginal does not carry into the tensor is an error.

    `bbs(x, center=True)` parsed, dropped `center`, and fitted an uncentered
    basis; the formula said one thing and the fit did another, with nothing to
    read that off.
    """
    data = _group_slope_data(n=50)
    with pytest.raises(ValueError, match="does not carry 'center'"):
        expand_interactions("y ~ bbs(x, center=True) %X% brandom(g)", data)
    with pytest.raises(ValueError, match="does not carry 'nonsense'"):
        expand_interactions("y ~ bbs(x, df=4, nonsense=2) %X% brandom(g)", data)
    with pytest.raises(ValueError, match="does not carry 'knots'"):
        expand_interactions("y ~ bols(x) %X% brandom(g, knots=10)", data)


def test_the_arguments_each_marginal_does_carry_are_accepted():
    """The refusal must not narrow what the marginals already accept."""
    data = _group_slope_data(n=50)
    for formula in (
        "y ~ bbs(x, knots=12, degree=3, differences=2, df=4) %X% brandom(g)",
        "y ~ bols(x, intercept=False) %X% brandom(g, df=3)",
        "y ~ bbs(x, lambda=2) %X% bols(x, lambda=1)",
    ):
        assert expand_interactions(formula, data)[0]


def _labelled_model():
    data = _surface_data(n=300)
    term = "bbs(x, df=3) %X% bbs(z, df=3)"
    fit = mb.gamboost(f"y ~ bbs(x) + {term}", data=data,
                      control=mb.boost_control(mstop=60, nu=0.1))
    return fit, term


def test_a_tensor_is_named_by_the_term_as_written():
    """The derived column `__ix0` is how the engine fits the term; every
    table and key a reader sees uses the term the formula wrote."""
    fit, term = _labelled_model()

    assert term in fit.base_learner_names()
    assert "__ix0" not in repr(fit)
    assert term in repr(fit)
    assert term in fit.coef()
    assert term in fit.varimp()["reduction"]
    assert fit.varimp()["variable_names"][term] == "x,z"
    assert term in fit.extract("lambda")
    row = [r for r in fit.selection_counts(output="records") if r["learner"] == term]
    assert row and row[0]["column"] == "x,z"
    assert "__ix0" not in fit.summary().as_text()


def test_the_derived_name_is_still_accepted_as_an_alias():
    fit, term = _labelled_model()

    assert fit.learner_index("__ix0") == fit.learner_index(term)
    by_alias = fit.partial_effects(which="__ix0")[term][1]
    by_label = fit.partial_effects(which=term)[term][1]
    np.testing.assert_array_equal(by_alias, by_label)
    assert list(fit.extract("df", which="__ix0")) == [term]


def test_a_refit_names_the_tensor_as_the_fit_does():
    """Refits go through the rewritten formula; they must still report the
    written term, or `which=` from the original fit misses every replicate."""
    fit, term = _labelled_model()

    refit = fit.update(control=mb.boost_control(mstop=20, nu=0.1))
    assert refit.base_learner_names() == fit.base_learner_names()


def test_an_unselected_learner_contributes_one_zero_per_row():
    """`predict(which=)` for a learner the fit never selected returned an
    array sized by the number of columns of a dict `newdata`."""
    rng = np.random.default_rng(0)
    n = 300
    data = {f"x{i}": rng.uniform(-2, 2, n) for i in range(1, 5)}
    data["y"] = np.sin(data["x1"]) + rng.normal(0, 0.3, n)
    fit = mb.gamboost("y ~ bbs(x1, df=4) + bbs(x2, df=4) + bbs(x3, df=4) + bbs(x4, df=4)",
                      data=data, control=mb.boost_control(mstop=30, nu=0.1))
    unselected = next(n for n in ("x2", "x3", "x4") if n not in fit.selected)
    new_rows = {k: v[:50] for k, v in data.items() if k != "y"}

    on_new = np.asarray(fit.predict(newdata=new_rows, which=unselected)).ravel()
    on_training = np.asarray(fit.predict(which=unselected)).ravel()
    np.testing.assert_array_equal(on_new, np.zeros(50))
    np.testing.assert_array_equal(on_training, np.zeros(n))


def test_a_tensor_of_string_factors_predicts_new_rows():
    """The fit builds a %X% design from the raw columns and then encodes the
    labels; prediction encoded first and rebuilt the design from codes, so
    every level read as unseen. Full and per-term predictions of new rows
    must equal the fitted values on the same rows."""
    import polars as pl

    rng = np.random.default_rng(0)
    n = 400
    d = pl.DataFrame({"g": rng.choice(["1", "2", "3", "4"], n),
                      "h": rng.choice(["a", "b", "c"], n), "x": rng.uniform(-1, 1, n)})
    effect = np.array([0, 1, -1, 2])[d["g"].cast(int).to_numpy() - 1] * (d["h"] == "b").to_numpy()
    d = d.with_columns(y=pl.Series(effect + np.sin(3 * d["x"].to_numpy())
                                   + rng.normal(0, 0.3, n)))
    model = mb.gamboost("y ~ bbs(x) + brandom(g) %X% brandom(h)", d)
    np.testing.assert_allclose(np.asarray(model.predict(newdata=d.head(6))).ravel(),
                               np.asarray(model.predict()).ravel()[:6], atol=1e-12)
    for name in model._learner_names:
        np.testing.assert_allclose(
            np.asarray(model.predict(newdata=d.head(6), which=name)).ravel(),
            np.asarray(model.predict(which=name)).ravel()[:6], atol=1e-12)


def test_a_tensor_margin_on_the_log_scale_predicts_beyond_the_training_range():
    """`bbs(log(a)) %X% bbs(b)` builds the margin on log(a) at fit and at
    prediction, so a new value far above the training range extrapolates on
    the log scale, as the main term `bbs(log(a))` does."""
    import numpy as np

    import mboostrs as mb

    rng = np.random.default_rng(0)
    n = 500
    a, b = np.exp(rng.normal(size=n)), rng.uniform(-1, 1, n)
    data = {"a": a, "b": b, "y": np.log(a) * b + rng.normal(scale=0.2, size=n)}
    model = mb.gamboost("y ~ bbs(log(a), df=2.5, knots=5) %X% bbs(b, df=2.5, knots=5)",
                        data, control=mb.boost_control(300, 0.1))
    new = {"a": np.array([np.e, 500.0]), "b": np.array([0.5, 0.5])}
    predicted = np.asarray(model.predict(newdata=new)).ravel()
    np.testing.assert_allclose(predicted, [0.5, 0.5 * np.log(500.0)], atol=0.25)
    refit = mb.gam_refit(model)
    assert np.isfinite(np.asarray(refit.predict(new))).all()
