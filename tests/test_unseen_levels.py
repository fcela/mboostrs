"""A `brandom` level the fit never saw, at prediction time.

Grouped cross-validation holds whole groups out, so every held-out row
carries a level its fold's fit never saw. `predict(newdata=)` refuses such a
level by default, and the error names `unseen_levels="zero_effect"`, which
gives the level a zero coefficient. `cvrisk` makes that same prediction: it
fits each fold on the full frame with the held-out group's rows at zero
weight, and a level with no weighted rows is estimated at zero. Found by the
agent pilot (`cases/agent_specification_search.ipynb`), where
`compare_specifications` could not score grouped folds at all.
"""

import numpy as np
import pytest

import mboostrs as mb

CONTROL = mb.boost_control(60, 0.1)


def _grouped(n_groups=8, per_group=15, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group).astype(float)
    x = rng.normal(size=g.size)
    y = (1.0 + x + rng.normal(size=n_groups)[g.astype(int)]
         + rng.normal(scale=0.5, size=g.size))
    return {"x": x, "g": g, "y": y}


def _take(frame, rows):
    return {name: np.asarray(values)[rows] for name, values in frame.items()}


def test_an_unseen_level_is_refused_and_the_error_names_the_remedy():
    frame = _grouped()
    held = frame["g"] < 2
    fit = mb.gamboost("y ~ bols(x) + brandom(g)", data=_take(frame, ~held), control=CONTROL)
    with pytest.raises(ValueError, match=r'unknown Brandom level 0 .*unseen_levels="zero_effect"'):
        fit.predict(newdata=_take(frame, held))


def test_an_unknown_policy_is_refused():
    frame = _grouped()
    fit = mb.gamboost("y ~ bols(x) + brandom(g)", data=frame, control=CONTROL)
    with pytest.raises(ValueError, match="unseen_levels must be 'error', 'zero_effect', 'average' or 'marginal'"):
        fit.predict(newdata=frame, unseen_levels="mean")


@pytest.mark.parametrize("term", ["brandom(g)", "brandom(g, center=TRUE)"])
def test_zero_effect_is_the_prediction_of_a_fit_with_the_group_at_zero_weight(term):
    """Exact for `bols` + `brandom`, whose bases do not move with the rows."""
    frame = _grouped()
    held = frame["g"] < 2
    formula = f"y ~ bols(x) + {term}"
    subset = mb.gamboost(formula, data=_take(frame, ~held), control=CONTROL)
    weighted = mb.gamboost(formula, data=frame, weights=(~held).astype(float), control=CONTROL)
    predicted = subset.predict(newdata=_take(frame, held), unseen_levels="zero_effect")
    np.testing.assert_allclose(predicted, np.asarray(weighted.predict())[held], rtol=0, atol=1e-10)


def test_zero_effect_leaves_seen_levels_unchanged():
    frame = _grouped()
    fit = mb.gamboost("y ~ bols(x) + brandom(g)", data=frame, control=CONTROL)
    np.testing.assert_array_equal(
        fit.predict(newdata=frame, unseen_levels="zero_effect"),
        fit.predict(newdata=frame),
    )


def test_refit_passes_unseen_policy_to_held_tree_predictor():
    rng = np.random.default_rng(9)
    x = rng.normal(size=120)
    group = np.repeat(np.array(["a", "b", "c", "d"]), 30)
    y = 3 * (group == "d") + x + rng.normal(0, .15, len(x))
    frame = {"x": x, "g": group, "y": y}
    fit = mb.gamboost(
        "y ~ btree(g, maxdepth=2, selection=sse) + bols(x)", frame,
        control=mb.boost_control(50))
    refit = mb.gam_refit(fit, which="all", unsupported="hold")
    assert refit.held == ("btree(g, maxdepth=2, selection=sse)",)

    unseen = {"x": np.array([-.5, .5]), "g": np.array(["q", "d"])}
    with pytest.raises(ValueError, match="not present when the model was fitted"):
        refit.predict(unseen, unseen_levels="error")
    predicted = refit.predict(unseen, unseen_levels="zero_effect")
    assert np.isfinite(predicted).all()
    assert np.isfinite(refit.effects(unseen, unseen_levels="zero_effect")["bols(x)"][0]).all()

    # The refitted linear effect is unchanged when only the group label changes.
    # Therefore the prediction difference must be the held tree's contribution.
    same_x = {"x": np.array([.5, .5]), "g": np.array(["q", "d"])}
    refit_gap = np.diff(refit.predict(same_x, unseen_levels="zero_effect"))[0]
    tree_gap = np.diff(fit.predict(same_x, which="g", unseen_levels="zero_effect"))[0]
    np.testing.assert_allclose(refit_gap, tree_gap, atol=1e-10, rtol=0)


def test_zero_effect_reaches_a_string_labelled_group_column():
    frame = _grouped()
    frame["g"] = np.array([f"site{int(value)}" for value in frame["g"]])
    held = np.isin(frame["g"], ["site0", "site1"])
    formula = "y ~ bols(x) + brandom(g)"
    subset = mb.gamboost(formula, data=_take(frame, ~held), control=CONTROL)
    with pytest.raises(ValueError, match='unseen_levels="zero_effect"'):
        subset.predict(newdata=_take(frame, held))
    weighted = mb.gamboost(formula, data=frame, weights=(~held).astype(float), control=CONTROL)
    predicted = subset.predict(newdata=_take(frame, held), unseen_levels="zero_effect")
    np.testing.assert_allclose(predicted, np.asarray(weighted.predict())[held], rtol=0, atol=1e-10)


@pytest.mark.parametrize("term", ["brandom(g)", "brandom(g, center=TRUE)"])
def test_contributions_and_path_grids_agree_with_the_prediction(term):
    frame = _grouped()
    held = frame["g"] < 2
    fit = mb.gamboost(f"y ~ bols(x) + {term}", data=_take(frame, ~held), control=CONTROL)
    new = _take(frame, held)
    full = np.asarray(fit.predict(newdata=new, unseen_levels="zero_effect"))

    total = np.full(held.sum(), fit.offset_)
    for name in sorted(set(fit.selected)):
        total += np.asarray(fit.predict(newdata=new, which=[name], unseen_levels="zero_effect"))
    np.testing.assert_allclose(total, full, rtol=0, atol=1e-10)

    steps = fit.prediction_contributions(newdata=new, aggregate="none", unseen_levels="zero_effect")
    np.testing.assert_allclose(np.sum(steps, axis=0) + fit.offset_, full, rtol=0, atol=1e-10)

    grid = fit.predict_over_mstop([10, 60], newdata=new, unseen_levels="zero_effect", output="numpy")
    np.testing.assert_allclose(grid[1], full, rtol=0, atol=1e-10)


def test_grouped_cross_validation_scores_through_predict_and_matches_cvrisk():
    frame = _grouped()
    folds = mb.grouped_folds(frame["g"], B=4)
    result = mb.evaluate_specification("bols(x) + brandom(g)", frame, "y", folds=folds, mstop=60,
                                       unseen_levels="zero_effect")
    fit = mb.gamboost("y ~ bols(x) + brandom(g)", data=frame, control=CONTROL)
    curve = np.asarray(fit.cvrisk(folds=np.asarray(folds)).aggregate)
    assert result.score == pytest.approx(curve[60], rel=1e-9)
    assert all(row["n_unseen_level_rows"] == row["n_evaluation"] for row in result.folds)

    with pytest.raises(ValueError, match='unseen_levels="zero_effect"'):
        mb.evaluate_specification("bols(x) + brandom(g)", frame, "y", folds=folds,
                                  mstop=60, unseen_levels="error")


def test_random_folds_that_keep_every_group_count_no_unseen_rows():
    frame = _grouped()
    result = mb.evaluate_specification("bols(x) + brandom(g)", frame, "y", folds=3, mstop=60)
    assert all(row["n_unseen_level_rows"] == 0 for row in result.folds)


def test_compare_specifications_scores_a_grouped_design():
    frame = _grouped()
    rows = mb.compare_specifications(
        {"fixed": "bols(x)", "random": "bols(x) + brandom(g)"},
        frame, "y", folds=mb.grouped_folds(frame["g"], B=4), mstop=60)
    assert [row["specification"] for row in rows] == ["fixed", "random"]
    assert all(np.isfinite(row["out-of-fold risk"]) for row in rows)


def test_a_distributional_model_takes_the_same_policy():
    frame = _grouped()
    held = frame["g"] < 2
    formula = "y ~ bols(x) + brandom(g)"
    control = mb.boost_control(40, 0.1)
    subset = mb.gamboostLSS(formula, data=_take(frame, ~held), families=mb.GaussianLSS(),
                            control=control)
    weighted = mb.gamboostLSS(formula, data=frame, families=mb.GaussianLSS(),
                              weights=(~held).astype(float), control=control)
    new = _take(frame, held)
    with pytest.raises(ValueError, match='unseen_levels="zero_effect"'):
        subset.predict(newdata=new, parameter="mu")
    for parameter in subset.param_names:
        predicted = subset.predict(newdata=new, parameter=parameter, unseen_levels="zero_effect")
        expected = np.asarray(weighted.predict(parameter=parameter))[held]
        np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-10, err_msg=parameter)
        effects = subset.partial_effects(parameter, newdata=new, unseen_levels="zero_effect")
        total = subset.coef(parameter)["(offset)"] + sum(effect for _, effect in effects.values())
        np.testing.assert_allclose(total, predicted, rtol=0, atol=1e-10, err_msg=parameter)


def test_average_predicts_a_new_group_with_the_mean_coefficient():
    frame = _grouped()
    held = frame["g"] < 2
    fit = mb.gamboost("y ~ bols(x) + brandom(g)", data=_take(frame, ~held), control=CONTROL)
    new = _take(frame, held)
    zero = np.asarray(fit.predict(newdata=new, unseen_levels="zero_effect"))
    average = np.asarray(fit.predict(newdata=new, unseen_levels="average"))
    levels = np.asarray(fit.coef(aggregate="sum")["g"], dtype=float)
    np.testing.assert_allclose(average - zero, levels.mean(), rtol=0, atol=1e-10)


def test_average_keeps_the_slope_a_random_slope_term_absorbed():
    """A random slope takes part of the average slope during boosting; a new
    group predicted with zero coefficients loses it, an average group keeps it."""
    n, k = 1500, 30
    def draw(seed, offset):
        r = np.random.default_rng(seed)
        codes = r.integers(0, k, n)
        x = r.normal(size=n)
        slopes = np.random.default_rng(99).normal(size=k)[codes] if offset == 0 else r.normal(size=n)
        return {"x": x, "clinic": codes.astype(float) + offset,
                "y": 1.0 * x + slopes * x + r.normal(scale=0.5, size=n)}
    train, test = draw(1, 0), draw(2, 1000)
    fit = mb.gamboost("y ~ bbs(x) + brandom(clinic, by=x)", data=train,
                      control=mb.boost_control(400, 0.1))
    zero = np.asarray(fit.predict(newdata=test, unseen_levels="zero_effect"))
    average = np.asarray(fit.predict(newdata=test, unseen_levels="average"))
    assert np.mean((test["y"] - average) ** 2) < np.mean((test["y"] - zero) ** 2)


def test_the_cross_validation_functions_default_to_the_average_group():
    frame = _grouped()
    folds = mb.grouped_folds(frame["g"], B=4)
    default = mb.cv_risk("bols(x) + brandom(g)", frame, "y", folds=folds, mstop=60)
    average = mb.cv_risk("bols(x) + brandom(g)", frame, "y", folds=folds, mstop=60,
                         unseen_levels="average")
    assert default == average


# -- the LSS prediction surface carries the same policy (W05/W06) ---------
#
# `predict` accepted `unseen_levels=` for an LSS model; `params`,
# `prediction_interval` (with its R aliases), `pit`, `quantile_residuals`,
# `residual_frame`, `predict_all` and `evaluate_distribution` did not, so the
# one policy a grouped LSS fit could state was lost at every diagnostic door.
# These propagate it unchanged, refuse by default as `predict` does, and
# `evaluate_distribution` names the policy it scored under in its assumptions.

def _lss_grouped_frame(seed=5):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 6, 240)
    x = rng.uniform(-2, 2, 240)
    y = np.sin(x) + 0.1 * g + rng.normal(size=240)
    return {"x": x, "g": g.astype(float), "y": y}


def _lss_grouped_fit():
    frame = _lss_grouped_frame()
    train = {name: values[frame["g"] < 4] for name, values in frame.items()}
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4) + brandom(g)",
                          "sigma": "y ~ 1"}, data=train,
                         family=mb.GaussianLSS(), control=CONTROL)
    held = {"x": frame["x"][frame["g"] >= 4][:6],
            "g": frame["g"][frame["g"] >= 4][:6],
            "y": frame["y"][frame["g"] >= 4][:6]}
    return fit, held


def test_lss_params_and_prediction_surfaces_propagate_the_policy():
    fit, held = _lss_grouped_fit()
    expected = np.asarray(fit.predict(newdata=held, type="response",
                                      unseen_levels="average")["mu"])
    assert np.allclose(np.asarray(fit.params(newdata=held,
                                             unseen_levels="average")["mu"]),
                       expected)
    for call in (fit.prediction_interval, fit.predint, fit.PI):
        rows = call(newdata=held, unseen_levels="average")
        assert len(rows) == 6
        assert np.all(np.isfinite(np.asarray(rows["lower"], dtype=float)))
        assert np.all(np.isfinite(np.asarray(rows["upper"], dtype=float)))
    assert np.all(np.isfinite(fit.pit(newdata=held, unseen_levels="average")))
    assert np.all(np.isfinite(fit.quantile_residuals(newdata=held,
                                                     unseen_levels="average")))
    assert len(fit.residual_frame(newdata=held, unseen_levels="average")) == 6
    assert len(fit.predict_all(newdata=held, unseen_levels="average")) == 6


def test_lss_prediction_surfaces_still_refuse_by_default():
    fit, held = _lss_grouped_fit()
    for call in (fit.params, fit.prediction_interval, fit.pit):
        with pytest.raises(ValueError, match="unknown Brandom level 4"):
            call(newdata=held)


def test_evaluate_distribution_scores_and_labels_unseen_group_rows():
    fit, held = _lss_grouped_fit()
    result = mb.evaluate_distribution(fit, held, response="y",
                                      data_role="external",
                                      unseen_levels="average")
    assert np.isfinite(result.metrics["mean_log_score"])
    assert any("average group" in line and "unseen_levels='average'" in line
               for line in result.assumptions)
    zero = mb.evaluate_distribution(fit, held, response="y",
                                    data_role="external",
                                    unseen_levels="zero_effect")
    assert any("zero group effect" in line for line in zero.assumptions)
    # The default is still the refusal, matching `predict`.
    with pytest.raises(ValueError, match="unknown Brandom level 4"):
        mb.evaluate_distribution(fit, held, response="y", data_role="external")
    # A model with no group terms does not carry the label.
    plain = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                           data={"x": held["x"], "y": held["y"]} | {},
                           family=mb.GaussianLSS(), control=CONTROL)
    plain_result = mb.evaluate_distribution(
        plain, {"x": held["x"], "y": held["y"]}, response="y",
        data_role="external", unseen_levels="average")
    assert not any("unseen_levels" in line for line in plain_result.assumptions)


def test_zero_effect_reaches_a_tensor_interaction_marginal():
    """A %X% marginal is a nested replay: the policy must survive the trip.

    Found by the 24-dataset runner on Brazilian houses: `predict(newdata=,
    unseen_levels=...)` threaded the policy to `brandom` terms but not into
    `_replay_interactions`, so a held-back level in an interaction marginal
    raised even under `zero_effect` (bathroom=9 in the artifact).
    """
    frame = _grouped()
    held = frame["g"] < 2
    fit = mb.gamboost("y ~ bols(x) + bbs(x, df=4) %X% brandom(g)",
                      data=_take(frame, ~held), control=CONTROL)
    with pytest.raises(ValueError, match="not seen at fit time"):
        fit.predict(newdata=_take(frame, held))
    predicted = np.asarray(fit.predict(newdata=_take(frame, held),
                                       unseen_levels="zero_effect"), float)
    assert np.isfinite(predicted).all()
    averaged = np.asarray(fit.predict(newdata=_take(frame, held),
                                      unseen_levels="average"), float)
    assert np.isfinite(averaged).all()


def test_a_distributional_model_threads_the_policy_through_its_tensors():
    frame = _grouped()
    held = frame["g"] < 2
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4) %X% brandom(g)",
                          "sigma": "y ~ 1"},
                         data=_take(frame, ~held), family=mb.GaussianLSS(),
                         control=CONTROL)
    with pytest.raises(ValueError, match="not seen at fit time"):
        fit.params(newdata=_take(frame, held))
    values = fit.params(newdata=_take(frame, held), unseen_levels="zero_effect")
    assert np.isfinite(np.asarray(values["mu"], float)).all()
