"""Correctness and resampling gates for compact binned base learners."""

import importlib
from collections import Counter

import numpy as np
import pytest

import mboostrs as mb


def _data(n=360):
    row = np.arange(n, dtype=np.float64)
    x1 = np.sin(0.071 * row) + 0.002 * row
    x2 = np.cos(0.043 * row + 0.3) - 0.001 * row
    x3 = np.sin(0.019 * row + 1.2)
    y = 1.2 * x1 - 0.7 * x2 + 0.25 * x3 + ((37 * row) % 97 / 97 - 0.5) * 0.15
    weights = 0.5 + row % 7 / 5
    return {"y": y, "x1": x1, "x2": x2, "x3": x3}, weights


def _formula(bins=None):
    suffix = "" if bins is None else f", bins={bins}"
    return (
        f"y ~ bols(x1, intercept=False{suffix})"
        f" + bols(x2, intercept=False{suffix})"
        f" + bols(x3, intercept=False{suffix})"
    )


def test_binned_bols_validates_supported_initial_surface():
    with pytest.raises(ValueError, match="intercept=False"):
        mb.Bols("x", bins=100)
    with pytest.raises(ValueError, match="by="):
        mb.Bols("x", intercept=False, by="z", bins=100)
    with pytest.raises(ValueError, match="must be >= 1"):
        mb.Bols("x", intercept=False, bins=0)
    with pytest.raises(ValueError, match="by="):
        mb.Bbs("x", by="z", bins=100)
    with pytest.raises(ValueError, match="shape constraints"):
        mb.Bbs("x", constraint="increasing", bins=100)


def test_exact_bins_match_weighted_selection_fitted_and_risk():
    data, weights = _data()
    control = mb.boost_control(80, 0.1)
    exact = mb.gamboost(
        _formula(), data=data, weights=weights, family=mb.Gaussian(), control=control
    )
    binned = mb.gamboost(
        _formula(len(data["y"])),
        data=data,
        weights=weights,
        family=mb.Gaussian(),
        control=control,
    )
    assert exact.selected == binned.selected
    np.testing.assert_allclose(binned.fitted_, exact.fitted_, rtol=0, atol=2e-12)
    np.testing.assert_allclose(binned.risk_, exact.risk_, rtol=0, atol=1e-9)
    assert binned.execution_report().route == "general"
    assert (
        binned.execution_report().retained_learner_bytes
        < exact.execution_report().retained_learner_bytes
    )


def test_binned_approximation_converges_with_more_bins():
    n = 1200
    x = np.linspace(-3, 3, n)
    data = {"x": x, "y": 0.4 + 1.3 * x + 0.2 * np.sin(2 * x)}
    control = mb.boost_control(100, 0.1)
    exact = mb.gamboost(
        "y ~ bols(x, intercept=False)", data=data, family=mb.Gaussian(), control=control
    )

    def error(bins):
        model = mb.gamboost(
            f"y ~ bols(x, intercept=False, bins={bins})",
            data=data,
            family=mb.Gaussian(),
            control=control,
        )
        return float(np.sqrt(np.mean((model.fitted_ - exact.fitted_) ** 2)))

    assert error(160) < error(16) / 5


def test_binned_bbs_exact_path_and_training_replay():
    n = 320
    row = np.arange(n, dtype=np.float64)
    x1 = np.sin(0.031 * row) + 0.003 * row
    x2 = np.cos(0.047 * row) - 0.002 * row
    y = (1.1 * x1 - 0.8 * x2 + 0.2 * np.sin(2 * x1) > 0.25).astype(float)
    data = {"y": y, "x1": x1, "x2": x2}
    exact_formula = (
        "y ~ bbs(x1, center=True, lambda_value=0.7)"
        " + bbs(x2, center=True, lambda_value=0.7)"
    )
    binned_formula = (
        f"y ~ bbs(x1, center=True, lambda_value=0.7, bins={n})"
        f" + bbs(x2, center=True, lambda_value=0.7, bins={n})"
    )
    control = mb.boost_control(60, 0.1)
    exact = mb.gamboost(exact_formula, data=data, family=mb.Binomial(), control=control)
    binned = mb.gamboost(binned_formula, data=data, family=mb.Binomial(), control=control)
    assert exact.selected == binned.selected
    np.testing.assert_allclose(binned.fitted_, exact.fitted_, rtol=0, atol=2e-10)
    np.testing.assert_allclose(binned.risk_, exact.risk_, rtol=0, atol=2e-9)
    np.testing.assert_allclose(
        binned.predict(newdata={"x1": x1, "x2": x2}),
        binned.fitted_,
        rtol=0,
        atol=2e-10,
    )


def test_binned_bspatial_exact_path_and_training_replay():
    n = 140
    row = np.arange(n, dtype=np.float64)
    x = np.sin(0.11 * row) + 0.001 * row
    z = np.cos(0.07 * row + 0.2) - 0.002 * row
    y = (0.9 * x - 0.6 * z + 0.4 * x * z > 0.1).astype(float)
    data = {"y": y, "x": x, "z": z}
    exact_formula = (
        "y ~ bspatial(x, z, knots=4, degree=2, differences=1, "
        "center=True, lambda_value=0.4)"
    )
    binned_formula = (
        f"y ~ bspatial(x, z, knots=4, degree=2, differences=1, "
        f"center=True, lambda_value=0.4, bins={n})"
    )
    control = mb.boost_control(35, 0.1)
    exact = mb.gamboost(exact_formula, data=data, family=mb.Binomial(), control=control)
    binned = mb.gamboost(binned_formula, data=data, family=mb.Binomial(), control=control)
    assert exact.selected == binned.selected
    np.testing.assert_allclose(binned.fitted_, exact.fitted_, rtol=0, atol=2e-9)
    np.testing.assert_allclose(
        binned.predict(newdata={"x": x, "z": z}), binned.fitted_, rtol=0, atol=2e-9
    )


def test_binned_brandom_is_exact_and_replays_levels():
    n = 240
    group = np.arange(n) % 6
    y = ((group == 1) | (group == 4)).astype(float)
    data = {"y": y, "group": group.astype(float)}
    control = mb.boost_control(40, 0.1)
    exact = mb.gamboost(
        "y ~ brandom(group, lambda_value=0.6)",
        data=data,
        family=mb.Binomial(),
        control=control,
    )
    binned = mb.gamboost(
        "y ~ brandom(group, lambda_value=0.6, bins=6)",
        data=data,
        family=mb.Binomial(),
        control=control,
    )
    assert exact.selected == binned.selected
    np.testing.assert_allclose(binned.fitted_, exact.fitted_, rtol=0, atol=2e-10)
    np.testing.assert_allclose(
        binned.predict(newdata={"group": group.astype(float)}),
        binned.fitted_, rtol=0, atol=2e-10,
    )


def test_binned_cvrisk_uses_compact_spec_boundary(monkeypatch):
    data, weights = _data(240)
    model = mb.gamboost(
        _formula(80),
        data=data,
        weights=weights,
        family=mb.Gaussian(),
        control=mb.boost_control(50, 0.1),
    )
    module = importlib.import_module("mboostrs.gamboost")
    native = module._cvrisk_gamboost_specs
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[0])
        return native(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "_cvrisk_gamboost_specs",
        observed,
    )
    folds = mb.cv_weights(len(data["y"]), type="subsampling", B=4, seed=7)
    result = model.cvrisk(folds=folds, parallel=False)
    assert calls and len(calls[0]) == len(model._learners_spec)
    assert result.risk_matrix.shape == (4, 51)
    assert 1 <= result.optimal_mstop <= 50


def test_binned_stabsel_uses_shared_spec_selected_engine(monkeypatch):
    data, weights = _data(240)
    model = mb.gamboost(
        _formula(80),
        data=data,
        weights=weights,
        family=mb.Gaussian(),
        control=mb.boost_control(60, 0.1),
    )
    module = importlib.import_module("mboostrs.stabsel")
    monkeypatch.setattr(
        module,
        "_run_refit_selected",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Python refit used")),
    )
    # Folds carry the model's *effective* weights, which the upstream policy
    # rescaled from the fractional vector passed to `weights=`. Drawing them
    # from the raw input instead would put more weight on a row than the model
    # holds, and the complement of that fold would be negative.
    folds = mb.cv_weights(
        len(data["y"]), type="subsampling", B=4, seed=11, weights=model.weights
    )
    result = mb.stabsel(
        model, q=2, PFER=1.0, folds=folds, B=4, verbose=False, n_jobs=1
    )
    assert result.phat.shape[0] == 3
    assert np.all(np.isfinite(result.max_))


def test_global_bins_and_diagnostic_refit():
    data, _ = _data(260)
    model = mb.gamboost(
        "y ~ bbs(x1, center=True) + bbs(x2, center=True)",
        data=data,
        bins=40,
        family=mb.Gaussian(),
        control=mb.boost_control(45, 0.1),
    )
    assert [learner.bins for learner in model._learners_spec] == [40, 40]
    diagnostic = model.check_binning(n=180, seed=3, mstop=30)
    assert diagnostic["n"] == 180
    assert diagnostic["mstop"] == 30
    assert np.isfinite(diagnostic["relative_to_residual_sd"])
    assert isinstance(diagnostic["selection_sequence_identical"], bool)
    assert diagnostic["binned_retained_learner_bytes"] > 0
    assert diagnostic["exact_retained_learner_bytes"] > diagnostic["binned_retained_learner_bytes"]


def test_global_bins_skips_terms_with_no_binned_design():
    """`bins=` binned what it can and named what it could not.

    Before, one intercept-carrying `bols` in a formula made `gamboost(bins=)`
    raise, so the argument was unusable on any formula written the ordinary
    way.
    """
    data, _ = _data(260)
    with pytest.warns(UserWarning, match="intercept=False"):
        model = mb.gamboost(
            "y ~ bbs(x1, center=True) + bols(x2) + bols(x3, intercept=False)",
            data=data,
            bins=40,
            family=mb.Gaussian(),
            control=mb.boost_control(30, 0.1),
        )
    assert [learner.bins for learner in model._learners_spec] == [40, None, 40]
    assert np.all(np.isfinite(model.fitted()))


def test_lss_global_bins_matches_the_exact_fit_and_is_inherited():
    # A dense continuous predictor, so 100 quantile bins resolve the design far
    # past what a df=4 spline can use. `_data`'s deterministic sinusoid over 360
    # points is the opposite case: there, binning is a real approximation and
    # the exact and binned fits differ by a fifth of the response's spread on
    # the `gamboost` path too.
    rng = np.random.RandomState(4)
    n = 3000
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    data = {
        "x1": x1,
        "x2": x2,
        "y": np.sin(3.0 * x1) + 0.5 * x2 + rng.normal(0.0, 0.3, n),
    }
    formula = "y ~ bbs(x1, center=True, df=4) + bbs(x2, center=True, df=4)"
    control = mb.boost_control(40, 0.1)

    exact = mb.gamboostLSS(formula, data=data, control=control)
    binned = mb.gamboostLSS(formula, data=data, control=control, bins=100)

    assert [learner.bins for learner in binned._params[0]["learners"]] == [100, 100]
    # Binning changes the training storage, not the model: both parameters pick
    # the same learners in the same order, and the fitted mu agrees to a few
    # percent of the residual scale. The remaining gap is the binning
    # approximation itself, which falls off as 1/bins.
    for parameter in ("mu", "sigma"):
        # The order differs at near-ties, where a step's two candidates are
        # separated by less than the binning error, and the pair then swaps
        # back at the next step. How often each learner is chosen over the
        # path does not move.
        assert (Counter(exact.selected[parameter])
                == Counter(binned.selected[parameter]))
    residual_sd = float(np.std(np.asarray(data["y"])))
    difference = exact.fitted()["mu"] - binned.fitted()["mu"]
    assert np.sqrt(np.mean(difference**2)) < 0.05 * residual_sd

    exact_bytes = exact.execution_report().retained_learner_bytes
    binned_bytes = binned.execution_report().retained_learner_bytes
    assert exact_bytes > 0
    assert binned_bytes < exact_bytes

    # A refit for stabsel/cvrisk inherits the value rather than rebuilding the
    # exact design the fit chose not to build.
    assert binned._refit_arguments()["bins"] == 100
    assert "bins" not in exact._refit_arguments()
    assert binned.update(mstop=45)._bins == 100


def test_lss_stabsel_reports_the_fits_it_ran():
    data, _ = _data(300)
    model = mb.gamboostLSS(
        "y ~ bbs(x1, center=True) + bbs(x2, center=True)",
        data=data,
        control=mb.boost_control(30, 0.1),
        bins=50,
    )
    result = model.stabsel(q=2, PFER=1.0, B=3)
    # SS sampling fits each half-sample and its complement.
    assert result.n_fits == 6
    assert result["n_fits"] == 6
    assert "n_fits=6" in repr(result)
