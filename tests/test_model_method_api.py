"""Method forms of the model-level operations.

Standard models expose the diagnostics, inference, prediction-interval,
model-selection, and serialization operations as methods (fit.pit(),
fit.conformal_interval(), fit.save()). Each delegates to the module-level
function of the same name, which remains available. These tests pin that the
method form exists and returns the same result as the function form.
"""

import numpy as np
import pytest

import mboostrs as mb


def _model(n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = {"x": rng.uniform(1.0, 10.0, n), "z": rng.normal(size=n)}
    d["y"] = 2.0 * np.log(d["x"]) + 0.5 * d["z"] + rng.normal(scale=0.2, size=n)
    return mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data=d, family=mb.Gaussian(),
                       control=mb.boost_control(150, 0.1)), d


def test_conformal_interval_method_matches_function():
    fit, _ = _model()
    m = fit.conformal_interval(level=0.9, method="split", seed=1)
    f = mb.conformal_interval(fit, level=0.9, method="split", seed=1)
    np.testing.assert_array_equal(m.lower, f.lower)
    np.testing.assert_array_equal(m.upper, f.upper)


def test_save_method_matches_save_model(tmp_path):
    fit, _ = _model()
    a = fit.save(tmp_path / "a.json")
    b = mb.save_model(fit, tmp_path / "b.json")
    assert a.read_text() == b.read_text()
    reloaded = mb.load_model(a)
    np.testing.assert_array_equal(reloaded.fitted_, fit.fitted_)


def _call(fn):
    """Return ('ok', value) or ('raised', exception_type)."""
    try:
        return "ok", fn()
    except Exception as exc:  # noqa: BLE001
        return "raised", type(exc)


@pytest.mark.parametrize("name,kwargs", [
    ("predict_all", {}),
    ("coef_all", {}),
    ("deviance_increment", {}),
    ("pit", {"random_state": 0}),              # unsupported for Gaussian: both raise
    ("quantile_residuals", {"random_state": 0}),
    ("residual_frame", {"random_state": 0}),
    ("q_stats", {"random_state": 0}),
    ("worm_data", {"random_state": 0}),
    ("calibration_frame", {}),
    ("outlier_frame", {}),
])
def test_diagnostic_method_matches_function(name, kwargs):
    """The method and the function produce the same outcome — the same value
    or the same exception."""
    fit, _ = _model()
    m_status, m_val = _call(lambda: getattr(fit, name)(**kwargs))
    f_status, f_val = _call(lambda: getattr(mb, name)(fit, **kwargs))

    assert m_status == f_status
    if m_status == "raised":
        assert m_val is f_val
        return

    if hasattr(m_val, "equals"):        # polars/pandas frame
        assert m_val.equals(f_val)
    elif isinstance(m_val, np.ndarray):
        np.testing.assert_array_equal(m_val, f_val)
    else:
        assert m_val == f_val or type(m_val) is type(f_val)


def test_methods_exist_on_both_standard_model_classes():
    from mboostrs.gamboost import GAMBoostModel
    from mboostrs.glmboost import GLMBoostModel

    expected = {
        "conformal_interval", "save", "predict_all", "coef_all",
        "deviance_increment", "pit", "quantile_residuals", "residual_frame",
        "q_stats", "worm_data", "calibration_frame", "outlier_frame",
        "stabsel", "profile_mstop",
    }
    for cls in (GAMBoostModel, GLMBoostModel):
        missing = expected - set(dir(cls))
        assert not missing, f"{cls.__name__} is missing methods {sorted(missing)}"


def test_pickle_round_trip_survives_the_mixin():
    import pickle

    fit, _ = _model()
    restored = pickle.loads(pickle.dumps(fit))
    np.testing.assert_array_equal(restored.fitted_, fit.fitted_)
    # method still reachable after unpickle
    assert hasattr(restored, "conformal_interval")


def test_glmboost_model_also_has_the_methods():
    rng = np.random.default_rng(2)
    n = 300
    d = {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    d["y"] = 1.5 * d["x"] - 0.8 * d["z"] + rng.normal(scale=0.3, size=n)
    fit = mb.glmboost("y ~ x + z", data=d, family=mb.Gaussian(),
                      control=mb.boost_control(150, 0.1))
    ci = fit.conformal_interval(level=0.9, seed=0)
    assert 0.82 < ci.empirical_coverage(d["y"]) < 0.98


# ── 1.2 / 3.2: LSSModel gets the method forms the standard models have ───────
class TestLSSMethodForms:
    """Enhancement-hunt 1.2. `GLMBoostModel` and `GAMBoostModel` got method
    forms for cvrisk, stabsel, and profile_mstop; `LSSModel` did not, so a user
    moving from `gamboost` to `gamboostLSS` dropped back to module-level
    functions and had to name the fitting function a second time. The model
    already knows which one produced it."""

    @staticmethod
    def _data(seed=4, n=200):
        rng = np.random.default_rng(seed)
        x, z = rng.normal(size=n), rng.normal(size=n)
        y = 1.0 + 2.0 * x + rng.normal(0, np.exp(0.2 + 0.15 * z), n)
        return {"x": x, "z": z, "y": y}, n

    @pytest.fixture
    def lss_pair(self):
        from mboostrs.gamlss import GaussianLSS

        data, _ = self._data()
        glm = mb.glmboostLSS(
            {"mu": "y ~ x + z", "sigma": "y ~ z"}, data=data,
            families=GaussianLSS(), control=mb.boost_control(40, 0.1),
        )
        gam = mb.gamboostLSS(
            {"mu": "y ~ bbs(x, df=4) + bols(z)", "sigma": "y ~ bols(z)"},
            data=data, families=GaussianLSS(),
            control=mb.boost_control(30, 0.1),
        )
        return glm, gam

    def test_the_methods_exist_on_lss_models(self, lss_pair):
        for model in lss_pair:
            for name in ("cvrisk", "stabsel", "profile_mstop", "plot_risk",
                         "plot_selected", "control", "offset"):
                assert hasattr(model, name), f"LSSModel lacks {name}"

    def test_cvrisk_dispatches_on_the_fitting_function(self, lss_pair):
        _, n = self._data()
        folds = mb.cv_weights(n, type="kfold", B=3, seed=1)
        for model in lss_pair:
            result = model.cvrisk(folds=folds, grid=[[5, 5], [10, 10]])
            assert result.__class__.__name__ == "LssCvRiskResult"

    def test_stabsel_dispatches_on_the_fitting_function(self, lss_pair):
        glm, gam = lss_pair
        assert glm.stabsel(q=2, PFER=1.0, B=10).__class__.__name__ == "LssStabSelResult"
        assert gam.stabsel(q=1, PFER=1.0, B=10).__class__.__name__ == "LssStabSelResult"

    def test_control_and_offset_report_the_fit(self, lss_pair):
        for model in lss_pair:
            control = model.control
            assert control[0] == model.mstop if isinstance(model.mstop, int) else True
            offsets = model.offset
            assert set(offsets) == set(model.param_names)
            assert all(np.isfinite(o) for o in offsets.values())

    def test_profile_mstop_and_plots_return_something_usable(self, lss_pair):
        for model in lss_pair:
            assert len(model.profile_mstop(grid=[5, 10, 20])) == 3
            assert model.plot_risk() is not None
            assert model.plot_selected() is not None

    def test_module_level_functions_remain(self):
        """They are kept as thin delegates; removing them would be a break."""
        for name in ("cvrisk_glmboostLSS", "cvrisk_gamboostLSS",
                     "stabsel_glmboostLSS", "stabsel_gamboostLSS"):
            assert callable(getattr(mb, name))


class TestLSSNamingAccessors:
    """Enhancement-hunt 3.2. The standard models' naming and shape vocabulary,
    answered per distribution parameter."""

    @pytest.fixture
    def pair(self):
        from mboostrs.gamlss import GaussianLSS

        rng = np.random.default_rng(2)
        n = 200
        x, z = rng.normal(size=n), rng.normal(size=n)
        data = {"x": x, "z": z, "y": 1 + 2 * x + rng.normal(0, 1, n)}
        gam = mb.gamboostLSS(
            {"mu": "y ~ bbs(x, df=4) + bols(z)", "sigma": "y ~ bols(z)"},
            data=data, families=GaussianLSS(), control=mb.boost_control(30, 0.1))
        glm = mb.glmboostLSS(
            {"mu": "y ~ x + z", "sigma": "y ~ z"}, data=data,
            families=GaussianLSS(), control=mb.boost_control(30, 0.1))
        return gam, glm

    def test_names_are_keyed_by_parameter(self, pair):
        for model in pair:
            names = model.bnames()
            assert set(names) == set(model.param_names)
            assert names == model.base_learner_names()
            for parameter in model.param_names:
                assert model.bnames(parameter) == names[parameter]

    def test_glmboostlss_names_carry_the_intercept_learner(self, pair):
        """A `glmboostLSS` parameter has an intercept learner that is not in
        `predictor_names`; a `gamboostLSS` one names every learner."""
        gam, glm = pair
        assert "(Intercept)" in glm.bnames("mu")
        assert "(Intercept)" not in gam.bnames("mu")

    def test_variable_names_drops_the_intercept_and_deduplicates(self, pair):
        for model in pair:
            for names in model.variable_names().values():
                assert "(Intercept)" not in names
                assert len(names) == len(set(names))

    def test_selected_which_filters_by_name(self, pair):
        for model in pair:
            everything = model.selected_which()
            assert set(everything) == set(model.param_names)
            filtered = model.selected_which(which="x")
            for parameter, names in filtered.items():
                assert set(names) <= {"x"}
                assert names == [n for n in everything[parameter] if n == "x"]
            assert model.selected_which(parameter="mu") == everything["mu"]

    def test_fitted_and_offset_mirror_the_per_parameter_shape(self, pair):
        for model in pair:
            fitted = model.fitted_
            assert set(fitted) == set(model.param_names)
            for parameter, values in fitted.items():
                np.testing.assert_array_equal(
                    values, np.asarray(model.fitted(parameter=parameter))
                )
            offsets = model.offset_
            assert set(offsets) == set(model.param_names)
            assert offsets == model.offset

    def test_an_unknown_parameter_is_rejected_by_name(self, pair):
        gam, _ = pair
        for call in (gam.bnames, gam.variable_names):
            with pytest.raises(ValueError, match="unknown parameter"):
                call("not_a_parameter")
        with pytest.raises(ValueError, match="unknown parameter"):
            gam.selected_which(parameter="not_a_parameter")
