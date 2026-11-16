"""One model surface, one result surface.

The three fitted-model classes disagreed about the form of three accessors:
`control` and `offset` were methods on `glmboost`/`gamboost` and properties on
the distributional classes, and `selected` was the other way round. The same
expression therefore returned a value on one fit and a bound method on
another, and `if fit.selected:` was true for every distributional fit whether
or not anything had been selected.

Two accessors also handed out the model's own arrays. `f = fit.fitted_;
f -= f.mean()` changed what `fit.residuals()` reported afterwards while
`fit.predict()` went on returning the original values, so the fit was
internally inconsistent and nothing said so.

Result objects differed the same way: the distributional ones answered
`result["aggregate"]` and the standard ones raised `TypeError`, and only three
of five took `to_frame(output=...)`. None of them recorded what produced them,
so a curve read back a week later could not say how many rows, which family,
or what seed it came from.
"""

import warnings

import numpy as np
import pytest
from mboostrs._protocols import BoostModel, LSSBoostModel, StandardBoostModel
from mboostrs._results import Provenance

import mboostrs as mb

MSTOP = 25


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    n = 200
    x, z = rng.normal(size=n), rng.normal(size=n)
    return {"x": x, "z": z, "y": 2.0 * x + 0.5 * z + rng.normal(size=n)}


@pytest.fixture(scope="module")
def glm(data):
    return mb.glmboost("y ~ x + z", data=data,
                       control=mb.boost_control(MSTOP, 0.1))


@pytest.fixture(scope="module")
def gam(data):
    return mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data=data,
                       control=mb.boost_control(MSTOP, 0.1))


@pytest.fixture(scope="module")
def lss(data):
    return mb.glmboostLSS("y ~ x + z", data=data, families=mb.GaussianLSS(),
                          control=mb.boost_control(MSTOP, 0.1))


@pytest.fixture(scope="module")
def every_model(glm, gam, lss):
    return {"glmboost": glm, "gamboost": gam, "glmboostLSS": lss}


# ─── the protocols hold ────────────────────────────────────────────


def test_every_model_satisfies_the_shared_protocol(every_model):
    for name, model in every_model.items():
        assert isinstance(model, BoostModel), name


def test_the_standard_and_distributional_protocols_separate_them(glm, gam, lss):
    assert isinstance(glm, StandardBoostModel)
    assert isinstance(gam, StandardBoostModel)
    assert not isinstance(lss, StandardBoostModel)
    assert isinstance(lss, LSSBoostModel)
    assert not isinstance(glm, LSSBoostModel)
    assert not isinstance(gam, LSSBoostModel)


def test_a_protocol_member_missing_from_a_class_is_a_failure_here():
    """The point of the conformance tests above: they fail when a member is
    deleted from one class and not the other, rather than surfacing later as
    an `AttributeError` inside a helper."""

    class Incomplete:
        mstop = 1

    assert not isinstance(Incomplete(), BoostModel)


def test_runtime_branching_does_not_go_through_the_protocols(glm, lss):
    """`_dispatch` compares against the real classes. A protocol check is
    `hasattr` for every member, which is the mechanism the dispatch module
    exists to avoid."""
    from mboostrs import _dispatch

    assert _dispatch.is_lss(lss) and not _dispatch.is_lss(glm)
    assert _dispatch.is_standard(glm) and not _dispatch.is_standard(lss)


# ─── control, offset, selected: one form each ──────────────────────


@pytest.mark.parametrize("name", ["control", "offset", "selected", "mstop"])
def test_the_state_accessors_are_properties_everywhere(every_model, name):
    for kind, model in every_model.items():
        attribute = getattr(type(model), name, None)
        assert isinstance(attribute, property), f"{kind}.{name} is not a property"


def test_control_reports_the_same_thing_on_every_class(every_model):
    for kind, model in every_model.items():
        control = model.control
        # 6 since the step rule: mstop, nu, risk, trace, step, step_iterations.
        assert isinstance(control, tuple) and len(control) == 6, kind
        assert control[0] == MSTOP
        assert control[1] == pytest.approx(0.1)


def test_offset_and_offset_underscore_are_the_same_value(every_model):
    """They are one quantity under two names. On the LSS classes `offset` was
    a list in `param_names` order while `offset_` was a mapping, so the two
    indexed differently."""
    for kind, model in every_model.items():
        assert model.offset == model.offset_, kind


def test_selected_is_the_whole_path_on_every_class(glm, gam, lss):
    for model in (glm, gam):
        assert isinstance(model.selected, list)
        assert len(model.selected) == MSTOP
    assert set(lss.selected) == set(lss.param_names)
    for path in lss.selected.values():
        assert len(path) == MSTOP


def test_canonical_property_values_have_documented_containers(every_model):
    """The accessors exist everywhere, but the containers they hand back are
    part of the contract too: `control` is a named record with fields, `risk`
    is an array the length of the path, `selected` is a list on the standard
    classes and the per-parameter mapping on the LSS ones, and `offset` is the
    scalar on one half and the parameter mapping on the other."""
    from mboostrs._control import BoostControl

    for kind, model in every_model.items():
        control = model.control
        assert isinstance(control, BoostControl), kind
        assert (control.mstop, control.nu) == (MSTOP, pytest.approx(0.1)), kind

        risk, risk_ = np.asarray(model.risk), np.asarray(model.risk_)
        assert risk.shape == (MSTOP + 1,), kind
        np.testing.assert_allclose(risk, risk_)

        assert isinstance(model.mstop, int) and model.mstop == MSTOP, kind

    for kind in ("glmboost", "gamboost"):
        model = every_model[kind]
        assert isinstance(model.offset, float), kind
        assert isinstance(model.selected, list), kind
        assert all(isinstance(name, str) for name in model.selected), kind

    lss = every_model["glmboostLSS"]
    assert isinstance(lss.offset, dict), "glmboostLSS"
    assert set(lss.offset) == set(lss.param_names), "glmboostLSS"
    assert isinstance(lss.selected, dict), "glmboostLSS"
    assert set(lss.selected) == set(lss.param_names), "glmboostLSS"


# ─── the method forms still work, and say they are going ───────────


def test_the_old_call_forms_still_work_and_warn(glm, gam, lss):
    for model in (glm, gam):
        with pytest.warns(FutureWarning, match=r"fit\.control\(\) is deprecated"):
            assert model.control() == model.control
        with pytest.warns(FutureWarning, match=r"fit\.offset\(\) is deprecated"):
            assert model.offset() == model.offset
    with pytest.warns(FutureWarning, match="distributional model is deprecated"):
        assert lss.selected(parameter="mu") == lss.selected["mu"]


def test_a_deprecation_warning_names_the_release_that_removes_it():
    from mboostrs._compat import DEPRECATIONS

    assert DEPRECATIONS, "expected a registry, not an empty dict"
    for key, entry in DEPRECATIONS.items():
        message = entry.message()
        assert entry.since in message and entry.removed_in in message, key
        assert entry.new in message, key


def test_reading_the_property_does_not_warn(every_model):
    """Only calling it is deprecated; the property form is the new one."""
    for model in every_model.values():
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            _ = model.control, model.offset, model.selected, model.mstop


def test_selected_which_covers_what_the_call_form_did(lss):
    """`selected(merge=True)` moved to `selected_which(merge=True)`, so the
    deprecated form leaves nothing behind."""
    merged = lss.selected_which(merge=True)
    assert all(name.rsplit(".", 1)[-1] in lss.param_names for name in merged)
    assert len(merged) == sum(len(p) for p in lss.selected.values())
    with pytest.warns(FutureWarning):
        assert lss.selected(merge=True) == merged


def test_merge_and_parameter_together_are_rejected(lss):
    with pytest.raises(ValueError, match="nothing to do once parameter="):
        lss.selected_which(parameter="mu", merge=True)


# ─── copy semantics ────────────────────────────────────────────────


def test_writing_to_the_stored_state_is_refused(glm, gam, lss):
    for model in (glm, gam):
        with pytest.raises(ValueError, match="read-only"):
            model.fitted_[0] = 0.0
        with pytest.raises(ValueError, match="read-only"):
            model.risk_[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        lss.risk_[0] = 0.0
    for values in lss.fitted_.values():
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0.0


def test_the_refusal_is_what_used_to_be_silent_corruption(data):
    """`f -= f.mean()` on `fit.fitted_` used to change `fit.residuals()` while
    leaving `fit.predict()` alone, so the fit disagreed with itself."""
    model = mb.glmboost("y ~ x", data=data, control=mb.boost_control(20, 0.1))
    before = model.residuals(type="response").copy()
    f = model.fitted_
    with pytest.raises(ValueError, match="read-only"):
        f -= f.mean()
    np.testing.assert_array_equal(model.residuals(type="response"), before)


def test_a_copy_of_the_stored_state_is_writable(glm):
    """Read-only is not a dead end: the caller who wanted their own array
    still gets one."""
    copy = glm.fitted_.copy()
    copy -= copy.mean()
    assert copy.flags.writeable
    assert not np.allclose(copy, glm.fitted_)


def test_the_method_forms_return_a_fresh_array(glm):
    """A method computes; there is no stored object behind it to protect."""
    values = glm.fitted()
    assert values.flags.writeable
    values[0] += 1.0
    assert glm.fitted()[0] != values[0]
    weights = glm.model_weights()
    assert weights.flags.writeable


# ─── result objects ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def results(glm, lss, data):
    n = len(data["y"])
    folds = mb.cv_weights(n, type="kfold", B=3, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "CvRiskResult": glm.cvrisk(),
            "StabSelResult": glm.stabsel(q=2, PFER=1.0, B=6),
            "AicResult": glm.aic(),
            "LssCvRiskResult": lss.cvrisk(folds=folds, grid=[[5, 5], [10, 10]]),
            "LssStabSelResult": lss.stabsel(q=2, PFER=1.0, B=6),
        }


@pytest.mark.parametrize("output", ["records", "polars", "pandas"])
def test_every_result_takes_the_same_output_argument(results, output):
    """Two of the five took no `output=` at all, so the same call worked for a
    standard fit and raised `TypeError` for a distributional one."""
    expected = {"records": "builtins", "polars": "polars", "pandas": "pandas"}
    for name, result in results.items():
        frame = result.to_frame(output=output)
        assert type(frame).__module__.split(".")[0] == expected[output], name


def test_every_result_reads_as_a_mapping(results):
    for name, result in results.items():
        keys = list(result.keys())
        assert keys, name
        assert dict(result).keys() == set(keys)
        for key in keys:
            assert key in result
            result[key]


def test_an_unknown_key_names_the_keys_that_exist(results):
    for result in results.values():
        with pytest.raises(KeyError, match="it answers to"):
            result["not_a_field"]


def test_the_legacy_r_style_keys_still_resolve(results):
    """`cvrisk_gamboostLSS` used to return an R-style list; callers wrote
    `result["risk"]` and `result["mstop"]` against it."""
    cv = results["LssCvRiskResult"]
    np.testing.assert_array_equal(cv["risk"], cv.risk_matrix)
    assert cv["mstop"] == cv.optimal_mstop
    stab = results["LssStabSelResult"]
    assert stab["names"] == stab.learner_names
    assert set(stab["max"]) == set(stab.learner_names)


def test_the_standard_results_answer_the_same_keys(results):
    """Generic code written against a distributional result used to raise
    `TypeError: 'CvRiskResult' object is not subscriptable`."""
    cv = results["CvRiskResult"]
    np.testing.assert_array_equal(cv["risk"], cv.risk_matrix)
    assert cv["mstop"] == cv.optimal_mstop
    stab = results["StabSelResult"]
    assert stab["names"] == stab.learner_names
    assert set(stab["max"]) == set(stab.learner_names)


# ─── provenance ────────────────────────────────────────────────────


def test_every_result_says_what_produced_it(results):
    for name, result in results.items():
        record = result.provenance
        assert isinstance(record, Provenance), name
        assert record.operation in ("cvrisk", "stabsel", "aic"), name
        assert record.model in ("glmboost", "gamboost",
                               "glmboostLSS", "gamboostLSS"), name
        assert record.family, name
        assert record.n_obs == 200, name
        assert record.mstop == MSTOP, name
        assert record.weight_policy == "upstream", name
        assert record.version, name


def test_provenance_records_the_resampling_scheme(results):
    """A curve read back a week later has to say what folds it came from;
    `fold_type` alone did not name the seed or the fit count."""
    stab = results["StabSelResult"]
    assert stab.provenance.seed == 42
    assert "pairs" in stab.provenance.folds
    assert stab.provenance.n_fits == 2 * stab.B
    assert results["AicResult"].provenance.folds is None


def test_provenance_records_the_effective_weights(data):
    """Fractional weights are rescaled under the upstream policy, so the
    effective total is not the sum of what the caller passed. Two results with
    different totals were not computed on the same sample."""
    rng = np.random.default_rng(1)
    w = rng.uniform(0.2, 0.9, size=len(data["y"]))
    model = mb.glmboost("y ~ x", data=data, weights=w,
                        control=mb.boost_control(10, 0.1))
    record = model.aic().provenance
    assert record.effective_weight_total == pytest.approx(len(w))
    assert record.effective_weight_total != pytest.approx(w.sum())

    raw = mb.glmboost("y ~ x", data=data, weights=w, weight_policy="raw",
                      control=mb.boost_control(10, 0.1))
    assert raw.aic().provenance.effective_weight_total == pytest.approx(w.sum())
    assert raw.aic().provenance.weight_policy == "raw"


def test_describe_prints_only_the_fields_the_operation_has(results):
    text = results["AicResult"].describe()
    assert "aic (mboostrs" in text
    assert "observations: 200" in text
    assert "folds" not in text  # an AIC path does not resample
    assert "seed" not in text


def test_provenance_is_a_value_not_a_live_view(glm):
    """It is frozen, so a result cannot be edited into claiming a different
    provenance than the one it was built with."""
    record = glm.aic().provenance
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError
        record.n_obs = 1
