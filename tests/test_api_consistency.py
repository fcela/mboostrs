"""The same user task, spelled the same way, on every model class.

The 2026-09-04 audit found the three fitted-model classes disagreeing about
their own vocabulary: `control` and `offset` were methods on `glmboost` and
`gamboost` and properties on the distributional ones, `selected` was the other
way round, `risk` existed only on the distributional class as a method whose
two arguments were both refusals, and `boost_control()` returned an anonymous
4-tuple with no docstring. A helper written against one class returned a bound
method on another, and `if fit.selected:` was true for every distributional fit
whatever it had selected.

This is the matrix D0 asks for: it runs the same task on all three classes and
compares the *form* of the answer -- property against method, the arguments the
fitting functions take, the shape of what comes back, and the exception class
on the same mistake. It is not a test of the numbers; it is a test that the
package can be described in one sentence per operation.
"""

import inspect
import pickle
import warnings

import numpy as np
import pytest

import mboostrs as mb

N = 150


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    x, z = rng.normal(size=N), rng.normal(size=N)
    return {"x": x, "z": z, "y": 2.0 * x + 0.5 * z + rng.normal(size=N)}


@pytest.fixture(scope="module")
def models(data):
    control = mb.boost_control(30, 0.1)
    return {
        "glmboost": mb.glmboost("y ~ x + z", data, control=control),
        "gamboost": mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data,
                                control=control),
        "glmboostLSS": mb.glmboostLSS("y ~ x + z", data,
                                      families=mb.GaussianLSS(),
                                      control=control),
        "gamboostLSS": mb.gamboostLSS("y ~ bbs(x, df=4)", data,
                                      families=mb.GaussianLSS(),
                                      control=control),
    }


FITTING_FUNCTIONS = (mb.glmboost, mb.gamboost, mb.glmboostLSS, mb.gamboostLSS)


# ─── the fitting signatures ────────────────────────────────────────


@pytest.mark.parametrize("fit", FITTING_FUNCTIONS, ids=lambda f: f.__name__)
def test_every_fitting_function_takes_formula_and_data_then_keywords(fit):
    """One rule: `fit(formula, data, *, ...)`. The distributional entry points
    were positional all the way down, and the standard ones required
    `data=`."""
    parameters = list(inspect.signature(fit).parameters.values())
    positional = [p.name for p in parameters
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == ["formula", "data"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
               for p in parameters[2:])


@pytest.mark.parametrize("fit", FITTING_FUNCTIONS, ids=lambda f: f.__name__)
def test_every_fitting_function_takes_the_same_core_arguments(fit):
    names = set(inspect.signature(fit).parameters)
    assert {"control", "weights", "offset", "weight_policy"} <= names


def test_the_distributional_family_argument_has_one_documented_spelling(data):
    """`families=` is the documented one -- it matches `gamboostLSS` in R and
    the R frontend here, and this codebase uses it 254 times against 11 for
    the alias. `family=` is accepted; both together is a `TypeError`, because
    they would have to agree and nothing checks that they do."""
    control = mb.boost_control(10, 0.1)
    plural = mb.glmboostLSS("y ~ x", data, families=mb.GaussianLSS(),
                            control=control)
    singular = mb.glmboostLSS("y ~ x", data, family=mb.GaussianLSS(),
                              control=control)
    np.testing.assert_allclose(plural.risk, singular.risk)
    with pytest.raises(TypeError, match="not both"):
        mb.glmboostLSS("y ~ x", data, families=mb.GaussianLSS(),
                       family=mb.GaussianLSS(), control=control)
    assert "Alias for ``families=``" in mb.glmboostLSS.__doc__


# ─── the control record ────────────────────────────────────────────


def test_boost_control_is_a_named_record_that_is_still_a_tuple():
    control = mb.boost_control(50, 0.05)
    assert isinstance(control, tuple)
    # The step rule grew the record by two defaulted fields; the first four
    # are the positions every existing `control[0]`-style reader uses.
    assert control == (50, 0.05, "inbag", False, "fixed", 4)
    assert (control.mstop, control.nu, control.risk, control.trace) == (
        50, 0.05, "inbag", False)
    assert mb.boost_control.__doc__
    assert mb.BoostControl.__doc__


def test_boost_control_still_validates_where_it_always_did():
    with pytest.raises(ValueError, match="mstop must be > 0"):
        mb.boost_control(0, 0.1)
    with pytest.raises(ValueError, match="risk must be"):
        mb.boost_control(10, 0.1, risk="oops")


def test_every_model_reports_the_control_it_was_fitted_with(models):
    for name, model in models.items():
        control = model.control
        assert control[:2] == (30, 0.1), name
        assert control.mstop == 30, name
        assert control.nu == 0.1, name


def test_transitional_property_values_pickle_as_ordinary_public_values(models):
    """The one-release call shim must not leak an unimportable dynamic type."""
    from mboostrs._control import BoostControl

    for kind, model in models.items():
        control = pickle.loads(pickle.dumps(model.control))
        assert type(control) is BoostControl, kind
        assert control == model.control

        offset = pickle.loads(pickle.dumps(model.offset))
        assert offset == model.offset, kind

        risk = pickle.loads(pickle.dumps(model.risk))
        np.testing.assert_array_equal(np.asarray(risk), np.asarray(model.risk),
                                      err_msg=kind)

        selected = pickle.loads(pickle.dumps(model.selected))
        assert selected == model.selected, kind


# ─── property against method, decided once ─────────────────────────


STATE_ACCESSORS = ("mstop", "control", "offset", "offset_", "selected",
                   "risk", "risk_", "fitted_", "weights", "weight_policy")


@pytest.mark.parametrize("name", STATE_ACCESSORS)
def test_the_state_accessors_are_properties_on_every_class(models, name):
    for kind, model in models.items():
        attribute = inspect.getattr_static(type(model), name, None)
        assert isinstance(attribute, property), f"{kind}.{name}"


COMPUTING_METHODS = ("fitted", "predict", "residuals", "coef", "summary",
                     "logLik", "cvrisk", "stabsel", "update", "with_mstop",
                     "varimp", "path_frame", "parity_notes")


@pytest.mark.parametrize("name", COMPUTING_METHODS)
def test_the_computing_operations_are_methods_on_every_class(models, name):
    for kind, model in models.items():
        attribute = inspect.getattr_static(type(model), name, None)
        assert attribute is not None, f"{kind} has no {name}"
        assert not isinstance(attribute, property), f"{kind}.{name}"
        assert callable(attribute), f"{kind}.{name}"


def test_risk_and_risk_underscore_are_the_same_path(models):
    for kind, model in models.items():
        np.testing.assert_array_equal(np.asarray(model.risk),
                                      np.asarray(model.risk_), err_msg=kind)


def test_offset_and_offset_underscore_are_the_same_value(models):
    for kind, model in models.items():
        assert model.offset == model.offset_, kind


# ─── the same task, on every class ─────────────────────────────────


def test_truncating_a_path_is_spelled_the_same_way(models):
    """`fit[:k]` on all four. A standard fit slices; a distributional one
    refits, because a cyclic path is not prefix-consistent -- but the caller
    writes the same thing."""
    for kind, model in models.items():
        shorter = model[:10]
        budget = shorter.mstop
        assert budget == 10 or set(budget.values()) == {10}, kind
        assert model.with_mstop(10).mstop == budget, kind
        assert model.update(mstop=10).mstop == budget, kind


def test_slicing_from_the_middle_is_refused_everywhere(models):
    for model in models.values():
        with pytest.raises(ValueError, match="from the start"):
            model[5:10]


def test_cross_validation_needs_no_arguments_on_any_class(models):
    for kind, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = model.cvrisk()
        assert result.risk_matrix.ndim == 2, kind
        assert np.all(np.isfinite(result.aggregate)), kind


def test_every_result_carries_the_same_provenance_fields(models):
    for kind, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            record = model.cvrisk().provenance
        assert record.operation == "cvrisk", kind
        assert record.n_obs == N, kind
        assert record.family, kind


def test_a_prediction_interval_has_one_documented_name(models):
    """`PI` and `predint` are R-parity aliases of `prediction_interval`, and
    say so in their own docstrings rather than being three peers."""
    from mboostrs.gamlss import LSSModel

    for kind, model in models.items():
        if not isinstance(model, LSSModel):
            continue
        assert "Alias for :meth:`prediction_interval`" in model.PI.__doc__, kind
        assert "Alias for :meth:`prediction_interval`" in model.predint.__doc__


@pytest.mark.parametrize("output", ["records", "polars", "pandas"])
def test_a_table_honours_output_on_every_class(models, output):
    expected = {"records": "builtins", "polars": "polars", "pandas": "pandas"}
    for kind, model in models.items():
        frame = model.parity_notes(output=output)
        assert type(frame).__module__.split(".")[0] == expected[output], kind


def test_the_same_mistake_raises_the_same_class(models):
    """An unknown output format is a `ValueError` everywhere, not a `KeyError`
    on one class and a `TypeError` on another."""
    for model in models.values():
        with pytest.raises(ValueError, match="output must be"):
            model.parity_notes(output="frame")


def test_reading_stored_state_never_mutates_the_model(models):
    """The trailing-underscore accessors hand out read-only views, so a caller
    who edits "their copy" is told rather than silently changing the fit."""
    for kind, model in models.items():
        risk = np.asarray(model.risk_)
        with pytest.raises(ValueError, match="read-only"):
            risk[0] = 0.0
        assert risk.copy().flags.writeable, kind


def test_a_deprecated_call_form_warns_on_every_class_that_had_one(models):
    from mboostrs._compat import DEPRECATIONS
    from mboostrs.gamlss import LSSModel

    for model in models.values():
        if isinstance(model, LSSModel):
            with pytest.warns(FutureWarning):
                model.selected(parameter=model.param_names[0])
        else:
            with pytest.warns(FutureWarning):
                model.control()
            with pytest.warns(FutureWarning):
                model.offset()
    # Every registered deprecation names its removal release.
    for key, entry in DEPRECATIONS.items():
        assert entry.removed_in and entry.since, key


def test_selection_counts_names_learners_apart_on_a_shared_column():
    """Two learners can read one column -- the linear-plus-deviation split --
    and `selected`, keyed by column, merges them. The per-learner table
    counts them separately, disambiguated the way `base_learner_names` is,
    and includes a term that never earned an update with zero rather than
    dropping it."""
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=n)
    data = {"x": x, "y": 1.5 * x + rng.normal(scale=0.2, size=n)}
    fit = mb.gamboost("y ~ bbs(x, df=4) + bols(x, intercept=False)",
                      data=data, control=mb.boost_control(40))
    rows = fit.selection_counts(output="records")

    names = [row["learner"] for row in rows]
    assert "x (bbs)" in names and "x (bols)" in names
    assert sum(row["updates"] for row in rows) == fit.mstop
    # Most updates first, and a never-selected learner is present with zero.
    updates = [row["updates"] for row in rows]
    assert updates == sorted(updates, reverse=True)
    assert min(row["share"] for row in rows) >= 0.0
    assert fit.selection_counts(output="polars").columns == [
        "learner", "column", "updates", "share"]
