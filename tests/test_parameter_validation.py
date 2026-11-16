"""Every public parameter is checked before it reaches a numerical routine.

A constructor is the last place a bad value can be named against the argument
the caller wrote. Past that point the value becomes a wrong number rather than
an error: ``Huber(delta=-1)`` makes the loss decrease with the residual and
reports a negative value as risk; ``Cindex(sigma=-1)`` inverts the smoothing
sigmoid so the loss rewards discordant pairs; ``PropOdds(n_levels=1)`` and
``Multinomial(n_classes=1)`` describe a categorical response with one
category, which the binding used to repair with ``.max(2)`` and ``.max(3)``
and then fit a different model than the caller asked for.

The same table is applied at three places, because a family reaches the engine
by three routes: a Python constructor, the binding dispatch used by
cross-validation and stability selection, and artifact deserialization. A
value rejected when typed is rejected when loaded.
"""

import json

import numpy as np
import pytest
from mboostrs._rust import bbs_raw_design, family_diagnostics, lss_family_diagnostics
from mboostrs._validate import (
    check_boundary_knots,
    check_finite_number,
    check_int_at_least,
    check_matrix,
    check_name_list,
    check_positive,
    check_unit_interval,
)

import mboostrs as mb

# ─── the checkers ──────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_unit_interval_rejects_everything_outside_it(value):
    with pytest.raises(ValueError):
        check_unit_interval("tau", value)


@pytest.mark.parametrize("value", [0.001, 0.5, 0.999])
def test_unit_interval_accepts_the_open_interval(value):
    assert check_unit_interval("tau", value) == pytest.approx(value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("-inf")])
def test_positive_rejects_zero_and_below(value):
    with pytest.raises(ValueError):
        check_positive("delta", value)


def test_finite_number_names_the_type_it_got():
    with pytest.raises(TypeError, match="delta must be a finite number, got str"):
        check_finite_number("delta", "big")
    with pytest.raises(TypeError, match="got bool"):
        check_finite_number("delta", True)


def test_none_is_accepted_only_where_it_is_meaningful():
    assert check_positive("delta", None, allow_none=True) is None
    with pytest.raises(ValueError, match="got None"):
        check_positive("delta", None)


def test_int_at_least_narrows_an_integral_float():
    """A value read back from JSON or from R arrives as a double."""
    assert check_int_at_least("n_classes", 3.0, 3) == 3
    with pytest.raises(ValueError, match="whole number"):
        check_int_at_least("n_classes", 3.5, 3)
    with pytest.raises(ValueError, match="at least 3, got 2"):
        check_int_at_least("n_classes", 2, 3)


def test_name_list_accepts_a_bare_string():
    assert check_name_list("Btree names", "x") == ["x"]
    assert check_name_list("Btree names", ["x", "y"]) == ["x", "y"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "at least one variable"),
        (["x", "x"], "must be distinct"),
        (["x", ""], "empty variable name"),
        ([1, 2], "got int"),
        (3, "got int"),
    ],
)
def test_name_list_rejects_unusable_shapes(value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        check_name_list("Btree names", value)


def test_matrix_accepts_array_likes_and_returns_an_array():
    out = check_matrix("X", [[1, 2], [3, 4]])
    assert isinstance(out, np.ndarray) and out.dtype == np.float64
    assert out.shape == (2, 2)


@pytest.mark.parametrize(
    ("kwargs", "value", "message"),
    [
        ({}, np.array([1.0, 2.0]), "must be 2-dimensional"),
        ({}, np.zeros((0, 3)), "must not be empty"),
        ({}, [[1.0, np.inf]], r"X\[0, 1\] is inf"),
        ({"square": True}, np.ones((2, 3)), "must be square"),
        ({"rows": 4}, np.ones((2, 2)), "must have 4 rows, got 2"),
        ({"cols": 4}, np.ones((2, 2)), "must have 4 columns, got 2"),
    ],
)
def test_matrix_rejects_wrong_shapes_and_values(kwargs, value, message):
    with pytest.raises(ValueError, match=message):
        check_matrix("X", value, **kwargs)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ((1.0,), "exactly two values"),
        ((1.0, 1.0), "strictly increasing"),
        ((5.0, 1.0), "strictly increasing"),
        ((np.nan, 1.0), "must be finite"),
    ],
)
def test_boundary_knots_must_bound_something(value, message):
    with pytest.raises(ValueError, match=message):
        check_boundary_knots("boundary_knots", value)


def test_boundary_knots_pass_through_as_plain_floats():
    assert check_boundary_knots("boundary_knots", (1, 5)) == (1.0, 5.0)
    assert check_boundary_knots("boundary_knots", None) is None


# ─── family constructors ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: mb.Quantile(qoffset=2), "qoffset must be in (0, 1), got 2.0"),
        (lambda: mb.Quantile(qoffset=0), "qoffset must be in (0, 1), got 0.0"),
        (lambda: mb.Quantile(tau=1.0), "tau must be in (0, 1), got 1.0"),
        (lambda: mb.Expectile(tau=-0.2), "tau must be in (0, 1), got -0.2"),
        (lambda: mb.Huber(delta=-1), "delta must be positive, got -1.0"),
        (lambda: mb.Huber(delta=0), "delta must be positive, got 0.0"),
        (lambda: mb.PropOdds(n_levels=1), "n_levels must be at least 3, got 1"),
        (lambda: mb.PropOdds(n_levels=2), "n_levels must be at least 3, got 2"),
        (lambda: mb.Multinomial(n_classes=1), "n_classes must be at least 3, got 1"),
        (lambda: mb.Multinomial(n_classes=2), "n_classes must be at least 3, got 2"),
        (lambda: mb.Cindex(sigma=-1), "sigma must be positive, got -1.0"),
        (lambda: mb.Cindex(sigma=0), "sigma must be positive, got 0.0"),
    ],
)
def test_invalid_family_parameters_are_rejected_at_construction(call, message):
    with pytest.raises(ValueError) as excinfo:
        call()
    assert message in str(excinfo.value)


def test_valid_family_parameters_still_construct():
    assert mb.Quantile(tau=0.9, qoffset=0.1).qoffset == pytest.approx(0.1)
    assert mb.Huber().delta is None
    assert mb.Huber(delta=2.0).delta == pytest.approx(2.0)
    assert mb.PropOdds().n_levels is None
    assert mb.PropOdds(n_levels=4).n_levels == 4
    assert mb.Multinomial(n_classes=3.0).n_classes == 3
    assert mb.Cindex().sigma == pytest.approx(0.1)


def test_a_fractional_category_count_is_rejected_not_truncated():
    with pytest.raises(ValueError, match="whole number"):
        mb.Multinomial(n_classes=3.5)
    with pytest.raises(ValueError, match="whole number"):
        mb.PropOdds(n_levels=4.2)


# ─── the binding dispatch ──────────────────────────────────────────
#
# A family reaches the engine from cross-validation, stability selection, and
# a deserialized artifact without passing a Python constructor, so the check
# lives on both sides rather than being trusted from one.


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("quantreg", {"tau": 0.0}, "tau must be in (0, 1)"),
        ("quantreg", {"qoffset": 2.0}, "qoffset must be in (0, 1)"),
        ("expectreg", {"tau": 1.0}, "tau must be in (0, 1)"),
        ("huber", {"delta": -1.0}, "delta must be positive"),
        ("cindex", {"sigma": -1.0}, "sigma must be positive"),
        ("hingeloss", {"alpha_cost": 1.5}, "alpha_cost must be in (0, 1)"),
        ("propodds", {"n_levels": 1.0}, "n_levels must be at least 3"),
        ("multinomial", {"n_classes": 2.0}, "n_classes must be at least 3"),
        ("multinomial", {"n_classes": 3.5}, "n_classes must be a whole number"),
    ],
)
def test_the_binding_rejects_what_the_constructor_rejects(name, params, message):
    y = np.array([1.0, 2.0, 3.0])
    f = np.zeros(3)
    w = np.ones(3)
    with pytest.raises(ValueError) as excinfo:
        family_diagnostics(name, params, y, f, w)
    assert message in str(excinfo.value)
    assert name in str(excinfo.value), "the message should name the family"


@pytest.mark.parametrize("name", ["quantreg", "expectreg", "huber"])
def test_the_binding_still_accepts_the_defaults(name):
    """Rejecting bad values must not have made the defaults unreachable."""
    y = np.array([1.0, 2.0, 3.0])
    f = np.zeros(3)
    w = np.ones(3)
    _offset, gradient, risk = family_diagnostics(name, {}, y, f, w)
    assert np.all(np.isfinite(gradient))
    assert np.isfinite(risk)


@pytest.mark.parametrize(
    ("y", "f", "w", "message"),
    [
        ([1.0, 2.0], [0.0], [1.0], "implies 2 fitted values"),
        ([1.0, 2.0], [0.0, 0.0], [1.0], "one weight per observation"),
        ([1.0], [np.nan], [1.0], "finite"),
        ([1.0], [0.0], [-1.0], "non-negative"),
        ([1.0], [0.0], [0.0], "positive sum"),
    ],
)
def test_family_diagnostics_validates_shapes_and_values(y, f, w, message):
    with pytest.raises(ValueError, match=message):
        family_diagnostics(
            "gaussian",
            {},
            np.asarray(y, dtype=float),
            np.asarray(f, dtype=float),
            np.asarray(w, dtype=float),
        )


def test_family_diagnostics_runs_family_response_validation():
    with pytest.raises(ValueError, match="family 'Binomial'"):
        family_diagnostics(
            "binomial",
            {},
            np.asarray([-0.2, 1.0]),
            np.zeros(2),
            np.ones(2),
        )


def test_family_diagnostics_accepts_strided_arrays():
    y = np.arange(1.0, 7.0)[::2]
    f = np.zeros(6)[::2]
    w = np.ones(6)[::2]
    offset, gradient, risk = family_diagnostics("gaussian", {}, y, f, w)
    assert np.isfinite(offset)
    assert len(gradient) == 3
    assert np.isfinite(risk)


def test_lss_family_diagnostics_accepts_strided_arrays():
    y = np.arange(1.0, 7.0)[::2]
    w = np.ones(6)[::2]
    eta = np.zeros(6)[::2]
    offsets, gradients, risk = lss_family_diagnostics(
        "gaussianlss", 2, y, [eta, eta], w
    )
    assert len(offsets) == 2
    assert all(len(gradient) == 3 for gradient in gradients)
    assert np.isfinite(risk)


def test_lss_family_diagnostics_rejects_invalid_dirichlet_dimension_cleanly():
    with pytest.raises(ValueError, match="DirichletLSS.*K must be in"):
        lss_family_diagnostics(
            "dirichletlss", 1, np.asarray([1.0]), [np.asarray([0.0])],
            np.ones(1),
        )


def test_raw_bbs_design_accepts_strides_and_rejects_invalid_inputs():
    x = np.linspace(0.0, 1.0, 12)[::2]
    flat, width = bbs_raw_design(x, (0.0, 1.0), 4, 3)
    assert len(flat) == len(x) * width

    with pytest.raises(ValueError, match="n_knots"):
        bbs_raw_design(x, (0.0, 1.0), 0, 3)
    with pytest.raises(ValueError, match="increasing"):
        bbs_raw_design(x, (1.0, 0.0), 4, 3)
    with pytest.raises(ValueError, match="NaN or infinity"):
        bbs_raw_design(np.asarray([0.0, np.nan]), (0.0, 1.0), 4, 3)


@pytest.mark.parametrize(
    ("y", "etas", "w", "message"),
    [
        ([1.0, 2.0], [[0.0], [0.0]], [1.0], "response values"),
        ([1.0], [[0.0]], [1.0], "eta arrays"),
        ([1.0, 2.0], [[0.0], [0.0]], [1.0, 1.0], "eta 0"),
        ([1.0], [[0.0], [0.0]], [-1.0], "non-negative"),
    ],
)
def test_lss_family_diagnostics_validates_shapes_and_values(y, etas, w, message):
    with pytest.raises(ValueError, match=message):
        lss_family_diagnostics(
            "gaussianlss",
            2,
            np.asarray(y, dtype=float),
            [np.asarray(eta, dtype=float) for eta in etas],
            np.asarray(w, dtype=float),
        )


# ─── learner constructors ──────────────────────────────────────────


@pytest.mark.parametrize(
    "learner",
    ["Bbs", "Bols", "Bmono", "Bmrf", "Brandom", "Bns", "Bkernel",
     "Bconstrained", "Buser"],
)
def test_a_scalar_name_learner_takes_one_non_empty_string(learner):
    """A list or a non-string used to reach the fit path unchecked.

    There it became a dict key (``unhashable type: 'list'``) or a column
    lookup that could not match anything, neither of which names the argument
    the caller wrote.
    """
    cls = getattr(mb, learner)
    extra = {}
    if learner == "Buser":
        extra = {"X": np.eye(2), "K": np.eye(2)}
    elif learner == "Bconstrained":
        extra = {"C": np.eye(3)}
    elif learner == "Bmrf":
        extra = {"K": np.array([[1.0, -1.0], [-1.0, 1.0]])}

    with pytest.raises(TypeError, match="takes one column name"):
        cls(["x", "y"], **extra)
    with pytest.raises(TypeError, match="takes a column name"):
        cls(3, **extra)
    with pytest.raises(ValueError, match="column name must not be empty"):
        cls("  ", **extra)
    assert cls("x", **extra).name == "x"


def test_bspatial_checks_both_of_its_names():
    with pytest.raises(TypeError, match="Bspatial name1 takes one column name"):
        mb.Bspatial(["x"], "z")
    with pytest.raises(TypeError, match="Bspatial name2 takes a column name"):
        mb.Bspatial("x", 2)
    assert mb.Bspatial("x", "z").name == "x,z"


def test_btree_names_must_be_a_usable_split_set():
    with pytest.raises(ValueError, match="at least one variable"):
        mb.Btree(names=[])
    with pytest.raises(ValueError, match="must be distinct"):
        mb.Btree(names=["x", "x"])
    assert mb.Btree(names="x").names == ["x"]


def test_buser_accepts_array_likes_and_checks_conformability():
    learner = mb.Buser("u", X=[[1.0, 2.0], [3.0, 4.0]], K=[[1.0, 0.0], [0.0, 1.0]])
    assert isinstance(learner.X, np.ndarray) and isinstance(learner.K, np.ndarray)
    with pytest.raises(ValueError, match="Buser K must have 2 rows, got 3"):
        mb.Buser("u", X=[[1.0, 2.0], [3.0, 4.0]], K=np.eye(3))
    with pytest.raises(ValueError, match="Buser K must be square"):
        mb.Buser("u", X=np.eye(5), K=np.ones((5, 3)))


def test_buser_rejects_a_non_finite_design_by_position():
    with pytest.raises(ValueError, match=r"Buser X\[1, 0\] is nan"):
        mb.Buser("u", X=[[1.0, 2.0], [np.nan, 4.0]], K=np.eye(2))


def test_bconstrained_accepts_array_likes():
    learner = mb.Bconstrained("x", C=[[1.0, -1.0, 0.0]])
    assert isinstance(learner.C, np.ndarray) and learner.C.shape == (1, 3)


@pytest.mark.parametrize("knots", [(1.0, 1.0), (5.0, 1.0), (np.nan, 2.0), (1.0,)])
def test_bbs_boundary_knots_must_bound_an_interval(knots):
    with pytest.raises(ValueError, match="boundary_knots"):
        mb.Bbs("x", boundary_knots=knots)


def test_bbs_boundary_knots_survive_as_plain_floats():
    assert mb.Bbs("x", boundary_knots=(0, 10)).boundary_knots == (0.0, 10.0)
    assert mb.Bbs("x").boundary_knots is None


# ─── artifacts ─────────────────────────────────────────────────────


def test_an_artifact_with_an_invalid_family_parameter_fails_on_load(tmp_path):
    """The check runs before any numerical routine, not after a bad fit.

    A family in an artifact is rebuilt through its constructor, so a payload
    edited to hold an impossible parameter is refused at load rather than
    producing a model that reports a negative risk.
    """
    rng = np.random.default_rng(3)
    n = 60
    data = {"x": rng.normal(size=n)}
    data["y"] = 2.0 * data["x"] + rng.normal(size=n)
    model = mb.glmboost("y ~ x", data=data, family=mb.Huber(delta=1.5),
                        control=mb.boost_control(20, 0.1))
    path = tmp_path / "model.json"
    model.save(path)

    payload = json.loads(path.read_text())
    found = _set_family_param(payload, "delta", -1.0)
    assert found, "the fixture must contain the family's delta"
    # Restamp the digest, as a writer of this payload would, so the check
    # under test is the parameter check rather than the checksum.
    from mboostrs.serialize import _content_digest

    payload["content_sha256"] = _content_digest(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="delta must be positive"):
        mb.load_model(path)


def _set_family_param(node, key, value):
    """Rewrite ``key`` inside the artifact's encoded family payload.

    Constructor arguments are stored as an encoded mapping
    (``{"__type__": "dict", "items": {...}}``), not as a bare object.
    """
    if isinstance(node, dict):
        if node.get("__type__") == "family":
            items = node.get("params", {}).get("items", {})
            if key in items:
                items[key] = value
                return True
        return any(_set_family_param(v, key, value) for v in node.values())
    if isinstance(node, list):
        return any(_set_family_param(v, key, value) for v in node)
    return False
