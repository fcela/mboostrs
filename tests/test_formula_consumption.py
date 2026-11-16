"""A formula term is understood whole, or it is rejected.

Each learner parser looped a keyword regex over the text inside its
parentheses. A regex reports what it matched and says nothing about what it
did not, so three things passed silently:

- **Extra positional variables.** ``bbs(x, y, z)`` matched with ``x`` as the
  variable and ``"y, z"`` as the keyword string, where no keyword matched. The
  learner was ``bbs(x)``; ``y`` and ``z`` vanished. A caller writing what they
  believed was a multi-column term got a single-column fit with nothing to
  indicate it.
- **Duplicate keywords.** ``bbs(x, df=3, df=4)`` took the last one quietly,
  and ``center=True, center=False`` took ``False``.
- **Malformed booleans.** ``center=maybe`` became ``center=False``, because
  the test was ``value in ("True", "TRUE", "true")`` and everything else fell
  through. Centering changes the fit, so a typo changed the model.

The tests below are grouped by what the parser must prove: that it consumed
the whole term, that it read each value as the type the argument takes, and
that a term it does accept still means what it meant.
"""

import numpy as np
import pytest
from mboostrs.formula import parse_gamboost_formula

import mboostrs as mb

#: One learner per shape the parsers take: a single variable, two variables,
#: a variable list, and one whose arguments resolve from `env`.
SINGLE_VARIABLE_LEARNERS = [
    "bbs", "bols", "bmono", "brandom", "bns", "bkernel",
]


def _parse(formula, **kwargs):
    return parse_gamboost_formula(formula, **kwargs)


# ─── the whole term is consumed ────────────────────────────────────


@pytest.mark.parametrize("learner", SINGLE_VARIABLE_LEARNERS)
def test_extra_positional_variables_are_rejected(learner):
    """The worst of the three: the extra columns simply disappeared.

    `bols` had its own check already, with a message naming R's `bols(a, b)`;
    the rest fall to the consumption check, whose message quotes the text it
    could not read. Either is acceptable, so long as the column is not
    silently dropped.
    """
    with pytest.raises(ValueError) as excinfo:
        _parse(f"y ~ {learner}(x, z)")
    message = str(excinfo.value)
    assert "'z'" in message, "the refusal must name the column it could not use"


def test_a_dropped_variable_would_have_changed_the_fit():
    """What the silent drop cost, made concrete.

    `bbs(x, z)` used to fit `bbs(x)` alone. The two are different models, so
    the caller was not getting an approximation of what they asked for.
    """
    rng = np.random.default_rng(4)
    n = 120
    data = {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    data["y"] = 2 * data["x"] - data["z"] + rng.normal(size=n)

    with pytest.raises(ValueError, match="did not understand"):
        mb.gamboost("y ~ bbs(x, z)", data=data,
                    control=mb.boost_control(20, 0.1))

    one_column = mb.gamboost("y ~ bbs(x, df=3)", data=data,
                             control=mb.boost_control(20, 0.1))
    both = mb.gamboost("y ~ bbs(x, df=3) + bbs(z, df=3)", data=data,
                       control=mb.boost_control(20, 0.1))
    assert not np.allclose(one_column.fitted(), both.fitted(), atol=1e-3)


def test_trailing_text_inside_the_parentheses_is_rejected():
    with pytest.raises(ValueError, match="did not understand"):
        _parse("y ~ bbs(x, df=3, leftover)")


def test_the_refusal_points_at_the_alternatives():
    with pytest.raises(ValueError) as excinfo:
        _parse("y ~ bbs(x, z)")
    message = str(excinfo.value)
    assert "one term per column" in message
    assert "buser()" in message


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ bbs(x, df=3) extra_text",
        "y ~ bbs(x df=3)",
        "y ~ bbs(x, df=3",
        "y ~ bbs(x)) + bols(z)",
        "y ~ bbs(bbs(x))",
        'y ~ bbs("x")',
        "y ~ bols(x) * bols(z)",
    ],
)
def test_malformed_term_syntax_is_rejected(formula):
    with pytest.raises(ValueError):
        _parse(formula)


# ─── duplicates ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ bbs(x, df=3, df=4)",
        "y ~ bbs(x, center=True, center=False)",
        "y ~ bbs(x, df=3, deriv=1, df=4)",
    ],
)
def test_a_repeated_keyword_is_rejected(formula):
    with pytest.raises(ValueError) as excinfo:
        _parse(formula)
    assert "more than once" in str(excinfo.value)
    assert "does not pick one silently" in str(excinfo.value)


# ─── values are read as the type the argument takes ────────────────


@pytest.mark.parametrize("value", ["maybe", "1", "yes", "Y", "none"])
def test_a_boolean_argument_rejects_anything_that_is_not_one(value):
    """`center=maybe` became `center=False`, silently changing the fit."""
    with pytest.raises(ValueError) as excinfo:
        _parse(f"y ~ bbs(x, center={value})")
    assert "is not a boolean" in str(excinfo.value)


def test_a_boolean_argument_with_no_value_is_rejected():
    """`center=` matches no keyword at all, so the consumption check has it."""
    with pytest.raises(ValueError, match="did not understand"):
        _parse("y ~ bbs(x, center=)")


@pytest.mark.parametrize("value", ["True", "TRUE", "true", "T"])
def test_true_is_accepted_in_the_spellings_users_write(value):
    _response, learners = _parse(f"y ~ bbs(x, center={value})")
    assert learners[0].center is True


@pytest.mark.parametrize("value", ["False", "FALSE", "false", "F"])
def test_false_is_accepted_in_the_spellings_users_write(value):
    _response, learners = _parse(f"y ~ bbs(x, center={value})")
    assert learners[0].center is False


def test_a_numeric_argument_names_the_learner_when_it_will_not_convert():
    """The bare `could not convert string to float` named neither."""
    with pytest.raises(ValueError) as excinfo:
        _parse("y ~ bbs(x, df=notanumber)")
    message = str(excinfo.value)
    assert "bbs(..., df=notanumber)" in message
    assert "is not a number" in message


def test_an_integer_argument_says_it_wants_an_integer():
    with pytest.raises(ValueError, match="is not an integer"):
        _parse("y ~ bbs(x, knots=c(1,2,3))")


def test_an_unknown_keyword_is_still_rejected():
    with pytest.raises(ValueError, match="unknown bbs parameter"):
        _parse("y ~ bbs(x, bogus=1)")


# ─── terms that are valid still parse ──────────────────────────────


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ bbs(x, df=3)", ["x"]),
        ("y ~ bbs(x, df=3, center=True)", ["x"]),
        ("y ~ bols(x, intercept=False)", ["x"]),
        ("y ~ bbs(x, df=3) + bols(z)", ["x", "z"]),
        ("y ~ bbs(x, boundary.knots=c(0,10))", ["x"]),
        ("y ~ bbs(x, df=3, degree=2, differences=1)", ["x"]),
    ],
)
def test_a_well_formed_term_still_parses(formula, expected):
    _response, learners = _parse(formula)
    assert [learner.name for learner in learners] == expected


def test_a_multi_variable_learner_still_takes_its_variables():
    """`bspatial` and `btree` genuinely take more than one column.

    The consumption check runs on the *keyword* text, after the term regex has
    taken the variables each learner declares, so it does not catch these.
    """
    _response, learners = _parse("y ~ bspatial(x, z, df=5)")
    assert learners[0].name == "x,z"
    _response, learners = _parse("y ~ btree(x, z)")
    assert learners[0].names == ["x", "z"]


def test_c_of_several_values_survives_the_consumption_check():
    """Commas inside `c(...)` must not read as argument separators."""
    _response, learners = _parse("y ~ bbs(x, boundary.knots=c(0, 10), df=4)")
    assert learners[0].boundary_knots == (0.0, 10.0)
    assert learners[0].df == 4


# ─── the decisions P0.6 asked for ──────────────────────────────────


@pytest.mark.parametrize("formula", ["y ~ x - 1", "y ~ x + 0", "y ~ 0 + x"])
def test_glmboost_fits_without_an_intercept(formula):
    """`- 1` drops the intercept learner, as in R (#101); gamboost has no
    global intercept candidate and fits the formula without the term."""
    rng = np.random.default_rng(5)
    n = 80
    data = {"x": rng.normal(size=n)}
    data["y"] = 2 * data["x"] + rng.normal(size=n)
    with pytest.warns(UserWarning, match="does not contain intercept"):
        fit = mb.glmboost(formula, data=data, control=mb.boost_control(10, 0.1))
    assert list(fit.coef()) == ["x"]
    gam = mb.gamboost(formula.replace("x", "bols(x)"), data=data,
                      control=mb.boost_control(10, 0.1))
    assert gam._global_intercept is False


def test_a_joint_multi_column_bols_is_refused_by_name():
    """Decided: unsupported. R's `bols(a, b)` is one learner over both
    columns; mboostrs fits one learner per column, and says so rather than
    dropping the second."""
    with pytest.raises(ValueError) as excinfo:
        _parse("y ~ bols(x, z)")
    message = str(excinfo.value)
    assert "names a second column" in message
    assert "bols(a, b)" in message, "the refusal should name the R form"


def test_a_bare_categorical_in_gamboost_fits_the_level_design():
    """R falls back to `bols` on a factor, which is least squares on its
    dummy design. mboostrs fits the same column space (one effect per
    level), never the integer codes as a measurement."""
    rng = np.random.default_rng(6)
    n = 90
    data = {
        "g": np.array(["a", "b", "c"] * 30),
        "y": rng.normal(size=n),
    }
    bare = mb.gamboost("y ~ g", data=data, control=mb.boost_control(10, 0.1))
    levels = mb.gamboost("y ~ brandom(g, lambda=0)", data=data,
                         control=mb.boost_control(10, 0.1))
    np.testing.assert_allclose(bare.fitted(), levels.fitted(), atol=1e-12)


def test_glmboostLSS_rejects_the_env_it_never_read():
    """Decided: refused. It was accepted and ignored.

    `env` resolves symbols inside learner calls; a linear formula has none, so
    there is nothing for it to resolve and accepting it read as support for
    something that did not happen.
    """
    rng = np.random.default_rng(7)
    n = 80
    data = {"x": rng.normal(size=n)}
    data["y"] = rng.normal(1 + data["x"], 1.0)
    with pytest.raises(TypeError) as excinfo:
        mb.glmboostLSS("y ~ x", data=data, families=mb.GaussianLSS(),
                       env={"K": np.eye(2)}, control=mb.boost_control(5, 0.1))
    assert "does not use env=" in str(excinfo.value)
    assert "gamboostLSS" in str(excinfo.value)

    # Without it the fit is unaffected.
    model = mb.glmboostLSS("y ~ x", data=data, families=mb.GaussianLSS(),
                           control=mb.boost_control(5, 0.1))
    assert model.mstop == 5


def test_a_factor_by_still_fits_in_gamboost():
    """Decided: supported on the gamboost path, and exercised here so the
    decision is not recorded as 'unsupported' by omission."""
    rng = np.random.default_rng(8)
    n = 120
    data = {
        "x": rng.normal(size=n),
        "g": np.array(["a", "b"] * 60),
    }
    data["y"] = 2 * data["x"] + rng.normal(size=n)
    model = mb.gamboost("y ~ bbs(x, df=4, by=g)", data=data,
                        control=mb.boost_control(15, 0.1))
    assert model.mstop == 15
    assert np.all(np.isfinite(model.fitted()))
