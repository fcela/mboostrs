"""Backtick escaping of awkward column names in formula text (issue #101, item 4).

A name that is not a plain identifier is written in backticks, as in R:
``bbs(`house age`) + bols(`price/sqft`)``. Inside backticks a backslash escapes
a backtick or a backslash, which is R's rule. Every place that writes a formula
quotes the names that need it, and every parser reads the quoted form back to
the column with that exact name.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._names import mask_names, needs_quoting, quote_name, unquote_name
from mboostrs.formula import (
    expand_formula,
    parse_formula,
    parse_gamboost_formula,
    term_variables,
)

# (category, column name)
AWKWARD = [
    ("space", "house age"),
    ("space", "  padded  "),
    ("slash", "price/sqft"),
    ("plus", "a+b"),
    ("minus", "net-gain"),
    ("star", "a*b"),
    ("colon", "a:b"),
    ("caret", "area^2"),
    ("percent", "pct%"),
    ("tilde", "a~b"),
    ("parens", "f(x)"),
    ("paren-open", "(adj"),
    ("comma", "a,b"),
    ("single-quote", "it's"),
    ("double-quote", 'say "hi"'),
    ("backtick", "a`b"),
    ("backslash", "a\\b"),
    ("equals", "a=1"),
    ("digit", "1st"),
    ("digits", "2020"),
    ("reserved", "bbs"),
    ("reserved", "bols"),
    ("reserved", "brandom"),
    ("reserved", "I"),
    ("reserved", "log"),
    ("reserved", "scale"),
    ("reserved", "if"),
    ("reserved", "TRUE"),
    ("call-like", "log(x)"),
    ("unicode", "prix €"),
]
NAMES = [name for _, name in AWKWARD]
IDS = [f"{category}-{i}" for i, (category, _) in enumerate(AWKWARD)]


def _data(name, n=240, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.0 + np.sin(x) + 0.5 * z + 0.3 * rng.normal(size=n)
    return {"y": y, name: x, "z": z}


def test_there_are_enough_awkward_names():
    assert len(AWKWARD) >= 15
    assert all(needs_quoting(name) for name in NAMES)


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_quote_round_trip(name):
    token = quote_name(name)
    assert token.startswith("`")
    assert unquote_name(token) == name
    # The masked token contains no operator, bracket or comma.
    assert set(mask_names(token)) == {"x"}


def test_plain_names_stay_bare():
    for name in ["x", "x1", "house.age", "age_2", "Temp"]:
        assert quote_name(name) == name


def test_backtick_escape_follows_r():
    # R writes a backtick inside a name as backslash-backtick.
    assert quote_name("a`b") == "`a\\`b`"
    assert quote_name("a\\b") == "`a\\\\b`"
    assert unquote_name("`a\\`b`") == "a`b"


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_parsers_read_the_column(name):
    q = quote_name(name)
    response, terms = parse_formula(f"y ~ {q} + z")
    assert (response, terms) == ("y", [name, "z"])
    response, learners = parse_gamboost_formula(
        f"y ~ bbs({q}, df=4) + bols(z, by={q}) + brandom({q}) + bmono({q})")
    assert [term.name for term in learners] == [name, "z", name, name]
    assert learners[1].by == name
    assert term_variables(f"bbs({q}, df=4)") == (name,)


def test_response_and_survival_names():
    q = quote_name("my y")
    assert parse_gamboost_formula(f"{q} ~ x")[0] == "my y"
    assert parse_gamboost_formula("Surv(`t 1`, `st/2`) ~ x")[0] == ("t 1", "st/2")
    assert parse_gamboost_formula("cbind(`a b`, c) ~ x")[0] == ("a b", "c")


def test_dot_expansion_quotes_generated_terms():
    data = {"y": [1.0, 2.0, 3.0], "house age": [1.0, 2.0, 4.0], "a+b": [3.0, 1.0, 2.0],
            "z": [0.0, 1.0, 5.0]}
    assert expand_formula("y ~ .", data) == "y ~ `house age` + `a+b` + z"
    assert expand_formula("y ~ . - `a+b`", data) == "y ~ `house age` + z"
    assert expand_formula("y ~ btree(.)", data) == "y ~ btree(`house age`, `a+b`, z)"


# Names written without backticks that need them. Each raises, and the message
# shows the backticked spelling of the name.
UNESCAPED = [
    ("y ~ bbs(house age)", "bbs(`house age`)"),
    ("y ~ bols(price/sqft)", "bols(`price/sqft`)"),
    ("y ~ house age", "`house age`"),
    ("y ~ price/sqft + z", "`price/sqft`"),
    ("y ~ bbs(area^2, df=4)", "bbs(`area^2`, df=4)"),
    ("y ~ bols(pct%)", "bols(`pct%`)"),
    ("y ~ bbs(net-gain)", "bbs(`net-gain`)"),
]


@pytest.mark.parametrize(("text", "shown"), UNESCAPED)
def test_unescaped_name_error_shows_backticks(text, shown):
    with pytest.raises(ValueError) as caught:
        parse_gamboost_formula(text)
    assert shown in str(caught.value)


def test_unescaped_message_example():
    with pytest.raises(ValueError, match=r"bbs\(`house age`\)"):
        parse_gamboost_formula("y ~ bbs(house age)")
    with pytest.raises(ValueError, match=r"`price/sqft`"):
        parse_gamboost_formula("y ~ price/sqft")


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_gamboost_end_to_end(name):
    data = _data(name)
    q = quote_name(name)
    formula = f"y ~ bbs({q}, df=4) + bols(z)"
    model = mb.gamboost(formula, data=data)
    assert model.formula == formula
    coef = model.coef()
    assert name in coef and "z" in coef
    newdata = _data(name, n=30, seed=9)
    pred = np.asarray(model.predict(newdata))
    assert pred.shape == (30,) and np.all(np.isfinite(pred))
    effects = model.partial_effects(which=name, newdata=newdata)
    assert effects is not None
    assert name in str(model.summary(as_dict=True)) or name in repr(model.summary())

    # The formula string re-parses to the same model.
    again = mb.gamboost(model.formula, data=data)
    np.testing.assert_allclose(np.asarray(again.predict(newdata)), pred)


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_glmboost_end_to_end(name):
    data = _data(name)
    q = quote_name(name)
    model = mb.glmboost(f"y ~ {q} + z", data=data)
    coef = model.coef()
    assert name in coef and "z" in coef
    newdata = _data(name, n=30, seed=9)
    pred = np.asarray(model.predict(newdata))
    assert np.all(np.isfinite(pred))
    again = mb.glmboost(model.formula, data=data)
    np.testing.assert_allclose(np.asarray(again.predict(newdata)), pred)
    # Dot expansion names the column in backticks.
    dotted = mb.glmboost("y ~ .", data=data)
    assert name in dotted.coef()
    np.testing.assert_allclose(np.asarray(dotted.predict(newdata)), pred, rtol=1e-8)


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_gamboost_lss_end_to_end(name):
    data = _data(name)
    q = quote_name(name)
    formula = f"y ~ bbs({q}, df=4) + bols(z)"
    model = mb.gamboostLSS(formula, data=data, families=mb.GaussianLSS(), mstop=30)
    newdata = _data(name, n=30, seed=9)
    mu = np.asarray(model.predict(newdata, parameter="mu"))
    assert mu.shape == (30,) and np.all(np.isfinite(mu))
    linear = mb.glmboostLSS(f"y ~ {q} + z", data=data, families=mb.GaussianLSS(), mstop=30)
    assert np.all(np.isfinite(np.asarray(linear.predict(newdata, parameter="mu"))))


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_gamlss_reml_end_to_end(name):
    data = _data(name)
    q = quote_name(name)
    fit = mb.gamlss(
        {"mu": f"y ~ bbs({q}, df=4) + bols(z)", "sigma": f"y ~ bols({q})"},
        data=data, family=mb.GaussianLSS())
    newdata = _data(name, n=30, seed=9)
    mu = np.asarray(fit.predict(newdata, parameter="mu"))
    assert mu.shape == (30,) and np.all(np.isfinite(mu))


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_autospec_formula_reparses_and_refits(name):
    data = _data(name)
    rec = mb.autospec(data, "y")
    formula = rec.formula
    assert quote_name(name) in formula
    _, learners = parse_gamboost_formula(formula)
    assert name in [learner.name for learner in learners]
    model = rec.fit(data)
    refit = mb.gamboost(formula, data=data, family=rec.family, control=rec.control)
    newdata = _data(name, n=30, seed=9)
    np.testing.assert_allclose(
        np.asarray(refit.predict(newdata)), np.asarray(model.predict(newdata)), rtol=1e-6)


def _triples():
    pool = [name for name in NAMES if name not in ("log", "bbs", "if", "TRUE", "I", "scale")]
    return [tuple(pool[i:i + 3]) for i in range(0, len(pool) - 2, 3)]


def _triple_data(names, n=300, seed=5):
    rng = np.random.default_rng(seed)
    columns = {name: rng.normal(size=n) for name in names}
    first, second, third = (columns[name] for name in names)
    y = np.sin(first) + 0.5 * second * third + 0.3 * rng.normal(size=n)
    return {"y": y, **columns}


@pytest.mark.parametrize("names", _triples(), ids=lambda names: "|".join(names))
def test_suggest_specifications_and_discover_formulas_reparse_and_refit(names):
    data = _triple_data(names)
    base = mb.autospec(data, "y").formula
    report = mb.term_report(base, data)
    suggestions = mb.suggest_specifications(report)
    assert suggestions
    for suggestion in suggestions:
        specification = suggestion.specification
        if "%X%" not in specification:
            _, learners = parse_gamboost_formula("y ~ " + specification)
            used = {col for learner in learners
                    for col in (getattr(learner, "names", None) or [learner.name])}
            assert used <= set(names)
        fitted = mb.gamboost("y ~ " + specification, data=data)
        assert np.all(np.isfinite(np.asarray(fitted.predict(data))))

    found = mb.discover(data, "y", seed=0)
    discovered = found.formula if "~" in mask_names(found.formula) else "y ~ " + found.formula
    if "%X%" not in discovered:
        _, learners = parse_gamboost_formula(discovered)
        used = {col for learner in learners
                for col in (getattr(learner, "names", None) or [learner.name])}
        assert used <= set(names)
    refit = mb.gamboost(discovered, data=data)
    assert np.all(np.isfinite(np.asarray(refit.predict(data))))
