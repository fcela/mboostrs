"""Rerun 4 friction items: string factors in the bootstrap, the GamRefit
coefficient table, the gam_refit type message, and the agent-guide topic sizes."""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.agent_docs import TOPICS, agent_guide
from mboostrs.selection import gam_refit

def _factor_frame(levels, n=300, seed=1):
    rng = np.random.default_rng(seed)
    eth = rng.choice(levels, n).tolist()
    x = rng.uniform(size=n)
    shift = {level: 0.5 * i for i, level in enumerate(levels)}
    y = x + np.array([shift[e] for e in eth]) + rng.normal(scale=0.3, size=n)
    return pl.DataFrame({"y": y, "x": x, "ethnicity": eth})


# --- 1. confint(method="bootstrap") with a string factor -------------------


@pytest.mark.parametrize("levels", [["afam", "cauc"], ["a", "b", "c", "d"]])
def test_bootstrap_confint_keeps_string_levels_on_the_default_grid(levels):
    df = _factor_frame(levels)
    model = mb.gamboost("y ~ bbs(x, df=3) + bols(ethnicity)", data=df,
                        control=mb.boost_control(mstop=50))
    name = next(n for n in model.base_learner_names() if "ethnicity" in n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.confint(parm=name, B=10, B_mstop=0, seed=1)
    assert result["n_failed"] == 0
    assert result["n_used"] == 10
    lower = np.asarray(result["lower"][name])
    upper = np.asarray(result["upper"][name])
    assert lower.shape == (df.height,)
    assert np.all(lower <= upper)


@pytest.mark.parametrize("levels", [["afam", "cauc"], ["a", "b", "c", "d"]])
def test_bootstrap_confint_accepts_the_raw_frame_as_newdata(levels):
    df = _factor_frame(levels)
    model = mb.gamboost("y ~ bbs(x, df=3) + bols(ethnicity)", data=df,
                        control=mb.boost_control(mstop=50))
    name = next(n for n in model.base_learner_names() if "ethnicity" in n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = model.confint(parm=name, B=6, B_mstop=0, seed=2)
        explicit = model.confint(parm=name, B=6, B_mstop=0, seed=2, newdata=df)
    np.testing.assert_allclose(default["lower"][name], explicit["lower"][name])
    np.testing.assert_allclose(default["upper"][name], explicit["upper"][name])


def test_bootstrap_confint_with_missing_values_and_omit_keeps_row_count():
    df = _factor_frame(["afam", "cauc", "x"])
    holes = df.with_columns(
        pl.when(pl.int_range(pl.len()) % 17 == 0).then(None).otherwise(pl.col("x")).alias("x"))
    model = mb.gamboost("y ~ bbs(x, df=3) + bols(ethnicity)", data=holes,
                        control=mb.boost_control(mstop=50), na_action="omit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.confint(B=5, B_mstop=0, seed=1)
    assert result["n_failed"] == 0
    assert len(result["lower"]["x"]) == len(model._fitted)


# --- 2. GamRefit.coef / term_coefficients ----------------------------------


@pytest.fixture(scope="module")
def refit():
    rng = np.random.default_rng(3)
    n = 300
    g = rng.choice(["afam", "cauc", "hisp"], n).tolist()
    x = rng.uniform(size=n)
    z = rng.normal(size=n)
    shift = {"afam": 0.0, "cauc": 0.5, "hisp": -0.4}
    y = (np.sin(3 * x) + np.array([shift[v] for v in g]) + 0.6 * z
         + rng.normal(scale=0.3, size=n))
    df = pl.DataFrame({"y": y, "x": x, "z": z, "g": g})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x, df=4) + bols(g) + bols(z)", data=df,
                            control=mb.boost_control(mstop=200))
        fit = gam_refit(model, which="all")
    return fit, df


def test_coef_is_a_frame_in_vcov_order(refit):
    fit, _ = refit
    table = fit.coef()
    assert list(table.columns) == ["term", "coefficient", "estimate", "se", "lower", "upper"]
    assert table.height == fit.vcov().shape[0]
    assert table["term"][0] == "(Intercept)"
    assert table["coefficient"][-1] == "bols(z)[1]"
    np.testing.assert_allclose(table["estimate"].to_numpy(), np.asarray(fit._coef))
    np.testing.assert_allclose(table["se"].to_numpy(), np.sqrt(np.diag(fit.vcov())))


def test_coef_interval_is_symmetric_and_widens_with_level(refit):
    fit, _ = refit
    wide = fit.coef(level=0.95)
    narrow = fit.coef(level=0.8)
    est = wide["estimate"].to_numpy()
    np.testing.assert_allclose(wide["upper"].to_numpy() - est, est - wide["lower"].to_numpy())
    assert np.all((wide["upper"] - wide["lower"]).to_numpy()
                  > (narrow["upper"] - narrow["lower"]).to_numpy())
    with pytest.raises(ValueError, match="level"):
        fit.coef(level=1.5)


def test_factor_block_has_one_coefficient_per_level_after_the_first(refit):
    fit, df = refit
    rows = fit.term_coefficients("bols(g)")
    assert rows["coefficient"].to_list() == ["bols(g)[1]", "bols(g)[2]"]
    # The level effects are read from effects(), one value per row; the
    # coefficients are weights on the constrained basis, not level contrasts.
    effect = np.asarray(fit.effects(df, level=0.9)["bols(g)"][0])
    g = df["g"].to_numpy()
    assert set(np.round(effect[g == "afam"], 10)) != set(np.round(effect[g == "cauc"], 10))


def test_basis_coefficients_are_numbered_and_intercept_is_available(refit):
    fit, _ = refit
    spline = fit.term_coefficients("bbs(x, df=4)")
    assert spline["coefficient"][0] == "bbs(x, df=4)[1]"
    assert spline.height == 23
    intercept = fit.term_coefficients("(Intercept)")
    assert intercept.height == 1


def test_unknown_term_names_the_terms_of_the_refit(refit):
    fit, _ = refit
    with pytest.raises(ValueError, match="bols\\(g\\)"):
        fit.term_coefficients("bols(nope)")


def test_vcov_docstring_states_the_index_order():
    doc = mb.selection.GamRefit.vcov.__doc__
    assert "Index 0 is the intercept" in doc
    assert "coef" in doc


# --- 3. gam_refit on a formula string --------------------------------------


def test_gam_refit_on_a_formula_names_the_fix():
    with pytest.raises(TypeError) as info:
        gam_refit("y ~ x")
    message = str(info.value)
    assert "gam_refit works on gamboost models" in message
    assert "mb.gamboost(formula, data)" in message
    assert "then pass it to gam_refit" in message


# --- 4. agent-guide topic sizes --------------------------------------------


def test_every_topic_is_under_fifteen_thousand_characters():
    for topic in TOPICS:
        assert len(agent_guide(topic)) < 15_000, topic


def test_causal_overview_is_short_and_lists_every_causal_subtopic():
    # The limit is the guide's 9,000-character file limit: the overview and
    # the causal sub-topics carry the EconML/DoubleML route (issue #451).
    overview = agent_guide("causal")
    assert len(overview) < 9_000
    subtopics = [name for name in TOPICS if name.startswith("causal-")]
    assert len(subtopics) >= 4
    for name in subtopics:
        assert f"`{name}`" in overview, name
        assert len(agent_guide(name)) < 9_000, name


def test_skill_points_at_every_causal_subtopic():
    skill = agent_guide()
    for name in TOPICS:
        if name.startswith("causal-"):
            assert f'"{name}"' in skill, name
