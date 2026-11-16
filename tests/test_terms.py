"""Tests for choosing base learners from the data (`mboostrs.terms`)."""
import numpy as np
import pytest

import mboostrs as mboost
from mboostrs.terms import classify_columns, expand_pairs, screen_interactions, suggest_terms


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 400
    return {
        "height": rng.normal(170, 10, n),                 # continuous
        "site": np.array([10.0, 27.0, 53.0, 98.0])[rng.integers(0, 4, n)],
        "treated": rng.integers(0, 2, n).astype(float),   # binary
        "y": rng.normal(0, 1, n),
    }


def test_each_column_kind_gets_its_learner(frame):
    kinds = {c.name: c for c in classify_columns(frame, response="y")}
    assert kinds["treated"].kind == "binary"
    assert kinds["treated"].term == "bols(treated)"
    assert kinds["site"].kind == "categorical"
    assert kinds["site"].term.startswith("brandom(site")
    assert kinds["height"].kind == "continuous"
    assert kinds["height"].term.startswith("bbs(height")


def test_a_continuous_column_is_never_categorical(frame):
    """Few distinct values do not make a measurement into levels."""
    coarse = dict(frame)
    coarse["height"] = np.round(frame["height"] / 40) * 40.5   # 3-4 non-integer levels
    kinds = {c.name: c for c in classify_columns(coarse, response="y")}
    assert kinds["height"].kind == "continuous"


def test_numeric_three_level_codes_are_group_effects_not_a_slope():
    """Codes that do not read as a count ({10, 27, 53}, say) are labels: a
    slope across them would order them by an arbitrary number."""
    data = {"y": np.arange(12.0),
            "stage": np.tile(np.array([10.0, 27.0, 53.0]), 4)}
    stage = classify_columns(data, response="y")[0]
    assert stage.term == "brandom(stage, df=4.0)"
    assert any(term == "bols(stage)" and "one linear slope" in assumption
               for term, assumption in stage.alternatives)


def test_the_response_is_left_out(frame):
    assert "y" not in suggest_terms(frame, response="y")


def test_smooth_option_changes_only_continuous_columns(frame):
    trees = {c.name: c.term for c in classify_columns(frame, response="y", smooth="btree")}
    assert trees["height"] == "btree(height)"
    assert trees["treated"] == "bols(treated)"
    menu = {c.name: c.term for c in classify_columns(frame, response="y", smooth="menu")}
    assert all(part in menu["height"] for part in ("bols(", "bbs(", "btree("))


def test_smooth_option_is_validated(frame):
    with pytest.raises(ValueError, match="smooth must be"):
        classify_columns(frame, response="y", smooth="wavelet")


def test_suggested_terms_fit(frame):
    frame = dict(frame)
    frame["y"] = 0.05 * frame["height"] + 0.4 * frame["treated"] + frame["y"]
    model = mboost.gamboost(f"y ~ {suggest_terms(frame, response='y')}", data=frame,
                            family=mboost.Gaussian(),
                            control=mboost.boost_control(200, 0.1))
    assert np.asarray(model.predict()).shape[0] == len(frame["y"])


def test_pair_learner_follows_the_margins(frame):
    terms = expand_pairs(frame, response="y",
                         pairs=[("height", "site"), ("height", "treated"),
                                ("treated", "site")])
    assert "%X%" in terms[0]                       # two continuous-ish margins
    assert terms[1] == "bbs(height, by=treated, df=4.0)"   # one binary margin
    assert terms[2].startswith("bbs(site, by=treated")     # binary margin second


def test_tree_pairs_need_no_margin_logic(frame):
    terms = expand_pairs(frame, response="y", kind="tree", maxdepth=3,
                         pairs=[("height", "treated")])
    assert terms == ["btree(height, treated, maxdepth=3)"]


def test_expanding_every_pair_is_quadratic(frame):
    assert len(expand_pairs(frame, response="y")) == 3      # 3 predictors -> 3 pairs


def test_screen_ranks_a_real_interaction_first():
    rng = np.random.default_rng(1)
    n = 1500
    data = {c: rng.normal(0, 1, n) for c in ("a", "b", "c", "d")}
    interaction = data["a"] * data["b"]
    ranked = screen_interactions(data, residual=interaction + rng.normal(0, 0.3, n))
    assert {ranked[0].left, ranked[0].right} == {"a", "b"}
    assert ranked[0].removed > 0


def test_screen_scores_pure_noise_below_zero_on_average():
    """On pure noise a pair's score is negative in expectation: the held-out
    half pays the variance of the fit-half cell means, about
    (cells / fit rows) per observation, here 16 / 750 = 0.021. The score is
    not zero-centred, so a single pair can exceed zero by chance. Measured over
    2000 noise replicates of this design (n=1500, three columns, bins=4):
    per-pair mean -0.023, sd 0.013, about 2% of pair scores above zero, and the
    largest per-replicate maximum 0.019. The assertions check that mean and
    that bound, not a strict zero for each maximum."""
    scores, maxima = [], []
    for replicate in range(60):
        rng = np.random.default_rng(100 + replicate)
        n = 1500
        data = {c: rng.normal(0, 1, n) for c in ("a", "b", "c")}
        ranked = screen_interactions(data, residual=rng.normal(0, 1, n))
        values = [s.removed for s in ranked]
        scores.extend(values)
        maxima.append(max(values))
    assert np.mean(scores) < 0.0
    assert max(maxima) < 0.03


def test_screen_checks_its_input_length():
    data = {"a": np.zeros(10), "b": np.zeros(10)}
    with pytest.raises(ValueError, match="residual has 5 rows"):
        screen_interactions(data, residual=np.zeros(5))


def test_screened_pairs_can_be_expanded_directly(frame):
    residual = frame["height"] * frame["treated"]
    ranked = screen_interactions(frame, residual=residual, response="y")
    terms = expand_pairs(frame, response="y",
                         pairs=[(s.left, s.right) for s in ranked[:2]])
    assert len(terms) == 2


def test_tensor_df_defaults_above_the_null_space_floor():
    """A tensor asked for its null-space dimension collapses to bilinear."""
    from mboostrs.terms import TENSOR_MARGINAL_DF, TENSOR_MINIMUM_DF
    assert TENSOR_MARGINAL_DF ** 2 > TENSOR_MINIMUM_DF


def test_a_degenerate_tensor_df_warns(frame):
    with pytest.warns(UserWarning, match="null space"):
        expand_pairs(frame, response="y", pairs=[("height", "site")],
                     marginal_df=2.0)


def test_a_tensor_above_the_floor_does_not_warn(frame):
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        expand_pairs(frame, response="y", pairs=[("height", "site")])


def test_the_floor_only_applies_to_two_continuous_margins(frame):
    """A margin that is an indicator gives a varying coefficient, not a tensor."""
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        expand_pairs(frame, response="y", pairs=[("height", "treated")],
                     marginal_df=2.0)
