"""Mechanics and coverage of `split_refit` (issue #117, items 1 and 2).

The unit tests check the split, the frequency count, the aggregation rule and
the contrast arithmetic on small hand-made inputs. The simulation checks the
coverage of the procedure as a whole: selection, split, refit and interval,
repeated on fresh data. Item 3 (selective inference) is not implemented, so
nothing here tests it.
"""

import json

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.selection import _average_contrast
from mboostrs.split_refit import (
    ContrastSummary,
    SplitContrast,
    SplitFit,
    _aggregate_contrasts,
    _aggregate_terms,
    _split_rows,
    meinshausen_buhlmann_p,
)


def _data(n, seed, *, noise=1.0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 * x1 + rng.normal(scale=noise, size=n)
    return {"y": y, "x1": x1, "x2": x2}


def _contrast(variable, *, selected, estimate, half_width=0.5, p=0.5):
    return SplitContrast(variable, -1.0, 1.0, selected, estimate, estimate - half_width,
                         estimate + half_width, p)


def _split(repeat, selected, term_p, contrasts):
    return SplitFit(repeat, 10, 10, tuple(selected), dict(term_p), tuple(contrasts))


# --------------------------------------------------------------- split rows
def test_split_halves_are_disjoint_and_cover_every_row():
    rng = np.random.default_rng(0)
    selection, inference = _split_rows(101, 0.5, rng)
    assert len(selection) == round(0.5 * 101)
    assert set(selection).isdisjoint(inference)
    assert sorted(np.concatenate([selection, inference])) == list(range(101))


def test_split_rows_follow_the_seed():
    first = _split_rows(60, 0.4, np.random.default_rng(3))
    again = _split_rows(60, 0.4, np.random.default_rng(3))
    other = _split_rows(60, 0.4, np.random.default_rng(4))
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert not np.array_equal(first[0], other[0])


def test_inference_half_is_the_row_count_the_refit_sees():
    result = mb.split_refit("y ~ bbs(x1, df=4) + bbs(x2, df=4)", _data(120, 1), repeats=2,
                            fraction=0.4, seed=5)
    for split in result.splits:
        assert split.n_selection == round(0.4 * 120)
        assert split.n_selection + split.n_inference == 120


def test_same_seed_gives_the_same_result():
    data = _data(120, 2)
    kwargs = dict(repeats=2, seed=7, contrasts=[("x1", -1.0, 1.0)])
    first = mb.split_refit("y ~ bbs(x1, df=4) + bbs(x2, df=4)", data, **kwargs)
    again = mb.split_refit("y ~ bbs(x1, df=4) + bbs(x2, df=4)", data, **kwargs)
    assert first.to_dict() == again.to_dict()


# ------------------------------------------------------------ aggregation
def test_selection_frequency_counts_the_repeats_that_chose_each_term():
    splits = [
        _split(0, ["bbs(x1, df=4)", "bbs(x2, df=4)"], {"bbs(x1, df=4)": 0.01,
                                                       "bbs(x2, df=4)": 0.3}, []),
        _split(1, ["bbs(x1, df=4)"], {"bbs(x1, df=4)": 0.02}, []),
        _split(2, [], {}, []),
    ]
    terms = {t.term: t for t in _aggregate_terms(splits, 3, lss=False)}
    assert terms["bbs(x1, df=4)"].selection_frequency == pytest.approx(2 / 3)
    assert terms["bbs(x2, df=4)"].selection_frequency == pytest.approx(1 / 3)


def test_term_p_value_aggregates_with_unselected_repeats_counted_as_one():
    # x1 chosen in repeats 0 and 2 with p 0.01 and 0.2; repeat 1 did not choose it.
    splits = [
        _split(0, ["bbs(x1, df=4)"], {"bbs(x1, df=4)": 0.01}, []),
        _split(1, [], {}, []),
        _split(2, ["bbs(x1, df=4)"], {"bbs(x1, df=4)": 0.2}, []),
    ]
    (term,) = _aggregate_terms(splits, 3, lss=False)
    # Per-split p-values [0.01, 1.0, 0.2]; the median is 0.2 and twice it is 0.4.
    assert term.p_value == pytest.approx(0.4)


def test_meinshausen_buhlmann_rule_is_twice_the_median_at_gamma_half():
    assert meinshausen_buhlmann_p([0.02, 0.04, 0.5]) == pytest.approx(0.08)
    assert meinshausen_buhlmann_p([0.4, 0.6, 0.9]) == 1.0  # capped at 1
    assert meinshausen_buhlmann_p([0.3]) == 0.6


def test_meinshausen_buhlmann_rule_uses_the_gamma_quantile():
    # The gamma-quantile of p / gamma: at gamma = 0.25 the first quartile of 4p.
    values = np.array([0.01, 0.02, 0.03, 0.04])
    expected = np.quantile(values / 0.25, 0.25)
    assert meinshausen_buhlmann_p(values, gamma=0.25) == pytest.approx(expected)
    with pytest.raises(ValueError, match="gamma"):
        meinshausen_buhlmann_p(values, gamma=1.0)


def test_one_repeat_reports_the_split_p_value_unadjusted():
    split = _split(0, ["bbs(x1, df=4)"], {"bbs(x1, df=4)": 0.03},
                   [_contrast("x1", selected=True, estimate=2.0, p=0.03)])
    (term,) = _aggregate_terms([split], 1, lss=False)
    (contrast,) = _aggregate_contrasts([split], [("x1", -1.0, 1.0)], 1)
    assert term.p_value == pytest.approx(0.03)
    assert contrast.p_value == pytest.approx(0.03)


def test_contrast_aggregate_takes_medians_and_the_selection_frequency():
    splits = [
        _split(0, [], {}, [_contrast("x1", selected=False, estimate=0.0, half_width=0.0, p=1.0)]),
        _split(1, [], {}, [_contrast("x1", selected=True, estimate=1.0, half_width=0.2, p=0.1)]),
        _split(2, [], {}, [_contrast("x1", selected=True, estimate=3.0, half_width=0.4, p=0.2)]),
    ]
    (summary,) = _aggregate_contrasts(splits, [("x1", -1.0, 1.0)], 3)
    assert isinstance(summary, ContrastSummary)
    assert summary.selection_frequency == pytest.approx(2 / 3)
    assert summary.estimate == pytest.approx(1.0)   # median of 0, 1, 3
    assert summary.lower == pytest.approx(0.8)      # median of 0, 0.8, 2.6
    assert summary.upper == pytest.approx(1.2)      # median of 0, 1.2, 3.4
    assert summary.p_value == pytest.approx(min(1.0, 2 * 0.2))  # MMB on [1.0, 0.1, 0.2]


# ---------------------------------------------------------------- contrasts
def test_average_contrast_equals_the_mean_prediction_difference():
    data = _data(150, 3)
    model = mb.gamboost("y ~ bbs(x1, df=4) + bbs(x2, df=4)", data,
                        control=mb.boost_control(mstop=1))
    refit = mb.gam_refit(model, terms=["bbs(x1, df=4)", "bbs(x2, df=4)"])
    estimate, lower, upper = _average_contrast(refit, data, "x1", -1.0, 1.0, 0.9)
    high = {**data, "x1": np.ones(150)}
    low = {**data, "x1": -np.ones(150)}
    assert estimate == pytest.approx(float(np.mean(refit.predict(high) - refit.predict(low))),
                                     rel=1e-10)
    assert lower < estimate < upper


def test_contrast_of_a_variable_no_term_reads_is_exactly_zero():
    data = _data(150, 4)
    model = mb.gamboost("y ~ bbs(x1, df=4)", data, control=mb.boost_control(mstop=1))
    refit = mb.gam_refit(model, terms=["bbs(x1, df=4)"])
    assert _average_contrast(refit, data, "x2", -1.0, 1.0, 0.9) == (0.0, 0.0, 0.0)


def test_contrast_refuses_a_variable_read_by_two_terms():
    # A smooth in x1 and a smooth in x1 varying with z: gam_refit keeps both
    # (it merges only a bols into a bbs over the same column).
    data = _data(150, 5)
    data["z"] = np.random.default_rng(6).integers(0, 2, 150).astype(float)
    terms = ["bbs(x1, df=4)", "bbs(x1, df=4, by=z)"]
    model = mb.gamboost("y ~ " + " + ".join(terms), data, control=mb.boost_control(mstop=1))
    refit = mb.gam_refit(model, terms=terms)
    assert len(refit.terms) == 2
    with pytest.raises(NotImplementedError, match="x1"):
        _average_contrast(refit, data, "x1", -1.0, 1.0, 0.9)


# -------------------------------------------------------------- result shape
def test_to_dict_is_json_and_summary_names_each_term_and_contrast():
    result = mb.split_refit("y ~ bbs(x1, df=4) + bbs(x2, df=4)", _data(120, 6), seed=1,
                            contrasts=[("x1", -1.0, 1.0)])
    record = result.to_dict(include_arrays=True)
    assert json.loads(json.dumps(record)) == record
    assert record["splits"][0]["n_inference"] == 60
    text = result.summary()
    assert "x1" in text and "selection" in text.lower()
    assert "Average contrasts" in text


def test_to_dict_returns_the_aggregate_unless_arrays_are_requested():
    result = mb.split_refit("y ~ bbs(x1, df=4)", _data(100, 7), seed=1)
    assert "splits" not in result.to_dict()
    assert "splits" in result.to_dict(include_arrays=True)
    with pytest.raises(TypeError):
        result.to_dict(True)


def test_distributional_mapping_labels_terms_by_parameter():
    # Seed 4 gives a split on which gamlss_refit converges. On other splits of this
    # size gamlss_refit raises "the refit optimum cannot be re-evaluated".
    data = _data(300, 8)
    formula = "y ~ bbs(x1, df=4) + bbs(x2, df=4)"
    result = mb.split_refit({"mu": formula, "sigma": formula}, data, family=mb.GaussianLSS(),
                            seed=4, contrasts=[("x1", -1.0, 1.0)])
    assert result.scale == "response of mu"
    assert all(label.startswith(("mu: ", "sigma: ")) for label in result.splits[0].selected)
    assert all(t.p_value is None for t in result.terms)
    assert result.contrasts[0].estimate > 0.5


# ------------------------------------------------------------------- errors
@pytest.mark.parametrize("kwargs, message", [
    ({"select": "lasso"}, "select must be"),
    ({"repeats": 0}, "repeats"),
    ({"fraction": 1.0}, "fraction"),
    ({"level": 0.0}, "level"),
    ({"contrasts": [("x1", 1.0, 1.0)]}, "equal reference"),
    ({"contrasts": [("nope", 0.0, 1.0)]}, "not a column"),
    ({"contrasts": [("x1", 0.0)]}, "variable, reference, alternative"),
])
def test_bad_arguments_are_refused_before_any_fit(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mb.split_refit("y ~ bbs(x1, df=4)", _data(60, 9), **kwargs)


def test_stabsel_is_refused_for_a_mapping_with_a_remedy():
    formula = "y ~ bbs(x1, df=4)"
    with pytest.raises(NotImplementedError, match="deselect"):
        mb.split_refit({"mu": formula, "sigma": formula}, _data(60, 10),
                       family=mb.GaussianLSS(), select="stabsel")


def test_mapping_without_a_distributional_family_is_refused():
    formula = "y ~ bbs(x1, df=4)"
    with pytest.raises(ValueError, match="distributional family"):
        mb.split_refit({"mu": formula, "sigma": formula}, _data(60, 11))


def test_callable_must_return_terms_of_the_formula():
    with pytest.raises(ValueError, match="not terms of the formula"):
        mb.split_refit("y ~ bbs(x1, df=4)", _data(120, 12),
                       select=lambda data: ["bbs(x9, df=4)"])


# --------------------------------------------------------------- simulation
def test_split_refit_interval_covers_the_true_average_contrast():
    """One linear effect (beta = 1) and one null covariate. The contrast of x1
    from -1 to 1 averages to 2 * beta = 2 over the rows. Each replicate draws
    fresh data and runs the whole procedure with one split.

    Measured when this test was written (60 replicates, n = 300, seeds 1000+):
    coverage 0.933 at nominal 0.90; mean interval width 0.557; x2 chosen in 0.47
    of the replicates by the deselect rule. The assertion is the lower bound of
    0.80 the procedure must meet.
    """
    replicates, n, truth = 60, 300, 2.0
    covered = 0
    for replicate in range(replicates):
        data = _data(n, 1000 + replicate)
        result = mb.split_refit("y ~ bbs(x1, df=4) + bbs(x2, df=4)", data, seed=replicate,
                                level=0.9, contrasts=[("x1", -1.0, 1.0)])
        contrast = result.contrasts[0]
        covered += int(contrast.lower <= truth <= contrast.upper)
    assert covered / replicates >= 0.80
