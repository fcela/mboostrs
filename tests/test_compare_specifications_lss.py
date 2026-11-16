"""Distributional candidates in `compare_specifications` (W03/W01).

A practitioner can diagnose varying spread (`heteroscedasticity`) but could
not use the assistant to act on the diagnosis: named ``mu``/``sigma``
candidates were parsed as one formula string and raised
``cannot parse formula term: "{'mu': ...}"``. A candidate now carries one
right-hand side per distribution parameter, every candidate is scored by the
family's own joint proper score (negative log predictive density) on the same
folds, and the change against the reference carries its paired fold standard
error. There is no separate `autospecLSS`/`term_reportLSS` result family: the
same rows, the same fields, one more kind of candidate.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _varying_scale_frame(seed=21, n=300):
    """y with a smooth mean and a scale that grows with x, plus a noise
    column the true specifications never use."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    y = np.sin(x) + np.exp(-0.3 + 0.5 * x) * rng.normal(size=n)
    return pl.DataFrame({"x": x, "z": z, "y": y})


def test_named_parameter_candidates_are_scored_together():
    """A mu/sigma mapping candidate is fitted as one distributional model
    and scored by the joint proper score, on the same folds as every other
    candidate."""
    data = _varying_scale_frame()
    rows = mb.compare_specifications(
        {
            "constant scale": {"mu": "bbs(x, df=4)", "sigma": "1"},
            "varying scale": {"mu": "bbs(x, df=4)", "sigma": "1 + bbs(x, df=4)"},
        },
        data, "y", family=mb.GaussianLSS(), folds=4, mstop=150,
    )
    assert len(rows) == 2
    assert rows[0]["change"] == 0.0
    assert rows[0]["change se"] == 0.0
    # The scale truly varies with x, so the constant-scale candidate must
    # lose, and by more than its paired standard error.
    assert rows[1]["change"] < 0
    assert rows[1]["change"] < -4 * rows[1]["change se"]
    # The candidate row is still the editable specification.
    assert rows[0]["rhs"] == {"mu": "bbs(x, df=4)", "sigma": "1"}



def test_a_sequence_of_mapping_candidates_is_named_and_scored():
    """A plain list of mu/sigma mappings used to fail with ``unhashable type:
    'dict'`` (the row name was the dict). Each row is named by its parameters'
    right-hand sides, keeps the mapping as its editable ``rhs``, and scores
    exactly as the same candidates passed by name. A whole formula for the
    response inside a mapping is read as its right-hand side."""
    data = _varying_scale_frame()
    constant = {"mu": "bbs(x, df=4)", "sigma": "1"}
    varying = {"mu": "bbs(x, df=4)", "sigma": "1 + bbs(x, df=4)"}
    listed = mb.compare_specifications(
        [constant, {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1 + bbs(x, df=4)"}],
        data, "y", family=mb.GaussianLSS(), folds=4, mstop=150)
    named = mb.compare_specifications(
        {"constant": constant, "varying": varying},
        data, "y", family=mb.GaussianLSS(), folds=4, mstop=150)
    assert [r["specification"] for r in listed] == [
        "mu: bbs(x, df=4); sigma: 1", "mu: bbs(x, df=4); sigma: 1 + bbs(x, df=4)"]
    assert listed[1]["rhs"] == varying
    assert [r["out-of-fold risk"] for r in listed] == [r["out-of-fold risk"] for r in named]
    assert all(isinstance(name, str) for name in listed[0]["competitive set"])

def test_an_underfit_mean_costs_more_than_the_same_scale():
    """Holding the scale specification fixed, the stiffer mean loses held-out
    score: the comparison separates a scale problem from a mean problem."""
    data = _varying_scale_frame()
    rows = mb.compare_specifications(
        {
            "flexible mean": {"mu": "bbs(x, df=4)", "sigma": "1 + bbs(x, df=4)"},
            "one slope": {"mu": "bols(x)", "sigma": "1 + bbs(x, df=4)"},
        },
        data, "y", family=mb.GaussianLSS(), folds=4, mstop=150,
    )
    assert rows[1]["change"] > 0
    assert rows[1]["loss units"] == "nats_per_observation"


def test_inner_tuned_mstop_counts_its_fits_exactly():
    """mstop='cv' tunes inside each outer fold; the planned and spent fit
    counts agree, so `fit_budget` gates the truth."""
    data = _varying_scale_frame(n=120)
    rows = mb.compare_specifications(
        {"constant": {"mu": "bbs(x, df=4)", "sigma": "1"}},
        data, "y", family=mb.GaussianLSS(), folds=3, mstop="cv",
    )
    assert rows[0]["fit count"] == rows[0]["planned fit count"]
    assert rows[0]["fit count"] == 3 * (1 + 5)


def test_a_plain_rhs_is_shared_by_every_parameter():
    """A string candidate with a distributional family is what a
    single-formula gamboostLSS call means: the same right-hand side for
    every parameter."""
    data = _varying_scale_frame(n=120)
    shared = mb.cv_risk("bbs(x, df=4)", data, "y",
                         family=mb.GaussianLSS(), folds=3, mstop=100)
    spelled = mb.cv_risk({"mu": "bbs(x, df=4)", "sigma": "bbs(x, df=4)"},
                         data, "y", family=mb.GaussianLSS(), folds=3, mstop=100)
    assert shared == pytest.approx(spelled, abs=1e-12)


def test_a_mapping_candidate_refuses_a_standard_family():
    data = _varying_scale_frame(n=120)
    with pytest.raises(ValueError, match="distributional family"):
        mb.compare_specifications(
            {"candidate": {"mu": "bbs(x)", "sigma": "1"}},
            data, "y", family=mb.Gaussian(),
        )


def test_unknown_parameter_names_are_refused_at_the_call():
    data = _varying_scale_frame(n=120)
    with pytest.raises(ValueError, match="GaussianLSS takes"):
        mb.cv_risk({"mu": "bbs(x)", "scale": "1"}, data, "y",
                   family=mb.GaussianLSS())


def test_single_quantity_scores_refuse_a_distributional_family():
    """The joint proper score is the only comparable target: 'mse' would
    drop the fitted scale silently and rank by another quantity."""
    data = _varying_scale_frame(n=120)
    with pytest.raises(ValueError, match="joint proper score"):
        mb.cv_risk({"mu": "bbs(x)", "sigma": "1"}, data, "y",
                   family=mb.GaussianLSS(), score="mse")


def test_missing_rows_are_dropped_on_both_sides_under_omit():
    data = _varying_scale_frame(n=150)
    values = data["x"].to_numpy().copy()
    values[:15] = np.nan
    with_missing = data.with_columns(pl.Series("x", values))
    risk = mb.cv_risk({"mu": "bbs(x, df=4)", "sigma": "1"}, with_missing, "y",
                      family=mb.GaussianLSS(), folds=3, mstop=100,
                      na_action="omit")
    assert np.isfinite(risk)


def test_the_score_matches_the_family_density_directly():
    """The reported score is the equal-fold mean of per-record negative log
    densities; each record's prediction is one value per parameter."""
    data = _varying_scale_frame(n=120)
    result = mb.evaluate_specification(
        {"mu": "bbs(x, df=4)", "sigma": "1"}, data, "y",
        family=mb.GaussianLSS(), folds=3, mstop=100,
    )
    names = mb.GaussianLSS().param_names
    for record in result.records:
        prediction = record["prediction"]
        assert isinstance(prediction, list) and len(prediction) == len(names)
        assert record["loss"] > 0
    # Fold scores average to the reported score.
    assert result.score == pytest.approx(float(np.mean(result.fold_scores)))


def test_function_level_lss_cvrisk_builds_the_diagonal_grid_when_none_is_given():
    """`cvrisk_gamboostLSS` / `cvrisk_glmboostLSS` used to refuse a call
    without `grid=`; they now use the diagonal grid `LSSModel.cvrisk()`
    builds, and a missing `folds=` names the remedy."""
    import mboostrs as mb

    rng = np.random.default_rng(0)
    n = 300
    data = {"x": rng.uniform(-2, 2, n)}
    data["y"] = np.sin(data["x"]) + np.exp(-1 + 0.4 * data["x"]) * rng.normal(size=n)
    folds = mb.cv_weights(n, "kfold", 5, seed=0)

    smooth = mb.cvrisk_gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(x)"}, data,
                                   families=mb.GaussianLSS(), folds=folds)
    linear = mb.cvrisk_glmboostLSS({"mu": "y ~ x", "sigma": "y ~ x"}, data,
                                   families=mb.GaussianLSS(), folds=folds)
    assert len(smooth.grid) == 10 and len(linear.grid) == 10
    with pytest.raises(ValueError, match="folds= is required"):
        mb.cvrisk_gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(x)"}, data,
                              families=mb.GaussianLSS())
