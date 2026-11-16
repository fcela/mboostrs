"""aipw on thin overlap: the all-rows ATE is biased, and target="overlap" is not.

The design has a propensity that drops steeply above z = 0.8 (about 21% of the
rows, with propensity near 0.004), and an effect proportional to the baseline,
so the effect is largest where treated rows are rarest. The outcome model
extrapolates into that region, and the propensity clip changes the score
there. Fits use n = 3,000 and no inner tuning so the file runs in seconds.
"""

import inspect
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import causal

N = 3000
FAST = dict(folds=5, smooth=True, tune=False, mstop=200, level=0.9)


def sig(x):
    return 1 / (1 + np.exp(-x))


def thin_overlap(seed, *, effect=True, missing=0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=N)
    x = rng.normal(size=N)
    e = sig(-0.3 - 0.8 * z + 0.3 * x - 4.5 * (z > 0.8))
    a = rng.binomial(1, e).astype(float)
    base = 1000 * np.exp(0.9 * z) + 200 * x
    tau = 0.3 * base if effect else np.zeros(N)
    y = base + a * tau + rng.normal(scale=300, size=N)
    if missing:
        y = np.where(rng.random(N) < missing, np.nan, y)
    data = {"a": a, "y": y, "z": z, "x": x}
    truth_all = float(tau.mean())
    h = e * (1 - e)
    truth_overlap = float((h * tau).sum() / h.sum())
    return data, truth_all, truth_overlap


SPEC = mb.CausalSpec(
    treatment="a", outcome="y", covariates=("z", "x"), target_population="simulated thin-overlap population",
    unit="person", time_zero="treatment assignment", follow_up="one year",
    adjustment_rationale="z and x cause both treatment and outcome")


def fit(data, **options):
    with warnings.catch_warnings():
        # The sparse-arm warning is the subject of tests/test_aipw_auto_outcome.py.
        warnings.simplefilter("ignore")
        return mb.aipw(data, SPEC, **{**FAST, **options})


def test_default_target_is_the_ate_over_every_row_and_says_so():
    data, _, _ = thin_overlap(1)
    default = fit(data)
    explicit = fit(data, target="all")
    assert default.estimate == explicit.estimate
    assert default.diagnostics["target"] == "all"
    assert "over every row" in default.describe()
    assert inspect.signature(mb.aipw).parameters["target"].default == "all"


def test_all_rows_estimate_is_biased_on_this_design_and_the_result_flags_thin_overlap():
    # The documented limitation: the all-rows ATE misses its own truth by several
    # standard errors here, and describe() points to the overlap target.
    data, truth_all, _ = thin_overlap(1)
    result = fit(data)
    error = result.estimate - truth_all
    assert abs(error) > 3 * result.standard_error
    text = result.describe()
    assert "below 0.05 or above 0.95" in text
    assert "target='overlap'" in text


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_overlap_estimate_is_closer_to_its_truth_than_the_all_rows_estimate_is_to_its_own(seed):
    data, truth_all, truth_overlap = thin_overlap(seed)
    all_rows = fit(data)
    overlap = fit(data, target="overlap")
    assert abs(overlap.estimate - truth_overlap) < abs(all_rows.estimate - truth_all)


def test_overlap_interval_covers_its_truth_on_most_seeds():
    # Ten fresh draws with effect: the 90% interval covers the overlap truth
    # on at least eight. Seed 1 is one of the two misses on this design.
    covered = 0
    for seed in range(1, 11):
        data, _, truth_overlap = thin_overlap(seed)
        lo, hi = fit(data, target="overlap").interval
        covered += lo <= truth_overlap <= hi
    assert covered >= 8


def test_overlap_estimate_is_the_overlap_weighted_augmented_score():
    # An independent computation of the overlap score from the result's own
    # nuisance arrays, with missing outcomes so the response weights enter.
    data, _, _ = thin_overlap(2, missing=0.2)
    result = fit(data, target="overlap", outcome_missing="ipw")
    a = result.treatment
    e = result.propensity
    pi = result.response_propensity
    observed = result.observed
    y = np.where(observed == 1, result.outcome, 0.0)
    h = e * (1 - e)
    numerator = h * (result.mu1 - result.mu0) + observed / pi * (
        a * (1 - e) * (y - result.mu1) - (1 - a) * e * (y - result.mu0))
    expected = float(numerator.sum() / h.sum())
    assert result.estimate == pytest.approx(expected, rel=1e-10)
    assert abs(float(np.mean(result.influence))) < 1e-8 * max(1.0, abs(expected))


def test_overlap_target_reports_the_target_and_does_not_clip_the_propensity():
    data, _, _ = thin_overlap(1)
    result = fit(data, target="overlap")
    assert result.diagnostics["target"] == "overlap"
    assert result.diagnostics["propensity_clip_applied"] is None
    assert result.diagnostics["n_clipped_propensity"] == 0
    assert "overlap-weighted ATE" in result.describe()
    assert "not the ATE over every row" in result.describe()
    assert repr(result).startswith("AIPWResult(overlap=")
    assert result.nuisance["target"] == "overlap"
    # The overlap analysis weights are 1 - e for treated rows and e for controls.
    frame = result.overlap_frame(output="records")
    weights = np.array([row["analysis_weight"] for row in frame])
    expected = np.where(result.treatment == 1, 1 - result.propensity, result.propensity)
    assert np.allclose(weights, expected)


def test_overlap_target_restricted_estimate_is_not_reported():
    data, _, _ = thin_overlap(1)
    result = fit(data, target="overlap")
    report = causal.overlap_report(result)
    assert report.estimand == "overlap"
    assert report.estimate_inside is None


def test_target_argument_is_checked():
    data, _, _ = thin_overlap(1)
    with pytest.raises(ValueError, match="target must be 'all'"):
        fit(data, target="trimmed")
    att = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("z", "x"), estimand="att",
        target_population="simulated", unit="person", time_zero="assignment",
        follow_up="one year", adjustment_rationale="z and x cause both")
    with pytest.raises(ValueError, match="defined for estimand='ate' only"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mb.aipw(data, att, **{**FAST, "target": "overlap"})


def test_overlap_target_survives_a_confounding_benchmark_refit():
    data, _, _ = thin_overlap(3)
    result = fit(data, target="overlap")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bench = causal.confounding_benchmark(data, result)
    assert bench is not None
