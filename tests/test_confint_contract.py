"""What a bootstrap confidence interval is computed from, and what it says.

`confint()` fits `B` models on resampled weights and takes quantiles. Three
things about that were invisible to the caller.

A replicate that never selects a learner has a zero partial effect -- that is
what an unselected base-learner contributes, and it is what `mboost`'s
`confint` bootstraps, through `predict(which=)` returning zeros. Those
replicates were recorded as `NaN` and dropped by `np.nanquantile`, which
removes exactly the replicates where the effect is zero. Measured on a 150-row
fit where 39 of 300 replicates never selected `z`: the interval covered zero at
37% of grid points, against 100% once the zeros are counted.

A replicate that fails outright was also dropped, so an interval built from 40
surviving replicates was reported in the same shape, with the same `level`, as
one built from 1000. And an unknown learner name reached the loop, matched
nothing, and came back as an all-NaN interval -- a typo reported as a result.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def weak_second_learner():
    """A fit where `z` is selected overall but not in every resample."""
    rng = np.random.default_rng(11)
    n = 150
    x, z = rng.normal(size=n), rng.normal(size=n)
    data = {"x": x, "z": z, "y": 2.0 * x + 0.20 * z + rng.normal(size=n)}
    model = mb.gamboost("y ~ bbs(x, df=4) + bbs(z, df=4)", data=data,
                        control=mb.boost_control(25, 0.1))
    assert "z" in set(model.selected)
    return model


def test_a_replicate_that_never_selects_a_learner_contributes_zero(
    weak_second_learner,
):
    ci = weak_second_learner.confint(parm=["z"], B=300, B_mstop=0, seed=5)
    predictions = np.array(ci["boot_pred"]["z"])

    assert not np.isnan(predictions).any(), "no replicate should be missing"
    never_selected = int((np.abs(predictions).max(axis=1) == 0).sum())
    assert never_selected > 0, "fixture no longer exercises the case"

    lower, upper = ci["lower"]["z"], ci["upper"]["z"]
    covers_zero = float(np.mean((lower <= 0) & (upper >= 0)))
    assert covers_zero == 1.0, (
        f"a learner missed by {never_selected}/300 replicates should have an "
        f"interval containing zero everywhere, got {covers_zero:.2f}"
    )


def test_dropping_those_replicates_is_what_moved_the_interval(
    weak_second_learner,
):
    """The same numbers, with the zeros removed again, reproduce the old
    answer -- so the difference is the padding rule and nothing else."""
    ci = weak_second_learner.confint(parm=["z"], B=300, B_mstop=0, seed=5)
    predictions = np.array(ci["boot_pred"]["z"])
    unselected = np.abs(predictions).max(axis=1) == 0
    kept = predictions[~unselected]

    lower = np.quantile(kept, 0.025, axis=0)
    upper = np.quantile(kept, 0.975, axis=0)
    assert float(np.mean((lower <= 0) & (upper >= 0))) < 0.5


def test_an_unknown_learner_name_is_refused(weak_second_learner):
    with pytest.raises(ValueError, match=r"unknown base-learner 'bbs\(z\)'"):
        weak_second_learner.confint(parm=["bbs(z)"], B=4, B_mstop=0)


def test_the_result_says_how_many_replicates_it_used(weak_second_learner):
    ci = weak_second_learner.confint(parm=["z"], B=40, B_mstop=0, seed=2)
    assert ci["B"] == 40
    assert ci["n_used"] == 40
    assert ci["n_failed"] == 0
    assert ci["n_untuned"] == 0
    assert ci["seed"] == 2


def test_the_glmboost_interval_reports_the_same_counts():
    rng = np.random.default_rng(3)
    n = 120
    data = {"x": rng.normal(size=n)}
    data["y"] = 2.0 * data["x"] + rng.normal(size=n)
    model = mb.glmboost("y ~ x", data=data, control=mb.boost_control(20, 0.1))
    ci = model.confint(parm=["x"], B=30, B_mstop=0, seed=7)
    assert (ci["B"], ci["n_used"], ci["n_failed"]) == (30, 30, 0)
    assert ci["lower"]["x"] < ci["upper"]["x"]


# ─── failures are counted, not swallowed ───────────────────────────


def test_all_replicates_failing_raises_rather_than_returning_nan():
    from mboostrs._bootstrap import FAILED, BootstrapTally, bootstrap_outcome

    tally = BootstrapTally(B=5)
    for b in range(5):
        tally.mark(b, FAILED, ValueError("singular design"))
    with pytest.raises(RuntimeError, match="every one of the 5"):
        bootstrap_outcome(tally, B=5, what="coefficient")


def test_some_replicates_failing_warns_with_the_count_and_the_reason():
    from mboostrs._bootstrap import FAILED, BootstrapTally, bootstrap_outcome

    tally = BootstrapTally(B=10)
    for b in range(3):
        tally.mark(b, FAILED, ValueError("singular design"))
    with pytest.warns(RuntimeWarning, match=r"3 of 10 .*singular design"):
        counts = bootstrap_outcome(tally, B=10, what="coefficient")
    assert counts == {"B": 10, "n_failed": 3, "n_untuned": 0, "n_at_mstop_cap": 0,
                      "n_used": 7}


def test_a_failed_inner_cv_is_reported_separately():
    """Those replicates did produce coefficients, at the outer model's mstop
    rather than their own, which is a different fact from a missing one."""
    from mboostrs._bootstrap import UNTUNED, BootstrapTally, bootstrap_outcome

    tally = BootstrapTally(B=10)
    for b in range(2):
        tally.mark(b, UNTUNED, RuntimeError("cv failed"))
    with pytest.warns(RuntimeWarning, match="inner mstop cross-validation"):
        counts = bootstrap_outcome(tally, B=10, what="coefficient")
    assert counts["n_untuned"] == 2
    assert counts["n_failed"] == 0


def test_a_clean_run_says_nothing():
    from mboostrs._bootstrap import BootstrapTally, bootstrap_outcome

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        counts = bootstrap_outcome(BootstrapTally(B=10), B=10, what="coefficient")
    assert counts["n_used"] == 10


def test_replicates_at_the_mstop_cap_are_counted_and_warned_once():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    data = {"x": x, "y": 2.0 * x + rng.normal(scale=0.5, size=200)}
    # Ten steps are far too few, so every inner cross-validation stops at
    # the cap; that used to raise one warning per replicate.
    fit = mb.glmboost("y ~ x", data=data, control=mb.boost_control(10, 0.1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ci = fit.confint("x", B=10, B_mstop=5, seed=0)
    assert ci["n_at_mstop_cap"] == 10
    messages = [str(w.message) for w in caught]
    assert sum("largest mstop the model allows" in m for m in messages) == 1
    assert not any("still falling at mstop" in m for m in messages)
