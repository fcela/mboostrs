"""Selection reports: interchangeable predictors, equivalence sets, substitution.

The simulation has three predictive drivers and six noise columns. ``x1d`` is a
near-copy of ``x1`` (correlation about 0.99), ``x2p`` a correlated proxy of
``x2`` (correlation about 0.89), and ``x3`` a weak independent driver. Every
result here is a predictive statement about these rows and resamples: a group
of substitutes is a group of learners that carry the same information, not a
statement about which one causes the response.
"""

import json
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.equivalence import (
    EquivalenceSets,
    equivalence_sets,
    selection_report,
    substitution_sets,
)
from mboostrs.stabsel import StabSelResult

N = 300
MSTOP = 120
NOISE = [f"z{k}" for k in range(6)]
DRIVER_AXES = {"axis1": ("x1", "x1d"), "axis2": ("x2", "x2p"), "axis3": ("x3",)}


def _simulate(seed):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, N)
    x2 = rng.uniform(-2, 2, N)
    x3 = rng.uniform(-2, 2, N)
    data = {
        "x1": x1,
        "x1d": x1 + rng.normal(0, 0.1, N),
        "x2": x2,
        "x2p": x2 + rng.normal(0, 0.6, N),
        "x3": x3,
    }
    for name in NOISE:
        data[name] = rng.uniform(-2, 2, N)
    data["y"] = 0.8 * x1 + np.sin(1.2 * x2) + 0.3 * x3 + rng.normal(0, 0.7, N)
    return data


def _fit(seed, columns=("x1", "x1d", "x2", "x2p", "x3", *NOISE)):
    data = _simulate(seed)
    rhs = " + ".join(f"bbs({c}, df=4)" for c in columns)
    model = mb.gamboost(f"y ~ {rhs}", data=data,
                        control=mb.boost_control(mstop=MSTOP, nu=0.1))
    return data, model


def _stab(seed=0):
    """A stability run on the simulated fit, with the error bound set by PFER."""
    _, model = _fit(seed)
    return mb.stabsel(model, cutoff=0.6, PFER=2.0, B=50, verbose=False)


def _result(names, subsample_sets):
    """A `StabSelResult` with the given subsample selections and nothing else."""
    p = len(names)
    return StabSelResult(
        phat=np.zeros((p, 1)), selected=[], max_=np.zeros(p), cutoff=0.6, q=p,
        PFER=1.0, p=p, B=len(subsample_sets) // 2, sampling_type="SS",
        assumption="none", learner_names=list(names), n_fits=len(subsample_sets),
        fit_selected=tuple(tuple(sorted(s)) for s in subsample_sets),
    )


# ------------------------------------------------------------ equivalence sets


def test_near_copy_is_grouped_and_unrelated_learners_are_not():
    _, model = _fit(1, columns=("x1", "x1d", "x2", "z0", "z1"))
    eq = equivalence_sets(model, threshold=0.9)

    assert eq.groups == (("x1", "x1d"),)
    assert eq.overlap[("x1", "x1d")] > 0.95
    for pair, value in eq.overlap.items():
        if pair != ("x1", "x1d"):
            assert value < 0.5, pair


def test_exact_duplicate_columns_have_full_overlap():
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 2, N)
    data = {"x": x, "x_copy": x.copy(), "z": rng.uniform(-2, 2, N)}
    data["y"] = x + rng.normal(0, 0.5, N)
    model = mb.gamboost("y ~ bbs(x, df=4) + bbs(x_copy, df=4) + bbs(z, df=4)",
                        data=data, control=mb.boost_control(mstop=60, nu=0.1))
    eq = equivalence_sets(model, threshold=0.99)

    assert eq.overlap[("x", "x_copy")] > 0.999
    assert eq.groups == (("x", "x_copy"),)


def test_wide_bases_on_few_rows_warn_that_overlap_is_unreliable():
    # Independent normal predictors with a 24-column basis on 71 rows: the
    # overlap saturates (0.99 for one pair in a check), so the warning fires.
    rng = np.random.default_rng(7)
    n = 71
    data = {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)}
    data["y"] = data["a"] + rng.normal(size=n)
    model = mb.gamboost("y ~ bbs(a, df=4) + bbs(b, df=4) + bbs(c, df=4)", data=data,
                        control=mb.boost_control(mstop=50, nu=0.1))
    with pytest.warns(UserWarning, match="widest basis"):
        equivalence_sets(model)

    wide_rows = _fit(0)[1]        # 300 rows, about 24 columns per learner
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        equivalence_sets(wide_rows)


def test_equivalence_threshold_is_refused_outside_the_unit_interval():
    _, model = _fit(1, columns=("x1", "z0"))
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="threshold"):
            equivalence_sets(model, threshold=bad)


def test_equivalence_sets_refuse_a_distributional_fit():
    rng = np.random.default_rng(4)
    data = {"x": rng.normal(size=120), "y": rng.normal(size=120)}
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=3)", "sigma": "y ~ bbs(x, df=3)"},
                         data=data, control=mb.boost_control(mstop=20, nu=0.1))
    with pytest.raises(TypeError, match="distributional"):
        equivalence_sets(fit)


# ----------------------------------------------------------- substitution sets


def test_alternating_copies_form_one_substitution_group():
    # learner 0 and 1 take turns; learner 2 is always selected; learner 3 is
    # selected in a pattern unrelated to the others.
    subsamples = []
    for b in range(40):
        first = 0 if b % 2 == 0 else 1
        second = 3 if b % 3 == 0 else None
        s = {first, 2}
        if second is not None:
            s.add(second)
        subsamples.append(s)
    result = _result(["a", "a_copy", "b", "c"], subsamples)

    sub = substitution_sets(result)

    assert [g.members for g in sub.groups] == [("a", "a_copy")]
    group = sub.groups[0]
    assert group.group_frequency == pytest.approx(1.0)
    assert group.member_frequency["a"] == pytest.approx(0.5)
    assert group.member_frequency["a_copy"] == pytest.approx(0.5)
    assert sub.group_of("b") is None
    assert sub.pairs[0]["frequency_both"] == pytest.approx(0.0)


def test_always_together_learners_are_not_substitutes():
    # 0 and 1 are selected together in 95% of subsamples. Learner 2 falls below
    # min_share, so it cannot be a substitute of either.
    subsamples = [{0, 1}] * 38 + [{2}] * 2
    sub = substitution_sets(_result(["a", "b", "c"], subsamples))
    assert sub.groups == ()
    assert sub.pairs == ()


def test_overlap_screen_keeps_unrelated_alternates_apart():
    # The same alternation as above, but the fit's design spaces say the two
    # learners overlap at 0.1: they are not substitutes.
    subsamples = [{0} if b % 2 == 0 else {1} for b in range(40)]
    result = _result(["a", "b"], subsamples)
    eq = EquivalenceSets(groups=(), overlap={("a", "b"): 0.1}, threshold=0.9,
                         measured=("a", "b"))

    assert substitution_sets(result).groups != ()
    screened = substitution_sets(result, equivalence=eq, min_overlap=0.5)
    assert screened.groups == ()
    assert screened.min_overlap == 0.5


def test_substitution_sets_refuse_a_result_without_subsample_selections():
    result = _result(["a", "b"], [{0}, {1}])
    result.fit_selected = ()
    with pytest.raises(ValueError, match="Run stabsel on a GAMBoostModel"):
        substitution_sets(result)


def test_stabsel_keeps_one_selection_per_subsample_fit():
    stab = _stab()
    assert len(stab.fit_selected) == stab.n_fits
    assert all(tuple(sorted(s)) == s for s in stab.fit_selected)
    assert all(set(s) <= set(range(stab.p)) for s in stab.fit_selected)


# ------------------------------------------------------------ selection report


def test_dropped_copy_is_reported_with_its_kept_equivalent():
    data, model = _fit(0)
    dsel = mb.deselect(model, folds=5, seed=0)
    report = selection_report(model, deselection=dsel)

    dropped_copy = [row for row in report.dropped_with_substitute
                    if row["dropped"] == "bbs(x1d, df=4)"]
    assert dropped_copy and dropped_copy[0]["kept_equivalent"] == ["x1"]
    assert all(row["dropped"] != "bbs(x1d, df=4)" for row in report.dropped_alone)


def test_selection_report_serialises_and_states_its_scope():
    _, model = _fit(0)
    dsel = mb.deselect(model, folds=5, seed=0)
    report = selection_report(model, deselection=dsel, stability=_stab(0))
    record = report.to_dict()
    json.dumps(record)

    assert "not a causal claim" in report.scope
    assert record["equivalence"]["groups"] == [["x1", "x1d"]]
    assert record["substitution"] is not None


# ------------------------------------------------------- duplicated drivers


SEEDS = range(20)


def test_duplicated_drivers_power_false_selections_and_substitutions_together():
    """Power, false selections and substitution sets on the same fits.

    Per seed, for deselection (threshold 1%): whether each driver axis keeps a
    learner (a copy counts for its axis), the noise columns kept, and how many
    dropped copies the report explains by a kept equivalent. For stability
    selection (cutoff 0.6): whether the axis-1 copies reach the cutoff, and the
    share of subsamples that selected at least one of them. The last number
    stays high when the two copies split the selections between them.
    """
    deselect_axes_kept, noise_kept, explained_drops = [], [], []
    stab_axis1_missed, axis1_any_frequency = [], []
    substitution_found = []
    for seed in SEEDS:
        _, model = _fit(seed)
        dsel = mb.deselect(model, folds=5, seed=seed, mstop=400)
        kept = {c for c in ("x1", "x1d", "x2", "x2p", "x3", *NOISE)
                if any(f"({c}," in t for t in dsel.kept)}
        deselect_axes_kept.append(all(kept & set(members) for members in DRIVER_AXES.values()))
        noise_kept.append(sum(c in kept for c in NOISE))

        stab = mb.stabsel(model, cutoff=0.6, PFER=2.0, B=50, verbose=False)
        stab_axis1_missed.append(not set(stab.selected_names) & {"x1", "x1d"})
        axis1_any_frequency.append(
            sum(bool({0, 1} & set(s)) for s in stab.fit_selected) / stab.n_fits)

        report = selection_report(model, deselection=dsel, stability=stab)
        groups = {g.members for g in report.substitution.groups}
        substitution_found.append(("x1", "x1d") in groups)
        explained_drops.append(len(report.dropped_with_substitute))
        dropped_alone = {r["dropped"] for r in report.dropped_alone}
        for row in report.dropped_with_substitute:
            assert row["kept_equivalent"], row
            assert row["dropped"] not in dropped_alone

    assert all(deselect_axes_kept), deselect_axes_kept
    assert sum(noise_kept) <= 2, noise_kept
    # Where stability selection misses the axis, the copies were still selected
    # together in nearly every subsample.
    assert all(f >= 0.9 for f in axis1_any_frequency), axis1_any_frequency
    assert sum(substitution_found) >= 10, substitution_found
    assert sum(explained_drops) >= 1
