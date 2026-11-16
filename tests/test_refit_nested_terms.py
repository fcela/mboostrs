"""A pair term and the main effects nested in it, in refits and LSS reports.

A tensor ``bbs(x1) %X% bbs(x2)`` reproduces the cubic polynomials of each
margin, which a ``bbs(x1)`` main effect also contains. The REML refits drop
the repeated columns (mgcv's ``gam.side``); before they did, the Gaussian
refit returned a negative EDF for ``bbs(x1)`` and the distributional refit
failed to evaluate its criterion. LSS models fit a pair as a derived design
(``__ix0``) and report it by the term the user wrote.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

PAIR = "bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5)"


def _data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.uniform(-1, 1, (3, n))
    noise = np.exp(0.3 * x3 + 0.4 * x1 * x2) * 0.3 * rng.normal(size=n)
    y = np.sin(3 * x1) + x2 + 1.2 * x1 * x2 + noise
    return pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def test_gaussian_refit_with_a_pair_and_its_main_effects_is_identified():
    data = _data()
    formula = f"y ~ bbs(x1, df=4, knots=10) + bbs(x2, df=4, knots=10) + {PAIR}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.tune_mstop(formula, data, seed=0).model
        refit = mb.gam_refit(model)
    assert refit.convergence["converged"]
    assert all(edf > 0.5 for edf in refit.edf.values()), refit.edf
    fitted = np.asarray(refit.predict(data))
    signal = (np.sin(3 * data["x1"]) + data["x2"] + 1.2 * data["x1"] * data["x2"]).to_numpy()
    assert np.sqrt(np.mean((fitted - signal) ** 2)) < 0.12


@pytest.fixture(scope="module")
def lss_fit():
    data = _data()
    formula = f"y ~ bbs(x1) + bbs(x2) + bbs(x3) + {PAIR}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboostLSS(formula, data, family=mb.GaussianLSS(),
                              control=mb.boost_control(400, 0.1), method="noncyclic")


def test_lss_names_a_pair_by_the_written_term(lss_fit):
    for names in lss_fit.base_learner_names().values():
        assert PAIR in names
        assert not any(str(n).startswith("__ix") for n in names)
    selected = {n for path in lss_fit.selected_which().values() for n in path}
    assert not any(str(n).startswith("__ix") for n in selected)
    rows = lss_fit.path_frame(output="records")
    assert not any(str(r["learner"]).startswith("__ix") for r in rows)
    # The derived name is still accepted where a learner is looked up.
    written = lss_fit.partial_effects(which=PAIR, parameter="mu")
    derived = lss_fit.partial_effects(which="__ix0", parameter="mu")
    np.testing.assert_allclose(np.asarray(written), np.asarray(derived))


def test_lss_deselect_reports_the_written_pair_and_refits_it(lss_fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kept = mb.deselect(lss_fit, seed=0)
        refit = mb.gamlss_refit(kept.model)
    text = repr(kept.kept) + repr(kept.dropped) + repr(kept.formula) + repr(kept.shares)
    assert "__ix" not in text and "buser" not in text
    assert (("mu", PAIR) in kept.kept) == (PAIR in refit.terms["mu"])
    assert all("__ix" not in v for vs in kept.variables.values() for v in vs)
    assert refit.convergence["converged"]
    for edfs in refit.edf.values():
        assert all(edf > 0.5 for edf in edfs.values()), refit.edf


def test_lss_selection_chart_labels_match_the_selections(lss_fit):
    from mboostrs.plot import plot_lss_selected

    chart = plot_lss_selected(lss_fit)
    rows = chart.data.to_dicts()
    for parameter, path in lss_fit.selected_which().items():
        for row in rows:
            if row["parameter"] == parameter:
                assert row["frequency"] == pytest.approx(path.count(row["learner"]) / len(path))
