"""`term_report` and `tune_mstop` for distributional specifications (W01/W03).

The evaluate-to-modify loop stopped at the LSS fit: `term_report` and
`tune_mstop` took a single-formula specification and refused everything
distributional. Both now accept the mapping `gamboostLSS` semantics use —
the existing functions, extended, not parallel `term_reportLSS` /
`tune_mstopLSS` families:

* `term_report({"mu": "bbs(x, df=4)", "sigma": "1 + bbs(z, df=4)"}, data,
  "y", family=GaussianLSS())` — what each term of each parameter earns on the
  joint proper score, with paired fold standard errors, and whether another
  shape of it earns more. A parameter that loses its last term becomes the
  estimated constant `1` in the removal candidate.
* `tune_mstop({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"}, data,
  family=GaussianLSS())` — the doubling grid read off the LSS diagonal, whose
  optimum is exactly what the fitted model's own `cvrisk` reports.
"""

import numpy as np
import pytest

import mboostrs as mb


def _varying_scale_frame(seed=11, n=240):
    """Mean smooth in x; scale grows with x; z is noise in both parameters."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    y = np.sin(x) + np.exp(-0.2 + 0.3 * x) * rng.normal(size=n)
    return {"y": y, "x": x, "z": z}


SPEC = {"mu": "bbs(x, df=4)", "sigma": "1 + bbs(z, df=4)"}


def test_a_signal_term_earns_and_a_noise_term_costs():
    data = _varying_scale_frame()
    report = mb.term_report(SPEC, data, "y", family=mb.GaussianLSS(),
                            folds=4, mstop=150)
    by_term = {(row["parameter"], row["term"]): row
               for row in report.contributions}
    mean_x = by_term[("mu", "bbs(x, df=4)")]
    # x drives the mean; removing its term costs at least two paired SEs.
    assert mean_x["earns"] > 2 * mean_x["se"]
    assert mean_x["earns"] > 0.1
    # z is noise in the scale; removing it does not raise the joint score.
    scale_z = by_term[("sigma", "bbs(z, df=4)")]
    assert scale_z["earns"] < 0
    # The explicit constant beside a bbs earns nothing: the spline's own
    # basis spans the intercept, so the two represent it twice.
    constant = by_term[("sigma", "1")]
    assert constant["earns"] == pytest.approx(0.0, abs=1e-12)
    assert all(row["parameter"] in ("mu", "sigma")
               for row in report.contributions)


def test_a_parameter_that_loses_its_last_term_becomes_a_constant():
    data = _varying_scale_frame(n=120)
    report = mb.term_report({"mu": "bbs(x, df=4)", "sigma": "bbs(x, df=4)"},
                            data, "y", family=mb.GaussianLSS(),
                            folds=3, mstop=100)
    by_term = {(row["parameter"], row["term"]): row
               for row in report.contributions}
    # Removing mu's only term is a candidate: mu fitted as the constant 1.
    assert ("mu", "bbs(x, df=4)") in by_term
    assert by_term[("mu", "bbs(x, df=4)")]["earns"] > 0
    # An intercept that is a parameter's only term is not a removal
    # candidate: nothing would remain to hold the parameter constant.
    report_constant = mb.term_report({"mu": "bbs(x, df=4)", "sigma": "1"},
                                      data, "y", family=mb.GaussianLSS(),
                                      folds=3, mstop=100)
    assert all((row["parameter"], row["term"]) != ("sigma", "1")
               for row in report_constant.contributions)


def test_the_shape_sweep_scores_within_the_terms_own_parameter():
    data = _varying_scale_frame()
    report = mb.term_report(SPEC, data, "y", family=mb.GaussianLSS(),
                            folds=4, mstop=150, shapes=True)
    assert report.shapes
    for row in report.shapes:
        assert row["parameter"] in ("mu", "sigma")
        assert set(row) >= {"term", "variable", "current", "best",
                            "gain over current", "se"}


def test_distributional_report_refusals():
    data = _varying_scale_frame(n=80)
    with pytest.raises(ValueError, match="needs a distributional family"):
        mb.term_report(SPEC, data, "y", family=mb.Gaussian())
    with pytest.raises(ValueError, match="GaussianLSS takes"):
        mb.term_report({"mu": "bbs(x)", "scale": "1"}, data, "y",
                       family=mb.GaussianLSS())
    with pytest.raises(ValueError, match="omits parameters"):
        mb.term_report({"mu": "bbs(x, df=4)"}, data, "y",
                       family=mb.GaussianLSS())
    with pytest.raises(ValueError, match="joint proper score"):
        mb.term_report(SPEC, data, "y", family=mb.GaussianLSS(),
                       folds=2, mstop=50, score="mse")


def test_tune_mstop_matches_the_models_own_diagonal_optimum():
    data = _varying_scale_frame(n=200)
    tuning = mb.tune_mstop({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                           data, family=mb.GaussianLSS(), folds=4,
                           start=60, nu=0.1)
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"}, data,
                         family=mb.GaussianLSS(),
                         control=mb.boost_control(60, 0.1))
    folds = mb.cv_weights(200, type="kfold", B=4, seed=0)
    reference = fit.cvrisk(folds=folds,
                           grid=mb.diagonal_grid(60, 2, length_out=61))
    assert tuning.mstop == reference.optimal_mstop[0]
    assert not tuning.at_boundary
    # One sweep of one fit per fold; the risk turned inside the grid, so
    # no doubling happened.
    assert tuning.fits == 4
    assert tuning.risk[0] > min(tuning.risk)


def test_tune_mstop_extends_when_the_risk_is_still_falling():
    data = _varying_scale_frame(n=200)
    tuning = mb.tune_mstop({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                           data, family=mb.GaussianLSS(), folds=4,
                           start=4, nu=0.1, extensions=5)
    # The first grid ended at 4 and the risk was still falling there, so the
    # grid doubled; the curve turned inside the doubled budget.
    assert tuning.largest > 4
    assert tuning.first_largest == 4
    assert not tuning.at_boundary
    # Each fold is fitted once and its path resumed across the grids, plus the
    # one full-data fit the search's refit arguments come from.
    assert tuning.fits == 4
    assert tuning.full_data_fits == 1


def test_tune_mstop_refuses_a_mapping_without_a_distributional_family():
    data = _varying_scale_frame(n=80)
    with pytest.raises(ValueError, match="needs a distributional family"):
        mb.tune_mstop({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                      data, family=mb.Gaussian())


def test_the_tuned_stop_refits_through_the_same_grid():
    """The tuned value is usable: refitting to it is the mstop the report
    recommends, and the LSS mstop vector spelling works with `update`."""
    data = _varying_scale_frame(n=200)
    tuning = mb.tune_mstop({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                           data, family=mb.GaussianLSS(), folds=4,
                           start=60, nu=0.1)
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"}, data,
                         family=mb.GaussianLSS(),
                         control=mb.boost_control(60, 0.1))
    truncated = fit.with_mstop([tuning.mstop] * 2)
    assert truncated.mstop == tuning.mstop
