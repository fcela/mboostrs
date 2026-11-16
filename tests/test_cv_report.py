"""One report shape for standard and distributional cross-validation (#170).

`CvRiskResult.report()` and `LssCvRiskResult.report()` return a `CvReport`:
fold design, fold count, grid, optimum (an int, or {parameter: steps}), mean
held-out risk at the optimum, the unit of that risk, and whether the optimum
sits on the grid's edge.
"""

import numpy as np

import mboostrs as mb
from mboostrs.cvrisk import CvReport


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return {"x": x, "y": rng.normal(np.sin(x), np.exp(0.3 * x))}


def test_standard_and_lss_reports_share_a_shape():
    d = _data()
    folds = mb.repeated_kfold_folds(200, k=5, repeats=1, seed=1)
    gam = mb.gamboost("y ~ bbs(x)", data=d, control=mb.boost_control(60))
    lss = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bols(x)"}, data=d,
                         families=mb.GaussianLSS(), control=mb.boost_control(40))
    standard = gam.cvrisk(folds=folds)
    distributional = lss.cvrisk(folds=folds)
    for result, report in ((standard, standard.report()),
                           (distributional, distributional.report())):
        assert isinstance(report, CvReport)
        assert report.n_folds == 5 and "5-fold" in report.folds
        assert report.risk_at_optimum == min(np.asarray(result.aggregate))
        assert "held-out" in report.score and report.at_boundary == result.at_boundary()
    assert standard.report().optimum == standard.optimal_mstop
    assert standard.report().grid == tuple(range(61))
    assert set(distributional.report().optimum) == {"mu", "sigma"}
    assert distributional.report().grid[-1] == (40, 40)


def test_noncyclic_reports_its_total_budget():
    d = _data()
    lss = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bols(x)"}, data=d,
                         families=mb.GaussianLSS(), control=mb.boost_control(60),
                         method="noncyclic")
    report = lss.cvrisk(folds=mb.repeated_kfold_folds(200, k=4, repeats=1, seed=2)).report()
    assert list(report.optimum) == ["total"] and report.n_folds == 4
