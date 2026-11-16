"""confint under-coverage warning and the REML refit route (issue #245).

Bootstrap intervals on a boosted fit under-cover (74% empirical at nominal
90% on the book's additive benchmark) because resampling reproduces the
shrinkage early stopping induces. `confint()` must say so at call time, and
`confint(method="refit")` must route through the REML refit whose Bayesian
bands average close to nominal.
"""

import numpy as np
import pytest

import mboostrs as mb


def _fit(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(-2, 2, n)
    y = 2 * np.sin(1.5 * x0) + 1.5 * (2 * np.exp(-x1 ** 2) - 1) + rng.normal(0, 0.5, n)
    data = {"x0": x0, "x1": x1, "y": y}
    fit = mb.gamboost("y ~ bbs(x0, df=4) + bbs(x1, df=4)", data=data,
                      control=mb.boost_control(400, 0.3))
    return fit, data


def test_bootstrap_confint_warns_about_undercoverage():
    fit, _ = _fit()
    with pytest.warns(UserWarning, match="under-cover"):
        fit.confint(B=10, B_mstop=0, seed=0)


def test_bootstrap_confint_warns_every_call():
    # The warning is the point of the call, not a one-time notice.
    fit, _ = _fit()
    with pytest.warns(UserWarning, match="under-cover"):
        fit.confint(B=10, B_mstop=0, seed=0)
    with pytest.warns(UserWarning, match="under-cover"):
        fit.confint(B=10, B_mstop=0, seed=1)


def test_refit_route_does_not_warn_and_reports_method():
    import warnings

    fit, _ = _fit()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = fit.confint(method="refit", level=0.9)
    assert not [w for w in record if "under-cover" in str(w.message)]
    assert out["method"] == "refit"
    assert out["level"] == 0.9
    assert set(out["lower"]) == set(out["upper"]) == set(out["estimate"])
    assert out["held"] == ()


def test_refit_route_hits_nominal_coverage():
    # Small Gaussian task; the refit's Bayesian bands should average close
    # to nominal (issue acceptance: >= 88% at nominal 90%).
    rng = np.random.default_rng(3)
    n = 400
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(-2, 2, n)
    f0 = 2 * np.sin(1.5 * x0)
    f1 = 1.5 * (2 * np.exp(-x1 ** 2) - 1)
    y = f0 + f1 + rng.normal(0, 0.5, n)
    data = {"x0": x0, "x1": x1, "y": y}
    fit = mb.gamboost("y ~ bbs(x0, df=4) + bbs(x1, df=4)", data=data,
                      control=mb.boost_control(400, 0.3))
    grid = {"x0": np.linspace(-1.9, 1.9, 25), "x1": np.linspace(-1.9, 1.9, 25)}
    out = fit.confint(method="refit", level=0.9, newdata=grid)
    assert out["method"] == "refit"
    # Match refit terms to the simulated components by variable name; the
    # truth is evaluated on the grid rows, and both bands and truth are
    # centred over the grid for the comparison.
    truths = {
        "x0": 2 * np.sin(1.5 * grid["x0"]),
        "x1": 1.5 * (2 * np.exp(-grid["x1"] ** 2) - 1),
    }
    truths = {v: t - t.mean() for v, t in truths.items()}
    covered = []
    for term in out["estimate"]:
        var = next(v for v in truths if v in term)
        est_mean = out["estimate"][term].mean()
        lo_c = out["lower"][term] - est_mean
        hi_c = out["upper"][term] - est_mean
        t = truths[var]
        covered.append(np.mean((lo_c <= t) & (t <= hi_c)))
    assert covered, "no refit terms matched the simulated components"
    coverage = float(np.mean(covered))
    assert coverage >= 0.88, coverage


def test_refit_refused_for_unsupported_family():
    rng = np.random.default_rng(0)
    n = 150
    data = {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Quantile(0.5),
                      control=mb.boost_control(50, 0.3))
    with pytest.raises(NotImplementedError, match="gam_refit supports"):
        fit.confint(method="refit")


def test_refit_held_terms_listed_not_refused():
    rng = np.random.default_rng(0)
    n = 200
    data = {"x": rng.normal(size=n), "z": rng.normal(size=n), "y": rng.normal(size=n)}
    fit = mb.gamboost("y ~ bbs(x, df=4) + btree(z)", data=data,
                      control=mb.boost_control(100, 0.3))
    out = fit.confint(method="refit", level=0.9)
    assert out["method"] == "refit"
    # btree cannot be refit: held at its boosted contribution, not an error.
    assert len(out["held"]) > 0


def test_glmboost_refit_refused_by_name():
    rng = np.random.default_rng(0)
    n = 150
    data = {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    fit = mb.glmboost("y ~ bols(x)", data=data, control=mb.boost_control(50, 0.3))
    with pytest.raises(NotImplementedError, match="gamboost"):
        fit.confint(method="refit")
    with pytest.warns(UserWarning, match="under-cover"):
        fit.confint(B=10, B_mstop=0, seed=0)


def test_bad_method_rejected():
    fit, _ = _fit()
    with pytest.raises(ValueError, match="method"):
        fit.confint(method="magic")
