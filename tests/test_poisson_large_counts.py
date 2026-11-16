"""Poisson fits on a count response in the tens to hundreds.

On 2,000 rows with a median of about 50 and a maximum of about 300,
``autospec(data, "y", budget=0)`` inferred Poisson and recommended the fixed
step nu = 0.3. No fit runs at budget 0, so nothing checked that step, and
``rec.fit(data)`` raised "risk became non-finite". The Poisson fixed step is
not scale-invariant: its update is nu times a least-squares fit to
``y - exp(f)``, which is on the count scale, so the linear predictor moves by
an amount that grows with the mean.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._errors import MboostrsRuntimeError


def _large_counts(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    mu = np.exp(3.9 + 0.8 * np.sin(x1) + 0.5 * x2)
    # Integer dtype, so autospec reads the response as counts without the
    # whole-number-stored-as-float warning.
    return {"y": rng.poisson(mu), "x1": x1, "x2": x2}


def _small_counts(n=600, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    return {"y": rng.poisson(3 * np.exp(0.6 * x)), "x": x}


def test_budget_zero_recommendation_on_large_counts_fits():
    data = _large_counts()
    rec = mb.autospec(data, "y", budget=0)
    assert type(rec.family).__name__ == "Poisson"
    assert rec.control.step == "line_search"
    fit = rec.fit(data)
    assert np.isfinite(fit.risk_).all()
    assert fit.risk_[-1] < fit.risk_[0]


def test_budget_zero_recommendation_records_the_reason():
    rec = mb.autospec(_large_counts(), "y", budget=0)
    assert "line_search" in rec.evidence["step"]
    assert "Poisson" in rec.evidence["step"]


def test_budget_zero_small_counts_also_take_the_line_search():
    # At a mean near 3 the fixed step did not overflow, but its held-out
    # deviance was well above the line search's, so the rule covers it too.
    rec = mb.autospec(_small_counts(), "y", budget=0)
    assert type(rec.family).__name__ == "Poisson"
    assert rec.control.step == "line_search"


def test_an_explicit_fixed_step_is_honoured_by_autospec():
    rec = mb.autospec(_large_counts(), "y", budget=0, step="fixed")
    assert rec.control.step == "fixed"


def test_fixed_step_overflow_names_iteration_family_and_remedy():
    data = _large_counts()
    with pytest.raises(MboostrsRuntimeError) as info:
        mb.gamboost("y ~ bbs(x1) + bbs(x2)", data=data, family=mb.Poisson(),
                    control=mb.boost_control(100, 0.3))
    message = str(info.value)
    assert "Poisson risk became non-finite after boosting iteration" in message
    assert "nu = 0.3" in message
    assert 'step="line_search"' in message


def test_line_search_fit_on_large_counts_stays_finite_and_descends():
    data = _large_counts()
    fit = mb.gamboost("y ~ bbs(x1) + bbs(x2)", data=data, family=mb.Poisson(),
                      control=mb.boost_control(100, 0.3, step="line_search"))
    assert np.isfinite(fit.risk_).all()
    assert np.max(np.diff(fit.risk_)) <= 1e-9 * np.abs(fit.risk_[0])
    fitted = np.asarray(fit.predict(type="response"))
    assert np.all(np.isfinite(fitted)) and np.all(fitted > 0)


@pytest.mark.r_parity
def test_r_mboost_also_stops_on_the_fixed_step_for_large_counts():
    """R mboost stops with "Infinite residuals" at the same step, so the
    fixed-step overflow is R's behaviour and not a regression of this port."""
    import rpy2.robjects as ro

    data = _large_counts()
    ro.r("suppressMessages(library(mboost))")
    ro.globalenv["d"] = ro.DataFrame({
        "y": ro.FloatVector(data["y"].astype(float)),
        "x1": ro.FloatVector(data["x1"]),
        "x2": ro.FloatVector(data["x2"]),
    })
    with pytest.raises(Exception, match="Infinite residuals"):
        ro.r("mboost::gamboost(y ~ bbs(x1) + bbs(x2), data = d, family = mboost::Poisson(),"
             " control = mboost::boost_control(mstop = 100, nu = 0.3))")
    with pytest.raises(MboostrsRuntimeError, match="non-finite"):
        mb.gamboost("y ~ bbs(x1) + bbs(x2)", data=data, family=mb.Poisson(),
                    control=mb.boost_control(100, 0.3))
