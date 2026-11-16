"""CQR's internal step-count search must not leak its doubling warnings."""

import warnings

import numpy as np

import mboostrs as mb
from mboostrs import conformal


def _heteroscedastic_fit(seed=0, n=400):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    sd = 0.3 + 1.5 * np.abs(x1)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + sd * rng.standard_normal(n)
    data = {"y": y, "x1": x1, "x2": x2}
    return mb.gamboost("y ~ bbs(x1, df=4) + bols(x2)", data, family=mb.Gaussian())


def _search_range_warnings(record):
    return [w for w in record if "search range was exhausted" in str(w.message)]


def test_cqr_doubling_emits_no_search_range_warning():
    fit = _heteroscedastic_fit()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.conformal_interval(fit, level=0.9, seed=1, method="cqr")
    assert _search_range_warnings(record) == []


def test_cqr_warns_when_quantile_step_cap_is_reached(monkeypatch):
    fit = _heteroscedastic_fit()
    # The search starts at 200 and the cap is 200, so the first grid point is the cap.
    monkeypatch.setattr(conformal._rust, "conformal_tuned_mstop_start", lambda model_mstop: 200)
    monkeypatch.setattr(conformal, "_QUANTILE_MSTOP_CAP", 200)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.conformal_interval(fit, level=0.9, seed=1, method="cqr")
    capped = [w for w in record if "reached the cap" in str(w.message)]
    assert _search_range_warnings(record) == []
    assert len(capped) == 1


def test_cap_summary_is_per_call_and_respects_warning_errors(monkeypatch):
    import pytest

    fit = _heteroscedastic_fit()
    monkeypatch.setattr(conformal._rust, "conformal_tuned_mstop_start", lambda _: 20)
    monkeypatch.setattr(conformal, "_QUANTILE_MSTOP_CAP", 20)
    for _ in range(2):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            mb.conformal_interval(fit, method="cqr", seed=1)
        assert sum("reached the cap" in str(w.message) for w in record) == 1
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="CQR quantile step count reached the cap.*")
        with pytest.raises(UserWarning, match="reached the cap"):
            mb.conformal_interval(fit, method="cqr", seed=1)
