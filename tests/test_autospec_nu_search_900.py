"""The bounded step-length search on the finished specification (#900).

The autospec route used to fit every probe at one step length and read
only the step count off the path. `probe_nu_search` re-reads the finished
specification at a small grid of other step lengths, one fold path per
value, charged to the same fit budget; `autospec` adopts the measured
best. These tests pin the grid, the budget counting, and the adoption
rule (a candidate must clear a stated margin over the incumbent, so a
flat-curve tie at a compensating step count changes nothing).
"""
import numpy as np
import pytest

import mboostrs as mb
from mboostrs._autospec import MstopEvidence, probe_mstop, probe_nu_search


def _evidence(chosen, risk_at_chosen, *, largest=100, fits=5):
    risk = np.full(largest + 1, risk_at_chosen + 1.0)
    risk[chosen] = risk_at_chosen
    return MstopEvidence(chosen=chosen, grid=tuple(range(largest + 1)),
                         risk=tuple(float(v) for v in risk),
                         at_boundary=False, fits=fits, sentence="fake")


def _scripted_probe(risks, calls):
    def fake(data, response, formula, family, *, folds, largest, nu,
             env=None, extend=0, fold_scheme=None):
        calls.append(nu)
        return _evidence(10, risks[nu], largest=largest)
    return fake


def _run_search(monkeypatch, risks, *, incumbent_risk=1.0, max_fits=None):
    calls = []
    monkeypatch.setattr("mboostrs._autospec.probe_mstop",
                        _scripted_probe(risks, calls))
    data = {"x": np.arange(20.0), "y": np.arange(20.0)}
    folds = np.ones((20, 5))
    incumbent = _evidence(40, incumbent_risk)
    best, best_nu, spent, records = probe_nu_search(
        data, "y", "y ~ bols(x)", mb.Gaussian(), folds=folds,
        incumbent=incumbent, nu=0.3, max_fits=max_fits)
    return best, best_nu, spent, records, calls


def test_the_search_probes_every_grid_value_and_counts_each_path(monkeypatch):
    risks = {0.1: 0.99, 0.2: 0.98, 0.5: 0.97, 0.7: 0.96, 1.0: 0.95}
    _best, _nu, spent, records, calls = _run_search(monkeypatch, risks)
    assert calls == [0.1, 0.2, 0.5, 0.7, 1.0]
    assert [r["nu"] for r in records] == [0.3, 0.1, 0.2, 0.5, 0.7, 1.0]
    assert records[0]["incumbent"] and records[0]["fits"] == 0
    # One path per fold per probed value: five folds, five fits each.
    assert all(r["fits"] == 5 for r in records[1:])
    assert spent == sum(r["fits"] for r in records) == 25


def test_the_search_picks_the_cv_best_step_length(monkeypatch):
    risks = {0.1: 0.99, 0.2: 0.98, 0.5: 0.90, 0.7: 0.96, 1.0: 0.95}
    best, best_nu, _spent, _records, _calls = _run_search(monkeypatch, risks)
    assert best_nu == 0.5
    assert best.chosen == 10
    assert best.risk[best.chosen] == pytest.approx(0.90)


def test_a_gain_below_the_margin_keeps_the_incumbent(monkeypatch):
    # The best candidate is lower by 5e-5 on a risk of 1.0: below the
    # 1e-4 * max(1, |risk|) margin, so the step length stands.
    risks = {0.1: 1.01, 0.2: 1.01, 0.5: 0.99995, 0.7: 1.01, 1.0: 1.01}
    best, best_nu, _spent, _records, _calls = _run_search(monkeypatch, risks)
    assert best_nu == 0.3
    assert best.chosen == 40


def test_the_search_stops_when_the_budget_cannot_pay(monkeypatch):
    risks = {0.1: 0.99, 0.2: 0.98, 0.5: 0.97, 0.7: 0.96, 1.0: 0.95}
    _best, _nu, spent, records, calls = _run_search(
        monkeypatch, risks, max_fits=10)
    assert calls == [0.1, 0.2]
    assert spent == 10
    assert len(records) == 3


def test_the_search_skips_the_incumbents_own_step_length(monkeypatch):
    # When the route's step length was lengthened to 0.5, the grid value
    # 0.5 is the incumbent and is not probed again.
    calls = []
    risks = {0.1: 0.99, 0.2: 0.98, 0.7: 0.96, 1.0: 0.95}
    monkeypatch.setattr("mboostrs._autospec.probe_mstop",
                        _scripted_probe(risks, calls))
    data = {"x": np.arange(20.0), "y": np.arange(20.0)}
    incumbent = _evidence(40, 1.0)
    _best, best_nu, spent, records = probe_nu_search(
        data, "y", "y ~ bols(x)", mb.Gaussian(), folds=np.ones((20, 5)),
        incumbent=incumbent, nu=0.5)
    assert 0.5 not in calls
    assert records[0]["nu"] == 0.5
    assert spent == 20


def _sin_data():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    return {"x": x, "y": np.sin(1.5 * x) * 3 + rng.normal(scale=0.2, size=n)}


def test_autospec_records_the_search_and_charges_its_fits():
    rec = mb.autospec(_sin_data(), "y", budget=400, mstop=100)
    search = rec.evidence["nu_search"]
    assert len({r["nu"] for r in search["candidates"]}) >= 4
    assert search["fits"] == sum(r["fits"] for r in search["candidates"])
    assert search["fits"] > 0
    assert rec.fits <= 400
    assert rec.control.nu == search["chosen_nu"]
    assert rec.mstop == rec.evidence["mstop"].chosen
    if search["adopted"]:
        assert rec.control.nu != search["incumbent_nu"]
        chosen = next(r for r in search["candidates"]
                      if r["nu"] == search["chosen_nu"])
        assert chosen["risk"] < search["candidates"][0]["risk"]


def test_an_explicit_step_length_is_never_searched():
    rec = mb.autospec(_sin_data(), "y", budget=400, mstop=100, nu=0.1)
    assert rec.control.nu == 0.1
    assert "nu_search" not in rec.evidence


def test_a_budget_with_no_room_skips_the_search():
    # 25 fits: the family and spread probes (the spread probe validates a
    # tree scale equation) and the stopping sweep leave no room for it.
    rec = mb.autospec(_sin_data(), "y", budget=25, mstop=100)
    assert "nu_search" not in rec.evidence
    assert rec.fits <= 25 + 25  # the search adds nothing when it cannot pay


def test_probe_nu_search_on_real_fits_varies_the_step_length():
    data = _sin_data()
    folds = mb.cv_weights(300, type="kfold", B=5, seed=0)
    incumbent = probe_mstop(data, "y", "y ~ bbs(x, df=4)", mb.Gaussian(),
                            folds=folds, largest=200, nu=0.3, extend=0)
    assert incumbent is not None
    best, best_nu, spent, records = probe_nu_search(
        data, "y", "y ~ bbs(x, df=4)", mb.Gaussian(), folds=folds,
        incumbent=incumbent, nu=0.3)
    assert {r["nu"] for r in records} == {0.3, 0.1, 0.2, 0.5, 0.7, 1.0}
    assert spent == sum(r["fits"] for r in records)
    assert best.risk[best.chosen] <= incumbent.risk[incumbent.chosen] + 1e-12
    assert best_nu in {r["nu"] for r in records}
