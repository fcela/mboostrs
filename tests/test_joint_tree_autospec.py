"""Joint geographic trees are discoverable, budgeted and replayable."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.autospec import _joint_tree_step


def _geography(n=600, seed=7, smooth=False):
    rng = np.random.default_rng(seed)
    lon, lat = rng.uniform(-1, 1, (2, n))
    signal = 2 * lon + lat if smooth else 3 * np.where(lon * lat > 0, 1, -1)
    return {"lon": lon, "lat": lat, "y": signal + rng.normal(0, 0.2, n)}


def test_joint_tree_formula_matches_explicit_learner_and_predicts_new_rows():
    data = _geography()
    control = mb.boost_control(80, nu=0.1)
    formula = mb.gamboost("y ~ btree(lon, lat, maxdepth=6, selection=sse)",
                          data, control=control)
    explicit = mb.gamboost("y ~ .", data,
                           learners=[mb.Btree(["lon", "lat"], maxdepth=6, selection="sse")],
                           control=control)
    dot = mb.gamboost("y ~ btree(., maxdepth=6, selection=sse)", data, control=control)
    newdata = {"lon": np.array([-0.8, -0.8, 0.8, 0.8]),
               "lat": np.array([-0.8, 0.8, -0.8, 0.8])}
    np.testing.assert_allclose(formula.predict(newdata), explicit.predict(newdata), atol=1e-12)
    np.testing.assert_allclose(formula.predict(newdata), dot.predict(newdata), atol=1e-12)
    assert np.mean((formula.predict(newdata) - [3, -3, -3, 3]) ** 2) < 0.1


def test_autospec_discovers_joint_signal_without_marginal_effects():
    data = _geography()
    rec = mb.autospec(data, "y", family=mb.Gaussian(), budget=120, mstop=80,
                      preexploration="off", deselect=False)
    evidence = rec.evidence["joint_tree"]
    assert evidence["adopted"]
    assert "btree(lon, lat," in rec.formula
    assert len(rec.learners or []) == 0  # rec.fit replays the formula itself
    assert rec.fits <= 120
    assert len(evidence["candidates"]) == 2
    assert "joint_tree" in rec.measured
    assert "joint tree" in rec.report() and "separate test data" in rec.report()
    new = _geography(n=1000, seed=31)
    prediction = rec.fit(data).predict(new)
    assert np.mean((prediction - new["y"]) ** 2) < 0.2
    assert rec.control.nu == evidence["nu"]
    assert rec.mstop == rec.evidence["mstop"].chosen


def test_an_additive_search_can_disable_joint_replacement():
    rec = mb.autospec(_geography(), "y", family=mb.Gaussian(), budget=120,
                      preexploration="off", deselect=False, joint_tree=False)
    assert "joint_tree" not in rec.evidence
    assert "btree(lon, lat," not in rec.formula
    assert len(rec.learners) == 2


def test_search_session_forwards_the_joint_tree_switch():
    session = mb.SearchSession(_geography(), "y", family=mb.Gaussian())
    rec = session.run("autospec", budget=120, preexploration="off", joint_tree=False).value
    assert "joint_tree" not in rec.evidence


def _probe(data, *, budget=21, declared=(), mandatory=(), forbidden=(), heredity="none",
           scheme=None):
    terms = "bbs(lon, df=4.0) + bbs(lat, df=4.0)"
    if "z" in data:
        terms += " + bmono(z, constraint=increasing)"
    folds = mb.cv_weights(len(data["y"]), type="kfold", B=5, seed=0)
    columns = [SimpleNamespace(name=n, kind="numeric") for n in data if n != "y"]
    evidence = {}
    result = _joint_tree_step(
        data, "y", "y ~ " + terms, None, mb.Gaussian(), columns, evidence,
        folds if scheme is None else scheme.weights, scheme, 80, 0.1, None, None,
        budget, 0, declared=declared, mandatory=mandatory, forbidden=forbidden,
        heredity=heredity, risk_floor=0.0)
    return result, evidence["joint_tree"]


def test_joint_tree_declines_for_a_linear_surface():
    result, evidence = _probe(_geography(smooth=True))
    assert not evidence["adopted"]
    assert "bbs(lon" in result[0]
    assert len(evidence["candidates"]) == 2
    assert result[-1] == 21


@pytest.mark.parametrize("kwargs, reason", [
    ({"budget": 20}, "budget"),
    ({"declared": ("lon", "lat")}, "declared"),
    ({"mandatory": ("lon",)}, "declared"),
    ({"forbidden": (frozenset(("lon", "lat")),)}, "constraints"),
    ({"heredity": "strong"}, "constraints"),
])
def test_joint_tree_skips_without_spending_fits(kwargs, reason):
    result, evidence = _probe(_geography(), **kwargs)
    assert result[-1] == 0
    assert not evidence["candidates"]
    assert reason in evidence["reason"]


def test_joint_tree_preserves_a_declared_shape_and_its_replay():
    data = _geography()
    rng = np.random.default_rng(8)
    data["z"] = rng.uniform(-1, 1, len(data["y"]))
    data["y"] += data["z"]
    result, evidence = _probe(data, declared=("z",))
    assert evidence["adopted"]
    assert "bmono(z, constraint=increasing)" in result[0]
    assert "btree(lon, lat," in result[0]
    fit = mb.gamboost(result[0], data, control=mb.boost_control(result[2], nu=result[3]))
    assert np.all(np.isfinite(fit.predict(data)))


def test_joint_tree_passes_the_actual_fold_scheme_to_every_sweep(monkeypatch):
    """Unjudged future rows must not become evaluation rows during a replacement."""
    from mboostrs.folds import rolling_folds

    data = _geography()
    scheme = rolling_folds(len(data["y"]), B=5, initial=300, horizon=60)
    calls = []

    def sweep(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(risk=(10.0, 1.0), chosen=1, at_boundary=False,
                               fits=scheme.weights.shape[1], _fit=None)

    monkeypatch.setattr(sys.modules["mboostrs.autospec"], "probe_mstop", sweep)
    result, evidence = _probe(data, budget=3 * (scheme.weights.shape[1] + 2), scheme=scheme)
    assert len(calls) == 3
    assert all(call["fold_scheme"] is scheme for call in calls)
    assert len({call["largest"] for call in calls}) == 1
    assert len({call["nu"] for call in calls}) == 1
    assert not evidence["adopted"]
    assert result[-1] == 3 * (scheme.weights.shape[1] + 2)


def test_joint_tree_compares_selected_steps_and_keeps_the_incumbents_step_length(monkeypatch):
    calls = []

    def sweep(*args, **kwargs):
        calls.append(kwargs)
        # Step zero must not decide the comparison: tune_mstop deliberately
        # selects a positive iteration, even when the offset scores better.
        score = 1.0 if len(calls) == 1 else 0.7
        return SimpleNamespace(risk=(0.5, score), chosen=1, at_boundary=False,
                               fits=5, _fit=None)

    monkeypatch.setattr(sys.modules["mboostrs.autospec"], "probe_mstop", sweep)
    data = _geography()
    columns = [SimpleNamespace(name=n, kind="numeric") for n in ("lon", "lat")]
    evidence = {}
    result = _joint_tree_step(
        data, "y", "y ~ bbs(lon) + bbs(lat)", None, mb.Gaussian(), columns,
        evidence, mb.cv_weights(600, type="kfold", B=5), None, 80, 0.3,
        None, None, 21, 0, declared=(), mandatory=(), forbidden=(),
        heredity="none", risk_floor=0.0)
    assert calls[0]["nu"] == 0.3
    assert calls[1]["nu"] == calls[2]["nu"] == 0.1
    assert evidence["joint_tree"]["adopted"]
    assert evidence["joint_tree"]["relative_gain"] == pytest.approx(0.3)
    assert result[3] == 0.1


def _geography_with_main_effect(n=600, seed=7):
    rng = np.random.default_rng(seed)
    lon, lat, z = rng.uniform(-1, 1, (3, n))
    signal = 3 * np.where(lon * lat > 0, 1, -1) + z
    return {"lon": lon, "lat": lat, "z": z, "y": signal + rng.normal(0, 0.2, n)}


def _budgeted_search(budget):
    return mb.autospec(_geography_with_main_effect(), "y", family=mb.Gaussian(),
                       budget=budget, mstop=80, preexploration="off", deselect=False)


@pytest.mark.filterwarnings("ignore:optimal mstop")
def test_joint_tree_fits_are_reserved_before_the_earlier_steps_spend(monkeypatch):
    """#807: at budget 40 on five folds the earlier steps used to leave 20
    fits, one short of the 21 the comparison needs, so the step was skipped."""
    rec = _budgeted_search(40)
    evidence = rec.evidence["joint_tree"]
    assert evidence["reserved"] == 21
    assert len(evidence["candidates"]) == 2
    assert "joint_tree" in rec.measured
    assert rec.fits <= 40

    # Without the reservation the same search reaches the step short of fits.
    monkeypatch.setattr(sys.modules["mboostrs.autospec"], "_joint_tree_may_run",
                        lambda *args, **kwargs: False)
    unreserved = _budgeted_search(40)
    skipped = unreserved.evidence["joint_tree"]
    assert not skipped["candidates"]
    assert "insufficient budget" in skipped["reason"]
    assert "reserved" not in skipped
    assert 40 - unreserved.fits < 21


@pytest.mark.parametrize("budget", [60, 70])
@pytest.mark.filterwarnings("ignore:optimal mstop")
def test_reservation_yields_to_the_coordinate_surface_step(budget):
    """#864: at budgets 60 and 70 the reservation used to leave the
    coordinate-surface step unpaid -- it skipped for budget while the
    joint tree was adopted, at a wide held-out cost on this fixture
    (0.477539 with the reservation against 0.112311 without it on a fresh
    sample). The reservation now yields to the surface step, which is
    scored and adopted; the joint tree then keeps its own budget check."""
    rec = _budgeted_search(budget)
    surface = rec.evidence["coordinate_surface"]
    assert surface["candidates"]
    assert surface["adopted"]
    assert surface["joint_reserve_released"] == 21
    assert "btree(lon, lat, maxdepth=3)" in rec.formula
    assert "bbs(z" in rec.formula
    joint = rec.evidence["joint_tree"]
    assert not joint["candidates"]
    assert "insufficient budget" in joint["reason"]
    assert "reserved" not in joint
    assert rec.fits <= budget
    new = _geography_with_main_effect(n=1000, seed=31)
    prediction = rec.fit(_geography_with_main_effect()).predict(new)
    assert np.mean((prediction - new["y"]) ** 2) < 0.2


@pytest.mark.filterwarnings("ignore:optimal mstop")
def test_joint_tree_skip_is_recorded_when_the_budget_cannot_hold_the_reservation():
    # After the depth sweep 19 fits remain: fewer than the 21 the comparison
    # needs plus one five-fold sweep for the earlier steps.
    rec = _budgeted_search(24)
    evidence = rec.evidence["joint_tree"]
    assert "reserved" not in evidence
    assert not evidence["candidates"]
    assert "insufficient budget" in evidence["reason"]
    assert rec.fits <= 24


@pytest.mark.parametrize("kwargs", [
    {"declared": ("lon",)},
    {"mandatory": ("lat",)},
    {"forbidden": (frozenset(("lon", "lat")),)},
    {"heredity": "weak"},
])
def test_no_reservation_when_the_joint_tree_step_would_refuse(kwargs):
    from mboostrs.autospec import _joint_tree_may_run

    columns = [SimpleNamespace(name=n, kind="numeric") for n in ("lon", "lat", "z")]
    terms = ["bbs(lon, df=4.0)", "bbs(lat, df=4.0)", "bbs(z, df=4.0)"]
    settings = {"declared": (), "mandatory": (), "forbidden": (), "heredity": "none"}
    assert _joint_tree_may_run(columns, terms, mb.Gaussian(), **settings)
    assert not _joint_tree_may_run(columns, terms, mb.Poisson(), **settings)
    settings.update(kwargs)
    assert not _joint_tree_may_run(columns, terms, mb.Gaussian(), **settings)


@pytest.mark.parametrize("lat_term", [
    "brandom(lat, df=4.0)",
    "bmono(lat, constraint=increasing)",
    "bols(lat, by=z)",
])
def test_reservation_reads_the_call_sites_column_set(lat_term):
    """#880: a coordinate whose term the joint tree does not replace leaves
    no pair for the call site, so the reservation check refuses as well."""
    from mboostrs.autospec import _joint_tree_columns, _joint_tree_may_run

    columns = [SimpleNamespace(name=n, kind="numeric") for n in ("lon", "lat", "z")]
    terms = ["bbs(lon, df=4.0)", lat_term, "bbs(z, df=4.0)"]
    assert [c.name for c in _joint_tree_columns(columns, terms)] == ["lon", "z"]
    assert not _joint_tree_may_run(columns, terms, mb.Gaussian(), declared=(),
                                   mandatory=(), forbidden=(), heredity="none")


def _geography_with_coded_latitude(n=600, seed=7):
    # Six latitude bands coded 10-15: autospec reads a whole-number column
    # with 3-12 distinct values as group labels and gives it a brandom term
    # (codes starting at 0 or 1 would read as a count and get a bbs term).
    rng = np.random.default_rng(seed)
    lon, z = rng.uniform(-1, 1, (2, n))
    lat = rng.integers(10, 16, n).astype(float)
    signal = 3 * np.where(lon * (lat - 12.5) > 0, 1, -1) + z
    return {"lon": lon, "lat": lat, "z": z, "y": signal + rng.normal(0, 0.2, n)}


@pytest.mark.filterwarnings("ignore:optimal mstop")
def test_no_reservation_when_a_coordinate_is_coded_as_groups(monkeypatch):
    """#880: the reservation check used every numeric column, so it reserved
    fits for a step the call site then refused because ``brandom(lat)`` is
    outside the joint tree's column set."""
    module = sys.modules["mboostrs.autospec"]
    original = module._joint_tree_may_run
    decisions = []

    def spy(*args, **kwargs):
        decisions.append(original(*args, **kwargs))
        return decisions[-1]

    monkeypatch.setattr(module, "_joint_tree_may_run", spy)
    rec = mb.autospec(_geography_with_coded_latitude(), "y", family=mb.Gaussian(),
                      budget=40, mstop=80, preexploration="off", deselect=False)
    assert "brandom(lat" in rec.formula
    assert decisions == [False]
    assert "reserved" not in rec.evidence.get("joint_tree", {})
    assert rec.fits <= 40


@pytest.mark.parametrize("failure_call, charged", [(1, 7), (2, 21)])
def test_failed_joint_sweeps_are_accounted_and_do_not_replace_the_incumbent(
        monkeypatch, failure_call, charged):
    calls = []

    def sweep(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == failure_call:
            return None
        return SimpleNamespace(risk=(2.0, 1.0), chosen=1, at_boundary=False,
                               fits=5, _fit=None)

    monkeypatch.setattr(sys.modules["mboostrs.autospec"], "probe_mstop", sweep)
    result, evidence = _probe(_geography())
    assert not evidence["adopted"]
    assert result[-1] == charged
    assert "bbs(lon" in result[0]
