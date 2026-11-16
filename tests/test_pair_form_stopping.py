"""The specification adopted on CV must replay its own measured path (#535)."""
import warnings

import numpy as np
import pytest

import mboostrs as mb
import mboostrs._autospec as evidence_module


@pytest.mark.parametrize("kind", ["tensor", "joint_tree"])
def test_pair_form_replays_its_own_measured_stopping_path(monkeypatch, kind):
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(-1, 1, (n, 4))
    interaction = (1.5 * x[:, 0] * x[:, 1] if kind == "tensor"
                   else 3 * np.sign(x[:, 0] * x[:, 1]))
    y = np.sin(2 * x[:, 0]) + x[:, 1] + interaction + rng.normal(0, .3, n)
    data = {f"x{j + 1}": x[:, j] for j in range(4)} | {"y": y}
    original = evidence_module.score_coordinate_candidates
    comparisons = []

    def record(*args, **kwargs):
        scores, fits = original(*args, **kwargs)
        comparisons.append((scores, fits))
        return scores, fits

    monkeypatch.setattr(evidence_module, "score_coordinate_candidates", record)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", family=mb.Gaussian(), budget=150, seed=0,
                          interactions="adopt", preexploration="off")
    scores, fits = comparisons[-1]
    form = next(row for key, row in rec.evidence.items() if key.startswith("pair_form_"))
    assert form["winner"] == kind
    stopping = scores[kind].stopping
    assert stopping is not None
    assert rec.mstop == stopping.chosen
    assert rec.evidence["mstop"] is stopping
    assert rec.mstop > 0
    assert rec.fits <= 150
    assert fits == form["picker_fits"] == 15
    assert form["mstop"] == stopping.chosen
    assert form["search_cap"] == stopping.grid[-1]
    assert form["at_boundary"] == stopping.at_boundary
    assert stopping.risk[stopping.chosen] == pytest.approx(scores[kind].risk)

    # Compare the public recommendation replay with the exact full-data path
    # whose fold risks justified adoption, at that path's measured step.
    fitted = rec.fit(data)
    expected = stopping._fit.with_mstop(stopping.chosen)
    np.testing.assert_allclose(fitted.predict(data), expected.predict(data), atol=1e-12)
    assert set(fitted.base_learner_names()) == set(expected.base_learner_names())


def test_coordinate_candidate_failure_has_no_replay_path(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("failed path")

    monkeypatch.setattr(evidence_module, "_path_held_out_evidence", fail)
    scores, fits = evidence_module.score_coordinate_candidates(
        {}, "y", {"bad": "y ~ bbs(x)"}, mb.Gaussian(), np.ones((10, 2)), 20, .1)
    assert fits == 2
    assert not np.isfinite(scores["bad"].risk)
    assert scores["bad"].stopping is None


def test_coordinate_path_retains_structured_judge_and_selected_fold_risks(monkeypatch):
    from mboostrs.folds import rolling_folds
    from mboostrs.gamboost import GAMBoostModel

    rng = np.random.default_rng(20)
    x = rng.normal(size=120)
    data = {"x": x, "y": 2 * x + rng.normal(0, .5, 120)}
    scheme = rolling_folds(120, B=2, initial=60, horizon=20)
    original = GAMBoostModel.cvrisk
    calls = []

    def record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append((kwargs["folds"], result))
        return result

    monkeypatch.setattr(GAMBoostModel, "cvrisk", record)
    scores, fits = evidence_module.score_coordinate_candidates(
        data, "y", {"linear": "y ~ bols(x)"}, mb.Gaussian(), scheme.weights, 20, .1,
        fold_scheme=scheme)
    assert len(calls) == 1 and calls[0][0] is scheme
    cv = calls[0][1]
    row = scores["linear"]
    assert fits == 2
    assert row.stopping.chosen == cv.optimal_mstop > 0
    assert row.stopping.risk == tuple(cv.aggregate)
    np.testing.assert_array_equal(row.fold_risk, cv.risk_matrix[:, cv.optimal_mstop])


def test_candidate_path_cap_policy():
    # The candidate stopping search's cap: floored by the requested budget and
    # the house 200 (never only by an early additive optimum), grown from the
    # incumbent when the incumbent itself needed a long path, and ceiled by
    # _MSTOP_EXTENSION_CAP (#536).
    from mboostrs._autospec import _MSTOP_EXTENSION_CAP
    from mboostrs.autospec import _candidate_path_cap

    assert _candidate_path_cap(200, 9) == 200        # the floor binds
    assert _candidate_path_cap(500, 10) == 500       # the requested budget binds
    assert _candidate_path_cap(200, 3000) == 6000    # the incumbent grows the cap
    assert _candidate_path_cap(200, 100_000) == _MSTOP_EXTENSION_CAP


def _boundary_data(seed=7, n=1500):
    """A weak additive signal and a strong boundary interaction.

    The additive specification stops after a handful of steps -- there is
    almost nothing for it to fit -- while the joint-tree candidate needs a
    longer path to build the sign(x1 * x2) boundary. This is the frozen
    joint_boundary structure at unit scale (#536)."""
    rng = np.random.default_rng(seed)
    columns = {f"x{j + 1}": rng.uniform(-1, 1, n) for j in range(6)}
    y = (3.0 * np.sign(columns["x1"] * columns["x2"]) + 0.5 * columns["x3"]
         + 0.5 * rng.normal(size=n))
    return dict(columns, y=y)


def _adopt(data, **kwargs):
    """Run autospec with the picker's `score_coordinate_candidates` recorded."""
    calls = []
    original = evidence_module.score_coordinate_candidates

    def record(*args, **kw):
        calls.append(args)
        return original(*args, **kw)

    evidence_module.score_coordinate_candidates = record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return mb.autospec(data, "y", family=mb.Gaussian(), budget=150,
                               seed=0, interactions="adopt", preexploration="off",
                               mstop=200, **kwargs), calls
    finally:
        evidence_module.score_coordinate_candidates = original


def test_pair_form_candidates_are_searched_past_the_additive_early_stop():
    # #536: the picker used to cap every competing form at twice the additive
    # incumbent's own optimum. Here the incumbent stops at a handful of
    # steps, so that cap sat far below the requested budget, and the joint
    # candidate -- whose own optimum lies beyond it -- was judged on a path
    # it had not finished using.
    data = _boundary_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        additive = mb.autospec(data, "y", family=mb.Gaussian(), budget=150,
                               seed=0, interactions="report",
                               preexploration="off", mstop=200)
    # The incumbent converges early; twice its optimum sits below the floor.
    assert 2 * additive.mstop < 200

    rec, calls = _adopt(data)
    assert calls, "the competing pair-form picker did not run"
    largest = calls[0][5]
    # The policy: floored by the requested budget and the house 200, never
    # by the incumbent alone, and ceiled by the hard cap.
    assert largest >= max(200, 200)
    assert largest > 2 * additive.mstop
    assert largest <= 10_000

    form = next(row for key, row in rec.evidence.items()
                if key.startswith("pair_form_"))
    assert form["winner"] == "joint_tree"
    # The candidate was judged at its own optimum, which lies beyond the old
    # additive-derived cap, and its evidence records the search cap it used.
    assert form["mstop"] > 2 * additive.mstop
    assert form["search_cap"] == largest
    assert form["at_boundary"] is False
    assert "btree(x1, x2" in rec.formula
    # The cap sets path length, not fit count: one fit per candidate per fold.
    assert form["picker_fits"] == 15
    assert rec.fits <= 150


@pytest.mark.parametrize("seed", [11, 12])
def test_pair_form_controls_adopt_nothing(seed):
    # The raised cap creates no adoption where no interaction exists: on a
    # smooth-additive response and on pure noise the screen proposes no
    # competing pair, the picker never runs and the formula keeps its main
    # effects only.
    rng = np.random.default_rng(seed)
    n = 1200
    columns = {f"x{j + 1}": rng.uniform(-1, 1, n) for j in range(4)}
    additive = np.sin(2 * columns["x1"]) + columns["x2"] ** 2
    data = dict(columns, y=additive + 0.3 * rng.normal(size=n))
    noise = dict(columns, y=rng.normal(size=n))
    for frame in (data, noise):
        rec, calls = _adopt(frame)
        assert calls == []
        assert not any(key.startswith("pair_form_") for key in rec.evidence)
        assert rec.evidence.get("adopted_interactions", ()) == ()
        assert "%X%" not in rec.formula
        assert "btree(" not in rec.formula
