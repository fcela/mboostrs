"""autospec compares its finished specification with simple models (#804).

After the search, the specification it settled on is scored on the same
folds as two prespecified rivals: the starting specification with each
one-column smooth written as ``bols`` ("linear"), and the starting
specification itself ("additive"). A rival replaces the searched
specification only when its held-out risk is lower by more than one standard
error of the paired fold difference.
"""

import importlib

import numpy as np

import mboostrs as mb

module = importlib.import_module("mboostrs.autospec")


def _linear_data(seed, n=300):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    return {"y": 3.0 * x1 - 2.0 * x2 + rng.normal(scale=0.3, size=n),
            "x1": x1, "x2": x2}


def _curved_data(seed, n=300):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    return {"y": 2.0 * np.sin(2.0 * x1) + x2 ** 2 + rng.normal(scale=0.3, size=n),
            "x1": x1, "x2": x2}


def test_linear_rival_writes_one_column_smooths_as_bols():
    rival = module._linear_rival(
        "y ~ bbs(x, df=4) + btree(z) + bbs(log(w), df=4) + brandom(g, df=4) "
        "+ btree(a, b, maxdepth=3) + bmono(m, constraint='increasing')")
    assert rival == ("y ~ bols(x) + bols(z) + bols(log(w)) + brandom(g, df=4) "
                     "+ btree(a, b, maxdepth=3) + bmono(m, constraint='increasing')")
    # The two parts of a Multinomial decomposition become one linear term.
    assert module._linear_rival("y ~ bols(x) + bbs(x, center=TRUE, df=1)") == "y ~ bols(x)"


def test_a_simple_model_that_clearly_wins_replaces_the_searched_one(monkeypatch):
    """Stumps on a linear signal. The shape probe is switched off so the
    search finishes with the stumps it started from; the linear rival has far
    less held-out risk and replaces them."""
    monkeypatch.setattr(module, "probe_shapes", lambda *args, **kwargs: ((), 0))
    data = _linear_data(1)
    stumps = "y ~ btree(x1) + btree(x2)"
    rec = mb.autospec(data, "y", smooth="btree", budget=200, seed=0,
                      preexploration="off", deselect=False, joint_tree=False)
    record = rec.evidence["whole_model"]

    assert record["compared"]
    assert record["chosen_formula"] == stumps
    linear = record["rivals"]["linear"]
    assert linear["formula"] == "y ~ bols(x1) + bols(x2)"
    assert record["rivals"]["additive"]["same_as_chosen"]
    assert linear["difference"] < 0
    assert -linear["difference"] > linear["difference_se"] > 0
    assert linear["beats_chosen"]
    assert record["replaced"] and record["winner"] == "linear"
    assert rec.formula == "y ~ bols(x1) + bols(x2)"
    assert rec.mstop == linear["mstop"] == rec.control.mstop
    assert "whole_model" in rec.measured
    assert rec.fits <= 200
    assert "replaces the searched specification" in rec.report()
    # The returned specification fits as recorded.
    fit = rec.fit(data)
    assert fit.mstop == rec.mstop


def test_the_searched_model_stands_when_it_beats_the_simple_models():
    data = _curved_data(2)
    rec = mb.autospec(data, "y", budget=200, seed=0, preexploration="off")
    record = rec.evidence["whole_model"]

    assert record["compared"]
    assert not record["replaced"] and record["winner"] == "searched"
    assert rec.formula == record["chosen_formula"]
    linear = record["rivals"]["linear"]
    assert linear["formula"] == "y ~ bols(x1) + bols(x2)"
    # The linear rival misses both curves and loses by many standard errors.
    assert linear["difference"] > 0 and not linear["beats_chosen"]
    assert np.isfinite(linear["difference_se"]) and linear["difference_se"] > 0
    assert record["folds"] == 5
    assert "searched specification stands" in record["sentence"]
    assert rec.fits <= 200


def test_a_distributional_recommendation_states_why_there_is_no_record():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.5, 1.5, 400)
    z = rng.normal(size=400)
    mu = np.exp(0.4 + 1.2 * (x > 0.3) + 0.2 * z)
    shape = np.exp(0.8 + 0.7 * x)
    data = {"y": rng.gamma(shape, mu / shape), "x": x, "z": z}
    rec = mb.autospec(data, "y", budget=200, mstop=100, seed=0)

    assert isinstance(rec.family, mb.GammaLSS)
    record = rec.evidence["whole_model"]
    assert record["compared"] is False
    assert "log score" in record["reason"]
    assert "finished_routes" in record["reason"]
    assert "whole model: not compared" in rec.report()
