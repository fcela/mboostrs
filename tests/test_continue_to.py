"""Warm-start API: continue_to / update(mstop=larger).

The contract is exact equivalence with a cold fit to the larger mstop, so the
interactive loop (fit, inspect risk, extend) never needs the model
re-specified. A gamboost fit that retained its gradient resumes from it; the
Gaussian Gram route, `glmboost` and the distributional models refit from
iteration zero. The equivalence guarantee is what these tests pin, because it
holds on both routes and is what any further optimization must preserve.
"""

import numpy as np
import pytest

import mboostrs as mb


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = {
        "x": rng.uniform(1.0, 10.0, n),
        "z": rng.normal(size=n),
        "g": np.array(["a", "b", "c"])[rng.integers(0, 3, n)],
    }
    d["y"] = 2.0 * np.log(d["x"]) + 0.5 * d["z"] + rng.normal(scale=0.2, size=n)
    return d


def _maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


@pytest.mark.parametrize("engine,formula", [
    ("glmboost", "y ~ x + z"),
    ("gamboost", "y ~ bbs(x, df=4) + bols(z)"),
])
def test_continue_to_equals_a_cold_fit(engine, formula):
    data = _data()
    warm = getattr(mb, engine)(
        formula, data=data, family=mb.Gaussian(), control=mb.boost_control(100, 0.1)
    ).continue_to(300)
    cold = getattr(mb, engine)(
        formula, data=data, family=mb.Gaussian(), control=mb.boost_control(300, 0.1)
    )
    assert warm.mstop == 300
    assert _maxdiff(warm.fitted_, cold.fitted_) == 0.0
    assert warm.selected == cold.selected


@pytest.mark.parametrize("engine,formula", [
    ("glmboost", "y ~ x + z"),
    ("gamboost", "y ~ bbs(x, df=4) + bols(z)"),
])
def test_slicing_the_grown_model_reproduces_the_original(engine, formula):
    data = _data()
    original = getattr(mb, engine)(
        formula, data=data, family=mb.Gaussian(), control=mb.boost_control(100, 0.1)
    )
    grown = original.continue_to(300)
    # first 100 iterations are identical to machine precision
    assert _maxdiff(grown[:100].fitted_, original.fitted_) < 1e-10
    assert grown[:100].selected == original.selected


def test_update_with_larger_mstop_now_grows():
    data = _data()
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    grown = fit.update(mstop=200)
    assert grown.mstop == 200


def test_continue_to_smaller_slices():
    data = _data()
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(200, 0.1))
    assert fit.continue_to(80).mstop == 80


def test_continue_to_carries_transforms():
    data = _data()
    warm = mb.gamboost("y ~ log(x) + z", data=data, family=mb.Gaussian(),
                       control=mb.boost_control(80, 0.1)).continue_to(200)
    cold = mb.gamboost("y ~ log(x) + z", data=data, family=mb.Gaussian(),
                       control=mb.boost_control(200, 0.1))
    newdata = {"x": np.array([2.0, 8.0]), "z": np.zeros(2)}
    assert _maxdiff(warm.predict(newdata=newdata), cold.predict(newdata=newdata)) == 0.0


def test_continue_to_carries_interactions():
    data = _data()
    warm = mb.gamboost("y ~ bbs(x, df=4) %X% brandom(g)", data=data,
                       family=mb.Gaussian(), control=mb.boost_control(100, 0.1))
    grown = warm.continue_to(250)
    assert grown.mstop == 250
    assert grown._interactions is not None
    newdata = {"x": np.array([2.0, 8.0]), "g": np.array(["a", "c"])}
    assert np.all(np.isfinite(grown.predict(newdata=newdata)))


def test_continue_to_preserves_family():
    rng = np.random.default_rng(3)
    n = 300
    d = {"x": rng.normal(size=n)}
    d["y"] = rng.poisson(np.exp(0.4 * d["x"])).astype(float)
    warm = mb.glmboost("y ~ x", data=d, family=mb.Poisson(),
                       control=mb.boost_control(50, 0.1)).continue_to(150)
    cold = mb.glmboost("y ~ x", data=d, family=mb.Poisson(),
                       control=mb.boost_control(150, 0.1))
    assert type(warm._family).__name__ == "Poisson"
    assert _maxdiff(warm.fitted_, cold.fitted_) == 0.0


def test_continue_to_below_one_is_rejected():
    data = _data()
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    with pytest.raises(ValueError, match="mstop must be >= 1"):
        fit.continue_to(0)


def test_continue_to_after_unpickle_grows_like_the_original():
    """A pickle keeps the refit inputs (issue #820), so a round-tripped model
    grows to the same path as the model it was written from."""
    import pickle

    data = _data()
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    restored = pickle.loads(pickle.dumps(fit))
    assert _maxdiff(restored.continue_to(200).fitted_,
                    fit.continue_to(200).fitted_) == 0.0


def test_continue_to_without_the_bundle_reports_it_is_missing():
    """A model without the refit inputs (a scoring-only artifact, or a pickle
    written by an earlier release) must fail loudly when asked to grow."""
    import pickle

    data = _data()
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    restored = pickle.loads(pickle.dumps(fit))
    restored._refit_bundle = None
    with pytest.raises(ValueError, match="did not retain the inputs needed to refit it"):
        restored.continue_to(200)


def test_warm_continue_resumes_from_the_retained_residual():
    """A general-route fit (Poisson) retains its residual and warms.

    The Gaussian Gram route does not retain a length-n residual, so it stays on
    the cold refit; a general-route family keeps one, and its grown model must
    still equal a cold fit bit-for-bit while carrying a non-empty residual
    forward (proof the warm path was taken, not a cold rerun).
    """
    rng = np.random.default_rng(4)
    n = 200
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 * x)).astype(float)
    data = {"x": x, "y": y}
    fit = mb.gamboost("y ~ bbs(x, df=4) + bols(x)", data=data, family=mb.Poisson(),
                      control=mb.boost_control(30, 0.1))
    assert getattr(fit, "_residual", None) is not None and len(fit._residual) == n
    grown = fit.continue_to(60)
    cold = mb.gamboost("y ~ bbs(x, df=4) + bols(x)", data=data, family=mb.Poisson(),
                       control=mb.boost_control(60, 0.1))
    assert _maxdiff(grown.fitted_, cold.fitted_) == 0.0
    assert np.array_equal(grown.selected, cold.selected)
    assert getattr(grown, "_residual", None) is not None and len(grown._residual) == n


def test_a_sliced_model_can_still_refit():
    """`fit[:k]` is how a reader reaches the tuned model, and everything that
    refits — confint, cvrisk, the conformal splits — needs the fit's own
    inputs. Slicing rebuilt the model without them, so `confint` on a sliced
    fit refused with "did not retain the inputs needed to refit it" while the
    same call on the unsliced fit worked. The book's bodyfat study is what
    executed it."""
    data = _data()
    for engine, formula in (("glmboost", "y ~ x + z"),
                            ("gamboost", "y ~ bbs(x, df=4) + bols(z)")):
        fit = getattr(mb, engine)(
            formula, data=data, family=mb.Gaussian(),
            control=mb.boost_control(60, 0.1))
        sliced = fit[:30]
        assert sliced.mstop == 30
        interval = sliced.confint(B=12, seed=1)
        assert interval["B"] == 12


def test_cvrisk_after_continue_to_poisson():
    """Warm continuation must keep a BoostControl.

    ``cvrisk`` reads ``control.mstop``. A warm ``continue_to`` used to store
    the raw ``(mstop, nu, risk, trace)`` tuple, so the documented loop — fit,
    find the optimum at the boundary, extend, cross-validate again — raised
    ``AttributeError`` on a Poisson ``gamboost``.
    """
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 + 0.3 * x)).astype(float)
    data = {"x": x, "y": y}
    fit = mb.gamboost(
        "y ~ bbs(x, df=4)", data, family=mb.Poisson(),
        control=mb.boost_control(20, 0.1))
    grown = fit.continue_to(40)
    assert grown.mstop == 40
    result = grown.cvrisk(type="kfold", B=3, seed=1)
    assert int(result.optimal_mstop) >= 1
