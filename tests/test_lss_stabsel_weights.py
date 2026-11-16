"""Weighted LSS stability selection uses the weights.

P0.1's remaining closure condition: "route weighted LSS stability selection
through the fitted model's effective weights. Its current refit arguments omit
`weights`, and its module-level SS and MB paths construct unit-weight folds.
Test that every SS pair sums to the effective weights and that non-unit
weights reach every LSS refit."

Both halves were true. `stabsel_glmboostLSS` and `stabsel_gamboostLSS` had no
`weights` parameter at all, and called
`make_folds(n, B, sampling_type=..., seed=...)` — unit half-samples whatever
the model was fitted on. `LSSModel._refit_arguments()` had no weights to pass
even if they had.

So a weighted distributional model's selection frequencies described a sample
the caller never fitted, and nothing said so: the run completed, the numbers
were plausible, and every fold silently weighted each row 1.

`cvrisk` was already right, by a different route — the method builds its folds
from `model_weights()` and hands them over, so the weights travel inside the
fold matrix. That asymmetry is why this one went unnoticed.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.stabsel import make_folds

N = 120


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    frame = {"x": rng.normal(size=N), "z": rng.normal(size=N)}
    frame["y"] = frame["x"] + rng.normal(scale=0.4, size=N)
    return frame


@pytest.fixture(scope="module")
def weights():
    return np.random.default_rng(5).uniform(0.5, 2.0, N)


def fit_lss(data, weights=None, kind="gamboostLSS"):
    formula = ("y ~ bbs(x, df=4) + bbs(z, df=4)" if kind == "gamboostLSS"
               else "y ~ x + z")
    entry = getattr(mb, kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return entry(formula, data, family=mb.GaussianLSS(),
                     control=mb.boost_control(30, 0.1), weights=weights)


@pytest.mark.parametrize("kind", ["glmboostLSS", "gamboostLSS"])
def test_the_entry_points_take_weights(kind):
    import inspect

    entry = getattr(mb, f"stabsel_{kind}")
    assert "weights" in inspect.signature(entry).parameters


def test_every_ss_pair_sums_to_the_effective_weights(data, weights):
    """The complementary-pairs contract, on a weighted sample: column `b` and
    column `B + b` together are the model's own weight vector."""
    fit = fit_lss(data, weights)
    effective = np.asarray(fit.weights, dtype=float)
    assert not np.allclose(effective, 1.0), "the fixture should be weighted"

    B = 6
    folds = make_folds(N, B, sampling_type="SS", weights=effective, seed=1)
    assert folds.shape == (N, 2 * B)
    for b in range(B):
        assert np.allclose(folds[:, b] + folds[:, B + b], effective), b


def test_a_weighted_model_passes_its_weights_to_stabsel(data, weights,
                                                        monkeypatch):
    """The route, checked at the call rather than through the answer: a
    selection result that happens to agree would prove nothing."""
    fit = fit_lss(data, weights)
    seen = {}

    import mboostrs.gamlss as gamlss

    original = gamlss.stabsel_gamboostLSS

    def recording(*args, **kwargs):
        seen["weights"] = kwargs.get("weights")
        return original(*args, **kwargs)

    monkeypatch.setattr(gamlss, "stabsel_gamboostLSS", recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit.stabsel(q=2, PFER=1, B=4, seed=1)

    assert seen["weights"] is not None, "stabsel was called with no weights"
    assert np.allclose(np.asarray(seen["weights"], dtype=float),
                       np.asarray(fit.weights, dtype=float))


@pytest.mark.parametrize("sampling_type", ["SS", "MB"])
def test_weights_reach_the_folds_for_both_sampling_types(data, weights,
                                                         sampling_type):
    """The condition names the SS *and* MB paths. Both built unit folds."""
    fit = fit_lss(data, weights)
    effective = np.asarray(fit.weights, dtype=float)
    folds = make_folds(N, 5, sampling_type=sampling_type, weights=effective,
                       seed=2)
    drawn = folds[folds > 0]
    assert not np.allclose(drawn, 1.0), (
        f"{sampling_type} folds are unit-valued, so the weights were dropped")
    assert np.all(folds <= effective[:, None] + 1e-12)


@pytest.mark.parametrize("kind", ["glmboostLSS", "gamboostLSS"])
def test_a_weighted_run_completes_and_reports_its_fits(data, weights, kind):
    fit = fit_lss(data, weights, kind=kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit.stabsel(q=2, PFER=1, B=5, seed=1)
    assert result.n_fits == 10, "SS fits both halves of every pair"
    assert result.B == 5, "the bound is computed on the pair count"
    assert set(result.selected) <= set(result.learner_names)


def test_a_wrong_length_weight_vector_is_refused(data):
    with pytest.raises(ValueError, match="one entry per row"):
        mb.stabsel_gamboostLSS(
            "y ~ bbs(x, df=4)", data, family=mb.GaussianLSS(),
            control=mb.boost_control(20, 0.1), q=1, PFER=1, B=3,
            weights=np.ones(N - 1))


def test_unit_weights_are_unchanged(data):
    """The fix must not move an unweighted run, which is every existing
    result."""
    fit = fit_lss(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = fit.stabsel(q=2, PFER=1, B=5, seed=1)
        direct = mb.stabsel_gamboostLSS(
            "y ~ bbs(x, df=4) + bbs(z, df=4)", data, family=mb.GaussianLSS(),
            control=mb.boost_control(30, 0.1), q=2, PFER=1, B=5, seed=1)
    assert theirs.selected == direct.selected
    assert np.allclose(theirs.phat, direct.phat)
