"""Interaction adoption on small n (issue #246).

With many columns and few rows the interaction screen passes chance pairs,
and the fixed 2-SE adoption bar lets some through. The bar now scales as
2*sqrt(log(candidates)) and the offered pairs are capped at O(n/50). On a
small-n design with no true interactions, the exhaustive-pairs route must
not lose to its own additive fit by more than the screen's 2% margin.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.autospec import _adoption_z_bar


def test_adoption_z_bar_scales_with_candidates():
    # One candidate on trial: the G-35 rule, unscaled.
    assert _adoption_z_bar(1) == 2.0
    # More candidates: the bar grows as 2*sqrt(log(n)).
    assert _adoption_z_bar(6) == max(2.0, 2.0 * np.sqrt(np.log(6)))
    assert _adoption_z_bar(6) > 2.0
    assert _adoption_z_bar(100) > _adoption_z_bar(6)


def _small_n_no_interaction_data(seed=0, n=120, p=10):
    """Many columns, few rows, pure main effects: every pair the screen
    offers is a chance pair."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(scale=0.5, size=n)
    return {f"x{j}": X[:, j] for j in range(p)} | {"y": y}


def _test_mse(formula, train, test):
    fit = mb.gamboost(formula, data=train, family=mb.Gaussian(),
                      control=mb.boost_control(200, 0.1))
    pred = np.asarray(fit.predict(newdata=test)).ravel()
    y = np.asarray(test["y"], dtype=float)
    return float(np.mean((y - pred) ** 2))


def test_adopt_does_not_lose_to_additive_on_small_n():
    data = _small_n_no_interaction_data()
    train = {k: np.asarray(v)[:80] for k, v in data.items()}
    test = {k: np.asarray(v)[80:] for k, v in data.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec_adopt = mb.autospec(train, "y", budget=150, seed=0,
                                interactions="adopt", preexploration="off")
        rec_add = mb.autospec(train, "y", budget=150, seed=0,
                              interactions="report", preexploration="off")
    mse_adopt = _test_mse(rec_adopt.formula, train, test)
    mse_add = _test_mse(rec_add.formula, train, test)
    # The screen's margin is 2%: the pairs route may not lose to its own
    # additive fit by more than that.
    assert mse_adopt <= mse_add * 1.02, (mse_adopt, mse_add)


@pytest.mark.parametrize("kind", ["tensor", "joint_tree"])
def test_adopted_pairs_carry_adoption_evidence(kind):
    """Adopted interactions are marked with (n_folds, rel, z, z_bar)."""
    rng = np.random.default_rng(1)
    n = 300
    X = rng.uniform(-1, 1, (n, 4))
    interaction = (1.5 * X[:, 0] * X[:, 1] if kind == "tensor"
                   else 3 * np.sign(X[:, 0] * X[:, 1]))
    y = np.sin(2 * X[:, 0]) + X[:, 1] + interaction
    data = {f"x{j + 1}": X[:, j] for j in range(4)} | {"y": y + rng.normal(0, 0.3, n)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", budget=150, seed=0,
                          interactions="adopt", preexploration="off")
    adopted = rec.evidence.get("adopted_interactions", ())
    evidence = rec.evidence.get("adoption_evidence", {})
    assert adopted, "the planted interaction must exercise the adoption route"
    assert any(row["winner"] == kind for key, row in rec.evidence.items()
               if key.startswith("pair_form_"))
    for term in adopted:
        assert term in evidence, f"{term} has no adoption evidence"
        n_folds, rel, z, z_bar = evidence[term]
        assert n_folds > 0 and z_bar >= 2.0
        assert rel > 0 or z >= z_bar
    assert "adopted with held-out evidence" in rec.report()
