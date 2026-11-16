"""Multinomial weights must reach the observation they belong to.

A Multinomial response is stored flat: observation ``i`` owns the ``K-1``
consecutive entries ``i*(K-1) .. (i+1)*(K-1)``, one per non-reference class,
and its weight is repeated over that block. Code that reads such a vector as
if the first ``n`` entries were the observations gives observation ``i`` the
weight of observation ``i // (K-1)``. That is invisible with unit weights and
wrong with any other, which is why every case here uses weights that differ
between observations.

The oracle is the boosting algorithm written out in NumPy: negative gradient
``y - p``, one weighted least-squares fit of ``[1, x]`` per class block, and
a shrunken update. It is independent of the engine rather than a second
spelling of it.
"""

import numpy as np
import pytest

import mboostrs as mb

NU = 0.1
MSTOP = 12


def make_data(seed=3, n=60, n_classes=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    data = {
        "y": np.array([chr(ord("a") + k) for k in range(n_classes)])[labels],
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    }
    return data, labels, n_classes


def indicators(labels, n_classes):
    """The flat response as an (n, K-1) matrix of non-reference indicators."""
    n_resp = n_classes - 1
    y = np.zeros((len(labels), n_resp))
    for i, label in enumerate(labels):
        if label < n_resp:
            y[i, label] = 1.0
    return y


def probabilities(eta):
    exp_eta = np.exp(np.clip(eta, -36.0, 36.0))
    return exp_eta / (1.0 + exp_eta.sum(axis=1, keepdims=True))


def partial_risk(eta, labels, weights, rows=None):
    """Upstream's reported risk: the NLL over non-reference observations."""
    rows = range(eta.shape[0]) if rows is None else rows
    log_denom = np.log1p(np.exp(np.clip(eta, -36.0, 36.0)).sum(axis=1))
    n_resp = eta.shape[1]
    return -sum(
        weights[i] * (eta[i, labels[i]] - log_denom[i])
        for i in rows
        if labels[i] < n_resp
    )


def full_nll(eta, labels, weights, rows=None):
    """The negative log-likelihood over every observation, reference class
    included: what `cvrisk` scores held-out rows with."""
    rows = range(eta.shape[0]) if rows is None else rows
    log_denom = np.log1p(np.exp(np.clip(eta, -36.0, 36.0)).sum(axis=1))
    n_resp = eta.shape[1]
    return -sum(
        weights[i] * ((eta[i, labels[i]] if labels[i] < n_resp else 0.0) - log_denom[i])
        for i in rows
    )


def oracle_path(designs, labels, weights, n_classes, mstop=MSTOP, nu=NU):
    """Component-wise boosting over Kronecker-lifted bols learners.

    Each learner is one design matrix shared by every class block, so its
    weighted least-squares fit solves for all ``K-1`` gradient columns at once.
    Returns the fitted predictor after every step and the learner selected at
    each step.
    """
    y = indicators(labels, n_classes)
    eta = np.zeros_like(y)
    path, selected = [eta.copy()], []
    for _ in range(mstop):
        u = y - probabilities(eta)
        best, best_rss, best_fit = None, np.inf, None
        for index, design in enumerate(designs):
            weighted = design.T * weights
            beta = np.linalg.solve(weighted @ design, weighted @ u)
            fitted = design @ beta
            rss = float((weights[:, None] * (u - fitted) ** 2).sum())
            if rss < best_rss:
                best, best_rss, best_fit = index, rss, fitted
        eta = eta + nu * best_fit
        path.append(eta.copy())
        selected.append(best)
    return path, selected


def bols_design(column):
    return np.column_stack([np.ones(len(column)), column])


def weight_cases(n, seed=11):
    rng = np.random.default_rng(seed)
    return {
        "integer": rng.integers(1, 5, size=n).astype(float),
        "zero-one": (rng.random(n) > 0.3).astype(float),
    }


def test_weighted_risk_path_weights_each_observation_once():
    """The reported risk path is the weighted partial likelihood.

    Reading the flat weights by observation index made an integer-weighted
    three-class fit report 98.440188 where the weighted likelihood is
    88.359482: every observation carried another observation's weight.
    """
    data, labels, n_classes = make_data()
    designs = [bols_design(data["x1"])]
    for name, weights in weight_cases(len(labels)).items():
        model = mb.gamboost("y ~ bols(x1)", data=data, family=mb.Multinomial(),
                            control=(MSTOP, NU), weights=weights)
        path, _ = oracle_path(designs, labels, weights, n_classes)
        want = np.array([partial_risk(eta, labels, weights) for eta in path])
        np.testing.assert_allclose(
            np.asarray(model.risk, dtype=float), want, rtol=1e-10,
            err_msg=f"{name} weights: reported risk path",
        )


def test_weighted_fit_matches_the_boosting_algorithm():
    """Weights reach the learner correctly, so the path is the oracle's."""
    data, labels, n_classes = make_data()
    designs = [bols_design(data["x1"])]
    for name, weights in weight_cases(len(labels)).items():
        model = mb.gamboost("y ~ bols(x1)", data=data, family=mb.Multinomial(),
                            control=(MSTOP, NU), weights=weights)
        path, _ = oracle_path(designs, labels, weights, n_classes)
        np.testing.assert_allclose(
            np.asarray(model.predict(type="link"), dtype=float), path[-1],
            atol=1e-12, err_msg=f"{name} weights: fitted predictor",
        )


def kfold_matrix(n, folds, seed=5):
    rng = np.random.default_rng(seed)
    fold_id = rng.integers(0, folds, size=n)
    return np.stack([(fold_id != b).astype(float) for b in range(folds)], axis=1)


@pytest.mark.parametrize("base", ["unit", "integer"])
def test_cvrisk_fits_each_fold_on_its_own_in_bag_weights(base):
    """Every fold's in-bag fit and out-of-bag risk use the fold's own rows.

    The weight contract is upstream's, with one addition: a supplied fold
    column is the in-bag weight vector, except that a 0/1 mask given to a
    weighted model is multiplied by the model's weights (upstream refits such
    a mask unweighted). The out-of-bag risk weights the rows that column
    zeroes by the model's own weights and divides by their total
    (`cvrisk.mboost`: `weights = folds[, i]`, `OOBweights[folds > 0] <- 0`,
    `oobrisk / colSums(OOBweights)`).

    The fold matrix reaches the core flattened, one block per observation.
    Taking its leading ``n`` entries as the observation weights trained fold
    ``b`` on a shifted subset, which agreed with a correct fit only at the
    offset step.
    """
    data, labels, n_classes = make_data()
    n = len(labels)
    weights = np.ones(n) if base == "unit" else weight_cases(n)["integer"]
    folds = kfold_matrix(n, 3)

    model = mb.gamboost("y ~ bols(x1)", data=data, family=mb.Multinomial(),
                        control=(MSTOP, NU), weights=weights)
    got = np.asarray(model.cvrisk(folds=folds).risk_matrix, dtype=float)

    designs = [bols_design(data["x1"])]
    for b in range(folds.shape[1]):
        # A 0/1 fold mask on a weighted model refits with the model's weights.
        in_bag = folds[:, b] * weights
        held_out = np.flatnonzero(folds[:, b] == 0)
        path, _ = oracle_path(designs, labels, in_bag, n_classes)
        divisor = weights[held_out].sum()
        # Held-out rows are scored with the full log-likelihood; upstream's
        # reported risk leaves the reference class out (see `Multinomial`'s
        # `held_out_risk`).
        want = np.array([
            full_nll(eta, labels, weights, rows=held_out) / divisor
            for eta in path
        ])
        np.testing.assert_allclose(got[b], want, rtol=1e-9,
                                   err_msg=f"{base} weights, fold {b}")


def test_selection_paths_use_each_fold_s_own_weights():
    """Stability selection's per-fold paths select on the fold's own fit.

    As in `cvrisk`, a 0/1 fold column times the model's weights is the in-bag
    weight vector for that refit, so the selected sequence is the one a fit on
    those weights makes.
    """
    from mboostrs.stabsel import _run_gamboost_specs_selected

    data, labels, n_classes = make_data()
    n = len(labels)
    weights = weight_cases(n)["integer"]
    folds = kfold_matrix(n, 3, seed=9)
    model = mb.gamboost("y ~ bols(x1) + bols(x2)", data=data,
                        family=mb.Multinomial(), control=(MSTOP, NU),
                        weights=weights)

    got = _run_gamboost_specs_selected(model, folds, MSTOP)

    designs = [bols_design(data["x1"]), bols_design(data["x2"])]
    for b in range(folds.shape[1]):
        _, selected = oracle_path(designs, labels, folds[:, b] * weights, n_classes)
        assert list(got[b]) == selected, f"fold {b} selection path"


def test_family_diagnostics_takes_one_weight_per_observation():
    """The diagnostic endpoint weights observations, not flat entries."""
    from mboostrs._rust import family_diagnostics

    data, labels, n_classes = make_data(seed=7, n=24, n_classes=4)
    n = len(labels)
    weights = weight_cases(n, seed=2)["integer"]
    rng = np.random.default_rng(1)
    eta = rng.normal(size=(n, n_classes - 1))
    y = indicators(labels, n_classes)

    _offset, gradient, risk = family_diagnostics(
        "multinomial", {"n_classes": float(n_classes)},
        y.ravel(), eta.ravel(), weights,
    )
    assert risk == pytest.approx(partial_risk(eta, labels, weights), rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(gradient, dtype=float).reshape(n, -1),
        y - probabilities(eta), atol=1e-12,
    )

    with pytest.raises(ValueError, match="one weight per observation"):
        family_diagnostics("multinomial", {"n_classes": float(n_classes)},
                           y.ravel(), eta.ravel(),
                           np.repeat(weights, n_classes - 1))
