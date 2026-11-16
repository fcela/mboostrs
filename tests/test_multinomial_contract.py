"""Multinomial contract checks kept apart by what each one establishes (issue #98).

New-data prediction, truncation, serialization and the full categorical
objective are separate claims; each test checks one of them against an
independent computation.
"""

import numpy as np
import pytest

import mboostrs as mb

CLASSES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 300
    x, z = rng.normal(size=n), rng.normal(size=n)
    eta = np.column_stack([np.sin(2 * x), -x + z, np.zeros(n)])
    p = np.exp(eta)
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(CLASSES, p=pi) for pi in p])
    d = {"y": y, "x": x, "z": z}
    model = mb.gamboost("y ~ bbs(x) + bols(z)", data=d, family=mb.Multinomial(),
                        control=mb.boost_control(80, 0.1))
    return model, d


def _softmax(link):
    full = np.column_stack([link, np.zeros(len(link))])
    full = np.exp(full - full.max(axis=1, keepdims=True))
    return full / full.sum(axis=1, keepdims=True)


def test_probabilities_are_the_softmax_of_the_link_with_the_last_class_as_reference(fitted):
    model, d = fitted
    link = np.asarray(model.predict(type="link"))
    probs = np.asarray(model.predict(type="response"))
    assert link.shape == (300, 2) and probs.shape == (300, 3)
    np.testing.assert_allclose(probs, _softmax(link), atol=1e-12)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)


def test_new_data_prediction_reproduces_the_fit(fitted):
    model, d = fitted
    np.testing.assert_allclose(np.asarray(model.predict(newdata=d, type="link")),
                               np.asarray(model.predict(type="link")), atol=1e-10)


def test_truncation_equals_a_shorter_fit(fitted):
    model, d = fitted
    short = mb.gamboost("y ~ bbs(x) + bols(z)", data=d, family=mb.Multinomial(),
                        control=mb.boost_control(30, 0.1))
    np.testing.assert_allclose(np.asarray(model[30].predict(newdata=d, type="link")),
                               np.asarray(short.predict(type="link")), atol=1e-10)


def test_serialization_round_trips_predictions(fitted, tmp_path):
    model, d = fitted
    path = tmp_path / "multinomial.json"
    mb.save_model(model, path)
    loaded = mb.load_model(path)
    np.testing.assert_allclose(np.asarray(loaded.predict(newdata=d, type="response")),
                               np.asarray(model.predict(newdata=d, type="response")),
                               atol=1e-12)


def test_held_out_risk_is_the_full_categorical_nll(fitted):
    # Scored with every class, reference included: the risk at step k equals
    # the mean negative log-probability of the observed class.
    model, d = fitted
    n = len(d["y"])
    folds = np.asarray(mb.cv_weights(n, "kfold", 4))
    risk = np.asarray(model.cvrisk(folds=folds).risk_matrix)
    codes = np.array([CLASSES.index(v) for v in d["y"]])
    weights = folds[:, 1]
    held = weights == 0
    refit = mb.gamboost("y ~ bbs(x) + bols(z)", data=d, family=mb.Multinomial(),
                        weights=weights, control=mb.boost_control(80, 0.1))
    probs = np.asarray(refit.predict(newdata=d, type="response"))
    nll = -np.log(probs[np.arange(n), codes])[held].mean()
    assert risk[1, 80] == pytest.approx(nll, rel=1e-9)


def test_relabelling_the_classes_permutes_the_probabilities(fitted):
    # With linear learners the fitted probabilities do not depend on which
    # class is the reference: boosting the K-1 contrasts reaches the same
    # multinomial logit fit when it is run to convergence.
    _, d = fitted
    relabel = {"a": "c", "b": "a", "c": "b"}
    d2 = {**d, "y": np.array([relabel[v] for v in d["y"]])}
    control = mb.boost_control(3000, 0.5)
    m1 = mb.gamboost("y ~ bols(x) + bols(z)", data=d, family=mb.Multinomial(), control=control)
    m2 = mb.gamboost("y ~ bols(x) + bols(z)", data=d2, family=mb.Multinomial(), control=control)
    p1 = np.asarray(m1.predict(type="response"))
    p2 = np.asarray(m2.predict(type="response"))
    # Column order is sorted labels: m2's columns a, b, c hold m1's b, c, a.
    np.testing.assert_allclose(p2, p1[:, [1, 2, 0]], atol=1e-4)


def test_a_boundary_the_kronecker_smooth_would_ignore_is_refused(fitted):
    _, d = fitted
    with pytest.raises(NotImplementedError, match="boundary_knots"):
        mb.gamboost("y ~ bbs(x, boundary.knots=c(-5, 5))", data=d,
                    family=mb.Multinomial(), control=mb.boost_control(10))
