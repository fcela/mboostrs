"""Fits and evaluations carry one data identity digest (issue #60).

`data_identity` is a SHA-256 digest of the observation IDs in row order, the
observed response values and the effective weights. Every fitted model records
it in its comparison identity, cross-validation and evaluation results carry
it, and comparisons refuse two rows whose digests differ. A model whose record
has no digest (a file written before the field existed) is compared through
the separate response, sample and weight digests, as before.
"""

import pickle

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._comparison_identity import data_identity, record_data_identity

# Short paths and small samples raise boundary and extrapolation warnings that
# are not under test here.
pytestmark = pytest.mark.filterwarnings("ignore")

CONTROL = mb.boost_control(40, 0.1)
N = 120


def _data(seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    z = rng.normal(size=N)
    return {"y": 0.5 + 0.8 * x + rng.normal(0, 0.3, N), "x": x, "z": z}


def _permuted(data, seed=5):
    out = dict(data)
    out["y"] = data["y"][np.random.default_rng(seed).permutation(N)]
    return out


def _weights():
    return 1.0 + np.random.default_rng(3).uniform(size=N)


def _glm(formula, data, **kwargs):
    return mb.glmboost(formula, data=data, family=mb.Gaussian(), control=CONTROL, **kwargs)


def _folds():
    return mb.cv_weights(N, type="kfold", B=5, seed=3)


def _gaic(*models):
    return mb.gaic_table(*models, output="records")


def _compare(*models):
    return mb.compare_models(*models, output="records")


HELPERS = pytest.mark.parametrize("helper", [_gaic, _compare],
                                  ids=["gaic_table", "compare_models"])


# ── the digest ───────────────────────────────────────────────────────────────

def test_digest_depends_on_row_order_response_and_weights():
    y = np.arange(6, dtype=float)
    w = np.ones(6)
    ids = np.arange(6)
    base = data_identity(y, w, ids)
    assert len(base) == 64
    assert data_identity(y.copy(), w.copy(), ids.copy()) == base
    assert data_identity(y[::-1], w, ids) != base
    assert data_identity(y, 2.0 * w, ids) != base
    assert data_identity(y, w, ids[::-1]) != base
    assert data_identity(y, w, ids, n_input=7) != base
    assert data_identity(y, w, ["a", "b", "c", "d", "e", "f"]) != base


def test_glm_gam_and_lss_fits_record_the_digest():
    data = _data()
    fits = [
        _glm("y ~ bols(x)", data),
        mb.gamboost("y ~ bbs(x, df=3)", data=data, family=mb.Gaussian(), control=CONTROL),
        mb.glmboostLSS("y ~ bols(x)", data, families=mb.GaussianLSS(), control=CONTROL),
    ]
    digests = {record_data_identity(fit._comparison_identity) for fit in fits}
    # One data set, one response, unit weights: every fit has the same digest.
    assert len(digests) == 1
    assert len(digests.pop()) == 64


def test_cv_and_evaluation_results_carry_the_digest():
    data = _data()
    fit = _glm("y ~ bols(x)", data)
    curve = fit.cvrisk(folds=_folds())
    assert curve.data_identity == record_data_identity(fit._comparison_identity)
    assert curve.to_dict()["data_identity"] == curve.data_identity
    lss = mb.glmboostLSS("y ~ bols(x)", data, families=mb.GaussianLSS(), control=CONTROL)
    assert len(lss.cvrisk(folds=_folds()).data_identity) == 64
    evaluation = mb.cv_risk("x", data, "y", folds=_folds(), mstop=20, score="mse",
                            return_result=True)
    assert len(evaluation.data_identity) == 64


# ── refusals ─────────────────────────────────────────────────────────────────

@HELPERS
def test_permuted_response_of_equal_length_is_refused(helper):
    data = _data()
    with pytest.raises(ValueError, match="different response values or row order"):
        helper(_glm("y ~ bols(x)", data), _glm("y ~ bols(x)", _permuted(data)))


@HELPERS
def test_same_data_under_different_weights_is_refused(helper):
    data = _data()
    with pytest.raises(ValueError, match="different effective weights"):
        helper(_glm("y ~ bols(x)", data), _glm("y ~ bols(x)", data, weights=_weights()))


@HELPERS
def test_a_digest_mismatch_alone_is_refused(helper):
    # The component digests agree, so only the data identity tells them apart.
    data = _data()
    first, second = _glm("y ~ bols(x)", data), _glm("y ~ bols(x) + bols(z)", data)
    second._comparison_identity = dict(second._comparison_identity, data_identity="0" * 64)
    with pytest.raises(ValueError, match="different data identity"):
        helper(first, second)


def test_cv_curves_on_a_permuted_response_or_other_weights_are_refused():
    data = _data()
    raw = _glm("y ~ bols(x)", data).cvrisk(folds=_folds())
    permuted = _glm("y ~ bols(x)", _permuted(data)).cvrisk(folds=_folds())
    weighted = _glm("y ~ bols(x)", data, weights=_weights()).cvrisk(folds=_folds())
    assert len({raw.data_identity, permuted.data_identity, weighted.data_identity}) == 3
    for other in (permuted, weighted):
        with pytest.raises(ValueError, match="cannot pair these curves"):
            raw.compare_to(other, mstop=20)


def test_evaluations_on_a_permuted_response_or_other_weights_are_not_comparable():
    data = _data()
    folds = _folds()

    def evaluate(frame, weights=None):
        return mb.cv_risk("x", frame, "y", folds=folds, mstop=20, score="mse",
                          weights=weights, return_result=True)

    base = evaluate(data)
    assert "data identity (observation IDs, response values or weights)" in (
        base.comparison_issues(evaluate(_permuted(data))))
    assert "data identity (observation IDs, response values or weights)" in (
        base.comparison_issues(evaluate(data, _weights())))
    assert base.comparison_issues(evaluate(data)) == ()


# ── unchanged behaviour ──────────────────────────────────────────────────────

@HELPERS
def test_identical_data_and_weights_rank_as_before(helper):
    data = _data()
    w = _weights()
    rows = helper(_glm("y ~ bols(x)", data, weights=w),
                  _glm("y ~ bols(x) + bols(z)", data, weights=w))
    assert len(rows) == 2
    assert all("comparison" not in row for row in rows)


def test_identical_cv_curves_pair_as_before():
    data = _data()
    a = _glm("y ~ bols(x)", data).cvrisk(folds=_folds())
    b = _glm("y ~ bols(x) + bols(z)", data).cvrisk(folds=_folds())
    assert a.data_identity == b.data_identity
    assert a.compare_to(b, mstop=20).inferential


def _without_digest(fit):
    record = {k: v for k, v in fit._comparison_identity.items() if k != "data_identity"}
    fit._comparison_identity = record
    return pickle.loads(pickle.dumps(fit))


# A pickled model keeps no learner store, so it cannot compute the AIC that
# gaic_table and compare_models rank by; compare_summaries reads the same
# shared check (`check_comparison`) without it.

def test_pickle_without_the_digest_loads_and_compares_as_before():
    data = _data()
    old = _without_digest(_glm("y ~ bols(x)", data))
    assert "data_identity" not in old._comparison_identity
    assert record_data_identity(old._comparison_identity) == ""
    mb.compare_summaries(old, _glm("y ~ bols(x) + bols(z)", data))
    # The component digests still refuse a permuted response and other weights.
    with pytest.raises(ValueError, match="different response values or row order"):
        mb.compare_summaries(old, _glm("y ~ bols(x)", _permuted(data)))
    with pytest.raises(ValueError, match="different effective weights"):
        mb.compare_summaries(old, _glm("y ~ bols(x)", data, weights=_weights()))


def test_live_model_without_any_identity_record_falls_back_to_retained_data():
    data = _data()
    fit = _glm("y ~ bols(x)", data)
    fit._comparison_identity = None
    mb.compare_summaries(fit, _glm("y ~ bols(x) + bols(z)", data))
    with pytest.raises(ValueError, match="response"):
        mb.compare_summaries(fit, _glm("y ~ bols(x)", _permuted(data)))


def test_pickle_without_any_identity_record_is_refused_as_before():
    # A pickle keeps no data frame, so the retained-data fallback has nothing
    # to compare and refuses, as it did before the digest existed.
    data = _data()
    fit = _glm("y ~ bols(x)", data)
    fit._comparison_identity = None
    old = pickle.loads(pickle.dumps(fit))
    assert old._comparison_identity is None
    with pytest.raises(ValueError, match="needs retained responses"):
        mb.compare_summaries(old, _glm("y ~ bols(x) + bols(z)", data))
