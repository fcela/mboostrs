"""The weight contract survives a round trip and shows up in `summary()`.

P0.2's closure audit named four conditions. Two are these:

  "standard-model pickle state and portable artifacts must preserve
  `_input_weights`, `_weight_policy`, and `_weights_rescaled`; the audited
  round trip changed `raw`/rescaled state to `upstream`/unknown"

  "`summary()` and serialized metadata must report the named policy, whether
  rescaling occurred, and which vector is effective"

Both held. `__getstate__` carried `weights` — the effective vector — and none
of the three fields that say what it is, so a restored model reported
`input_weights()` as `None` and `weight_policy` as `"upstream"` whatever the
fit had used. A model saved from a `raw` fit came back claiming the upstream
policy, with the rescaling flag gone.

`summary()` reported `n` and nothing else about the sample. A fit made with
fractional weights under the upstream policy ran on a rescaled vector summing
to `n`, and a summary giving only the row count described a sample nobody
fitted.
"""

import json
import pathlib
import pickle
import tempfile
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.gamboost import GAMBoostModel

# The public `gamboost` is a wrapper: `gamboost.__globals__` is the
# `_model_api` module and no longer holds `GAMBoostModel`.
_GAMBoostModel = GAMBoostModel

N = 60


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    frame = {"x": rng.normal(size=N), "z": rng.normal(size=N)}
    frame["y"] = frame["x"] + rng.normal(scale=0.3, size=N)
    return frame


@pytest.fixture(scope="module")
def fractional():
    return np.random.default_rng(9).uniform(0.3, 2.0, N)


def build(kind, data, weights, policy):
    formula = {"gamboost": "y ~ bols(x)", "glmboost": "y ~ x",
               "gamboostLSS": "y ~ bbs(x, df=4)"}[kind]
    extra = {"family": mb.GaussianLSS()} if kind == "gamboostLSS" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(mb, kind)(formula, data, control=mb.boost_control(20, 0.1),
                                 weights=weights, weight_policy=policy, **extra)


def state_of(model):
    return (model.weight_policy,
            None if model.input_weights() is None
            else float(np.sum(model.input_weights())),
            getattr(model, "_weights_rescaled", None),
            float(np.sum(np.asarray(model.weights, dtype=float))))


@pytest.mark.parametrize("kind", ["gamboost", "glmboost", "gamboostLSS"])
@pytest.mark.parametrize("policy", ["upstream", "raw"])
def test_pickle_preserves_the_whole_contract(kind, policy, data, fractional):
    fit = build(kind, data, fractional, policy)
    assert state_of(pickle.loads(pickle.dumps(fit))) == state_of(fit)


@pytest.mark.parametrize("policy", ["upstream", "raw"])
def test_the_portable_artifact_preserves_it_too(policy, data, fractional):
    fit = build("gamboost", data, fractional, policy)
    path = pathlib.Path(tempfile.mkdtemp(prefix="mboostrs-weights-")) / "m.json"
    fit.save(str(path))
    assert state_of(mb.load_model(str(path))) == state_of(fit)


def test_the_two_policies_stay_distinct_through_a_round_trip(data, fractional):
    """The failure the audit recorded: `raw` came back as `upstream`, which
    reads as a fit that was rescaled when it was not."""
    upstream = build("gamboost", data, fractional, "upstream")
    raw = build("gamboost", data, fractional, "raw")
    assert state_of(upstream) != state_of(raw)
    assert state_of(pickle.loads(pickle.dumps(raw)))[0] == "raw"
    assert pickle.loads(pickle.dumps(raw))._weights_rescaled is False


def test_state_written_before_the_contract_reads_as_upstream(data, fractional):
    """A restored model with no recorded policy has to assume something. It
    assumes the default, and only because nothing else was recorded."""
    fit = build("gamboost", data, fractional, "raw")
    state = fit.__getstate__()
    for key in ("input_weights", "weight_policy", "weights_rescaled"):
        state.pop(key)
    restored = _GAMBoostModel.__new__(_GAMBoostModel)
    restored.__setstate__(state)
    assert restored.weight_policy == "upstream"
    assert restored.input_weights() is None


@pytest.mark.parametrize("kind", ["gamboost", "glmboost"])
def test_summary_says_what_a_row_counted_for(kind, data, fractional):
    fit = build(kind, data, fractional, "upstream")
    block = fit.summary(as_dict=True)["weights"]
    assert block["policy"] == "upstream"
    assert block["rescaled"] is True
    assert block["input_total"] == pytest.approx(float(np.sum(fractional)))
    assert block["effective_total"] == pytest.approx(float(N))
    assert block["input_total"] != pytest.approx(block["effective_total"])
    assert "rescaled" in block["note"]


@pytest.mark.parametrize("kind", ["gamboost", "glmboost"])
def test_a_raw_fit_reports_no_rescaling(kind, data, fractional):
    block = build(kind, data, fractional, "raw").summary(as_dict=True)["weights"]
    assert block["rescaled"] is False
    assert block["input_total"] == pytest.approx(block["effective_total"])


def test_an_unweighted_fit_says_so(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboost("y ~ bols(x)", data,
                          control=mb.boost_control(20, 0.1))
    block = fit.summary(as_dict=True)["weights"]
    assert block["input_total"] is None, "no weights were given"
    assert block["effective_total"] == pytest.approx(float(N))
    assert block["rescaled"] is False


def test_the_summary_block_is_json_serializable(data, fractional):
    """It goes into serialized metadata, so it has to survive `json.dumps`
    without a custom encoder."""
    block = build("gamboost", data, fractional, "upstream").summary(
        as_dict=True)["weights"]
    assert json.loads(json.dumps(block)) == block
