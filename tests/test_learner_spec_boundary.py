"""The learner boundary between Python and Rust.

Enhancement-hunt 10.1. Learner specifications used to cross the FFI as ~80
parallel arrays feeding two 88-argument PyO3 functions, with index alignment
between the arrays enforced by nothing but care at each of three call sites
(`fit`, `cvrisk`, `predict`). That is the failure class the crate already
documents: `Family::encode_response`'s comment records that each binding used
to recode `y` itself and "the cvrisk path was missed, giving wrong
cross-validated risk for Binomial".

The R binding never had that shape — it passes one self-describing,
type-tagged record per learner — and so does the mixed-learner LSS path in this
binding. `fit_gamboost_specs` is the same boundary for `gamboost`: 14
arguments, one of which is the learner list.

These tests pin that the compact path accepts every learner kind and that its
direct low-level result agrees with the public model factory that uses it.
"""

import numpy as np
import pytest
from mboostrs._rust import fit_gamboost_specs
from mboostrs.gamlss import learner_spec

import mboostrs as mb


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.uniform(0, 1, n)
    z = rng.normal(size=n)
    g = rng.integers(0, 5, n).astype(float)
    y = np.sin(2 * np.pi * x) + 0.7 * z + rng.normal(0, 0.3, n)
    return {"x": x, "z": z, "g": g, "y": y}


def _fit_specs(learners, data, mstop=60, nu=0.1, family="gaussian"):
    specs = [learner_spec(learner, data) for learner in learners]
    weights = np.ones(len(data["y"]))
    return fit_gamboost_specs(
        specs, data["y"], weights, mstop, nu, family, {},
        None, None, "inbag", False, None, 0,
    )


LEARNER_CASES = {
    "bbs": [mb.Bbs("x", df=4)],
    "bols": [mb.Bols("x"), mb.Bols("z")],
    "bns": [mb.Bns("x", df=4)],
    "brandom": [mb.Brandom("g", df=4)],
    "btree": [mb.Btree(["x", "z"])],
    "bmono": [mb.Bmono("x", df=4)],
    "bkernel": [mb.Bkernel("x", df=4)],
    "mixed": [mb.Bbs("x", df=4), mb.Bols("z"), mb.Brandom("g", df=4)],
}


@pytest.mark.parametrize("label", sorted(LEARNER_CASES))
def test_every_learner_kind_crosses_as_one_record(label, data):
    """One dict per learner, not a slot in each of ~80 arrays."""
    offset, selected, coefs, risk, fitted, *_ = _fit_specs(LEARNER_CASES[label], data)
    assert len(selected) == 60
    assert np.isfinite(offset)
    assert np.all(np.isfinite(np.asarray(risk)))
    assert np.asarray(risk)[-1] < np.asarray(risk)[0]
    assert set(selected) <= set(range(len(LEARNER_CASES[label])))


@pytest.mark.parametrize(
    "formula, learners",
    [
        ("y ~ bbs(x, df=4)", [mb.Bbs("x", df=4)]),
        ("y ~ bbs(x, df=4) + bols(z)", [mb.Bbs("x", df=4), mb.Bols("z")]),
        ("y ~ bols(x) + bols(z)", [mb.Bols("x"), mb.Bols("z")]),
        ("y ~ bns(x, df=4)", [mb.Bns("x", df=4)]),
        ("y ~ brandom(g, df=4)", [mb.Brandom("g", df=4)]),
        ("y ~ btree(x)", [mb.Btree(["x"])]),
        ("y ~ bmono(x, df=4)", [mb.Bmono("x", df=4)]),
    ],
)
def test_the_spec_path_agrees_with_the_public_factory(formula, learners, data):
    """`gamboost()` and direct compact-boundary fits must agree.

    Single-learner fits are bit-exact. Multi-learner fits differ by at most
    ~1e-13 in the risk because the public factory performs its own state and
    result mapping around the native call. The selected sequence must be
    identical.
    """
    reference = mb.gamboost(formula, data=data, family=mb.Gaussian(),
                            control=mb.boost_control(mstop=60, nu=0.1))
    offset, selected, coefs, risk, fitted, *_ = _fit_specs(learners, data)

    assert offset == reference.offset_
    assert list(selected) == list(reference._selected)
    np.testing.assert_allclose(np.asarray(risk), np.asarray(reference.risk_),
                               rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.asarray(fitted).ravel(),
                               np.asarray(reference.fitted_).ravel(),
                               rtol=0, atol=1e-12)


def test_specs_are_in_formula_order(data):
    """A spec list is in formula order, so index `i` is the `i`-th term.

    This avoids the kind-grouped index remapping formerly required by the
    parallel-array boundary (AGENTS.md gotcha 13 records a bug in that map).
    """
    learners = [mb.Bbs("x", df=4), mb.Bols("z")]
    specs = [learner_spec(learner, data) for learner in learners]
    assert [s["kind"] for s in specs] == ["bbs", "bols"]
    assert [s["name"] for s in specs] == ["x", "z"]

    # y depends on x only, so the smooth term must win every round and it must
    # be index 0 under both paths.
    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(0, 1, n)
    smooth_only = {"x": x, "z": rng.normal(size=n),
                   "y": np.sin(3 * np.pi * x) + rng.normal(0, 0.2, n)}
    _, selected, *_ = _fit_specs(learners, smooth_only, mstop=80)
    assert set(selected) == {0}

    reference = mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data=smooth_only,
                            family=mb.Gaussian(),
                            control=mb.boost_control(mstop=80, nu=0.1))
    assert set(reference.selected) == {"x"}


def test_a_malformed_spec_names_the_learner_it_came_from(data):
    """The point of a self-describing record: an error is attributable. An
    aligned-array mistake could only ever produce a length complaint."""
    specs = [learner_spec(mb.Bbs("x", df=4), data)]
    specs[0]["kind"] = "not_a_learner"
    weights = np.ones(len(data["y"]))
    with pytest.raises(ValueError, match="unsupported LSS learner kind"):
        fit_gamboost_specs(specs, data["y"], weights, 10, 0.1, "gaussian", {},
                           None, None, "inbag", False, None, 0)

    # No learner is the offset-only model (#357): the fit runs and selects nothing.
    offset, selected, *_rest = fit_gamboost_specs(
        [], data["y"], weights, 10, 0.1, "gaussian", {},
        None, None, "inbag", False, None, 0)
    assert list(selected) == []


# ---------------------------------------------------------------------------
# The prediction half of the boundary
# ---------------------------------------------------------------------------
#
# `LearnerBlueprint::prepare` returns a `BasisInfo` — the core's own new-data
# replay recipe — alongside the prepared learner, and `BasisInfo::predict_into`
# covers all ten learner kinds. The array fit path discarded it and had Python
# rebuild a partial copy through `set_basis_info`, which covers bols and bbs
# only; everything else got `BasisInfo::Unsupported`, and prediction went back
# out through `_collect_predict_args` and the 44-argument `predict_gamboost`,
# with bkernel and buser computed in Python on top of that.
#
# `fit_gamboost_specs` keeps what `prepare` returned, so a store fitted through
# the spec path can predict any learner kind by itself.


def _replay_x(learners, data, buser_design=None):
    """The raw new-data slice each learner's replay recipe expects."""
    slices = []
    for learner in learners:
        if isinstance(learner, mb.Bspatial):
            slices.append(
                np.concatenate([data[learner.name1], data[learner.name2]]).tolist()
            )
        elif isinstance(learner, mb.Btree):
            slices.append(
                np.column_stack([data[name] for name in learner.names]).ravel().tolist()
            )
        elif isinstance(learner, mb.Buser):
            slices.append(np.asarray(buser_design, dtype=float).ravel().tolist())
        else:
            slices.append(np.asarray(data[learner.name], dtype=float).tolist())
    return slices


def _every_kind(data):
    """One case per learner kind, including the four the array predict path
    could not replay from the store."""
    # Graph Laplacian of a 5-node chain. The diagonal is the neighbour count,
    # so the two end nodes get 1 and the interior nodes 2 — using 2 throughout
    # (a second-difference penalty) leaves the end rows summing to 1.
    K = np.diag([1.0, 2.0, 2.0, 2.0, 1.0]) - np.eye(5, k=1) - np.eye(5, k=-1)
    design = np.column_stack([data["x"], data["x"] ** 2, data["z"]])
    return {
        "bols": ([mb.Bols("x"), mb.Bols("z")], None),
        "bbs": ([mb.Bbs("x", df=4)], None),
        "bns": ([mb.Bns("x", df=4)], None),
        "brandom": ([mb.Brandom("g", df=4)], None),
        "btree": ([mb.Btree(["x", "z"])], None),
        "bmono": ([mb.Bmono("x", df=4)], None),
        "bkernel": ([mb.Bkernel("x", df=4)], None),
        "bmrf": ([mb.Bmrf("g", K=K, df=3)], None),
        "bspatial": ([mb.Bspatial("x", "z", df=6)], None),
        "buser": ([mb.Buser("u", X=design, K=np.eye(3), df=2)], design),
        "mixed": (
            [
                mb.Bbs("x", df=4),
                mb.Bols("z"),
                mb.Brandom("g", df=4),
                mb.Btree(["x"]),
                mb.Bmono("z", df=4),
            ],
            None,
        ),
    }


@pytest.mark.parametrize("label", sorted(_every_kind({"x": np.zeros(1), "z": np.zeros(1)})))
def test_the_fitted_store_carries_a_replay_recipe_for_every_learner(label, data):
    learners, _ = _every_kind(data)[label]
    specs = [learner_spec(learner, data) for learner in learners]
    store = fit_gamboost_specs(
        specs, data["y"], np.ones(len(data["y"])), 40, 0.1, "gaussian", {},
        None, None, "inbag", False, None, 0,
    )[7]
    assert store.basis_info_len() == len(learners)


@pytest.mark.parametrize("label", sorted(_every_kind({"x": np.zeros(1), "z": np.zeros(1)})))
def test_store_replay_reproduces_the_fitted_values(label, data):
    """Replaying the path through the store's own recipes returns the fitted
    values the engine produced, for every learner kind."""
    learners, design = _every_kind(data)[label]
    specs = [learner_spec(learner, data) for learner in learners]
    offset, selected, coefs, _risk, fitted, *rest = fit_gamboost_specs(
        specs, data["y"], np.ones(len(data["y"])), 60, 0.1, "gaussian", {},
        None, None, "inbag", False, None, 0,
    )
    store = rest[2]
    x_data = _replay_x(learners, data, design)

    stepwise = np.asarray(
        store.predict_newdata(offset, selected, coefs, x_data, None, None)
    )
    grouped = np.asarray(
        store.predict_newdata_grouped(offset, selected, coefs, x_data, None, None)
    )
    fitted = np.asarray(fitted)
    # Not bit-identical: grouping sums a learner's coefficients before applying
    # its basis, which reassociates the sum. The gap is floating-point noise.
    assert np.allclose(stepwise, fitted, rtol=0, atol=1e-11)
    assert np.allclose(grouped, fitted, rtol=0, atol=1e-11)


def test_grouped_replay_cost_does_not_track_mstop():
    """Why the grouped variant exists.

    `predict_newdata` rebuilds a learner's basis on every step that selected
    it, so its cost grows with `mstop`; a comment in `gamboost.py` recorded
    that as the reason prediction could not route through the store. Summing a
    learner's coefficients first means each basis is built once, so the cost is
    driven by how many distinct learners were selected, plus a cheap
    O(mstop x p) summing pass.

    Measured here at n=500 with six bbs learners: raising mstop from 200 to
    2000 took the stepwise path from 5.7 ms to 55.7 ms (9.8x, tracking mstop)
    and the grouped path from 0.15 ms to 0.40 ms. The assertion compares the
    two paths at mstop=2000 only.
    """
    import time

    rng = np.random.default_rng(3)
    n = 500
    data = {f"x{j}": rng.normal(size=n) for j in range(6)}
    data["y"] = sum(np.sin(data[f"x{j}"]) for j in range(3)) + rng.normal(0, 0.3, n)
    learners = [mb.Bbs(f"x{j}", df=4) for j in range(6)]
    specs = [learner_spec(learner, data) for learner in learners]
    x_data = [np.asarray(data[learner.name]).tolist() for learner in learners]

    def timings(mstop):
        offset, selected, coefs, _r, _f, *rest = fit_gamboost_specs(
            specs, data["y"], np.ones(n), mstop, 0.1, "gaussian", {},
            None, None, "inbag", False, None, 0,
        )
        store = rest[2]

        def clock(call):
            # The minimum of several runs: other work on the machine (another
            # pytest worker) only ever adds time, so the minimum is the
            # closest reading of the call's own cost. A mean of three failed
            # under `pytest -n 8` (issue #162).
            call()
            best = float("inf")
            for _ in range(7):
                start = time.perf_counter()
                call()
                best = min(best, time.perf_counter() - start)
            return best

        return (
            clock(lambda: store.predict_newdata(offset, selected, coefs, x_data, None, None)),
            clock(lambda: store.predict_newdata_grouped(offset, selected, coefs, x_data, None, None)),
        )

    # One comparison with a wide margin. The growth ratios this test used to
    # assert divided four readings, two of them near 0.15 ms, and failed under
    # `pytest -n 8` with no code change (issue #162). At mstop=2000 the
    # stepwise path takes about 140 times as long as the grouped one; the
    # assertion needs 10.
    step_large, grouped_large = timings(2000)
    assert grouped_large < step_large / 10, (grouped_large, step_large)


def test_grouped_replay_equals_stepwise_replay():
    """Summing a learner's coefficients before building its basis gives the
    same prediction as replaying every step."""
    rng = np.random.default_rng(3)
    n = 300
    data = {f"x{j}": rng.normal(size=n) for j in range(4)}
    data["y"] = sum(np.sin(data[f"x{j}"]) for j in range(2)) + rng.normal(0, 0.3, n)
    learners = [mb.Bbs(f"x{j}", df=4) for j in range(4)]
    specs = [learner_spec(learner, data) for learner in learners]
    x_data = [np.asarray(data[learner.name]).tolist() for learner in learners]
    offset, selected, coefs, _r, _f, *rest = fit_gamboost_specs(
        specs, data["y"], np.ones(n), 300, 0.1, "gaussian", {},
        None, None, "inbag", False, None, 0,
    )
    store = rest[2]
    np.testing.assert_allclose(
        store.predict_newdata_grouped(offset, selected, coefs, x_data, None, None),
        store.predict_newdata(offset, selected, coefs, x_data, None, None),
        rtol=0, atol=1e-10)


def test_replay_rejects_a_wrongly_shaped_slice(data):
    """Shape errors name the learner index rather than failing silently."""
    learners = [mb.Bbs("x", df=4), mb.Bols("z")]
    specs = [learner_spec(learner, data) for learner in learners]
    offset, selected, coefs, _r, _f, *rest = fit_gamboost_specs(
        specs, data["y"], np.ones(len(data["y"])), 20, 0.1, "gaussian", {},
        None, None, "inbag", False, None, 0,
    )
    store = rest[2]
    x_data = _replay_x(learners, data)
    x_data[1] = x_data[1][:-1]
    with pytest.raises(ValueError, match=r"x_data\[1\] has length"):
        store.predict_newdata_grouped(offset, selected, coefs, x_data, None, None)


def test_a_deserialized_model_predicts_and_scores_through_the_core():
    """A pickle round trip used to lose the learner store, and with it `aic()`
    and `hatvalues()` — both documented as returning `None` afterwards, with
    "refit the model" as the advice. The store is rebuilt from the stored
    learner specifications and training weights instead."""
    import pickle

    rng = np.random.default_rng(4)
    n = 200
    data = {
        "x": rng.uniform(0, 1, n),
        "z": rng.normal(size=n),
        "g": rng.integers(0, 4, n).astype(float),
    }
    data["y"] = (
        np.sin(3 * data["x"]) + 0.5 * data["z"] + rng.normal(0, 0.2, n)
    )
    fit = mb.gamboost(
        "y ~ bbs(x, df=4) + bols(z) + brandom(g, df=3)",
        data=data,
        family=mb.Gaussian(),
        control=mb.boost_control(60, 0.1),
    )
    restored = pickle.loads(pickle.dumps(fit))

    np.testing.assert_allclose(
        np.asarray(restored.predict(newdata=data)),
        np.asarray(fit.predict(newdata=data)),
        rtol=0,
        atol=1e-12,
    )
    assert restored.aic() is not None
    assert restored.aic().mstop == fit.aic().mstop
    np.testing.assert_array_equal(
        np.asarray(restored.hatvalues()), np.asarray(fit.hatvalues())
    )
