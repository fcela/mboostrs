import numpy as np
import polars as pl

import mboostrs as mb
from mboostrs import _rust
from benches.probe_joint_regularization import null_projector, run_seed


def test_null_projector_is_centered_and_penalizes_only_restricted_nullspace():
    x = np.random.default_rng(9).beta(2, 5, 80) * 2 - 1
    d = mb.learner_design("bbs(x, df=4, knots=12)", {"x": x})
    B, K = np.asarray(d.design), np.asarray(d.penalty)
    P, C, U0 = null_projector(B, K)
    assert np.linalg.norm(P - P.T) < 1e-10
    assert np.linalg.norm(P @ P - P) < 1e-10
    assert np.linalg.norm(P @ (B.T @ np.ones(len(x)))) < 1e-8
    assert np.linalg.norm(C.T @ K @ C @ U0) < 1e-8
    assert np.linalg.norm(P @ C @ np.linalg.eigh(C.T @ K @ C)[1][:, -1]) < 1e-8


def test_native_single_penalty_matches_public_all_term_refit_at_1_and_100_steps():
    rng = np.random.default_rng(123)
    frame = pl.DataFrame({"x0": rng.uniform(-1, 1, 90), "x1": rng.uniform(-1, 1, 90)})
    y = np.sin(3 * frame["x0"].to_numpy()) + frame["x1"].to_numpy() + rng.normal(size=90)
    frame = frame.with_columns(pl.Series("y", y))
    terms = ["bbs(x0, df=4, knots=12)", "bbs(x1, df=4, knots=12)"]
    ds = [mb.learner_design(t, frame) for t in terms]
    core = _rust.gam_refit_core(
        terms, [d.design for d in ds], [[d.penalty] for d in ds], [None, None], y, variables=[["x0"], ["x1"]]
    )
    raw = [d.transform(frame) for d in ds]
    direct = core.prediction(raw, [None, None], [False, False])[0]
    predictions = []
    for budget in (1, 100):
        model = mb.gamboost("y ~ " + " + ".join(terms), frame, control=mb.boost_control(budget, 0.1))
        refit = mb.gam_refit(model, which="all")
        predictions.append(refit.predict(frame, extrapolation="allow"))
        np.testing.assert_allclose(predictions[-1], direct, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-10, atol=1e-10)


def test_small_probe_keeps_signal_and_failure_fields():
    rows = run_seed(1000, n=40, test_n=48)
    assert len(rows) == 16
    assert {r["scenario"] for r in rows} == {"sparse", "correlated", "dense", "null"}
    assert all(r["status"] == "ok" for r in rows), [
        (r["scenario"], r["method"], r["error"]) for r in rows if r["status"] != "ok"
    ]
    for row in rows:
        assert row["status"] in {"ok", "error"}
        assert "error" in row and "converged" in row and "warnings" in row
        if row["status"] == "ok":
            assert np.isfinite(row["signal_mse"])


def test_cli_refuses_mismatched_native_sources(monkeypatch):
    import pytest
    from benches import probe_joint_regularization as probe

    monkeypatch.setattr(probe, "native_source_digest", lambda: "sha256:wrong")
    with pytest.raises(RuntimeError, match="native source digest"):
        probe.main(["--seeds", "1"])
