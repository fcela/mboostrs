"""Tests for CvRiskResult.to_frame() and StabSelResult.to_frame()."""

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture
def glm_model():
    rng = np.random.default_rng(1)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3}
    model = mb.glmboost(
        "y ~ x1 + x2 + x3", data=data, family=mb.Gaussian(),
        control=mb.boost_control(30, 0.1),
    )
    return model, data


def test_cvrisk_to_frame_shape_and_columns(glm_model):
    model, _ = glm_model
    cvr = model.cvrisk(type="kfold", B=5, seed=1)
    frame = cvr.to_frame(output="records")
    assert len(frame) == cvr.risk_matrix.shape[0] * cvr.risk_matrix.shape[1]
    assert set(frame[0].keys()) == {"fold", "mstop", "risk", "aggregate_risk", "is_optimal"}
    optimal_rows = [r for r in frame if r["is_optimal"]]
    assert len(optimal_rows) == cvr.risk_matrix.shape[0]
    assert all(r["mstop"] == cvr.optimal_mstop for r in optimal_rows)


def test_cvrisk_to_frame_values_match_risk_matrix(glm_model):
    model, _ = glm_model
    cvr = model.cvrisk(type="kfold", B=5, seed=1)
    frame = cvr.to_frame(output="records")
    for row in frame:
        fold_idx = row["fold"] - 1
        col = list(cvr.grid).index(row["mstop"])
        assert row["risk"] == pytest.approx(cvr.risk_matrix[fold_idx, col])
        assert row["aggregate_risk"] == pytest.approx(cvr.aggregate[col])


def test_stabsel_to_frame_shape_and_columns(glm_model):
    model, _ = glm_model
    sel = mb.stabsel(model, q=2, PFER=1)
    frame = sel.to_frame(output="records")
    assert len(frame) == sel.p
    assert set(frame[0].keys()) == {"learner", "max_phat", "selected"}
    names = [row["learner"] for row in frame]
    assert names == sel.learner_names
    selected_names = {row["learner"] for row in frame if row["selected"]}
    assert selected_names == {sel.learner_names[i] for i in sel.selected}


def test_to_frame_supports_polars_and_pandas_when_available(glm_model):
    model, _ = glm_model
    cvr = model.cvrisk(type="kfold", B=3, seed=1)
    sel = mb.stabsel(model, q=2, PFER=1)

    pl = pytest.importorskip("polars")
    assert isinstance(cvr.to_frame(output="polars"), pl.DataFrame)
    assert isinstance(sel.to_frame(output="polars"), pl.DataFrame)

    pd = pytest.importorskip("pandas")
    assert isinstance(cvr.to_frame(output="pandas"), pd.DataFrame)
    assert isinstance(sel.to_frame(output="pandas"), pd.DataFrame)
