"""Discoverability contracts from issue #79.

Covers three items:

- Empty result tables: every public result class with a ``to_frame`` method
  returns a frame with the same columns and zero rows when it holds no rows.
- sklearn ``LSSRegressor``: ``predict`` is the conditional mean, ``score``
  is computed from it, and the estimator works in pipelines and with
  sample-weight routing.
- The task-to-API map (``docs/skills/analyzing-data-with-mboostrs/topics/task-map.md``)
  names only functions that exist in Python, in the R NAMESPACE and in the Rust
  core. The map checks are at the end of this file.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mboostrs as mb
from mboostrs.sklearn import LSSRegressor

# --------------------------------------------------------------------------
# Item 3: zero-row tables
# --------------------------------------------------------------------------

#: Public result classes whose ``to_frame`` returns zero rows in a real state:
#: no candidate learners, no folds, no retained records, no observations.
ZERO_ROW_CLASSES = (
    "AicResult",
    "AIPWResult",
    "ContrastResult",
    "CvRiskResult",
    "DistributionEvaluationResult",
    "ElasticityResult",
    "EvaluationResult",
    "LssCvRiskResult",
    "LssStabSelResult",
    "MarginalEffectResult",
    "StabSelResult",
)

#: Public result classes whose constructor refuses zero rows (a grid, bin
#: edges, a fold matrix or an interval must be non-empty), so their table
#: always has rows and has no zero-row case to return.
NEVER_EMPTY_CLASSES = (
    "ALE1DResult",
    "ConformalInterval",
    "FoldScheme",
    "IVResult",
    "PredictionProfileResult",
)

#: Public result classes whose ``to_frame`` always has one row. They have no
#: zero-row case; the test checks the one-row table with the same columns.
ONE_ROW_CLASSES = ("SameFamilyGap",)


def _rng_data(n=160, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.uniform(1.0, 3.0, n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-x1))).astype(float)
    y = 1 + 0.5 * x1 - 0.3 * x2 + 0.2 * a + rng.normal(scale=0.3, size=n)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3, "a": a}


@pytest.fixture(scope="module")
def full_results():
    """One non-empty instance of every result class listed above."""
    data = _rng_data()
    n = len(data["y"])
    ctrl = mb.boost_control(40, 0.1)
    glm = mb.glmboost("y ~ x1 + x2", data=data, family=mb.Gaussian(), control=ctrl)
    positive = dict(data, y=data["y"] + 5.0)
    gpos = mb.glmboost("y ~ x1 + x2 + x3", data=positive, family=mb.Gaussian(),
                       control=ctrl)
    lss = mb.glmboostLSS("y ~ x1", data, families=mb.GaussianLSS(),
                         control=mb.boost_control(40, 0.1))
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"), estimand="ate",
        target_population="sample", unit="person", time_zero="assignment",
        follow_up="one year", adjustment_rationale="x1 and x2 precede a",
    )
    cvr = glm.cvrisk(type="kfold", B=3, seed=1, parallel=False)
    return {
        "AicResult": glm.aic(),
        "AIPWResult": mb.aipw(data, spec, folds=3, mstop=30, tune=False),
        "ALE1DResult": mb.accumulated_local_effect(glm, data, "x1", bins=4),
        "ContrastResult": mb.prediction_contrast(glm, data, "x1",
                                                 reference=-1.0, alternative=1.0),
        "ElasticityResult": mb.average_elasticity(gpos, positive, "x3"),
        "MarginalEffectResult": mb.average_marginal_effect(glm, data, "x1"),
        "PredictionProfileResult": mb.prediction_profile(glm, data, "x1", points=5),
        "CvRiskResult": cvr,
        "EvaluationResult": mb.evaluate_specification("x1 + x2", data, "y",
                                                      folds=3, mstop=30),
        "DistributionEvaluationResult": mb.evaluate_distribution(
            lss, data, "y", data_role="external"),
        "StabSelResult": mb.stabsel(glm, q=1, PFER=1, B=10, verbose=False, n_jobs=1),
        "LssCvRiskResult": mb.cvrisk_glmboostLSS(
            "y ~ x1 + x2", data, families=mb.GaussianLSS(), control=ctrl,
            folds=mb.cv_weights(n, "kfold", 5)),
        "LssStabSelResult": mb.stabsel_glmboostLSS(
            "y ~ x1 + x2", data, families=mb.GaussianLSS(), control=ctrl,
            B=6, q=1, PFER=1),
        "ConformalInterval": mb.conformal_interval(glm, newdata=data, level=0.9,
                                                   seed=0),
        "SameFamilyGap": mb.same_family_gap(glm),
        "FoldScheme": mb.loo_folds(10),
        "IVResult": mb.iv_effect(
            dict(data, z=data["a"] + np.random.default_rng(0).normal(size=n)),
            spec, "z", folds=3, tune=False, mstop=40),
    }


def _zero_rows(name, full):
    """The same result class holding no rows.

    Per-row arrays are cut to length zero; classes whose row count comes from
    a field other than the per-row arrays get that field set explicitly.
    """
    if name == "AicResult":
        return dataclasses.replace(full, aic=full.aic[:0], df=full.df[:0], mstop=0)
    if name == "CvRiskResult":
        return mb.CvRiskResult(
            risk_matrix=full.risk_matrix[:0], aggregate=full.aggregate,
            grid=full.grid, optimal_mstop=full.optimal_mstop,
            fold_type=full.fold_type, provenance=full.provenance,
        )
    if name == "LssCvRiskResult":
        return dataclasses.replace(full, risk_matrix=full.risk_matrix[:0])
    if name == "EvaluationResult":
        # The case a user meets with retain_arrays=False: no records, no folds.
        return dataclasses.replace(full, records=(), folds=())
    if name == "StabSelResult":
        return dataclasses.replace(
            full, p=0, phat=full.phat[:0], max_=full.max_[:0],
            learner_names=[], selected=[],
        )
    if name == "LssStabSelResult":
        return dataclasses.replace(
            full, phat=full.phat[:0], max_=full.max_[:0],
            learner_names=[], selected=[],
        )

    def sliced(value):
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return value[:0]
        if isinstance(value, Mapping):
            return {key: sliced(item) for key, item in value.items()}
        return value

    changes = {}
    for field in dataclasses.fields(full):
        value = getattr(full, field.name)
        if isinstance(value, (np.ndarray, Mapping)):
            changes[field.name] = sliced(value)
    return dataclasses.replace(full, **changes)


def _column_names(frame):
    return list(frame.columns)


@pytest.mark.parametrize("name", ZERO_ROW_CLASSES)
def test_zero_row_table_keeps_its_columns(full_results, name):
    pl = pytest.importorskip("polars")
    pd = pytest.importorskip("pandas")
    full = full_results[name]
    empty = _zero_rows(name, full)

    full_frame = full.to_frame(output="polars")
    assert isinstance(full_frame, pl.DataFrame)
    assert full_frame.height > 0, "fixture must hold rows"

    empty_polars = empty.to_frame(output="polars")
    assert isinstance(empty_polars, pl.DataFrame)
    assert empty_polars.height == 0
    assert _column_names(empty_polars) == _column_names(full_frame)

    empty_pandas = empty.to_frame(output="pandas")
    assert isinstance(empty_pandas, pd.DataFrame)
    assert len(empty_pandas) == 0
    assert list(empty_pandas.columns) == _column_names(full_frame)

    assert empty.to_frame(output="records") == []


@pytest.mark.parametrize("name", NEVER_EMPTY_CLASSES)
def test_constructor_refuses_zero_rows_so_the_table_always_has_rows(full_results, name):
    """No zero-row state exists for these classes; the constructor is what keeps it so."""
    with pytest.raises(ValueError):
        _zero_rows(name, full_results[name])


@pytest.mark.parametrize("name", ONE_ROW_CLASSES)
def test_one_row_table_keeps_its_columns_when_comparison_is_unavailable(full_results, name):
    pl = pytest.importorskip("polars")
    full = full_results[name]
    assert full.available, "fixture must hold a comparison"
    unavailable = dataclasses.replace(
        full, available=False, boost_loglik=None, comparison_loglik=None,
        gap=None, material=False,
    )
    frame = unavailable.to_frame(output="polars")
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 1
    assert _column_names(frame) == _column_names(full.to_frame(output="polars"))


def _result_classes_with_to_frame():
    found = set()
    for attr in dir(mb):
        if attr == "plot":  # loads the plotting module; not a result class
            continue
        obj = getattr(mb, attr)
        if isinstance(obj, type) and callable(getattr(obj, "to_frame", None)):
            found.add(obj.__name__)
    return found


def test_every_public_class_with_to_frame_is_covered_here():
    """A new result class with a ``to_frame`` must be added to the lists above."""
    covered = set(ZERO_ROW_CLASSES) | set(NEVER_EMPTY_CLASSES) | set(ONE_ROW_CLASSES)
    assert _result_classes_with_to_frame() == covered


# --------------------------------------------------------------------------
# Item 5: LSSRegressor predicts the conditional mean
# --------------------------------------------------------------------------


def _family_data(family, n=300, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, (n, 1))
    if family == "gaussian":
        y = 1.0 + 1.5 * X[:, 0] + rng.normal(scale=0.5, size=n)
    elif family == "studentt":
        y = 1.0 + 1.5 * X[:, 0] + rng.standard_t(8, size=n)
    elif family == "gamma":
        y = np.exp(0.3 + 0.6 * X[:, 0]) * rng.gamma(4.0, 0.25, size=n)
    elif family == "negativebinomial":
        y = rng.poisson(np.exp(0.5 + 0.6 * X[:, 0]) * rng.gamma(3.0, 1 / 3.0, size=n))
    else:  # pragma: no cover
        raise AssertionError(family)
    return X, y.astype(float)


@pytest.mark.parametrize("family", ["gaussian", "studentt", "gamma", "negativebinomial"])
def test_predict_is_the_family_conditional_mean(family):
    X, y = _family_data(family)
    est = LSSRegressor(family=family, mstop=60, nu=0.1).fit(X, y)
    pred = est.predict(X)
    expected = est.model_._family.mean(est.model_.params(newdata=est._frame(X)))
    np.testing.assert_allclose(pred, np.asarray(expected).ravel(), rtol=0, atol=1e-12)
    if family != "studentt":
        # For these families the location parameter mu is the mean.
        np.testing.assert_allclose(pred, est.predict_params(X)["mu"], rtol=0, atol=1e-12)


def test_studentt_mean_is_the_location_where_df_exceeds_one():
    """With df > 1 the Student-t mean equals its location mu."""
    X, y = _family_data("studentt")
    est = LSSRegressor(family="studentt", mstop=60).fit(X, y)
    params = est.predict_params(X)
    assert np.all(np.asarray(params["df"]) > 1)
    np.testing.assert_allclose(est.predict(X), np.asarray(params["mu"]).ravel(),
                               rtol=0, atol=1e-12)


def test_studentt_row_without_a_finite_mean_is_refused_with_remedy(monkeypatch):
    X, y = _family_data("studentt")
    est = LSSRegressor(family="studentt", mstop=60).fit(X, y)
    real_params = est.model_.params

    def params_with_df_at_most_one(newdata=None, **kwargs):
        out = dict(real_params(newdata=newdata, **kwargs))
        out["df"] = np.full(len(next(iter(out.values()))), 0.9)
        return out

    monkeypatch.setattr(est.model_, "params", params_with_df_at_most_one)
    with pytest.raises(ValueError, match=r"StudentTLSS.*predict_params.*predict_interval"):
        est.predict(X)
    with pytest.raises(ValueError, match="StudentTLSS"):
        est.score(X, y)


def test_score_is_r_squared_of_the_conditional_mean():
    X, y = _family_data("gamma")
    est = LSSRegressor(family="gamma", mstop=60).fit(X, y)
    assert est.score(X, y) == pytest.approx(r2_score(y, est.predict(X)))


def test_pipeline_and_cross_validation_run_on_the_mean_prediction():
    X, y = _family_data("gaussian")
    pipe = make_pipeline(StandardScaler(), LSSRegressor(mstop=60))
    scores = cross_val_score(pipe, X, y, cv=3)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    # The fold score is R-squared of the mean the pipeline predicts.
    fitted = clone(pipe).fit(X[:200], y[:200])
    assert fitted.score(X[200:], y[200:]) == pytest.approx(
        r2_score(y[200:], fitted.predict(X[200:])))


def test_grid_search_over_mstop_selects_on_the_mean():
    X, y = _family_data("gaussian")
    search = GridSearchCV(LSSRegressor(mstop=20), {"mstop": [20, 60]}, cv=3)
    search.fit(X, y)
    assert search.best_params_["mstop"] in (20, 60)
    assert np.isfinite(search.best_score_)


def test_sample_weights_are_routed_through_grid_search_when_requested():
    """Metadata routing delivers sample_weight to the estimator's fit."""
    X, y = _family_data("gaussian")
    weights = np.where(np.arange(len(y)) % 2 == 0, 1.0, 3.0)
    with config_context(enable_metadata_routing=True):
        # score takes sample_weight too; the search routes it to the scorer.
        estimator = (LSSRegressor(mstop=40)
                     .set_fit_request(sample_weight=True)
                     .set_score_request(sample_weight=True))
        search = GridSearchCV(estimator, {"mstop": [40]}, cv=3)
        search.fit(X, y, sample_weight=weights)
    direct = LSSRegressor(mstop=40).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(search.best_estimator_.predict(X), direct.predict(X),
                               rtol=0, atol=1e-10)


def test_sample_weights_change_the_fit():
    X, y = _family_data("gaussian")
    weights = np.where(np.arange(len(y)) % 2 == 0, 1.0, 5.0)
    plain = LSSRegressor(mstop=40).fit(X, y).predict(X)
    weighted = LSSRegressor(mstop=40).fit(X, y, sample_weight=weights).predict(X)
    assert not np.allclose(plain, weighted)


# --------------------------------------------------------------------------
# Item 2: the task-to-API map
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
TASK_MAP = REPO / "docs" / "skills" / "analyzing-data-with-mboostrs" / "topics" / "task-map.md"
R_NAMESPACE = (REPO / "mboostrs" / "NAMESPACE").read_text(encoding="utf-8")
RUST_CORE_SRC = REPO / "crates" / "mboostrs-core" / "src"

TASKS = ("Fit", "Tune", "Evaluate", "Diagnose", "Compare", "Interpret",
         "Predict with uncertainty", "Export")
_MISSING = object()
_NO_FUNCTION = "—"


def parse_task_map(text):
    """Rows of every table in the map, with the task section each sits under."""
    rows = []
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] == "Python" or set(cells[0]) <= set("-: "):
            continue  # header and separator rows
        if len(cells) != 4:
            raise ValueError(f"task-map row must have four cells: {line!r}")
        python, r, rust, notes = cells
        rows.append({"task": section, "python": python, "r": r, "rust": rust,
                     "notes": notes})
    return rows


def _listed_name(cell):
    """The backticked name in a table cell, or None for a frontend with no function."""
    if cell == _NO_FUNCTION:
        return None
    names = re.findall(r"`([^`]+)`", cell)
    if len(names) != 1:
        raise ValueError(f"cell {cell!r} must hold one backticked name or {_NO_FUNCTION}")
    return names[0]


def python_problem(cell):
    name = _listed_name(cell)
    if name is None:
        return None
    obj = mb
    for part in name.split("."):
        obj = getattr(obj, part, _MISSING)
        if obj is _MISSING:
            return f"Python name {name!r} is not defined on mboostrs"
    return None


def r_problem(cell, namespace=R_NAMESPACE):
    name = _listed_name(cell)
    if name is None:
        return None
    pattern = re.compile(
        rf'^(?:export|S3method)\(\s*"?{re.escape(name)}"?\s*[,)]', re.MULTILINE)
    if not pattern.search(namespace):
        return f"R name {name!r} is not exported or registered in mboostrs/NAMESPACE"
    return None


def rust_problem(cell, src=RUST_CORE_SRC):
    name = _listed_name(cell)
    if name is None:
        return None
    if name.count("::") != 1:
        return f"Rust entry point {name!r} must be written file.rs::name"
    file_name, identifier = name.split("::")
    path = src / file_name
    if not path.is_file():
        return f"Rust file {file_name!r} is not under crates/mboostrs-core/src"
    declared = re.search(
        rf"pub\s+(?:fn|struct|enum)\s+{re.escape(identifier)}\b",
        path.read_text(encoding="utf-8"))
    if declared is None:
        return f"{identifier!r} is not a pub fn, struct or enum in {file_name}"
    return None


# The task map is split across two topic files (interpret and export moved to
# task-map-interpret-export.md), so the table is read from both.
TASK_ROWS = parse_task_map(
    TASK_MAP.read_text(encoding="utf-8")
    + (TASK_MAP.parent / "task-map-interpret-export.md").read_text(encoding="utf-8"))


def _row_id(row):
    return f"{row['task']}-{row['python']}".replace("`", "")


def test_task_map_has_a_table_for_each_task():
    found = {row["task"] for row in TASK_ROWS}
    assert found == set(TASKS)
    assert len(TASK_ROWS) >= 40


@pytest.mark.parametrize("row", TASK_ROWS, ids=_row_id)
def test_task_map_python_name_exists(row):
    assert python_problem(row["python"]) is None


@pytest.mark.parametrize("row", TASK_ROWS, ids=_row_id)
def test_task_map_r_name_is_in_the_namespace(row):
    assert r_problem(row["r"]) is None


@pytest.mark.parametrize("row", TASK_ROWS, ids=_row_id)
def test_task_map_rust_entry_point_exists(row):
    assert rust_problem(row["rust"]) is None


@pytest.mark.parametrize("row", TASK_ROWS, ids=_row_id)
def test_task_map_explains_each_missing_frontend_function(row):
    """A dash in the R or Rust cell must come with the reason in Notes."""
    if row["r"] == _NO_FUNCTION or row["rust"] == _NO_FUNCTION:
        assert row["notes"] not in ("", _NO_FUNCTION), row["python"]


def test_task_map_checks_fail_on_a_name_that_is_not_there():
    """The checks above are only useful if they reject a missing name."""
    assert python_problem("`no_such_function`") is not None
    assert python_problem("`GLMBoostModel.no_such_method`") is not None
    assert r_problem("`no_such_function`") is not None
    assert rust_problem("`engine.rs::no_such_function`") is not None
    assert rust_problem("`no_such_file.rs::try_fit`") is not None


def test_task_map_checks_pass_on_names_that_are_there():
    assert python_problem("`glmboost`") is None
    assert python_problem("`GLMBoostModel.cvrisk`") is None
    assert r_problem("`cvrisk`") is None
    assert r_problem("`coef`") is None  # registered as S3method(coef, ...)
    assert rust_problem("`engine.rs::try_fit`") is None
