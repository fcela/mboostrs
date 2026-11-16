"""Prediction, calibration, structure and equal-work explanation measurements."""
from __future__ import annotations

import gzip
import pickle
import time

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, r2_score, roc_auc_score

from benches.shootout_data import truth
from benches.shootout_models import loss


def interval_radius(residuals, level=.9):
    """Finite-sample split-conformal order statistic, without interpolating."""
    residuals = np.asarray(residuals, float)
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("calibration residuals must be nonempty and finite")
    rank = min(residuals.size, int(np.ceil((residuals.size + 1) * level)))
    return float(np.partition(residuals, rank - 1)[rank - 1])


def evaluate(fitted, cal, test, train):
    y, yc = test["y"].to_numpy(), cal["y"].to_numpy()
    begin = time.perf_counter()
    p, pc = fitted.predict(test), fitted.predict(cal)
    scores = {"loss": loss(y, p, fitted.kind)}
    if fitted.kind == "binary":
        scores.update(brier=float(brier_score_loss(y, p)), auc=float(roc_auc_score(y, p)))
        def logits(prob):
            prob = np.clip(prob, 1e-8, 1 - 1e-8)
            return np.log(prob / (1 - prob))[:, None]
        calibration = LogisticRegression(C=1e6, max_iter=1000).fit(logits(pc), yc)
        calibrated = calibration.predict_proba(logits(p))[:, 1]
        scores["calibrated_loss"] = loss(y, calibrated, "binary")
    else:
        radius = interval_radius(np.abs(yc - pc))
        scores.update(r2=float(r2_score(y, p)), rmse=float(np.sqrt(scores["loss"])),
                      coverage90=float(np.mean(np.abs(y - p) <= radius)),
                      width90=float(2 * radius),
                      width90_sd=float(2 * radius / np.std(train["y"].to_numpy())))
    seconds = time.perf_counter() - begin
    batch = test.head(min(256, len(test)))
    fitted.predict(batch)  # warmup; the seven repetitions use identical inputs
    latency = []
    for _ in range(7):
        start = time.perf_counter()
        fitted.predict(batch)
        latency.append(time.perf_counter() - start)
    scores.update(calibration_prediction_seconds=seconds,
                  prediction_batch_rows=len(batch), prediction_seconds=latency)
    return scores, p


def structure(fitted):
    start = time.perf_counter()
    out = {"raw_inputs": len(fitted.names), "native_terms": None,
           "used_inputs": None, "univariate_terms": None, "joint_terms": None}
    try:
        out["gzip_pickle_bytes"] = len(gzip.compress(pickle.dumps(fitted, protocol=5)))
    except Exception as exc:
        out["storage_error"] = f"{type(exc).__name__}: {exc}"
    m = fitted.model
    if fitted.method.startswith("mboost"):
        imp = m.varimp()
        used = [term for term, value in imp["selfreq"].items() if value > 0]
        # Candidate columns per retained learner: a joint btree lists every
        # column it may split on, not just the one the label is named for.
        # Split-used columns narrow that to what the fitted trees actually
        # split on (varimp()["split_columns"]). Retained-input counting uses
        # the latter; the former is reported alongside for transparency.
        candidate = {c.strip() for term in used
                     for c in imp["variable_names"][term].split(",") if c.strip()}
        split_used = {c.strip() for term in used
                      for c in imp["split_columns"][term].split(",") if c.strip()}
        arity = [len([c for c in imp["variable_names"][term].split(",") if c.strip()])
                 for term in used]
        out.update(used_inputs=len(split_used), candidate_inputs=len(candidate),
                   native_terms=len(used),
                   univariate_terms=sum(a == 1 for a in arity),
                   joint_terms=sum(a > 1 for a in arity), mstop=int(m.mstop),
                   term_variables={term: imp["variable_names"][term] for term in used},
                   term_split_columns={term: imp["split_columns"][term] for term in used},
                   structural_unit="retained learner coefficient scalars")
        try:
            out["structure_size"] = sum(np.asarray(v).size for v in m.coef().values())
        except Exception as exc:
            out["structure_error"] = str(exc)
    elif fitted.method == "ebm":
        arity = [len(term) for term in m.term_features_]
        out.update(used_inputs=len(set(i for term in m.term_features_ for i in term)),
                   native_terms=len(arity), univariate_terms=sum(a == 1 for a in arity),
                   joint_terms=sum(a > 1 for a in arity),
                   structure_size=sum(np.asarray(a).size for a in m.term_scores_),
                   structural_unit="stored EBM score cells")
    elif fitted.method == "linear":
        estimator = m.steps[-1][1]
        out.update(used_inputs=len(fitted.names), native_terms=len(fitted.names),
                   univariate_terms=len(fitted.names), joint_terms=0,
                   structure_size=int(np.asarray(estimator.coef_).size),
                   structural_unit="encoded linear coefficients")
    elif fitted.method == "catboost":
        out.update(used_inputs=int(np.sum(m.feature_importances_ > 0)),
                   structure_size=int(np.sum(m.get_tree_leaf_counts())),
                   structural_unit="tree leaves", trees=int(m.tree_count_))
    elif fitted.method == "lightgbm":
        dump = m.booster_.dump_model()
        out.update(used_inputs=int(np.sum(m.feature_importances_ > 0)),
                   structure_size=sum(t["num_leaves"] for t in dump["tree_info"]),
                   structural_unit="tree leaves", trees=len(dump["tree_info"]))
    elif fitted.method == "xgboost":
        frame = m.get_booster().trees_to_dataframe()
        out.update(used_inputs=int(np.sum(m.feature_importances_ > 0)),
                   structure_size=int(np.sum(frame.Feature == "Leaf")),
                   structural_unit="tree leaves", trees=int(frame.Tree.nunique()))
    out["report_storage_seconds"] = time.perf_counter() - start
    return out


def explanations(fitted, train, task, seed):
    """Same background, permutation and marginal-response workload for every model."""
    begin = time.perf_counter()
    rng = np.random.default_rng(seed + 4000)
    rows = rng.choice(len(train), min(128, len(train)), replace=False)
    background = train[rows.tolist()]
    y = background["y"].to_numpy()
    baseline = loss(y, fitted.predict(background), fitted.kind)
    importance, curves, truth_curves = {}, {}, {}
    for name in fitted.names:
        permuted = background.with_columns(pl.Series(name, background[name].to_numpy()[rng.permutation(len(rows))]))
        importance[name] = loss(y, fitted.predict(permuted), fitted.kind) - baseline
        if name in fitted.cats:
            continue
        grid = np.quantile(train[name].to_numpy(), np.linspace(.05, .95, 9))
        values, targets = [], []
        for point in grid:
            changed = background.with_columns(pl.lit(float(point)).alias(name))
            values.append(float(fitted.predict(changed).mean()))
            if task in ("smooth_sparse", "joint_boundary"):
                targets.append(float(truth(changed.select(fitted.names).to_numpy(), task).mean()))
        centred = np.asarray(values) - np.mean(values)
        curves[name] = {"grid": grid.tolist(), "effect": centred.tolist()}
        if targets:
            truth_curves[name] = (np.asarray(targets) - np.mean(targets)).tolist()
    out = {"importance": importance, "seconds": time.perf_counter() - begin,
           "background_rows": len(rows), "curve_points": 9}
    native_start = time.perf_counter()
    m, X = fitted.model, fitted.data(background)
    if fitted.method.startswith("mboost"):
        parts = [np.asarray(m.predict(newdata=background, type="link", which=n,
                                     unseen_levels="zero_effect", extrapolation="allow"))
                 for n in m.base_learner_names()]
        reconstruction = np.sum(parts, axis=0) + np.asarray(m.offset_)
        reference = m.predict(newdata=background, type="link", unseen_levels="zero_effect",
                              extrapolation="allow")
        label = "native learner terms; a joint tree remains a joint prediction term"
    elif fitted.method == "ebm":
        reconstruction = m.eval_terms(X).sum(axis=1) + m.intercept_
        reference = m.decision_function(X) if fitted.kind == "binary" else m.predict(X)
        label = "native one- and two-variable score tables"
    elif fitted.method == "catboost":
        from catboost import Pool

        parts = m.get_feature_importance(Pool(X, cat_features=[fitted.names.index(c) for c in fitted.cats]),
                                         type="ShapValues", thread_count=1)
        reconstruction = parts.sum(axis=1)
        reference = m.predict(X, prediction_type="RawFormulaVal")
        label = "native SHAP feature attributions; post-hoc decomposition of an ensemble"
    elif fitted.method == "lightgbm":
        reconstruction = np.asarray(m.booster_.predict(X, pred_contrib=True)).sum(axis=1)
        reference = m.booster_.predict(X, raw_score=True)
        label = "native SHAP feature attributions; post-hoc decomposition of an ensemble"
    elif fitted.method == "xgboost":
        import xgboost as xgb

        matrix = xgb.DMatrix(X, enable_categorical=True)
        reconstruction = m.get_booster().predict(matrix, pred_contribs=True).sum(axis=1)
        reference = m.get_booster().predict(matrix, output_margin=True)
        label = "native SHAP feature attributions; post-hoc decomposition of an ensemble"
    else:
        estimator = m.steps[-1][1]
        encoded = m.steps[0][1].transform(X)
        reconstruction = np.asarray(encoded @ np.asarray(estimator.coef_).ravel()).ravel() + estimator.intercept_
        reference = m.decision_function(X) if fitted.kind == "binary" else m.predict(X)
        label = "native encoded linear terms"
    out["native_decomposition"] = {"kind": label,
                                    "max_abs_error": float(np.max(np.abs(reconstruction - reference))),
                                    "seconds": time.perf_counter() - native_start}
    out["seconds"] += out["native_decomposition"]["seconds"]
    if truth_curves:
        a = np.array([curves[n]["effect"] for n in truth_curves])
        b = np.array(list(truth_curves.values()))
        out["curve_nrmse"] = float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))
        out["truth_curves"] = truth_curves
    if task in ("smooth_sparse", "joint_boundary", "correlated_proxies"):
        groups = [[f"x{i}"] for i in range(12)]
        if task == "correlated_proxies":
            groups = [["x0", "x3"], ["x1", "x4"], ["x2"]] + [[f"x{i}"] for i in range(5, 12)]
            gains = []
            for group in groups:
                order = rng.permutation(len(rows))
                changed = background.with_columns(
                    [pl.Series(n, background[n].to_numpy()[order]) for n in group])
                gains.append(loss(y, fitted.predict(changed), fitted.kind) - baseline)
        else:
            gains = [importance[group[0]] for group in groups]
        ranked = sorted(range(len(groups)), key=lambda i: gains[i], reverse=True)
        out["driver_group_recall_at3"] = len(set(ranked[:3]) & {0, 1, 2}) / 3
        out["driver_groups"] = groups
        out["group_importance"] = gains
    top = sorted(importance, key=importance.get, reverse=True)[:5]
    if task in ("smooth_sparse", "joint_boundary", "correlated_proxies"):
        top = list(curves)
    out["curves"] = {n: curves[n] for n in top if n in curves}
    out["seconds"] = time.perf_counter() - begin
    return out
