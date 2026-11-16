"""Prespecified model searches with one shared validation partition."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, mean_squared_error

METHODS = ["linear", "mboost_additive", "mboost_hybrid", "mboost_auto",
           "ebm", "catboost", "lightgbm", "xgboost"]
GRIDS = {
    "linear": [{"penalty": p} for p in (.1, 1., 10., 100.)],
    "mboost_additive": [{"df": d, "nu": nu} for d, nu in ((4, .1), (8, .1), (4, .05), (8, .05))],
    "mboost_hybrid": [{"df": d, "nu": .05, "depth": depth}
                      for d, depth in ((4, 3), (4, 6), (8, 3), (8, 6))],
    "ebm": [{"interactions": inter, "max_leaves": leaves}
            for inter, leaves in ((0, 2), ("5x", 2), (0, 3), ("5x", 3))],
    "catboost": [{"depth": d, "l2_leaf_reg": reg} for d, reg in ((6, 3), (8, 3), (4, 3), (6, 10))],
    "lightgbm": [{"num_leaves": leaves, "reg_lambda": reg}
                 for leaves, reg in ((31, 0), (63, 0), (15, 1), (31, 10))],
    "xgboost": [{"max_depth": d, "reg_lambda": reg} for d, reg in ((6, 1), (8, 1), (3, 1), (6, 10))],
}


def loss(y, prediction, kind):
    p = np.asarray(prediction, float).ravel()
    if len(p) != len(y) or not np.isfinite(p).all():
        raise ValueError("non-finite or misaligned prediction")
    return float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])
                 if kind == "binary" else mean_squared_error(y, p))


@dataclass
class Fitted:
    model: object
    method: str
    kind: str
    cats: list
    names: list
    levels: dict
    config: dict
    terms: list

    def data(self, frame):
        if self.method.startswith("mboost"):
            return frame
        if self.method in ("catboost", "ebm"):
            return frame.select(self.names).to_numpy()
        if self.method == "linear":
            return frame.select(self.names).to_pandas()
        data = frame.select(self.names).to_pandas()
        import pandas as pd

        for name in self.cats:
            data[name] = pd.Categorical(data[name], categories=self.levels[name])
        return data

    def predict(self, frame):
        data = self.data(frame)
        if self.method.startswith("mboost"):
            return np.asarray(self.model.predict(newdata=data, type="response",
                              unseen_levels="zero_effect", extrapolation="allow"), float).ravel()
        if self.kind == "binary":
            return np.asarray(self.model.predict_proba(data)[:, 1], float)
        return np.asarray(self.model.predict(data), float).ravel()


def clean(train, *others, cats):
    """Fit imputation on training rows, then replay; vocabularies stay native."""
    names = [c for c in train.columns if c != "y"]
    medians = {}
    for name in names:
        if name not in cats:
            x = train[name].cast(pl.Float64).to_numpy()
            finite = x[np.isfinite(x)]
            medians[name] = float(np.median(finite)) if finite.size else 0.0
    out = []
    for frame in (train, *others):
        expressions = []
        for name in names:
            if name in cats:
                expressions.append(pl.col(name).cast(pl.String).fill_null("__missing__"))
            else:
                expressions.append(pl.when(pl.col(name).is_finite()).then(pl.col(name))
                                   .otherwise(medians[name]).cast(pl.Float64).alias(name))
        out.append(frame.with_columns(expressions))
    return out


def formula(frame, cats, df, hybrid_depth=None):
    terms = []
    for name in frame.columns:
        if name == "y":
            continue
        if name in cats:
            terms.append(f"brandom({name}, df={df})")
        elif frame[name].n_unique() <= 2:
            terms.append(f"bols({name})")
        else:
            terms.append(f"bbs({name}, df={df})")
    if hybrid_depth is not None:
        names = [c for c in frame.columns if c != "y"]
        terms.append(f"btree({', '.join(names)}, maxdepth={hybrid_depth}, selection=sse)")
    return "y ~ " + " + ".join(terms), terms


def fit_model(method, config, train, cats, kind, seed, *, validation=None, stop=None):
    """Fit or refit one configuration; `stop` is selected without test rows."""
    names = [c for c in train.columns if c != "y"]
    y = train["y"].to_numpy()
    levels = {c: sorted(train[c].unique().to_list()) for c in cats}
    wrapped = Fitted(None, method, kind, cats, names, levels, dict(config), [])
    valid = None if validation is None else wrapped.data(validation)
    yv = None if validation is None else validation["y"].to_numpy()
    if method.startswith("mboost"):
        import mboostrs as mb

        family = mb.Binomial() if kind == "binary" else mb.Gaussian()
        if method == "mboost_auto":
            dev = pl.concat([train, validation])
            weights = np.r_[np.ones(len(train)), np.zeros(len(validation))][:, None]
            rec = mb.autospec(dev, "y", family=family, budget=config["budget"],
                              folds=weights, nu=.1, mstop=200, seed=seed,
                              interactions="adopt", residual_tree="auto")
            model = rec.fit(dev)
            wrapped.config.update(formula=rec.formula, mstop=rec.mstop, nu=rec.control.nu,
                                  probe_fits=rec.fits)
            wrapped.terms = model.base_learner_names()
            wrapped.model = model
            return wrapped
        rhs, terms = formula(train, cats, config["df"], config.get("depth"))
        wrapped.terms = terms
        if stop is None:
            dev = pl.concat([train, validation])
            weights = np.r_[np.ones(len(train)), np.zeros(len(validation))][:, None]
            model = mb.gamboost(rhs, dev, family=family,
                                control=mb.boost_control(1200, nu=config["nu"]))
            cv = model.cvrisk(folds=weights, preprocessing="refit")
            stop = int(cv.optimal_mstop)
            # The native root path already fitted all development rows;
            # slice it to the step selected by the independent CV path.
            wrapped.model = model.with_mstop(stop)
            wrapped.config.update(mstop=stop, formula=rhs, validation_loss=float(cv.aggregate[stop]))
        else:
            wrapped.model = mb.gamboost(rhs, train, family=family,
                                        control=mb.boost_control(stop, nu=config["nu"]))
        wrapped.config.update(mstop=stop, formula=rhs)
        return wrapped
    if method == "linear":
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric = [c for c in names if c not in cats]
        pre = ColumnTransformer([("numeric", StandardScaler(), numeric),
                                 ("category", OneHotEncoder(handle_unknown="ignore"), cats)])
        estimator = (LogisticRegression(C=1 / config["penalty"], max_iter=2000)
                     if kind == "binary" else Ridge(alpha=config["penalty"]))
        wrapped.model = make_pipeline(pre, estimator).fit(wrapped.data(train), y)
    elif method == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

        cls = ExplainableBoostingClassifier if kind == "binary" else ExplainableBoostingRegressor
        model = cls(**config, outer_bags=8, max_rounds=1500, n_jobs=1, random_state=seed)
        if validation is not None:
            dev = pl.concat([train, validation])
            rng = np.random.default_rng(seed)
            bags = np.empty((len(dev), 8), dtype=np.int32)
            for b in range(8):
                bags[:len(train), b] = rng.multinomial(len(train), np.full(len(train), 1 / len(train)))
                bags[len(train):, b] = -1
            model.fit(wrapped.data(dev), dev["y"].to_numpy(), bags=bags)
        else:
            model.fit(wrapped.data(train), y)
        wrapped.model = model
    elif method == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if kind == "binary" else CatBoostRegressor
        model = cls(**config, iterations=stop or 1200, learning_rate=.05, thread_count=1,
                    random_seed=seed, verbose=False, allow_writing_files=False,
                    cat_features=[names.index(c) for c in cats],
                    loss_function="Logloss" if kind == "binary" else "RMSE")
        model.fit(wrapped.data(train), y,
                  eval_set=None if validation is None else (valid, yv),
                  early_stopping_rounds=None if validation is None else 60)
        wrapped.model = model
        wrapped.config["iterations"] = int(model.tree_count_)
    elif method == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if kind == "binary" else lgb.LGBMRegressor
        model = cls(**config, n_estimators=stop or 1200, learning_rate=.05, n_jobs=1,
                    random_state=seed, verbosity=-1)
        model.fit(wrapped.data(train), y,
                  eval_set=None if validation is None else [(valid, yv)],
                  callbacks=[] if validation is None else [lgb.early_stopping(60, verbose=False)])
        wrapped.model = model
        wrapped.config["iterations"] = int(model.best_iteration_ or model.n_estimators)
    elif method == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if kind == "binary" else XGBRegressor
        model = cls(**config, n_estimators=stop or 1200, learning_rate=.05, n_jobs=1,
                    random_state=seed, enable_categorical=True, tree_method="hist",
                    early_stopping_rounds=None if validation is None else 60,
                    eval_metric="logloss" if kind == "binary" else "rmse")
        model.fit(wrapped.data(train), y, eval_set=None if validation is None else [(valid, yv)],
                  verbose=False)
        wrapped.model = model
        wrapped.config["iterations"] = int(model.best_iteration + 1 if validation is not None
                                             else model.n_estimators)
    else:
        raise ValueError(method)
    return wrapped


def search(method, train, val, cats, kind, seed, *, tier=None):
    """First configuration and validation-selected configuration; no test outcomes."""
    start = time.perf_counter()
    configs = [{"budget": 32}, {"budget": 128}] if method == "mboost_auto" else GRIDS[method]
    if tier == "small":
        configs = configs[:1]
    elif tier == "tuned" and method == "mboost_auto":
        configs = configs[-1:]
    rows, candidates, search_seconds = [], [], []
    for config in configs:
        cleaned_train, cleaned_val = clean(train, val, cats=cats)
        fitted = fit_model(method, config, cleaned_train, cats, kind, seed, validation=cleaned_val)
        if method in ("mboost_additive", "mboost_hybrid"):
            score = fitted.config["validation_loss"]
        elif method == "mboost_auto":
            # Autospec selected its spec using only the supplied native held-
            # out paths. Its final full-data fit must never score this val set.
            score = None
        else:
            score = loss(val["y"].to_numpy(), fitted.predict(cleaned_val), kind)
        rows.append({"config": fitted.config, "validation_loss": score})
        candidates.append(fitted)
        search_seconds.append(time.perf_counter() - start)
    best = (len(candidates) - 1 if method == "mboost_auto" else
            min(range(len(rows)), key=lambda i: rows[i]["validation_loss"]))
    outputs = []
    selections = (("small", 0, search_seconds[0]), ("tuned", best, search_seconds[-1]))
    for selected_tier, i, seconds in selections:
        if tier is not None and selected_tier != tier:
            continue
        refit_start = time.perf_counter()
        selected = candidates[i]
        if method != "mboost_auto":
            dev = pl.concat([train, val])
            dev = clean(dev, cats=cats)[0]
            selected = fit_model(method, configs[i], dev, cats, kind, seed,
                                 stop=selected.config.get("mstop", selected.config.get("iterations")))
        outputs.append((selected_tier, selected, {"search_seconds": seconds,
                        "refit_seconds": time.perf_counter() - refit_start,
                        "candidates": rows[:1] if selected_tier == "small" else rows,
                        "selected_candidate": i}))
    return outputs
