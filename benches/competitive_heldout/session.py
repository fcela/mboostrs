"""D5 analyst session of PROTOCOL.md for one arm; prints one JSON line.

50,000 x 20 planted data (S1 structure plus 17 noise columns). Ten follow-up
fits, each followed by prediction on 2,000 rows and a 3-repeat permutation
importance on them. Same idiomatic calls as run_one.py, without intervals.
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

arm = sys.argv[1]
rng = np.random.default_rng(0)
n, p = 52000, 20
X = rng.uniform(-1, 1, (n, p))
y = 2 * np.sin(2 * X[:, 0]) + 2 * (X[:, 1] ** 2 - 1 / 3) + X[:, 2] + rng.normal(0, 0.5, n)
base = pl.DataFrame({f"x{i}": X[:, i] for i in range(p)} | {"y": y})
train_all, test_all = base[:50000], base[50000:]


def follow_ups():
    yield "base", train_all, test_all, False
    for k in range(5):
        yield f"drop_x{k}", train_all.drop(f"x{k}"), test_all.drop(f"x{k}"), False
    add = lambda f: f.with_columns((pl.col("x0") * pl.col("x1")).alias("x0_x1"))  # noqa: E731
    yield "add_x0_x1", add(train_all), add(test_all), False
    yield "subsample_50pct", train_all.sample(fraction=0.5, seed=1), test_all, False
    extra = lambda f, s: f.with_columns([pl.Series(f"z{j}", np.random.default_rng(s + j).uniform(-1, 1, f.height)) for j in range(5)])  # noqa: E731
    yield "five_more_noise", extra(train_all, 10), extra(test_all, 20), False
    thr = float(np.median(train_all["y"]))
    binarize = lambda f: f.with_columns((pl.col("y") > thr).cast(pl.Float64).alias("y"))  # noqa: E731
    yield "binary_response", binarize(train_all), binarize(test_all), True


def fit_predict(train, binary):
    cols = [c for c in train.columns if c != "y"]
    yv = train["y"].to_numpy()
    Xp = lambda f: pd.DataFrame({c: f[c].to_numpy() for c in cols})  # noqa: E731
    if arm == "mboostrs":
        import mboostrs as mb
        r = mb.fit(train, "y", seed=0)

        def pr(f):
            v = r.predict(f, type="response")
            return np.asarray(r.model.predict(f, type="mean") if isinstance(v, dict) else v, float).ravel()
        return pr
    if arm == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        m = (CatBoostClassifier if binary else CatBoostRegressor)(thread_count=1, random_seed=0, verbose=0,
                                                                    allow_writing_files=False).fit(Xp(train), yv)
    elif arm == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        m = (LGBMClassifier if binary else LGBMRegressor)(n_jobs=1, random_state=0, verbose=-1).fit(Xp(train), yv)
    elif arm == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        m = (XGBClassifier if binary else XGBRegressor)(n_jobs=1, random_state=0, tree_method="hist").fit(Xp(train), yv)
    elif arm == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
        m = (ExplainableBoostingClassifier if binary else ExplainableBoostingRegressor)(n_jobs=1, random_state=0).fit(Xp(train), yv)
    elif arm == "reference":
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000) if binary else RidgeCV()).fit(Xp(train), yv)
    return (lambda f: m.predict_proba(Xp(f))[:, 1]) if binary else (lambda f: m.predict(Xp(f)))


def loss(yt, pv, binary):
    if binary:
        q = np.clip(pv, 1e-12, 1 - 1e-12)
        return float(-np.mean(yt * np.log(q) + (1 - yt) * np.log(1 - q)))
    return float(np.mean((yt - pv) ** 2))


steps, prng = [], np.random.default_rng(5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, train, test, binary in follow_ups():
        t0 = time.perf_counter()
        status = "ok"
        try:
            pr = fit_predict(train, binary)
            yt = test["y"].to_numpy()
            b = loss(yt, pr(test), binary)
            for c in [c for c in test.columns if c != "y"]:
                for _ in range(3):
                    loss(yt, pr(test.with_columns(pl.Series(c, prng.permutation(test[c].to_numpy())))), binary)
        except Exception as exc:
            status = f"error: {type(exc).__name__}: {exc}"[:200]
        steps.append(dict(step=name, seconds=round(time.perf_counter() - t0, 2), status=status))
sec = np.array([s["seconds"] for s in steps])
print(json.dumps(dict(arm=arm, median_seconds=float(np.median(sec)), p90_seconds=float(np.percentile(sec, 90)),
                      total_seconds=float(sec.sum()), steps=steps)), flush=True)
