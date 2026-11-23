"""One (task, seed, arm) run of the competitive study; prints one JSON line.

See PROTOCOL.md. Real tasks (integer ids) come from ../proof/data with a
seeded 80/20 split; planted tasks (S*_g) from ./data with fixed rows.
"""
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

HERE = pathlib.Path(__file__).resolve().parent
REAL = HERE.parent / "fit_heldout" / "data"
tid, seed, arm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
planted = tid.startswith("S")

if planted:
    task = {t["id"]: t for t in json.load(open(HERE / "data/manifest_synth.json"))["tasks"]}[tid]
    frame = pl.read_parquet(HERE / f"data/{tid}.parquet")
    tr, te = np.arange(task["n_train"]), np.arange(task["n_train"], frame.height)
    known = []
else:
    did = int(tid)
    task = {t["id"]: t for f in ("manifest.json", "manifest_uci.json", "manifest_mb.json")
            for t in json.load(open(REAL / f))["tasks"]}[did]
    frame = pl.read_parquet(REAL / f"{did}.parquet")
    strat = frame["y"].to_numpy() if task["kind"] == "binary" else None
    tr, te = train_test_split(np.arange(frame.height), test_size=0.2, random_state=seed, stratify=strat)
    rng = np.random.default_rng(seed)
    if len(tr) > 4000:
        tr = rng.choice(tr, 4000, replace=False)
    if len(te) > 2000:
        te = rng.choice(te, 2000, replace=False)
    known = json.load(open(HERE.parent / "fit_heldout" / "known_effects.json")).get(str(did), {}).get("effects", [])
cats = task["categorical"]
for c in cats:
    frame = frame.with_columns(pl.col(c).cast(pl.Categorical))
train, test = frame[np.sort(tr).tolist()], frame[np.sort(te).tolist()]
binary = task["kind"] == "binary"
predictors = [c for c in frame.columns if c != "y"]
levels = {c: sorted(v for v in train[c].cast(pl.Utf8).unique().to_list() if v is not None) for c in cats}


def to_pandas(df, style):
    """Predictor frame for a competitor; style decides the categorical encoding."""
    out = {}
    for c in predictors:
        s = df[c]
        if c in cats:
            v = s.cast(pl.Utf8).to_list()
            if style == "catboost":
                out[c] = ["__NA__" if x is None else x for x in v]
            elif style == "category":
                out[c] = pd.Categorical(v, categories=levels[c])
            else:  # object strings with NaN
                out[c] = [np.nan if x is None else x for x in v]
        else:
            out[c] = s.cast(pl.Float64).to_numpy()
    return pd.DataFrame(out)


class Arm:
    interval_offered = True

    def fit(self):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError

    def interval(self, df):
        return None


class Mboostrs(Arm):
    def fit(self):
        import mboostrs as mb
        self.mb = mb
        self.result = mb.fit(train, "y", seed=seed)

    def predict(self, df):
        p = self.result.predict(df, type="response")
        if isinstance(p, dict):
            p = self.result.model.predict(df, type="mean")
        return np.asarray(p, float).ravel()

    def interval(self, df):
        ci = self.mb.conformal_interval(self.result.model, df, level=0.9, method="split", seed=seed)
        return np.asarray(ci.lower, float).ravel(), np.asarray(ci.upper, float).ravel()

    def claim(self):
        formula = self.result.trail.to_dict()["final"]["formula"]
        import re
        return sorted({c for c in predictors if re.search(rf"\b{re.escape(c)}\b", formula)})

    def trail(self):
        return True


class CatBoost(Arm):
    def _model(self, **kw):
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if binary and "loss_function" not in kw else CatBoostRegressor
        return cls(thread_count=1, random_seed=seed, verbose=0, cat_features=cats or None,
                   allow_writing_files=False, **kw)

    def fit(self):
        self.X = to_pandas(train, "catboost")
        self.m = self._model().fit(self.X, train["y"].to_numpy())

    def predict(self, df):
        X = to_pandas(df, "catboost")
        return self.m.predict_proba(X)[:, 1] if binary else self.m.predict(X)

    def interval(self, df):
        q = self._model(loss_function="MultiQuantile:alpha=0.05,0.95").fit(self.X, train["y"].to_numpy())
        p = q.predict(to_pandas(df, "catboost"))
        return p[:, 0], p[:, 1]


class LightGBM(Arm):
    def _model(self, **kw):
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if binary and "objective" not in kw else LGBMRegressor
        return cls(n_jobs=1, random_state=seed, verbose=-1, **kw)

    def fit(self):
        self.X = to_pandas(train, "category")
        self.m = self._model().fit(self.X, train["y"].to_numpy())

    def predict(self, df):
        X = to_pandas(df, "category")
        return self.m.predict_proba(X)[:, 1] if binary else self.m.predict(X)

    def interval(self, df):
        X = to_pandas(df, "category")
        lo = self._model(objective="quantile", alpha=0.05).fit(self.X, train["y"].to_numpy()).predict(X)
        hi = self._model(objective="quantile", alpha=0.95).fit(self.X, train["y"].to_numpy()).predict(X)
        return lo, hi


class XGBoost(Arm):
    def _model(self, **kw):
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if binary and "objective" not in kw else XGBRegressor
        return cls(n_jobs=1, random_state=seed, enable_categorical=True, tree_method="hist", **kw)

    def fit(self):
        self.X = to_pandas(train, "category")
        self.m = self._model().fit(self.X, train["y"].to_numpy())

    def predict(self, df):
        X = to_pandas(df, "category")
        return self.m.predict_proba(X)[:, 1] if binary else self.m.predict(X)

    def interval(self, df):
        q = self._model(objective="reg:quantileerror", quantile_alpha=np.array([0.05, 0.95]))
        p = q.fit(self.X, train["y"].to_numpy()).predict(to_pandas(df, "category"))
        return p[:, 0], p[:, 1]


class EBM(Arm):
    interval_offered = False

    def fit(self):
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
        cls = ExplainableBoostingClassifier if binary else ExplainableBoostingRegressor
        self.m = cls(n_jobs=1, random_state=seed).fit(to_pandas(train, "object"), train["y"].to_numpy())

    def predict(self, df):
        X = to_pandas(df, "object")
        return self.m.predict_proba(X)[:, 1] if binary else self.m.predict(X)


class Reference(Arm):
    def _pipe(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        num = [c for c in predictors if c not in cats]
        steps = []
        if num:
            steps.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                               StandardScaler()), num))
        if cats:
            steps.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                               OneHotEncoder(handle_unknown="ignore")), cats))
        return ColumnTransformer(steps, sparse_threshold=0)

    def fit(self):
        from sklearn.linear_model import LogisticRegression, RidgeCV
        self.t = self._pipe()
        X = self.t.fit_transform(to_pandas(train, "object"))
        self.Xtr = X
        self.m = (LogisticRegression(max_iter=2000) if binary else RidgeCV()).fit(X, train["y"].to_numpy())

    def predict(self, df):
        X = self.t.transform(to_pandas(df, "object"))
        return self.m.predict_proba(X)[:, 1] if binary else self.m.predict(X)

    def interval(self, df):
        from scipy.stats import t as student
        X = np.column_stack([np.ones(len(self.Xtr)), self.Xtr])
        Z = np.column_stack([np.ones(df.height), self.t.transform(to_pandas(df, "object"))])
        y = train["y"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        dof = max(len(y) - np.linalg.matrix_rank(X), 1)
        s = np.sqrt(np.sum((y - X @ beta) ** 2) / dof)
        G = np.linalg.pinv(X.T @ X)
        h = np.einsum("ij,jk,ik->i", Z, G, Z)
        half = student.ppf(0.95, dof) * s * np.sqrt(1 + h)
        centre = Z @ beta
        return centre - half, centre + half


ARMS = {"mboostrs": Mboostrs, "catboost": CatBoost, "lightgbm": LightGBM, "xgboost": XGBoost,
        "ebm": EBM, "reference": Reference}


def loss_of(y, p):
    if binary:
        q = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return float(np.mean((y - p) ** 2))


out = dict(id=tid, name=task["name"], kind=task["kind"], seed=seed, arm=arm,
           n_train=len(tr), n_test=len(te), planted=planted)
t0 = time.perf_counter()
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ARMS[arm]()
        a.fit()
        pred = np.asarray(a.predict(test), float).ravel()
        out["fit_predict_seconds"] = round(time.perf_counter() - t0, 3)
        y = test["y"].to_numpy().astype(float)
        base = loss_of(y, pred)
        if not np.isfinite(base):
            raise FloatingPointError("non-finite loss")
        out.update(status="ok", loss=base)
        if binary:
            bins = np.clip((pred * 10).astype(int), 0, 9)
            out["ece"] = float(sum(abs(pred[bins == b].mean() - y[bins == b].mean()) * np.mean(bins == b)
                                   for b in range(10) if np.any(bins == b)))
        # known effects (R tier)
        eff = []
        scale = 1.0 if binary else float(np.nanstd(train["y"].to_numpy()))
        for e in known:
            col = e["column"]
            vals = train[col].drop_nulls().to_numpy().astype(float)
            lo, hi = np.nanquantile(vals, [0.1, 0.9])  # missing values may be NaN, not null
            dt = train[col].dtype
            ch = float(np.mean(a.predict(test.with_columns(pl.lit(hi).cast(dt).alias(col)))
                               - a.predict(test.with_columns(pl.lit(lo).cast(dt).alias(col)))))
            eff.append(dict(variable=e["variable"], sign=e["sign"], change=ch,
                            recovered=bool(np.sign(ch) == e["sign"] and abs(ch) > 1e-8 * scale)))
        out["effects"] = eff
        # permutation importance (planted tasks)
        if planted:
            rng = np.random.default_rng(seed + 7)
            perm = {}
            for c in predictors:
                inc = []
                for _ in range(10):
                    shuffled = test.with_columns(pl.Series(c, rng.permutation(test[c].to_numpy())))
                    inc.append(loss_of(y, np.asarray(a.predict(shuffled), float).ravel()) - base)
                inc = np.asarray(inc)
                m, se = float(inc.mean()), float(inc.std(ddof=1) / np.sqrt(len(inc)))
                perm[c] = dict(mean=m, se=se, found=bool(m > 2 * se and m > 0.005 * base))
            out["permutation"] = perm
            if arm == "mboostrs":
                out["native_claim"] = a.claim()
        # prediction intervals (regression)
        if not binary:
            if not a.interval_offered:
                out["interval"] = "not offered"
            else:
                ti = time.perf_counter()
                lo, hi = a.interval(test)
                lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                cover = (y >= lo) & (y <= hi)
                iv = dict(coverage=float(cover.mean()),
                          median_width_over_sd=float(np.median(hi - lo) / np.std(y)),
                          seconds=round(time.perf_counter() - ti, 3))
                if planted and tid.startswith("S3"):
                    x2 = test["x2"].to_numpy()
                    edges = np.quantile(x2, [1 / 3, 2 / 3])
                    grp = np.digitize(x2, edges)
                    iv["coverage_by_x2_tercile"] = [float(cover[grp == g].mean()) for g in range(3)]
                out["interval"] = iv
        out["trail"] = bool(getattr(a, "trail", lambda: False)())
except Exception as exc:
    out.update(status=f"error: {type(exc).__name__}: {exc}"[:400], loss=None)
out["seconds"] = round(time.perf_counter() - t0, 2)
print(json.dumps(out), flush=True)
