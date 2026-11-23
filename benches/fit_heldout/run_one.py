"""One (dataset, seed, arm) run; prints one JSON line. See PREREGISTRATION.md."""
import importlib.util
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

HERE = pathlib.Path(__file__).resolve().parent
did, seed, arm = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
manifest = {t["id"]: t for f in ("manifest.json", "manifest_uci.json", "manifest_mb.json")
            for t in json.load(open(HERE / "data" / f))["tasks"]}
known = json.load(open(HERE / "known_effects.json")).get(str(did), {}).get("effects", [])
task = manifest[did]
frame = pl.read_parquet(HERE / f"data/{did}.parquet")
for c in task["categorical"]:
    frame = frame.with_columns(pl.col(c).cast(pl.Categorical))
y_all = frame["y"].to_numpy()
rows = np.arange(frame.height)
strat = y_all if task["kind"] == "binary" else None
tr, te = train_test_split(rows, test_size=0.2, random_state=seed, stratify=strat)
rng = np.random.default_rng(seed)
if len(tr) > 4000:
    tr = rng.choice(tr, 4000, replace=False)
if len(te) > 2000:
    te = rng.choice(te, 2000, replace=False)
train, test = frame[np.sort(tr).tolist()], frame[np.sort(te).tolist()]

import mboostrs as mb  # noqa: E402

if arm == "old":
    spec = importlib.util.spec_from_file_location("old_fit", HERE / "old_fit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fit_fn, kwargs = module.fit, {}
elif arm == "new":
    fit_fn, kwargs = mb.fit, {}
elif arm == "quick":
    fit_fn, kwargs = mb.fit, {"budget": "quick"}
else:
    raise SystemExit(f"unknown arm {arm}")

t0 = time.perf_counter()
out = dict(id=did, name=task["name"], kind=task["kind"], seed=seed, arm=arm,
           n_train=len(tr), n_test=len(te))
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_fn(train, "y", seed=seed, **kwargs)
        def predict_mean(rows):
            p = result.predict(rows, type="response")
            if isinstance(p, dict):  # distributional (LSS) model: score its mean
                p = result.model.predict(rows, type="mean")
            return np.asarray(p, float).ravel()
        pred = predict_mean(test)
        effects = []
        scale = 1.0 if task["kind"] == "binary" else float(np.nanstd(train["y"].to_numpy()))
        for e in known:
            col = e["column"]
            vals = train[col].drop_nulls().to_numpy().astype(float)
            lo, hi = np.nanquantile(vals, [0.1, 0.9])  # missing values may be NaN, not null
            dtype = train[col].dtype
            low = test.with_columns(pl.lit(lo).cast(dtype).alias(col))
            high = test.with_columns(pl.lit(hi).cast(dtype).alias(col))
            change = float(np.mean(predict_mean(high) - predict_mean(low)))
            effects.append(dict(variable=e["variable"], sign=e["sign"], change=change,
                                recovered=bool(np.sign(change) == e["sign"]
                                               and abs(change) > 1e-8 * scale)))
    y = test["y"].to_numpy().astype(float)
    if task["kind"] == "binary":
        q = np.clip(pred, 1e-12, 1 - 1e-12)
        loss = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    else:
        loss = float(np.mean((y - pred) ** 2))
    if not np.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    trail = result.trail.to_dict()
    out.update(status="ok", loss=loss, effects=effects, formula=trail["final"]["formula"],
               family=trail["final"]["family"], spent=trail["budget"].get("spent"))
except Exception as exc:
    out.update(status=f"error: {type(exc).__name__}: {exc}"[:400], loss=None)
out["seconds"] = round(time.perf_counter() - t0, 2)
print(json.dumps(out), flush=True)
