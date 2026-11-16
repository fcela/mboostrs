"""Frozen inputs and dependency-aware partitions for the competitive study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "benches/.cache/competitive_shootout"
TASKS = {
    "housing_iid": ("housing", "iid", "regression"),
    "housing_spatial": ("housing", "spatial", "regression"),
    "bike_iid": ("bike", "iid", "regression"),
    "bike_future": ("bike", "temporal", "regression"),
    "diamonds": (44979, "iid", "regression"),
    "protein": (44963, "iid", "regression"),
    "superconductivity": (44964, "iid", "regression"),
    "concrete": (44959, "iid", "regression"),
    "credit": (31, "iid", "binary"),
    "cancer": ("cancer", "iid", "binary"),
    "smooth_sparse": ("synthetic", "iid", "regression"),
    "joint_boundary": ("synthetic", "iid", "regression"),
    "correlated_proxies": ("synthetic", "iid", "regression"),
}


def truth(X, task):
    x = np.asarray(X, float)
    if task == "joint_boundary":
        return 3 * np.where(x[:, 0] * x[:, 1] > 0, 1, -1) + x[:, 2]
    return 2 * np.sin(2 * x[:, 0]) + 2 * (x[:, 1] ** 2 - 1 / 3) + x[:, 2]


def synthetic(task, seed, n=5000):
    rng = np.random.default_rng(20000 + seed)
    X = rng.uniform(-1, 1, (n, 12))
    if task == "correlated_proxies":
        X[:, 3] = X[:, 0] + rng.normal(0, .05, n)
        X[:, 4] = X[:, 1] + rng.normal(0, .05, n)
    data = {f"x{i}": X[:, i] for i in range(X.shape[1])}
    data["y"] = truth(X, task) + rng.normal(0, .5, n)
    return pl.DataFrame(data), [], {}


def prepare(task):
    """Download/prepare predictors once; never infer features from test scores."""
    import re

    from sklearn.datasets import fetch_openml, load_breast_cancer

    source, split, kind = TASKS[task]
    if source == "synthetic":
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{task}.parquet"
    if path.exists():
        return
    metadata = {"task": task, "source": source, "split": split, "kind": kind}
    cats, aux = [], {}
    if source == "housing":
        frame = pl.read_csv(ROOT / "docs/book/case_studies/california_housing_1990.csv")
        # The same semantic quantities as the housing study; totals per
        # household avoid giving population size an accidental scale effect.
        frame = frame.select(
            pl.col("longitude"), pl.col("latitude"), pl.col("housing_median_age").alias("age"),
            pl.col("median_income").alias("income"),
            (pl.col("total_rooms") / pl.col("households")).alias("rooms"),
            (pl.col("total_bedrooms") / pl.col("households")).alias("bedrooms"),
            pl.col("population"),
            (pl.col("population") / pl.col("households")).alias("occupancy"),
            pl.col("median_house_value").log1p().alias("y"))
        aux["groups"] = ((np.floor(frame["longitude"].to_numpy() * 2).astype(int) * 1000)
                         + np.floor(frame["latitude"].to_numpy() * 2).astype(int)).tolist()
        metadata["response"] = "log1p(median house value in USD); top-coded source"
    elif source == "bike":
        raw = pl.read_csv(ROOT / "docs/book/case_studies/bike_sharing_hour.csv")
        cats = ["season", "holiday", "workingday", "weathersit", "weekday"]
        frame = raw.select(*cats, "hr", "mnth", "yr", "temp", "hum", "windspeed",
                           pl.col("cnt").log1p().alias("y"))
        frame = frame.with_columns(pl.col(cats).cast(pl.String))
        aux["groups"] = raw["dteday"].to_list()
        metadata["response"] = "log1p(hourly total rentals)"
    elif source == "cancer":
        bunch = load_breast_cancer()
        frame = pl.DataFrame({f"x{i}": bunch.data[:, i] for i in range(bunch.data.shape[1])})
        frame = frame.with_columns(pl.Series("y", 1 - bunch.target))
        metadata["feature_labels"] = list(bunch.feature_names)
        metadata["response"] = "malignant = 1, benign = 0"
    else:
        bunch = fetch_openml(data_id=source, as_frame=True, parser="auto",
                             data_home=str(CACHE / "openml"))
        raw = bunch.data
        names = [re.sub(r"\W", "_", str(c)) for c in raw.columns]
        names = [n if n[0].isalpha() else "v_" + n for n in names]
        data = {}
        for old, name in zip(raw.columns, names, strict=True):
            values = raw[old]
            if values.dtype.kind in "fiub":
                data[name] = values.to_numpy().astype(float)
            else:
                cats.append(name)
                data[name] = values.astype("string").fillna("__missing__").to_list()
        y = np.asarray(bunch.target)
        if kind == "binary":
            y = (y.astype(str) == "bad").astype(float)
            metadata["response"] = "bad credit risk = 1"
        else:
            y = y.astype(float)
            if task == "diamonds":
                y = np.log1p(y)
            metadata["response"] = ("log1p(price in USD)" if task == "diamonds"
                                    else str(bunch.target_names))
        data["y"] = y
        frame = pl.DataFrame(data).filter(pl.col("y").is_finite())
        metadata["source_url"] = f"https://www.openml.org/d/{source}"
    frame.write_parquet(path)
    metadata.update(categorical=cats, auxiliary=aux, shape=list(frame.shape))
    (CACHE / f"{task}.json").write_text(json.dumps(metadata))


def _random_cut(rows, fraction, seed, y=None):
    a, b = train_test_split(rows, test_size=fraction, random_state=seed,
                            stratify=None if y is None else y[rows])
    return np.asarray(a), np.asarray(b)


def partitions(n, seed, kind, design, *, y=None, groups=None):
    """Fit/validation/calibration/test; dependency units never cross a boundary."""
    rows = np.arange(n)
    if design == "temporal":
        levels = np.unique(groups)
        # Three increasing origins with disjoint future horizons.
        cut = int(len(levels) * (.55 + .15 * seed))
        horizon = max(1, int(.12 * len(levels)))
        pool = levels[:cut]
        n_cal = max(1, int(.2 * len(pool)))
        n_val = max(1, int(.25 * (len(pool) - n_cal)))
        units = [pool[:-n_cal - n_val], pool[-n_cal - n_val:-n_cal], pool[-n_cal:],
                 levels[cut:cut + horizon]]
        return [rows[np.isin(groups, unit)] for unit in units]
    if design == "spatial":
        levels = np.unique(groups)
        pool, test = _random_cut(levels, .2, seed)
        dev, cal = _random_cut(pool, .2, seed + 101)
        fit, val = _random_cut(dev, .25, seed + 202)
        return [rows[np.isin(groups, unit)] for unit in (fit, val, cal, test)]
    labels = y if kind == "binary" else None
    pool, test = _random_cut(rows, .2, seed, labels)
    dev, cal = _random_cut(pool, .2, seed + 101, labels)
    fit, val = _random_cut(dev, .25, seed + 202, labels)
    return [fit, val, cal, test]


def payload(task, seed):
    source, design, kind = TASKS[task]
    if source == "synthetic":
        frame, cats, metadata = synthetic(task, seed)
    else:
        frame = pl.read_parquet(CACHE / f"{task}.parquet")
        metadata = json.loads((CACHE / f"{task}.json").read_text())
        cats = metadata["categorical"]
    groups = np.asarray(metadata["auxiliary"].get("groups")) if "auxiliary" in metadata else None
    cuts = partitions(len(frame), seed, kind, design, y=frame["y"].to_numpy(), groups=groups)
    fit, val, cal, test = cuts
    rng = np.random.default_rng(seed + 999)
    if len(fit) + len(val) > 4000:
        if design == "temporal":
            val = val[-1000:]
            fit = fit[-(4000 - len(val)):]
        else:
            val = rng.choice(val, min(1000, len(val)), replace=False)
            fit = rng.choice(fit, 4000 - len(val), replace=False)
    if len(cal) > 1000:
        cal = rng.choice(cal, 1000, replace=False)
    if len(test) > 2000:
        test = rng.choice(test, 2000, replace=False)
    cuts = [fit, val, cal, test]
    frames = [frame[cut.tolist()] for cut in cuts]
    return {"frames": frames, "indices": [c.tolist() for c in cuts],
            "categorical": cats, "kind": kind, "task": task, "seed": seed,
            "metadata": metadata}
