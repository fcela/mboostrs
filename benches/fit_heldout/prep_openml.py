"""Load every cached OpenML dataset offline and write a clean parquet per task.

Regression: numeric target. Binary: exactly two target classes (second sorted
class = 1). Multiclass and unreadable datasets are skipped and listed.
Columns are renamed to safe identifiers c0..ck; categoricals become strings.
"""
import json
import pathlib
import sys

import pandas as pd
import polars as pl
from sklearn.datasets import fetch_openml

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
EXCLUDE = {44959, 44963, 44964, 44979, 31}  # used to choose the budget mapping
cache = pathlib.Path.home() / "scikit_learn_data/openml/openml.org/api/v1/json/data"
ids = sorted(int(p.name.split(".")[0]) for p in cache.glob("*.gz") if p.name.split(".")[0].isdigit())
manifest, skipped = [], []
for did in ids:
    if did in EXCLUDE:
        skipped.append((did, "used in budget selection"))
        continue
    try:
        b = fetch_openml(data_id=did, as_frame=True, parser="auto")
    except Exception as exc:
        skipped.append((did, f"load: {type(exc).__name__}"))
        continue
    X, y = b.data.copy(), b.target
    if isinstance(y, pd.DataFrame):
        skipped.append((did, "multi-target"))
        continue
    y = pd.Series(y)
    keep = y.notna().to_numpy()
    X, y = X[keep], y[keep]
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 2:
        kind, yv = "regression", y.astype(float).to_numpy()
    elif y.nunique() == 2:
        classes = sorted(y.astype(str).unique())
        kind, yv = "binary", (y.astype(str) == classes[1]).astype(float).to_numpy()
    else:
        skipped.append((did, f"{y.nunique()} classes"))
        continue
    data, cats = {}, []
    for i, col in enumerate(X.columns):
        s = X[col]
        name = f"c{i}"
        if not pd.api.types.is_numeric_dtype(s) or s.dtype == bool:
            data[name] = pl.Series(name, [None if pd.isna(v) else str(v) for v in s], dtype=pl.Utf8)
            cats.append(name)
        else:
            data[name] = s.astype(float).to_numpy()
    data["y"] = yv
    frame = pl.DataFrame(data)
    for c in cats:
        frame = frame.with_columns(pl.col(c).cast(pl.Utf8))
    path = OUT / f"{did}.parquet"
    frame.write_parquet(path)
    manifest.append(dict(id=did, name=b.details.get("name"), kind=kind, n=frame.height,
                         p=frame.width - 1, categorical=cats, source="OpenML",
                         columns={f"c{i}": str(c) for i, c in enumerate(X.columns)}))
    print(did, b.details.get("name"), kind, frame.shape, flush=True)
json.dump(dict(tasks=manifest, skipped=skipped), open(OUT / "manifest.json", "w"), indent=1)
print("tasks", len(manifest), "skipped", skipped)
