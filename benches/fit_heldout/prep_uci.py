"""Fetch UCI datasets with ucimlrepo and write them in the study format.

Task id = 100000 + UCI id. Regression: numeric target with more than 10
distinct values. Binary: exactly two distinct target values. Everything else
is skipped and listed. Columns are renamed c0..ck; non-numeric columns are
strings. Identifier columns declared by UCI (`index_col`, role "ID") are
dropped, because they are row labels, not predictors.
"""
import concurrent.futures as cf
import json
import pathlib

import pandas as pd
import polars as pl
from ucimlrepo import fetch_ucirepo

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "data"
meta = [m for m in json.load(open(HERE / "uci/metadata.json")) if "error" not in m]
EXCLUDE = {1, 2, 20, 17, 27, 143, 144, 159, 162, 165, 186, 275, 291, 464, 147}


def eligible(x):
    t, ch = set(x.get("tasks") or []), set(x.get("characteristics") or [])
    n, p = x.get("num_instances") or 0, x.get("num_features") or 0
    return (t & {"Regression", "Classification"} and ch & {"Tabular", "Multivariate"}
            and x.get("target_col") and len(x["target_col"]) == 1
            and 300 <= n <= 500000 and 2 <= p <= 300 and x["uci_id"] not in EXCLUDE)


def load(x):
    uid = x["uci_id"]
    try:
        d = fetch_ucirepo(id=uid)
        X, Y = d.data.features, d.data.targets
        if Y is None or Y.shape[1] != 1:
            return uid, x["name"], None, "not one target column"
        y = Y.iloc[:, 0]
        ids = set(d.data.ids.columns) if d.data.ids is not None else set()
        X = X[[c for c in X.columns if c not in ids]]
        keep = y.notna().to_numpy()
        X, y = X[keep], y[keep]
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10 and "Regression" in (x.get("tasks") or []):
            kind, yv = "regression", y.astype(float).to_numpy()
        elif y.nunique() == 2:
            classes = sorted(y.astype(str).unique())
            kind, yv = "binary", (y.astype(str) == classes[1]).astype(float).to_numpy()
        else:
            return uid, x["name"], None, f"{y.nunique()} target values"
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
        tid = 100000 + uid
        frame.write_parquet(OUT / f"{tid}.parquet")
        return uid, x["name"], dict(id=tid, name=f"uci:{x['name'].strip()}", kind=kind,
                                    n=frame.height, p=frame.width - 1, categorical=cats,
                                    source="UCI", uci_id=uid,
                                    columns={f"c{i}": str(c) for i, c in enumerate(X.columns)}), None
    except Exception as exc:
        return uid, x["name"], None, f"{type(exc).__name__}: {exc}"[:200]


cands = [x for x in meta if eligible(x)]
with cf.ThreadPoolExecutor(6) as ex:
    results = list(ex.map(load, cands))
tasks = [r[2] for r in results if r[2]]
skipped = [(r[0], r[1], r[3]) for r in results if not r[2]]
json.dump(dict(tasks=tasks, skipped=skipped), open(OUT / "manifest_uci.json", "w"), indent=1)
print(len(cands), "candidates;", len(tasks), "kept;",
      sum(t["kind"] == "binary" for t in tasks), "binary;",
      sum(t["kind"] == "regression" for t in tasks), "regression")
for t in tasks:
    print(t["id"], t["name"], t["kind"], t["n"], t["p"])
