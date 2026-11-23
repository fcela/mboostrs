"""onceover scan of every study dataset, before any study run reads results.

Per dataset, records the findings that bear on how a fitted model should be
read: exact identities between the target and a column (target derived from a
predictor), leakage findings, key/identifier columns, target predictability
(onceover's holdout R² from its own boosted screen), and missingness. The
flags are fixed here and used only to stratify reporting, never to drop data.
"""
import concurrent.futures as cf
import json
import pathlib
import re

import onceover
import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
ids = sorted(int(p.stem) for p in (HERE / "data").glob("*.parquet"))


def scan(did):
    df = pl.read_parquet(HERE / f"data/{did}.parquet")
    if df.height > 50000:
        df = df.sample(50000, seed=0)
    try:
        r = onceover.profile(df, target="y", deadline_seconds=120)
    except Exception as exc:
        return did, {"error": f"{type(exc).__name__}: {exc}"[:200]}
    out = {"identity_with_target": [], "leakage": [], "keys": [], "predictable_r2": None,
           "missing_cell_share": float(sum(df[c].null_count() for c in df.columns) / (df.height * df.width))}
    for f in r.findings:
        cols = set((f.get("subject") or {}).get("columns") or [])
        cat, typ, claim = f.get("category"), f.get("type"), f.get("claim", "")
        if cat == "dependencies" and typ in ("affine", "functional", "identity") and "y" in cols:
            out["identity_with_target"].append(claim[:200])
        if "leak" in (cat or "") or "leak" in (typ or ""):
            out["leakage"].append(claim[:200])
        if cat == "keys":
            out["keys"].append(claim[:150])
        if typ == "predictive" and (f.get("subject") or {}).get("target") == "y":
            m = re.search(r"holdout R² = ([0-9.]+)", claim)
            if m:
                out["predictable_r2"] = float(m.group(1))
    return did, out


with cf.ThreadPoolExecutor(4) as ex:
    scans = dict(ex.map(scan, ids))
json.dump({str(k): v for k, v in scans.items()}, open(HERE / "onceover_scan.json", "w"), indent=1)


def flag(v, k):
    return bool(v.get(k)) if "error" not in v else None


print("datasets", len(scans), "errors", sum("error" in v for v in scans.values()))
print("identity_with_target", [k for k, v in scans.items() if flag(v, "identity_with_target")])
print("leakage", [k for k, v in scans.items() if flag(v, "leakage")])
print("keys", sum(bool(flag(v, "keys")) for v in scans.values()))
