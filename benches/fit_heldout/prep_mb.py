"""Write the three mboostrs.datasets tables of the held-out study.

Munich rent (response `rent`; `rentsqm` is rent divided by area and is
dropped), India stunting (response `stunting`; `mcdist_lab` is the label of
`mcdist` and is dropped) and bodyfat (response `DEXfat`). Task ids 200001-3.
"""
import json
import pathlib

import polars as pl

import mboostrs.datasets as d

OUT = pathlib.Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
specs = {200001: ("mb:munich_rent", d.munich_rent(), "rent", ["rentsqm"]),
         200002: ("mb:india", d.india(), "stunting", ["mcdist_lab"]),
         200003: ("mb:bodyfat", d.bodyfat(), "DEXfat", [])}
tasks = []
for tid, (name, frame, target, drop) in specs.items():
    if not isinstance(frame, pl.DataFrame):
        frame = pl.from_pandas(frame)
    predictors = [c for c in frame.columns if c not in drop + [target]]
    out, cats = {}, []
    for i, col in enumerate(predictors):
        s = frame[col]
        if s.dtype in (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean):
            out[f"c{i}"] = s.cast(pl.Utf8)
            cats.append(f"c{i}")
        else:
            out[f"c{i}"] = s.cast(pl.Float64)
    out["y"] = frame[target].cast(pl.Float64)
    pl.DataFrame(out).write_parquet(OUT / f"{tid}.parquet")
    tasks.append(dict(id=tid, name=name, kind="regression", n=frame.height, p=len(predictors),
                      categorical=cats, source="mboostrs.datasets", target=target,
                      columns={f"c{i}": c for i, c in enumerate(predictors)}))
json.dump(dict(tasks=tasks, skipped=[]), open(OUT / "manifest_mb.json", "w"), indent=1)
print(len(tasks), "tasks")
