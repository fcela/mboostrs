"""Readers for the public-data benchmark artifacts (``bench_public_*.json``).

The notebooks in this part only read recorded runs; they never refit. Every
summary keeps failed fits visible: a method's mean is taken over its
successful seeds, and the count of failures is reported beside it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cases"))

from bench_tables import table  # noqa: E402,F401  (re-exported for the notebooks)


def load(task: str):
    """The artifact and its result rows as a Polars frame."""
    bench = json.loads((HERE / f"bench_public_{task}.json").read_text())
    rows = [{key: value for key, value in row.items() if not isinstance(value, (list, dict))}
            for row in bench["results"]]
    return bench, pl.DataFrame(rows, infer_schema_length=None)


def summary(frame: pl.DataFrame, metrics: list[str], by=("method",)) -> pl.DataFrame:
    """Mean and Monte Carlo standard error of each metric across seeds,
    successful-fit counts, and median whole-workflow seconds."""
    ok = frame.filter(pl.col("status") == "success")
    stats = ok.group_by(*by).agg(
        *[pl.col(m).mean().alias(m) for m in metrics],
        *[(pl.col(m).std() / pl.len().sqrt()).alias(f"{m}_se") for m in metrics[:1]],
        pl.col("seconds").median().alias("seconds"),
    )
    counts = frame.group_by(*by).agg(
        pl.len().alias("planned"), (pl.col("status") != "success").sum().alias("failed"))
    return counts.join(stats, on=list(by), how="left")


def paired(frame: pl.DataFrame, metric: str, reference: str, by=()) -> pl.DataFrame:
    """Per-seed difference ``method - reference`` on ``metric``, averaged over
    the seeds where both succeeded, with its paired standard error."""
    ok = frame.filter(pl.col("status") == "success")
    keys = ["seed", *by]
    base = ok.filter(pl.col("method") == reference).select(*keys, pl.col(metric).alias("_ref"))
    joined = ok.filter(pl.col("method") != reference).join(base, on=keys)
    return (joined.with_columns((pl.col(metric) - pl.col("_ref")).alias("difference"))
            .group_by("method", *by)
            .agg(pl.len().alias("pairs"), pl.col("difference").mean().alias("difference"),
                 (pl.col("difference").std() / pl.len().sqrt()).alias("difference_se")))


def failures(frame: pl.DataFrame) -> pl.DataFrame:
    """Every unsuccessful fit with its recorded error, first line only."""
    if "error" not in frame.columns:  # every fit succeeded
        return pl.DataFrame(schema={"method": pl.String, "seed": pl.Int64, "error": pl.String})
    columns = [c for c in ("method", "variant", "seed", "error") if c in frame.columns]
    return (frame.filter(pl.col("status") != "success").select(columns)
            .with_columns(pl.col("error").str.split("\n").list.first().str.slice(0, 160)))


def memory(bench: dict) -> pl.DataFrame:
    """Peak resident-memory growth of one fit in a fresh process (seed 0)."""
    return pl.DataFrame([{"method": r["method"], "seconds": r.get("seconds"),
                          "peak_rss_growth_mb": r.get("peak_rss_growth_mb"),
                          "status": r.get("status")} for r in bench["memory"]])


def warnings_by_method(bench: dict) -> pl.DataFrame:
    """Distinct warning messages each method raised across all seeds."""
    seen: dict[str, set[str]] = {}
    for row in bench["results"]:
        for message in row.get("warnings", []):
            seen.setdefault(row["method"], set()).add(message.split(":", 1)[0] + ": "
                                                      + message.split(":", 1)[-1].strip()[:90])
    return pl.DataFrame([{"method": m, "distinct_warnings": len(v),
                          "example": sorted(v)[0]} for m, v in seen.items()]
                        or [{"method": "", "distinct_warnings": 0, "example": ""}])


def protocol(bench: dict) -> pl.DataFrame:
    """What was run, on what, by which code: one row per fact."""
    design, prov = bench["design"], bench["provenance"]
    versions = ", ".join(f"{k} {v}" for k, v in bench["versions"].items())
    facts = {
        "source": design["source"],
        "source sha256": design["sha256"][:16] + "…",
        "train fraction": str(design["train_frac"]),
        "seeds (random splits)": str(len(design["seeds"])),
        "trees (fixed setting)": "{rounds} rounds, learning rate {learning_rate}".format(
            **design["tree_settings"]),
        "threads": "1 (RAYON, OMP, OpenBLAS, MKL)",
        "versions": versions,
        "commit": prov["commit"][:10] + (" + uncommitted runner" if prov["working_tree"]["dirty"]
                                         else ""),
        "runner sha256": prov["runner_sha256"][:16] + "…",
        "measured at (UTC)": prov["created_at"],
        "machine": f"{prov['platform']['cpu']}, {prov['threads']['logical_cpus']} logical CPUs",
    }
    return pl.DataFrame({"item": list(facts), "value": list(facts.values())})


def lines(bench: dict) -> pl.DataFrame:
    """Non-blank, non-comment source lines of each method's function."""
    return pl.DataFrame({"method": list(bench["lines_of_code"]),
                         "lines": list(bench["lines_of_code"].values())})
