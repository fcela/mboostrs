"""Replace the mboostrs rows of the distributional comparison with a new run.

The distributional comparison (``benches/bench_distributional.py``) fits three
mboostrs procedures, R gamboostLSS, mssm and NGBoost. Only the mboostrs rows
depend on mboostrs, so a refresh reruns those three procedures on the same
draws and keeps every other row, and the original provenance, unchanged:

    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python -m benches.bench_distributional --repeats 5 \
        --libraries mboostrs "mboostrs + REML refit" \
        "mboostrs autospec pairs + REML refit" --out NEW.json
    python benches/refresh_distributional.py NEW.json

The refreshed document records the new run's provenance under ``refresh``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/book/cases/bench_distributional.json"
KEY = ("scenario", "track", "library", "repeat")


def main(new_path: str) -> None:
    source_bytes = ARTIFACT.read_bytes()
    document = json.loads(source_bytes)
    new = json.loads(Path(new_path).read_text())
    libraries = sorted({row["library"] for row in new["results"]})
    if not all(library.startswith("mboostrs") for library in libraries):
        raise ValueError(f"the new run holds non-mboostrs rows: {libraries}")
    by_key = {tuple(row[k] for k in KEY): row for row in new["results"]}
    old_keys = {tuple(row[k] for k in KEY) for row in document["results"]
                if row["library"] in libraries}
    if old_keys != set(by_key):
        raise ValueError("the new run does not cover the same draws as the artifact")
    for a, b in ((row, by_key[tuple(row[k] for k in KEY)]) for row in document["results"]
                 if row["library"] in libraries):
        if a["seed"] != b["seed"] or a["input_digest"] != b["input_digest"]:
            raise ValueError(f"draw {tuple(a[k] for k in KEY)} differs from the artifact")
    document["results"] = [by_key.get(tuple(row[k] for k in KEY), row)
                           for row in document["results"]]
    document["refresh"] = {
        "libraries": libraries,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "created_at": new["created_at"],
        "commit": new["commit"],
        "working_tree": new["working_tree"],
        "versions": new["versions"],
        "retained": "rows of the other libraries, with the original provenance above",
    }
    ARTIFACT.write_text(json.dumps(document, indent=1) + "\n")
    print(f"replaced {len(by_key)} rows of {', '.join(libraries)}")


if __name__ == "__main__":
    main(sys.argv[1])
