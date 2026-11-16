#!/usr/bin/env python3
"""Generate the packaged datasets' provenance cards from the data itself.

D3 asks for "data cards for source, license, retrieval date, curation script,
exclusions, units, missingness, and known biases", noting that the packaged
datasets "rely mostly on loader docstrings rather than a machine-readable
provenance record".

The split here is deliberate. Fields that can be **measured** from the shipped
file -- row count, column dtypes, missing counts -- are generated, so they
cannot drift from the data they describe. Fields that cannot -- licence,
retrieval date, curation script, exclusions, units, known biases -- are left
`None` until somebody records them.

`None` means "not recorded", never "none apply". A licence invented here would
be worse than an absent one: a reader deciding whether they may redistribute a
dataset needs to see that the answer is missing, not read a guess. `--check`
reports how many fields are still unrecorded rather than failing on them, so
the gap is visible without blocking on research nobody has done yet.

    python tools/generate_data_cards.py
    python tools/generate_data_cards.py --check
"""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "python" / "mboostrs" / "datasets" / "data_cards.json"

sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from mboostrs import datasets as ds  # noqa: E402

#: Fields no measurement can supply. Recorded as `None` until researched.
UNRECORDED = ("license", "retrieved", "curation_script", "exclusions",
              "units", "known_biases")


def loaders() -> list[str]:
    """The packaged-data loaders: those taking only `output`.

    The simulators (`friedman1`, `sparse_linear`, ...) take `n` and `seed`
    and generate their data, so they have no provenance to record.
    """
    names = []
    for name in sorted(dir(ds)):
        if name.startswith("_"):
            continue
        function = getattr(ds, name)
        if not callable(function):
            continue
        try:
            parameters = list(inspect.signature(function).parameters)
        except (TypeError, ValueError):
            continue
        if parameters == ["output"]:
            names.append(name)
    return names


def measure(name: str) -> dict:
    frame = getattr(ds, name)(output="polars")
    doc = inspect.getdoc(getattr(ds, name)) or ""
    paragraphs = doc.split("\n\n")
    columns = {}
    for column in frame.columns:
        series = frame[column]
        missing = int(series.null_count())
        if series.dtype.is_numeric():
            values = series.drop_nulls().to_numpy()
            if values.dtype.kind == "f":
                missing += int(np.isnan(values).sum())
        columns[column] = {"dtype": str(series.dtype), "missing": missing}
    card = {
        "summary": paragraphs[0].replace("\n", " ").strip(),
        "provenance_note": (paragraphs[1].replace("\n", " ").strip()
                            if len(paragraphs) > 1 else ""),
        "rows": frame.height,
        "columns": columns,
    }
    card.update(dict.fromkeys(UNRECORDED))
    return card


def build(existing: dict | None = None) -> dict:
    """Measured fields regenerated; recorded ones carried forward."""
    existing = existing or {}
    cards = {}
    for name in loaders():
        card = measure(name)
        previous = existing.get(name, {})
        for field in UNRECORDED:
            if previous.get(field) is not None:
                card[field] = previous[field]
        cards[name] = card
    return cards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail when the committed cards are out of date")
    arguments = parser.parse_args()

    existing = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    cards = build(existing)
    rendered = json.dumps(cards, indent=2, ensure_ascii=False) + "\n"
    unrecorded = sum(1 for card in cards.values()
                     for field in UNRECORDED if card[field] is None)

    if arguments.check:
        if OUTPUT.read_text() != rendered:
            print(
                "python/mboostrs/datasets/data_cards.json is out of date: a "
                "dataset's rows, columns or missingness changed, or a loader "
                "was added. Run `python tools/generate_data_cards.py`.",
                file=sys.stderr,
            )
            return 1
        print(f"Data cards current: {len(cards)} datasets, "
              f"{unrecorded} field(s) not yet recorded")
        return 0

    OUTPUT.write_text(rendered)
    print(f"Data cards written: {len(cards)} datasets, "
          f"{unrecorded} field(s) left unrecorded "
          f"(licence, retrieval date and the rest need a person)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
