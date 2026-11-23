#!/usr/bin/env python3
"""Check that the R migration page names its deviations where they are used.

D8 item 3: "An R `mboost`/`gamboostLSS` user must find a concise migration
table whose examples either reproduce upstream or name the exact deviation at
the call."

The page had both halves and not the join. Its tables map R spellings to
mboostrs spellings, and a separate "What is different" section, several
thousand words later, lists the deviations. A reader porting a script reads
the row, copies the call, and never reaches the section -- which is the whole
failure mode "at the call" is guarding against. `gamboost(class ~ ...,
family = Multinomial())` had a row that looked like a rename and is not one:
the fit is gamboost-only here, and `aic()` and `hatvalues()` raise on it.

`mboostrs._parity_notes` already records every such difference in a structured
form, with a mechanism, a consequence, and whether upstream shares it. This
reads that registry and requires, for each deviation mboostrs does *not* share
with upstream, that a row mentioning the object also carries the deviation.
Deviations shared with upstream are exempt: a ported script behaves the same
in both packages, so the row is an accurate migration instruction even though
the object differs from its name.

`KEYWORDS` maps each note handle to the word a row must contain. A new note
with `shared_with_upstream=False` and no entry here fails the check rather
than passing silently, so adding a deviation forces a decision about whether
the migration page needs to say so.

Run from the repository root:

    python tools/check_migration_table.py
    python tools/check_migration_table.py --show
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "book" / "usage" / "r_migration.ipynb"

sys.path.insert(0, str(ROOT / "python"))

from mboostrs._parity_notes import FAMILY_NOTES, LEARNER_NOTES  # noqa: E402

#: What a table row must contain to count as naming the deviation, keyed by
#: the parity note's handle: `(keyword, anchor)`. The keyword is chosen so a
#: row cannot satisfy it by accident and so a rewrite that drops the warning
#: also drops the word.
#:
#: `anchor` overrides where the row is looked for. A note is filed under the
#: object that carries it, which is not always the call a reader is porting:
#: the Gaussian `logLik` difference belongs on the `logLik(fit)` row, not on
#: every row that constructs a `Gaussian()`, because that is the call whose
#: result differs.
KEYWORDS = {
    "multinomial is gamboost-only": (r"gamboost[- ]only", None),
    "multinomial cvrisk scores the full negative log-likelihood":
        (r"negative log-likelihood|NLL", None),
    "weighted auc is an mboostrs extension": (r"case weights|weights", None),
    "gehan offset is zero rather than arbitrary": (r"offset", None),
    "gaussian logLik is a real log-likelihood": (r"negative_risk|log-likelihood", r"\blogLik\b"),
    "bkernel is a kernel-ridge learner": (r"kernel[- ]ridge", None),
    "brad centre choice differs": (r"cent(?:re|er)s?", None),
    "btree splits on squared error, not conditional inference":
        (r"squared error|SSE", None),
    "bns is ridge-penalized, not unpenalized": (r"ridge", None),
    "extract('lambda') is unavailable for block learners":
        (r"cannot be read|unavailable|not available", r"extract\(fit, .lambda"),
    "adaptive huber cvrisk starts each fold afresh": (r"fold", None),
}

#: How an object's registry key appears in a migration row. A learner is
#: written lowercase in an R formula and capitalised as a Python class, and a
#: row may use either.
def name_pattern(key: str, kind: str) -> str:
    if kind == "learner":
        return rf"\b[{key[0]}{key[0].lower()}]{re.escape(key[1:])}\b"
    return rf"\b{re.escape(key)}\b"


def page_text() -> str:
    cells = json.loads(PAGE.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells)


def table_rows(text: str) -> list[tuple[int, str]]:
    """`(line number, row)` for every markdown table row on the page."""
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("|") and not re.fullmatch(r"\|[\s|:-]+\|", stripped):
            rows.append((number, stripped))
    return rows


def deviations() -> list[tuple[str, str, object]]:
    """`(key, kind, note)` for every difference mboostrs does not share."""
    found = []
    for kind, registry in (("family", FAMILY_NOTES), ("learner", LEARNER_NOTES)):
        for key, notes in registry.items():
            for note in notes:
                if not note.shared_with_upstream:
                    found.append((key, kind, note))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="print each deviation and where it is named")
    arguments = parser.parse_args()

    text = page_text()
    rows = table_rows(text)
    problems = []

    for key, kind, note in deviations():
        entry = KEYWORDS.get(note.name)
        if entry is None:
            problems.append(
                f"`{key}` carries the parity note {note.name!r}, which "
                f"mboostrs does not share with upstream, and "
                f"tools/check_migration_table.py has no keyword for it. Add "
                f"one and say so in the migration row, or record why the row "
                f"does not need it."
            )
            continue
        keyword, anchor = entry
        pattern = anchor or name_pattern(key, kind)
        mentions = [(number, row) for number, row in rows if re.search(pattern, row)]
        if not mentions:
            problems.append(
                f"nothing in the migration table maps `{key}`, which exists "
                f"in both packages and differs: {note.mechanism[:110]}... A "
                f"reader who does not find it here concludes it is missing."
            )
            continue
        naming = [number for number, row in mentions
                  if re.search(keyword, row, re.I)]
        if not naming:
            lines = ", ".join(str(number) for number, _ in mentions)
            problems.append(
                f"{PAGE.name}:{lines} maps `{key}` without naming the "
                f"deviation {note.name!r}. A reader porting that call gets "
                f"{note.consequence[:120]}... Add {keyword!r} to the row."
            )
        elif arguments.show:
            print(f"  {key}: {note.name} — named at line(s) "
                  f"{', '.join(map(str, naming))}")

    if problems:
        print("Migration table check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print(f"Migration table: {len(rows)} row(s), {len(deviations())} "
          f"upstream deviation(s), each named where it is mapped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
