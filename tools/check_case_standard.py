#!/usr/bin/env python3
"""How far each case study is from the reconstruction standard.

D3: "For every public case, state the practical target, observed response,
estimand, units, sampling unit, dependence structure, split/resampling unit,
tuning procedure, uncertainty procedure, and claim type."

Audited across the 29 studies, the gaps are wide and uneven:

    28/29  sampling unit      -- what one row is
    26/29  estimand           -- the quantity being estimated
    16/29  claim type         -- descriptive, predictive or causal
    10/29  dependence structure

Closing those means understanding each analysis, so this does not invent them
and does not fail on the backlog. It **ratchets**: the recorded baseline is
the current count, and the check fails when a study loses an element it had,
or a new study arrives below the standard. The backlog can only shrink.

`--update` rewrites the baseline after a genuine improvement.

    python tools/check_case_standard.py
    python tools/check_case_standard.py --report   # per-study detail
    python tools/check_case_standard.py --update   # after closing gaps
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"
BASELINE = ROOT / "docs" / "book" / "case_standard_baseline.json"

#: Element -> what counts as stating it. These are presence tests: a study
#: that says "each row is a district" covers the sampling unit. Whether it
#: says so *well* needs a reader, which is the standard's own point.
ELEMENTS = {
    "target": r"\btarget\b|question|aim|goal",
    "response": r"\bresponse\b|outcome|dependent",
    "estimand": r"estimand|quantity of interest|what is estimated",
    "units": r"\bunits?\b|measured in|per ",
    "sampling unit": r"sampling unit|each row is|one row per|observation is",
    "dependence": r"depend(ence|ency)|clustered|correlated|independent|i\.i\.d",
    "split unit": r"split|fold|resampl|held-?out|training set",
    "tuning": r"tuning|cvrisk|cross-?validat|mstop",
    "uncertainty": r"uncertaint|interval|standard error|confidence|bootstrap",
    # D3 names three claim types -- descriptive, predictive, causal -- and
    # asks that they be kept distinct. Five studies make a fourth kind:
    # `after_selection`, `learner_flexibility`, `stopping_rule`, `buser` and
    # `beyond_the_data` claim something about an *estimator*, demonstrated on
    # data whose truth the simulation sets. Calling that predictive or causal
    # would be the confusion the standard exists to prevent, so the vocabulary
    # gains the word rather than the studies losing the distinction.
    "claim type": r"descriptive|predictive|causal|association|methodological",
}


def case_pages() -> list[str]:
    import yaml

    config = yaml.safe_load((BOOK / "myst.yml").read_text(encoding="utf-8"))

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    return sorted(p for p in walk(config["project"]["toc"], [])
                  if p.startswith(("cases/", "case_studies/")) and not p.endswith("overview.ipynb"))


def audit() -> dict[str, list[str]]:
    """Missing elements per study."""
    result = {}
    for page in case_pages():
        notebook = json.loads((BOOK / page).read_text(encoding="utf-8"))
        text = "\n".join("".join(c["source"]) for c in notebook["cells"])
        result[page] = [name for name, pattern in ELEMENTS.items()
                        if not re.search(pattern, text, re.I)]
    return result


def load_baseline() -> dict[str, list[str]]:
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8"))["missing"]


def write_baseline(current: dict[str, list[str]]) -> None:
    total = sum(len(v) for v in current.values())
    BASELINE.write_text(json.dumps({
        "_comment": (
            "Gaps against the D3 case-study reconstruction standard. This is a "
            "ratchet, not a target: the check fails when a study loses an "
            "element it had, or a new study arrives below the standard. "
            "Closing a gap means understanding the analysis, so nothing here "
            "is generated prose -- shrink it by writing the missing statement "
            "and running --update."
        ),
        "studies": len(current),
        "open_gaps": total,
        "missing": {k: sorted(v) for k, v in sorted(current.items())},
    }, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--update", action="store_true")
    arguments = parser.parse_args()

    current = audit()
    total = sum(len(v) for v in current.values())

    if arguments.update:
        write_baseline(current)
        print(f"Baseline updated: {len(current)} studies, {total} open gap(s)")
        return 0

    baseline = load_baseline()
    if not baseline:
        write_baseline(current)
        print(f"Baseline created: {len(current)} studies, {total} open gap(s)")
        return 0

    regressions = []
    for page, missing in current.items():
        was = set(baseline.get(page, []))
        if page not in baseline:
            if missing:
                regressions.append(
                    f"{page} is new and does not state: {', '.join(missing)}"
                )
            continue
        lost = sorted(set(missing) - was)
        if lost:
            regressions.append(f"{page} no longer states: {', '.join(lost)}")

    if arguments.report:
        from collections import Counter

        counts = Counter(name for missing in current.values() for name in missing)
        print(f"{len(current)} case studies audited against the D3 standard\n")
        for name, count in counts.most_common():
            print(f"  {count:>3}/{len(current)}  {name}")
        print("\nper study:")
        for page, missing in sorted(current.items(), key=lambda kv: -len(kv[1])):
            if missing:
                print(f"  {len(missing)}  {page}: {', '.join(missing)}")

    if regressions:
        print("Case-standard ratchet failed:", file=sys.stderr)
        for regression in regressions:
            print(f"- {regression}", file=sys.stderr)
        print("Run --update only when a gap has genuinely been closed.",
              file=sys.stderr)
        return 1

    baseline_total = sum(len(v) for v in baseline.values())
    closed = baseline_total - total
    message = f"Case standard: {total} open gap(s) across {len(current)} studies"
    if closed > 0:
        message += f" ({closed} closed since the baseline; run --update)"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
