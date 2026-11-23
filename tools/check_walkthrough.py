#!/usr/bin/env python3
"""Verify the two core reader journeys execute and match the package contract.

1. **New-user journey:** concatenates the README Python blocks and the
   quickstart notebook code cells into one session, checks that install, fit,
   tune, truncate, diagnose, and predict appear as runnable calls, and executes
   the combined session cleanly.
2. **Experienced-user journey:** checks that the five model kinds (mean,
   quantile, survival, ranking, distributional) report prediction scales and
   diagnostics status in `family_capabilities()`, and that the decision pages
   exist and reference `family_capabilities` and `unavailable_operations`.

Run from the repository root:

    python tools/check_walkthrough.py
    python tools/check_walkthrough.py --show
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
QUICKSTART = ROOT / "docs" / "book" / "usage" / "quickstart.ipynb"

sys.path.insert(0, str(ROOT / "python"))

import mboostrs as mb  # noqa: E402


@dataclass(frozen=True)
class Operation:
    """One of the six executable steps in the new-user journey."""

    name: str
    call: str
    executable: bool = True


OPERATIONS = (
    Operation(name="install", call=r"pip install\s+[\"']?mboostrs", executable=False),
    Operation(name="fit", call=r"\bmb\.gamboost\("),
    Operation(name="tune", call=r"\.cvrisk\("),
    Operation(name="truncate", call=r"\.with_mstop\(|\bfit\[:"),
    Operation(
        name="diagnose",
        call=(
            r"\.residual_frame\(|\.plot_residuals\(|\.pit\(|\.worm_data\(|"
            r"\.q_stats\(|\.calibration_frame\(|\.outlier_frame\("
        ),
    ),
    Operation(name="predict", call=r"\.predict\("),
)

MODEL_KINDS = {
    "mean": "Gaussian",
    "quantile": "Quantile",
    "survival": "CoxPH",
    "ranking": "AUC",
    "distributional": "GaussianLSS",
}

DECISION_PAGES = (
    "usage/five_decisions.ipynb",
    "usage/families.ipynb",
    "usage/interface_map.ipynb",
)


def notebook_source(path: pathlib.Path) -> tuple[str, str]:
    """`(python source, markdown prose)` for a notebook, cells in order."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code, prose = [], []
    for cell in cells:
        text = "".join(cell["source"])
        (code if cell["cell_type"] == "code" else prose).append(text)
    return "\n".join(code), "\n".join(prose)


def markdown_blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, re.S)


def readme_source() -> tuple[str, str, str]:
    """`(python, shell, prose)` for the README."""
    text = README.read_text(encoding="utf-8")
    python = [
        block for block in markdown_blocks(text, "python")
        if "check-front-door: skip" not in block.splitlines()[0]
    ]
    shell = markdown_blocks(text, "bash") + markdown_blocks(text, "shell")
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    return "\n".join(python), "\n".join(shell), prose


def run_session(blocks: list[tuple[str, str]]) -> tuple[dict, list[str]]:
    """Execute the reader's blocks in order, in one namespace, in a scratch dir."""
    namespace: dict[str, object] = {}
    problems: list[str] = []
    scratch = tempfile.mkdtemp(prefix="mboostrs-walkthrough-")
    previous = os.getcwd()
    os.chdir(scratch)
    try:
        for origin, source in blocks:
            try:
                exec(compile(source, origin, "exec"), namespace)  # noqa: S102
            except Exception:
                problems.append(
                    f"{origin} failed while walking the journey:\n"
                    + "".join(traceback.format_exc(limit=3)).rstrip()
                )
    finally:
        os.chdir(previous)
        shutil.rmtree(scratch, ignore_errors=True)
    return namespace, problems


def check_new_user(show: bool) -> list[str]:
    readme_python, readme_shell, _ = readme_source()
    quickstart_code, _ = notebook_source(QUICKSTART)

    executable = readme_python + "\n" + quickstart_code
    shell = readme_shell

    problems = []
    for operation in OPERATIONS:
        haystack = executable if operation.executable else shell
        where = "a python block" if operation.executable else "a shell block"
        if not re.search(operation.call, haystack):
            problems.append(
                f"new-user journey cannot {operation.name}: no call matching "
                f"`{operation.call}` in {where} of the README or quickstart"
            )
            continue
        if show:
            print(f"  {operation.name}: called")

    blocks = [("README.md", readme_python), ("quickstart.ipynb", quickstart_code)]
    namespace, failures = run_session(blocks)
    problems.extend(failures)
    if show and not failures:
        print(f"  the combined session ran, leaving {len(namespace)} name(s)")
    return problems


def check_experienced_user(show: bool) -> list[str]:
    problems = []
    capabilities = {
        row["name"]: row
        for row in mb.family_capabilities(as_frame=False)
    }

    for kind, family in MODEL_KINDS.items():
        row = capabilities.get(family)
        if row is None:
            problems.append(
                f"{kind}: `{family}` is not in `family_capabilities()`, so a "
                f"reader has no source-free answer for it"
            )
            continue
        if not row.get("predict_response"):
            problems.append(
                f"{kind}: `{family}` reports no prediction scale in `family_capabilities()`"
            )
        if not row.get("diagnostics"):
            problems.append(
                f"{kind}: `{family}` reports no diagnostics status in `family_capabilities()`"
            )

    book = ROOT / "docs" / "book"
    for page in DECISION_PAGES:
        if not (book / page).exists():
            problems.append(f"the decision route names {page}, which is missing")

    prose = "\n".join(
        notebook_source(book / page)[1] + notebook_source(book / page)[0]
        for page in DECISION_PAGES if (book / page).exists()
    )
    for route, what in (
        (r"family_capabilities", "the per-family scale and diagnostics table"),
        (r"unavailable_operations", "what a fitted model refuses and why"),
    ):
        if not re.search(route, prose):
            problems.append(
                f"no decision page points at `{route}`, which is how a reader "
                f"finds {what} without opening the source"
            )

    for kind in MODEL_KINDS:
        if not re.search(rf"\b{kind}\b", prose, re.I):
            problems.append(f"the decision pages never name the {kind} case")
        elif show:
            print(f"  {kind}: named, with a scale and a diagnostics status")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="print each journey step as it passes")
    parser.add_argument("--journey", choices=("new-user", "experienced-user"),
                        help="walk only one of the two")
    arguments = parser.parse_args()

    problems = []
    if arguments.journey in (None, "new-user"):
        if arguments.show:
            print("new-user journey:")
        problems += check_new_user(arguments.show)
    if arguments.journey in (None, "experienced-user"):
        if arguments.show:
            print("experienced-user journey:")
        problems += check_experienced_user(arguments.show)

    if problems:
        print("Walkthrough check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print(f"Walkthrough: {len(OPERATIONS)} new-user operation(s) reachable, "
          f"{len(MODEL_KINDS)} model kind(s) decidable without the source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
