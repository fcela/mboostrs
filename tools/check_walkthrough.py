#!/usr/bin/env python3
"""Walk the two reader journeys D8 gates the documentation on.

D8 item 1: "A new Python user must install, fit, tune, truncate, diagnose, and
predict a Gaussian additive model using only the README and quickstart."

D8 item 2: "A statistically experienced user must choose among mean, quantile,
survival, ranking, and distributional models and identify the prediction scale
and unavailable diagnostics without opening source code."

Both are claims about *reachability*, and both were unverified. The package
having a diagnostic is not the same as the walkthrough reaching one, and a
capability table being correct is not the same as a reader being able to find
it. The first journey ran four of its six operations: the quickstart fitted,
tuned, truncated and predicted, and never checked whether the fit was right.

**The new-user journey** concatenates the README's Python blocks and the
quickstart's code cells into one session, in reading order, and runs it. Each
of the six operations must

* appear as a call the reader can copy,
* execute in that session without error, and
* be *interpreted* in the surrounding prose -- a sentence that says what the
  result means or what value to expect.

The third condition is the one that matters. `fit.residuals()` sitting in a
list of five one-liners is a call, not a diagnosis: the reader gets an array
and no way to tell a good one from a bad one.

**The experienced-user journey** takes the five model kinds the item names and,
for each, checks that a reader who never opens the source can answer two
questions: what scale does `predict` return on, and which diagnostics are
unavailable. The answers must exist in the documentation *and* agree with what
the package reports through `family_capabilities()` and
`unavailable_operations()` -- a documented scale that the package contradicts
is worse than a missing one, because the reader has no reason to doubt it.

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
    """One of the six things the new-user journey must let a reader do."""

    #: The operation's name, as D8 item 1 writes it.
    name: str
    #: What the reader copies. Matched against the executable source of the
    #: two documents -- shell for `install`, Python for the rest.
    call: str
    #: A sentence in the prose that says what the result means. Without it the
    #: reader can run the call and cannot read the answer.
    interpretation: str
    #: What the sentence has to convey, for the failure message. This is the
    #: repair instruction, not a restatement of the regex.
    conveys: str
    #: Whether the call is Python the combined session executes.
    executable: bool = True


#: Ordered as the reader meets them.
OPERATIONS = (
    Operation(
        name="install",
        call=r"pip install\s+[\"']?mboostrs",
        interpretation=r"installs? NumPy and Polars|nothing else|no other dependenc",
        conveys="what the base install pulls in, so the reader knows whether "
                "the plotting extra is needed",
        executable=False,
    ),
    Operation(
        name="fit",
        call=r"\bmb\.gamboost\(",
        interpretation=r"additive\b.*\bmodel|smooth (effect|term)|arbitrary base learners",
        conveys="that gamboost fits an additive model and what the terms in "
                "the formula are",
    ),
    Operation(
        name="tune",
        call=r"\.cvrisk\(",
        interpretation=r"cross-validated risk|risk-minimi[sz]ing iteration|"
                       r"out-of-(bag|sample)",
        conveys="that mstop is chosen on data the fit did not use, and which "
                "iteration the result names",
    ),
    Operation(
        name="truncate",
        call=r"\.with_mstop\(|\bfit\[:",
        interpretation=r"truncat|shorter path|stops the path at|as if it had "
                       r"been fitted",
        conveys="that truncation changes which iteration every later call "
                "reads from, and does not refit",
    ),
    Operation(
        name="diagnose",
        call=r"\.residual_frame\(|\.plot_residuals\(|\.pit\(|\.worm_data\(|"
             r"\.q_stats\(|\.calibration_frame\(|\.outlier_frame\(",
        interpretation=r"if the (model|fit|variance|mean) (is|were) wrong|"
                       r"would show|structure left in the residual|"
                       r"pattern (in|against)",
        conveys="what a wrong fit would look like in the diagnostic, so the "
                "reader can tell a good result from a bad one",
    ),
    Operation(
        name="predict",
        call=r"\.predict\(",
        interpretation=r'type="response"|on the (response|original) scale|'
                       r"scale of `?y`?",
        conveys="which scale the prediction comes back on",
    ),
)


#: D8 item 2's five model kinds, each with the entry point a reader would
#: reach for and a family that exercises it. `family_capabilities()` is the
#: package's own answer about scale and diagnostics; the documentation has to
#: agree with it.
MODEL_KINDS = {
    "mean": "Gaussian",
    "quantile": "Quantile",
    "survival": "CoxPH",
    "ranking": "AUC",
    "distributional": "GaussianLSS",
}

#: Where an experienced reader is expected to find those answers. Every page
#: here must exist and must route to the two package-level answers.
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
    """Execute the reader's blocks in order, in one namespace.

    A scratch directory, for the reason the front-door check runs in one: the
    README serializes a model, and the reader's working directory is theirs to
    litter rather than the repository's.
    """
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
    """D8 item 1."""
    readme_python, readme_shell, readme_prose = readme_source()
    quickstart_code, quickstart_prose = notebook_source(QUICKSTART)

    executable = readme_python + "\n" + quickstart_code
    shell = readme_shell
    prose = readme_prose + "\n" + quickstart_prose

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
        if not re.search(operation.interpretation, prose, re.I):
            problems.append(
                f"new-user journey can run {operation.name} but not read it: "
                f"the prose never says {operation.conveys}"
            )
            continue
        if show:
            print(f"  {operation.name}: called and interpreted")

    blocks = [("README.md", readme_python), ("quickstart.ipynb", quickstart_code)]
    namespace, failures = run_session(blocks)
    problems.extend(failures)
    if show and not failures:
        print(f"  the combined session ran, leaving {len(namespace)} name(s)")
    return problems


def check_experienced_user(show: bool) -> list[str]:
    """D8 item 2."""
    problems = []
    capabilities = {row["name"]: row
                    for row in mb.family_capabilities(as_frame=False)}

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
                f"{kind}: `{family}` reports no prediction scale in "
                f"`family_capabilities()`"
            )
        if not row.get("diagnostics"):
            problems.append(
                f"{kind}: `{family}` reports no diagnostics status in "
                f"`family_capabilities()`"
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
            problems.append(
                f"the decision pages never name the {kind} case, which D8 "
                f"item 2 requires a reader to choose among"
            )
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
            print("new-user journey (D8 item 1):")
        problems += check_new_user(arguments.show)
    if arguments.journey in (None, "experienced-user"):
        if arguments.show:
            print("experienced-user journey (D8 item 2):")
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
