#!/usr/bin/env python3
"""Hold `examples/python/` to the same standard as the book.

Eight public notebooks sat outside every factual check the book pages get.
Measured against the package they document, they called a function that does
not exist (`mb.plot_cvrisk`, which lives at `mb.plot.plot_cvrisk`), described
`bns` as having no penalty where it is ridge-penalised, called `bkernel`
Nadaraya-Watson smoothing where it is kernel ridge over an explicit design,
and said trees "handle ... missing data naturally" where every learner refuses
a `NaN`. Seven of the eight also carried stored outputs from an earlier
revision, which nothing regenerated and nothing compared with the current
package.

What this checks:

* **Every `mb.<name>` resolves.** In prose as well as code -- three of the
  four errors above were in markdown, so a check that read only code cells
  would have missed them.
* **No stored outputs.** They are a claim about what the package prints, made
  by a file nothing re-executes. `execute_examples.sh` runs the notebooks
  against the built wheel; that run is the evidence, not a saved cell.
* **Claims that contradict the package.** A short, specific list -- each entry
  is a statement measured false, not a phrase somebody disliked.

Run from the repository root:

    python tools/check_examples.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "python"

sys.path.insert(0, str(ROOT / "python"))

import mboostrs as mb  # noqa: E402

#: A match is not a problem when the surrounding text names upstream. Saying
#: "upstream's `bns` is unpenalised; this one is not" *is* the deviation being
#: explained, which is what a page should say -- and a rule that forbids the
#: word makes the correct sentence unwritable. This window is what separates
#: explaining a deviation from claiming it of mboostrs's own learner.
EXPLAINED_WINDOW = 80

#: Statements measured against the package and found false. The pattern is
#: what to look for; the reason is what the package actually does, so a
#: failure tells the author which way to correct it.
CONTRADICTED = (
    (r"\bmb\.plot_(cvrisk|risk|stabsel|aic)\b",
     "there is no top-level `mb.plot_*`; the functions live in `mb.plot`, and "
     "results carry a `.plot()` method"),
    # Both spellings: the examples said "no penalty" and the methodology page
    # headed a section "`bns` — unpenalised B-spline" directly above a body
    # saying "with an identity (ridge) penalty", so the first pattern caught
    # only one of them.
    (r"[Bb]ns\b[^.]{0,90}(\bno (difference )?penalt|unpenali[sz]ed)",
     "`bns` is ridge-penalised. Upstream's is unpenalised, which is a recorded "
     "parity deviation, not a shared property"),
    (r"[Bb]kernel\b[^.]{0,90}Nadaraya",
     "`bkernel` is kernel ridge over an explicit design, not a "
     "Nadaraya-Watson weighted average"),
    (r"[Tt]rees?\b[^.]{0,90}missing (data|values?)\s+naturally",
     "every learner requires finite input; a `NaN` raises. `na_action=\"omit\"` "
     "drops those rows"),
    (r"(complete|full|100%)\s+parity",
     "the capability contract records verified, partial and unsupported rows; "
     "parity is per row, not a property of the package"),
    # A *support* claim, not any sentence with the words in it. The first
    # version flagged "all base learners have identical df" in a gamlss
    # exercise, which is a condition on a derivation and says nothing about
    # what the package implements.
    (r"(all base learners and families"
     r"|(supports?|implements?|provides?|covers?|the same)\s+all\s+"
     r"(the\s+)?(families|base learners)"
     r"|all\s+(the\s+)?(families|base learners)\s+(are|is)\s+"
     r"(supported|implemented|available))",
     "`bss` and `blackboost` are not implemented, nor are the `FP`, `Hurdle` "
     "and `RCG` families"),
)


def notebooks() -> list[pathlib.Path]:
    """The eight public examples, plus every notebook in the book's ToC.

    A claim measured false is false wherever it appears. Restricting this to
    `examples/python` left `methodology/base_learners.ipynb` heading a section
    "`bns` — unpenalised B-spline" directly above a body saying "with an
    identity (ridge) penalty" -- a reader scanning headings got the opposite
    of the truth.

    Book pages are checked for contradicted claims and unresolved `mb.` names
    only. Their stored outputs are the book build's business, and
    `check_docs.py` already rules on those.
    """
    import yaml

    pages = sorted(EXAMPLES.glob("*.ipynb")) if EXAMPLES.is_dir() else []
    book = ROOT / "docs" / "book"
    config = yaml.safe_load((book / "myst.yml").read_text(encoding="utf-8"))

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    pages += [book / page for page in walk(config["project"]["toc"], [])
              if page.endswith(".ipynb")]
    return pages


def cells(notebook: dict, kind: str) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == kind)


def check(path: pathlib.Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = cells(notebook, "code")
    prose = cells(notebook, "markdown")
    problems: list[str] = []

    public = set(mb.__all__)
    for name in sorted(set(re.findall(r"\bmb\.(\w+)", code + "\n" + prose))):
        if name not in public and not name.startswith("_"):
            problems.append(f"`mb.{name}` is not a public name")

    # The book's own stored outputs are `check_docs.py`'s call; this rule is
    # about the standalone examples, which nothing else executes.
    in_examples = EXAMPLES in path.parents
    stored = sum(1 for c in notebook["cells"]
                 if c["cell_type"] == "code" and c.get("outputs"))
    if stored and in_examples:
        problems.append(
            f"{stored} cell(s) carry stored outputs. Nothing regenerates them, "
            f"so they are a claim about the package that no run supports; "
            f"`execute_examples.sh` is the evidence instead."
        )

    for pattern, reason in CONTRADICTED:
        for match in re.finditer(pattern, prose):
            window = prose[max(0, match.start() - EXPLAINED_WINDOW):match.end()]
            if "upstream" in window.lower():
                continue
            problems.append(f"{match.group(0)[:70]!r}: {reason}")

    return problems


def main() -> int:
    if not EXAMPLES.is_dir():
        print(f"no {EXAMPLES.relative_to(ROOT)} to check")
        return 0
    failed = 0
    for path in notebooks():
        problems = check(path)
        if problems:
            failed += 1
            print(f"\n=== {path.relative_to(ROOT)}")
            for problem in problems:
                print(f"  - {problem}")
    if failed:
        print(f"\n{failed} example notebook(s) need attention")
        return 1
    print(f"{len(notebooks())} notebooks (examples + book ToC): names "
          f"resolve, no stored outputs in the examples, no contradicted "
          f"claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
