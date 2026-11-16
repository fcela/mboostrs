#!/usr/bin/env python3
"""The family and learner pages cover what a reader has to know.

D2 sets a template for two page kinds:

    Every family page should show loss/NLL, link and inverse link, offset,
    negative gradient, response domain, prediction interpretation, variance or
    nuisance parameters, residuals, and failure modes. Every learner page
    should show design, penalty, null space, centering, extrapolation, `by`
    behavior, cost, and one case where it is the wrong choice.

Those are checkable, and checking them found gaps. `usage/families.ipynb` did
not mention the link or the inverse link at all — the two things that decide
whether `predict()` returns a probability or a log-odds. Neither learner page
mentioned extrapolation, for a policy the package ships and warns about.
`usage/base_learners.ipynb` had no case where a learner is the wrong choice,
which is the half of a catalogue that stops it being a menu.

This checks presence, not quality: a page mentioning extrapolation once passes.
That is the honest limit of a string match, and it is still worth having,
because the failures above were absences rather than bad explanations.

    python tools/check_page_templates.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"

#: Element name -> what counts as covering it. Alternatives are spellings the
#: book actually uses, not synonyms invented here.
FAMILY_ELEMENTS = {
    "loss/NLL": r"\bloss\b|negative log-?likelihood|\bNLL\b",
    "link": r"\blink\b",
    "inverse link": r"inverse link|response scale",
    "offset": r"\boffset\b",
    "negative gradient": r"negative gradient|ngradient|pseudo-?residual",
    "response domain": r"response (domain|support)|\bsupport\b|must be in",
    "prediction interpretation": r'predict\(.*type|type="response"|on the response scale',
    "nuisance": r"nuisance|dispersion|scale parameter|sigma",
    "residuals": r"residual",
    "failure modes": r"fail|refus|raise|not available|unsupported",
}

LEARNER_ELEMENTS = {
    "design": r"\bdesign\b|basis matrix",
    "penalty": r"\bpenalt",
    "null space": r"null space|unpenali[sz]ed part",
    "centering": r"cent(er|re)",
    "extrapolation": r"extrapolat|boundary knot|outside the (training |fitted )?range",
    "by= behaviour": r"`by`|by=|varying[- ]coefficient",
    "cost": r"\bcost\b|degrees of freedom|df=|memory|expensive",
    "wrong choice": r"wrong (choice|when)|not the right|do not use|avoid|inappropriate|when not to",
}

PAGES = {
    "usage/families.ipynb": FAMILY_ELEMENTS,
    "methodology/families.ipynb": FAMILY_ELEMENTS,
    "usage/base_learners.ipynb": LEARNER_ELEMENTS,
    "methodology/base_learners.ipynb": LEARNER_ELEMENTS,
}


def page_text(page: str) -> str:
    notebook = json.loads((BOOK / page).read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in notebook["cells"])


def missing(page: str, elements: dict[str, str]) -> list[str]:
    text = page_text(page)
    return [name for name, pattern in elements.items()
            if not re.search(pattern, text, re.I)]


def main() -> int:
    problems = []
    for page, elements in PAGES.items():
        gaps = missing(page, elements)
        if gaps:
            problems.append(f"{page} does not cover: {', '.join(gaps)}")
    if problems:
        print("Page template check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print(f"Page templates: {len(PAGES)} pages cover every required element")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
