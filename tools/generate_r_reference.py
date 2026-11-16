#!/usr/bin/env python3
"""Render the R frontend's reference page from its own help files.

D1's remaining item: "linking each entry to a minimal copyable example, and
the R/Rust halves of the reference. The Python surface is generated;
`_rust.pyi` is already generated ... and the R NAMESPACE/Rd surface is checked
by the capability contract but not rendered into the book."

So an R user reading the book had a migration table and no reference: to see
what `mboostrs::stabsel` takes they had to install the package and type `?`.
The information existed -- `NAMESPACE` says what is exported and `man/*.Rd`
carries the usage and the title -- and nothing brought the two into the book.

This reads both and renders one page. Nothing is written by hand, so the page
cannot drift from the package: a renamed argument changes the `\\usage` block,
which changes this page, and `--check` fails until it is regenerated.

Two things it deliberately leaves out:

* **internal helpers.** A page carrying `\\keyword{internal}` is documented
  for a maintainer, not offered to a user.
* **S3 methods**, individually. There are 144, and a user does not call
  `predict.mboostrs_gamboost`; they call `predict`. The generics are listed
  once with the classes they dispatch on.

Run from the repository root:

    python tools/generate_r_reference.py
    python tools/generate_r_reference.py --check
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMESPACE = ROOT / "mboostrs" / "NAMESPACE"
MAN = ROOT / "mboostrs" / "man"
OUTPUT = ROOT / "docs" / "book" / "appendix" / "r_reference.ipynb"

#: How the entries are grouped, in the order a reader meets them. A name
#: matching none of these falls into "Everything else", which is a signal that
#: either the grouping or the export needs a decision -- not a place to leave
#: things.
GROUPS = (
    ("Fitting", (r"^(glm|gam)boost(LSS)?$", r"^boost_control$")),
    ("Families", (r"LSS$", r"^(Gaussian|Binomial|Poisson|Laplace|Huber|"
                  r"Quantile|Expectile|AdaExp|NBinomial|Hurdle|PropOdds|CoxPH|"
                  r"Weibull|Loglog|Lognormal|Multinomial|GammaReg|GaussReg|"
                  r"AUC|Cindex|Gehan|HingeLoss)$")),
    ("Base learners", (r"^base_learners$",)),
    ("Model selection", (r"^(cvrisk|stabsel|cv_weights|loo_folds|mstop)",
                         r"^stabsel_params$", r"aic_path")),
    ("REML refits", (r"^gam(lss)?_refit$", r"^gamlss_draws$", r"mboostrs_gam(lss)?_refit$",
                     r"^concurvity")),
    ("Reading a fit", (r"^(coef|predict|fitted|residuals|risk|selected|"
                       r"varimp|summary|logLik|AIC|confint|extract|"
                       r"model_weights|nuisance|size|hatvalues|partial_effects|"
                       r"path_frame|execution_report|deviance_increment)",
                       r"^coefAll", r"^coef_path")),
    ("Distributional predictions", (r"^lss_", r"^(centiles|PI|predint)$",
                                    r"^make_grid_lss$")),
    ("Prediction intervals", (r"^(conformal_interval|rearrange_quantiles)$",)),
    ("Explaining a fit", (r"^llm_", r"^LLM_")),
    ("Diagnostics and plots", (r"plot", r"^mboostrs_palette$",
                               r"residual|calibration|pit|worm|q_stats",
                               r"^(centile_fan_data|density_overlay_data|"
                               r"deviance_profile|support_report)$")),
    ("Artifacts", (r"^(load_model|save_model)$",)),
    ("Build identity", (r"^mboostrs_build_info$",)),
    ("Capabilities", (r"capabilit", r"^parity", r"^validation")),
)


def shown(path: pathlib.Path) -> str:
    """The path as a reader would type it, or as given when it is outside the
    repository -- a `--check` run against a scratch file must report, not
    raise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def exported_names() -> set[str]:
    text = NAMESPACE.read_text(encoding="utf-8")
    return {name.strip().strip('"')
            for name in re.findall(r"^export\(([^)]+)\)", text, re.M)}


def s3_methods() -> dict[str, list[str]]:
    """`generic -> [class, ...]`, which is what a user needs: they call the
    generic."""
    text = NAMESPACE.read_text(encoding="utf-8")
    found: dict[str, list[str]] = collections.defaultdict(list)
    for generic, klass in re.findall(r"^S3method\(([^,]+),\s*([^)]+)\)", text,
                                     re.M):
        found[generic.strip().strip('"')].append(klass.strip().strip('"'))
    return {generic: sorted(classes) for generic, classes in sorted(found.items())}


#: Help pages whose aliases are deliberately *not* exported, with the reason.
#: `base_learners.Rd` documents `bols`, `bbs` and the rest as formula syntax
#: -- calling them at the prompt is an error, which that page says in as many
#: words -- so none of them is in NAMESPACE. It is also the page a formula
#: writer needs most, and excluding it because of how R models the surface
#: would be the tail wagging the dog.
UNEXPORTED_BUT_PUBLIC = {
    "base_learners": "documents the formula syntax; the learners are parsed "
                     "out of a formula rather than called",
}


def rd_block(text: str, command: str) -> str:
    """Read an Rd command's contents, including balanced nested markup."""
    match = re.search(r"\\" + re.escape(command) + r"\s*\{", text)
    if match is None:
        return ""
    start = match.end()
    depth = 1
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
        index += 1
    raise ValueError(f"unclosed Rd command: {command}")


def help_pages() -> list[dict]:
    """One record per public help page, in the order the reader will see."""
    exports = exported_names()
    pages = []
    for path in sorted(MAN.glob("*.Rd")):
        text = path.read_text(encoding="utf-8")
        if "\\keyword{internal}" in text:
            continue
        aliases = [alias.strip()
                   for alias in re.findall(r"\\alias\{([^}]+)\}", text)]
        public = [alias for alias in aliases if alias in exports]
        if not public and path.stem in UNEXPORTED_BUT_PUBLIC:
            public = [path.stem]
        if not public:
            continue
        pages.append({
            "name": public[0],
            "aliases": public,
            "title": clean(rd_block(text, "title")),
            "usage": rd_block(text, "usage").strip(),
        })
    return pages


def clean(text: str) -> str:
    """Rd markup to plain text, for a one-line title."""
    while True:
        previous = text
        text = re.sub(r"\\(code|pkg|emph|strong|dQuote|sQuote)\{([^{}]*)\}",
                      r"`\2`", text)
        text = re.sub(r"\\link\[[^\]]*\]\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
        if text == previous:
            break
    text = text.replace("\\%", "%").replace("\\", "")
    return " ".join(text.split())


def group_of(name: str) -> str:
    for label, patterns in GROUPS:
        if any(re.search(pattern, name) for pattern in patterns):
            return label
    return "Everything else"


def first_line(usage: str) -> str:
    """The first signature, on one line, with balanced parentheses.

    An Rd usage block wraps long calls over several lines and often lists
    several. Matching to the first `)` truncates every default that is itself
    a call -- `gamboost(formula, data, family = Gaussian()` -- so the depth is
    tracked instead.
    """
    if not usage:
        return ""
    joined = re.sub(r"\s+", " ", " ".join(line.strip()
                                          for line in usage.splitlines()))
    start = joined.find("(")
    if start == -1:
        return joined.strip()
    depth = 0
    for index in range(start, len(joined)):
        if joined[index] == "(":
            depth += 1
        elif joined[index] == ")":
            depth -= 1
            if depth == 0:
                return joined[:index + 1].strip()
    return joined.strip()


def render_markdown() -> str:
    pages = help_pages()
    methods = s3_methods()
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for page in pages:
        grouped[group_of(page["name"])].append(page)

    parts = [
        "# R reference",
        "",
        "`tools/generate_r_reference.py` reads exported names from `NAMESPACE` "
        "and call signatures from the R help files in `man/*.Rd`. "
        "Use `?name` in R for the corresponding argument descriptions.",
        "",
        f"{len(pages)} exported entries with a help page, and "
        f"{len(methods)} S3 generics. Internal helpers are left out: a page "
        "marked `\\keyword{internal}` is documented for a maintainer rather "
        "than offered to a user.",
        "",
        "The R frontend refuses `%X%`/`%O%` interactions, `bins=`, and "
        "inline transforms. Support for `by=` depends on the learner and "
        "modifier type; unsupported combinations raise an error. "
        "[](../usage/r_migration.ipynb) maps the two.",
        "",
    ]

    order = [label for label, _ in GROUPS] + ["Everything else"]
    for label in order:
        entries = grouped.get(label)
        if not entries:
            continue
        parts += [f"## {label}", ""]
        parts.append("| Call | What it is |")
        parts.append("|---|---|")
        for entry in sorted(entries, key=lambda item: item["name"]):
            signature = first_line(entry["usage"]) or entry["name"]
            extra = ""
            if len(entry["aliases"]) > 1:
                others = ", ".join(f"`{a}`" for a in entry["aliases"][1:])
                extra = f" Also documents {others}."
            parts.append(f"| `{signature}` | {entry['title']}{extra} |")
        parts.append("")

    parts += [
        "## S3 generics",
        "",
        "Call the generic, such as `predict(fit)`, rather than "
        "`predict.mboostrs_gamboost(fit)`. These are the generics the package "
        "extends and the classes each dispatches on.",
        "",
        "| Generic | Classes |",
        "|---|---|",
    ]
    for generic, classes in methods.items():
        shown = ", ".join(f"`{name}`" for name in classes)
        parts.append(f"| `{generic}` | {shown} |")
    parts.append("")
    return "\n".join(parts) + "\n"


def render_notebook() -> str:
    markdown = render_markdown()
    notebook = {
        "cells": [{"cell_type": "markdown", "metadata": {},
                   "source": markdown.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail when the committed page is out of date")
    arguments = parser.parse_args()

    generated = render_notebook()
    if arguments.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != generated:
            print(f"{shown(OUTPUT)} is stale. Regenerate it with "
                  f"tools/generate_r_reference.py", file=sys.stderr)
            return 1
        print(f"R reference is up to date: {len(help_pages())} entries")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(generated, encoding="utf-8")
    print(f"wrote {shown(OUTPUT)}: {len(help_pages())} entries, "
          f"{len(s3_methods())} generics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
