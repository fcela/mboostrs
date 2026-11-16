"""The landing page and the table of contents agree, and the counts are real.

Three claims a reader meets before any statistics, all measured wrong:

* The landing page offered three cards and called Implementation "Part III".
  The ToC makes Case studies Part III and Implementation Part IV, so a reader
  following the card for the implementation chapters arrived at case studies,
  and the 29 studies had no card at all.
* The case overview opened "Seventeen worked analyses". The ToC lists 30 case
  pages, one of which is that overview — 29 studies, stale by twelve.
* The quickstart sat *after* the interface map, so the first page after
  installing was a matrix of every entry point rather than one worked fit.

The counts here are read from `myst.yml`, so a page added or removed fails
this rather than leaving the prose quietly wrong.
"""

import json
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"

NUMBER_WORDS = {
    17: "Seventeen", 27: "Twenty-seven", 28: "Twenty-eight",
    29: "Twenty-nine", 30: "Thirty", 31: "Thirty-one", 32: "Thirty-two",
    33: "Thirty-three", 36: "Thirty-six", 38: "Thirty-eight",
    41: "Forty-one", 43: "Forty-three", 45: "Forty-five", 46: "Forty-six",
    51: "Fifty-one", 53: "Fifty-three", 55: "Fifty-five",
    56: "Fifty-six", 57: "Fifty-seven", 58: "Fifty-eight",
    59: "Fifty-nine", 60: "Sixty", 61: "Sixty-one", 68: "Sixty-eight", 69: "Sixty-nine",
    78: "Seventy-eight", 79: "Seventy-nine", 80: "Eighty", 81: "Eighty-one",
}


def toc():
    config = yaml.safe_load((BOOK / "myst.yml").read_text())

    def walk(entries, out):
        for entry in entries:
            if "title" in entry and "file" not in entry:
                out.append(("part", entry["title"]))
            if "file" in entry:
                out.append(("page", entry["file"]))
            if "children" in entry:
                walk(entry["children"], out)
        return out

    return walk(config["project"]["toc"], [])


def prose(page: str) -> str:
    notebook = json.loads((BOOK / page).read_text())
    return "\n".join("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "markdown")


@pytest.fixture(scope="module")
def parts():
    return [name for kind, name in toc() if kind == "part"]


@pytest.fixture(scope="module")
def case_pages():
    return [p for kind, p in toc() if kind == "page" and p.startswith(("cases/", "case_studies/"))]


# ─── the landing page matches the ToC ──────────────────────────────


def test_the_toc_numbers_its_parts_in_order(parts):
    numbered = [p for p in parts if p.startswith("Part ")]
    assert [p.split(" —")[0] for p in numbered] == [
        "Part I", "Part II", "Part III", "Part IV", "Part V",
    ]


def test_the_landing_page_names_each_part_as_the_toc_does(parts):
    """It called Implementation "Part III", where the ToC puts case studies."""
    landing = prose("intro.ipynb")
    for part in parts:
        if not part.startswith("Part "):
            continue
        number, title = part.split(" — ", 1)
        found = re.findall(rf"{number} — ([A-Za-z ]+)", landing)
        assert found, f"{number} is not named on the landing page"
        assert title.strip() in [f.strip() for f in found], (
            f"{number} is '{title}' in the ToC and {found} on the landing page"
        )


def test_the_landing_page_has_a_card_for_every_part(parts):
    landing = prose("intro.ipynb")
    cards = re.findall(r"grid-item-card\} (Part [IV]+)", landing)
    assert cards == [p.split(" —")[0] for p in parts if p.startswith("Part ")]


def test_no_stale_part_cross_reference():
    """The body said "Part III documents the places where ..." about the
    implementation chapters."""
    landing = prose("intro.ipynb")
    assert "Part III documents" not in landing


# ─── the case count is the ToC's ───────────────────────────────────


def test_the_overview_states_the_number_of_studies_it_has(case_pages):
    studies = [p for p in case_pages if not p.endswith("overview.ipynb")]
    word = NUMBER_WORDS.get(len(studies))
    assert word, f"add {len(studies)} to NUMBER_WORDS"
    opening = prose("cases/overview.ipynb").split("\n\n", 2)[1]
    assert word in opening, (
        f"the overview should open with {word} ({len(studies)} studies in the "
        f"ToC); it says: {opening[:90]!r}"
    )


def test_the_overview_claims_no_split_it_cannot_support(case_pages):
    """It said "Eleven use small real data sets ... Six use simulated", which
    summed to seventeen and was checked by nothing."""
    overview = prose("cases/overview.ipynb")
    assert "Eleven use small real data" not in overview
    assert "Six\nuse simulated" not in overview


# ─── the reader's first pages ──────────────────────────────────────


def test_install_then_quickstart_come_first():
    """The interface map sat between them, so the page after installing was a
    matrix of every entry point rather than one worked fit."""
    pages = [p for kind, p in toc() if kind == "page" and p.startswith("usage/")]
    assert pages[0] == "usage/install.ipynb"
    assert pages[1] == "usage/quickstart.ipynb"
    assert pages.index("usage/interface_map.ipynb") > pages.index("usage/quickstart.ipynb")


def test_the_landing_page_states_bounded_support():
    """The same claim the README carried: "the same base learners, and the
    same families as the R packages ... run several times faster"."""
    landing = prose("intro.ipynb")
    assert "the same base learners" not in landing
    assert "several times faster" not in landing
    assert "capability contract" in landing or "capabilities.ipynb" in landing


# ─── the R "same workflow" really is the same ──────────────────────


def test_the_r_block_performs_the_same_four_steps():
    """The quickstart's Python workflow is fit, tune `mstop`, truncate,
    interpret. The block headed "The same workflow in R" did the first and
    last only -- no `cvrisk`, no truncation -- so the heading was false.
    D6: the two must perform the same tuning and post-fit operations, or
    identify the difference."""
    text = prose("usage/quickstart.ipynb")
    block = text.split("## The same workflow in R", 1)[1].split("## ", 1)[0]
    assert "cvrisk(" in block, "no tuning step"
    assert "mstop(cvr)" in block, "nothing reads the chosen mstop"
    assert "fit[m_opt]" in block, "the path is never truncated"
    assert "coef(final)" in block, "it interprets the untruncated fit"
    for step in ("# 1.", "# 2.", "# 3.", "# 4."):
        assert step in block, step


def test_the_r_block_names_the_python_spellings():
    """A migration reader needs the mapping, not just the equivalence."""
    text = prose("usage/quickstart.ipynb")
    block = text.split("## The same workflow in R", 1)[1].split("## See also", 1)[0]
    assert "fit[:k]" in block
    assert "cvr.optimal_mstop" in block


# ─── the benchmarks intro carries the scorecard its pages cite ──────


def test_the_benchmarks_intro_names_the_whole_scorecard():
    """The Part V landing page says its comparisons are graded on the
    knowledge-discovery scorecard. It should name the version and every
    dimension, so the contract is on the page and not only in the notebooks."""
    text = prose("benchmarks/overview.ipynb")
    assert "mboostrs-scorecard/2.1" in text
    for dim in ["D1a", "D1b"] + [f"D{i}" for i in range(2, 12)]:
        assert f"{dim} " in text, f"{dim} is missing from the benchmarks intro"
