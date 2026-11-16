"""The deferred studies stay deferred, and the published ones stay honest.

P4.4: "Studies 05--07 remain references until their data are available and
their validation design is credible ... The deposit-elasticity extension
remains active research, not package evidence, until data access, panel
identification, and uncertainty are resolved."

Nothing here can be finished: the mortality, natality and Opportunity Atlas
studies wait on public data, a memory-bounded high-`n` pipeline, and a
disclosure review, none of which is code. What *is* actionable is a ratchet,
because the pressure on an item like this is one direction only — a study gets
promoted into the book when someone has a plot, and the framing that made it
honest goes with the draft it came from.

So: the three stay out of the table of contents, and the two published studies
that touch this territory keep the claim types that make them honest. The
deposit study's headline is a negative result — public aggregates cannot
identify the marginal cost — and the commuting-zone study is a descriptive
association, not the causal high-`p` selection study P4.4 defers.
"""

import json
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"

#: The three P4.4 holds back, by the words a page about them would use.
DEFERRED = ("mortality surface", "natality", "opportunity atlas")


def toc_pages() -> list[str]:
    config = yaml.safe_load((BOOK / "myst.yml").read_text())

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    return walk(config["project"]["toc"], [])


def page_text(relative: str) -> str:
    path = BOOK / relative
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8")
    return "\n".join("".join(cell["source"])
                     for cell in json.loads(path.read_text())["cells"])


def claim_type(relative: str) -> str:
    text = page_text(relative)
    line = next(line for line in text.splitlines() if "Claim type" in line)
    return line.split("|")[2].strip()


def test_the_deferred_studies_have_no_page():
    """A page in the ToC is the package saying "here is what we found"."""
    pages = toc_pages()
    for study in DEFERRED:
        slug = study.split()[0]
        assert not any(slug in page.lower() for page in pages), study


def test_no_published_study_is_titled_after_a_deferred_one():
    """The subtler failure: not a new page, but an existing one growing the
    claim the deferred study would have made."""
    for page in toc_pages():
        if not page.startswith("cases/"):
            continue
        heading = next((line for line in page_text(page).splitlines()
                        if line.startswith("# ")), "")
        for study in DEFERRED:
            assert study not in heading.lower(), (page, heading)


def test_the_deposit_study_reports_a_negative_result():
    """P4.4 wants it kept as active research rather than package evidence.
    It is: the recorded claim is that the identification fails."""
    claim = claim_type("cases/deposit_elasticity.ipynb")
    assert claim.startswith("descriptive")
    assert "cannot identify the marginal cost" in claim


def test_the_commuting_zone_study_stays_descriptive():
    """It uses Opportunity Atlas data, and is not the causal high-`p`
    stability-selection study P4.4 defers."""
    assert claim_type("cases/cz_known_variance.ipynb") == \
        "descriptive association"


@pytest.mark.parametrize("page", ["cases/deposit_elasticity.ipynb",
                                  "cases/cz_known_variance.ipynb"])
def test_neither_claims_causal_identification(page):
    claim = claim_type(page).lower()
    assert "causal" not in claim, claim
