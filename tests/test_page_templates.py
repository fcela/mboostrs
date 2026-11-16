"""The family and learner pages cover what a reader has to know.

D2 sets a template for each page kind. Checking it found real absences:

* `usage/families.ipynb` never mentioned the **link** or the **inverse link**
  — the two things that decide whether `predict()` hands back a probability or
  a log-odds — nor the negative gradient, nor where a family refuses.
* Neither learner page mentioned **extrapolation**, for a policy the package
  ships, warns about, and offers four settings for.
* `usage/base_learners.ipynb` had no case where a learner is the **wrong**
  choice, which is the half of a catalogue that stops it being a menu.

The check tests presence, not quality: a page mentioning extrapolation once
passes. That is the honest limit of a string match, and it was still worth
having, because every failure above was an absence rather than a poor
explanation.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_page_templates import (  # noqa: E402
    FAMILY_ELEMENTS,
    LEARNER_ELEMENTS,
    PAGES,
    missing,
    page_text,
)


@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_page_covers_its_template(page):
    assert missing(page, PAGES[page]) == []


def test_the_template_matches_what_d2_asks_for():
    """The element list is the item's, not a convenient subset."""
    assert set(FAMILY_ELEMENTS) == {
        "loss/NLL", "link", "inverse link", "offset", "negative gradient",
        "response domain", "prediction interpretation", "nuisance",
        "residuals", "failure modes",
    }
    assert set(LEARNER_ELEMENTS) == {
        "design", "penalty", "null space", "centering", "extrapolation",
        "by= behaviour", "cost", "wrong choice",
    }


def test_the_families_page_distinguishes_the_two_prediction_scales():
    """The specific gap: without the link and its inverse, a reader cannot
    tell what `predict()` returned."""
    text = page_text("usage/families.ipynb")
    assert "log-odds" in text
    assert 'type="response"' in text
    assert 'type="link"' in text


def test_the_learner_pages_document_the_extrapolation_policy():
    for page in ("usage/base_learners.ipynb", "methodology/base_learners.ipynb"):
        text = page_text(page)
        assert "extrapolat" in text.lower(), page
    usage = page_text("usage/base_learners.ipynb")
    for setting in ("warn", "error", "clip", "allow"):
        assert f'extrapolation="{setting}"' in usage or setting in usage, setting
    assert "range_report" in usage


def test_the_learner_page_says_when_each_is_wrong():
    text = page_text("usage/base_learners.ipynb")
    assert "wrong when" in text.lower()
    # A table naming several, not one throwaway line.
    assert text.lower().count("`bols`") > 2


def test_the_check_fails_on_a_missing_element(monkeypatch):
    """A check that cannot fail is not a check."""
    import check_page_templates

    monkeypatch.setattr(check_page_templates, "PAGES",
                        {"usage/families.ipynb": {"unicorn": r"\bunicorn\b"}})
    monkeypatch.setattr(sys, "argv", ["check_page_templates.py"])
    assert check_page_templates.main() == 1


def test_the_committed_pages_pass():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_page_templates.py")],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_ci_runs_the_template_check():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "check_page_templates.py" in workflow
