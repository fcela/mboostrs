"""How far the case studies are from the reconstruction standard, held steady.

D3 asks every public case to state its practical target, observed response,
estimand, units, sampling unit, dependence structure, split/resampling unit,
tuning procedure, uncertainty procedure, and claim type. Audited across the 29
studies, the gaps are wide and uneven:

    28/29  sampling unit   — what one row is
    26/29  estimand        — the quantity being estimated
    16/29  claim type      — descriptive, predictive or causal
    10/29  dependence structure

109 open gaps in total. Closing one means reading the analysis and working out
what it actually estimated, which is a person's job per study; generating
those sentences would produce 109 plausible statements about analyses nobody
checked, which is worse than the absence.

So the check ratchets instead of gating: it fails when a study loses an
element it had, or a new study arrives below the standard. The backlog is
recorded in the open, and it can only shrink.
"""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_case_standard import (  # noqa: E402
    BASELINE,
    ELEMENTS,
    audit,
    case_pages,
)


def test_every_case_study_is_audited():
    pages = case_pages()
    assert len(pages) == 81
    assert all(p.startswith(("cases/", "case_studies/")) for p in pages)
    assert not any("overview" in p for p in pages)


def test_the_elements_are_the_ones_d3_lists():
    assert set(ELEMENTS) == {
        "target", "response", "estimand", "units", "sampling unit",
        "dependence", "split unit", "tuning", "uncertainty", "claim type",
    }


def test_the_baseline_is_committed_and_current():
    assert BASELINE.exists()
    recorded = json.loads(BASELINE.read_text())
    assert recorded["studies"] == len(case_pages())
    assert set(recorded["missing"]) == set(audit())


def test_the_baseline_says_it_is_a_ratchet_not_a_target():
    """A committed list of gaps reads as an accepted state unless it says
    otherwise."""
    recorded = json.loads(BASELINE.read_text())
    assert "ratchet, not a target" in recorded["_comment"]
    assert "can only shrink" in recorded["_comment"] or "shrink" in recorded["_comment"]


def test_the_backlog_is_closed():
    """It opened at 109 gaps across 29 studies -- 28 not saying what one row
    was, 26 naming no estimand. Each study now carries a card stating the ten
    elements, read from that study's own opening, fit call and dataset. New
    studies must meet the same standard when they enter the table of contents."""
    recorded = json.loads(BASELINE.read_text())
    assert recorded["open_gaps"] == 0
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_case_standard.py")],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0


def test_every_study_states_its_design():
    import json as _json
    import re

    for page in case_pages():
        text = "\n".join("".join(c["source"]) for c in
                          _json.loads((ROOT / "docs" / "book" / page).read_text())["cells"])
        # The writing guide also permits a prose study-design paragraph.
        # The D3 audit above separately requires all ten design elements.
        assert ("What this study estimates" in text or "Study design" in text
                or re.search(r"\bestimands?\b", text, re.I)), page


def test_the_cards_are_not_boilerplate():
    """Ten identical cards would pass the checker and say nothing. Each field
    is read from the study it describes, so the estimands differ."""
    import json as _json
    import re

    estimands = set()
    for page in case_pages():
        text = "\n".join("".join(c["source"]) for c in
                          _json.loads((ROOT / "docs" / "book" / page).read_text())["cells"])
        row = [line for line in text.splitlines() if "**Estimand**" in line]
        if not row:
            row = [line for line in text.splitlines() if re.search(r"\bestimands?\b", line, re.I)]
        assert row, page
        estimands.add(row[0])
    assert len(estimands) == len(case_pages()), "some studies share an estimand line"


def test_methodological_is_an_accepted_claim_type():
    """Five studies claim something about an *estimator*, on data whose truth
    the simulation sets. Calling that predictive or causal would be the
    confusion the standard exists to prevent."""
    import re

    assert re.search(ELEMENTS["claim type"], "this is methodological")
    for page in ("cases/after_selection.ipynb", "cases/learner_flexibility.ipynb",
                 "cases/stopping_rule.ipynb", "cases/buser.ipynb",
                 "cases/beyond_the_data.ipynb"):
        assert page in case_pages()


def test_the_ratchet_fails_when_a_study_loses_an_element(tmp_path, monkeypatch):
    """A check that cannot fail is not a check. With the backlog closed, a
    study dropping an element is the regression this guards."""
    import check_case_standard

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"missing": {p: [] for p in case_pages()}}))
    monkeypatch.setattr(check_case_standard, "BASELINE", path)

    stripped = {p: ["estimand"] for p in case_pages()}
    monkeypatch.setattr(check_case_standard, "audit", lambda: stripped)
    monkeypatch.setattr(sys, "argv", ["check_case_standard.py"])
    assert check_case_standard.main() == 1


def test_a_new_study_below_the_standard_fails(tmp_path, monkeypatch):
    import check_case_standard

    pages = case_pages()
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"missing": {p: [] for p in pages[1:]}}))
    monkeypatch.setattr(check_case_standard, "BASELINE", path)
    # The omitted study is treated as new, and arrives with a gap.
    monkeypatch.setattr(check_case_standard, "audit",
                        lambda: {pages[0]: ["estimand"],
                                 **{p: [] for p in pages[1:]}})
    monkeypatch.setattr(sys, "argv", ["check_case_standard.py"])
    assert check_case_standard.main() == 1


def test_closing_a_gap_does_not_fail(tmp_path, monkeypatch):
    """Improvement must not look like a regression -- which is how the
    backlog got from 109 to 0."""
    import check_case_standard

    current = audit()
    loosened = {page: sorted(set(missing) | {"target"})
                for page, missing in current.items()}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"missing": loosened}))
    monkeypatch.setattr(check_case_standard, "BASELINE", path)
    monkeypatch.setattr(sys, "argv", ["check_case_standard.py"])
    assert check_case_standard.main() == 0


@pytest.mark.parametrize("flag", ["--report"])
def test_the_report_lists_the_worst_studies(flag):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_case_standard.py"), flag],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0
    assert "case studies audited" in result.stdout
    assert "per study:" in result.stdout


def test_ci_runs_the_ratchet():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "check_case_standard.py" in workflow
