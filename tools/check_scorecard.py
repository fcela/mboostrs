"""Check the single scorecard definition, registry, template and document links."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/scorecard"
CARD = DIRECTORY / "scorecard.md"
IDENTIFIER = "mboostrs-scorecard/2.1"
ORDER = ["D1a", "D1b"] + [f"D{i}" for i in range(2, 12)]


def check() -> list[str]:
    errors = []
    text = CARD.read_text()
    dimensions = [
        {"id": dim, "name": name.strip()}
        for dim, name in re.findall(r"^## (D\d+[a-z]?)\. (.*?) \[[^\]]+\]$", text, re.M)
    ]
    if [d["id"] for d in dimensions] != ORDER:
        errors.append("canonical dimensions must be D1a, D1b, D2 through D11 exactly once, in order")
    registry = json.loads((DIRECTORY / "dimensions.json").read_text())
    if registry != {"scorecard_id": IDENTIFIER, "canonical_document": "scorecard.md",
                    "dimensions": dimensions}:
        errors.append("dimension registry differs from the authoritative scorecard")
    template = json.loads((DIRECTORY / "report-template.json").read_text())
    if template.get("scorecard_id") != IDENTIFIER or IDENTIFIER not in text:
        errors.append("scorecard identifier is inconsistent")
    if template.get("scorecard_sha256") != hashlib.sha256(CARD.read_bytes()).hexdigest():
        errors.append("report template's scorecard hash is stale")
    if [{"id": d["id"], "name": d["name"]} for d in template["dimensions"]] != dimensions:
        errors.append("report template dimension names differ from the scorecard")
    if any(d["status"] != "unmeasured" or d["score"] is not None
           or d["evidence"] for d in template["dimensions"]) or template["aggregate"] is not None:
        errors.append("blank template must not invent scores, evidence or an aggregate")
    statuses = (DIRECTORY / "status.md").read_text()
    if re.findall(r"^\| \*\*(D\d+[a-z]?) ", statuses, re.M) != [d["id"] for d in dimensions]:
        errors.append("status page must cover every canonical dimension exactly once")
    weights = re.findall(r"^\| (D\d+[a-z]?) [^|]+\| [\d.]+× \| [\d.]+× \|$", text, re.M)
    if weights != [d["id"] for d in dimensions]:
        errors.append("mode weights must cover every canonical dimension exactly once")
    paper = (DIRECTORY / "paper.md").read_text()
    if re.search(r"^### D\d+\.", paper, re.M):
        errors.append("working paper must delegate normative dimensions to scorecard.md")
    paths = list(DIRECTORY.glob("*.md")) + [
        ROOT / "docs/benchmarks/agentic_hostile_workflows_report.md",
        ROOT / "docs/benchmarks/agentic_hostile_workflows/agentic_hostile_workflows_rubric.md",
    ]
    for path in paths:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            link = link.split("#", 1)[0]
            if not link or re.match(r"[a-zA-Z]+:", link):
                continue
            if not (path.parent / link).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link {link}")
    archived = paths[-1].read_text()
    if "Archived local criteria" not in archived or "migration.md" not in archived:
        errors.append("hostile-workflow local rubric must be explicitly archived")
    return errors


if __name__ == "__main__":
    problems = check()
    if problems:
        raise SystemExit("\n".join(problems))
    print("Scorecard 2.1: twelve consistent dimensions, template, weights and links")
