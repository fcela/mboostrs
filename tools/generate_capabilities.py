#!/usr/bin/env python3
# ruff: noqa: E501 -- validation node IDs and generated prose are intentionally literal
"""Validate the capability contract and regenerate all derived surfaces."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "python/mboostrs/capability_manifest.json"
R_OUTPUT = ROOT / "mboostrs/R/aaa-capabilities-generated.R"
BOOK_OUTPUT = ROOT / "docs/book/appendix/capabilities.ipynb"
README = ROOT / "README.md"
README_START = "<!-- capability-summary:start -->"
README_END = "<!-- capability-summary:end -->"
VALID_STATUSES = {
    "verified",
    # Some inputs work and some are refused, by a limitation the
    # implementation knows about and reports. Distinct from "unsupported",
    # which is the whole operation, and from "unverified", which is a gap in
    # the evidence rather than in the code. The row's note must say which
    # inputs are refused and what the caller sees.
    "partial",
    "smoke",
    "intentional-deviation",
    "unsupported",
    "unverified",
}
FAMILY_COLUMNS = (
    "name",
    "kind",
    "r_package",
    "r_name",
    "parameters",
    "links",
    "fit",
    "predict_link",
    "predict_response",
    "predict_class",
    "coef",
    "residuals",
    "aic",
    "loglik",
    "cdf",
    "pdf_or_pmf",
    "quantile",
    "simulate",
    "deviance_increment",
    "diagnostics",
    "known_deviation",
    "notes",
)
LEARNER_COLUMNS = (
    "name",
    "r_name",
    "glmboost",
    "gamboost",
    "glmboostLSS",
    "gamboostLSS",
    "newdata",
    "cvrisk",
    "stabsel",
    "extract_design",
    "penalty",
    "notes",
)
METHOD_COLUMNS = (
    "model_class",
    "method",
    "implemented",
    "r_equivalent",
    "partial",
    "notes",
)
OPERATIONS = ("fit", "predict", "cv", "stability_selection", "diagnostics")


class ContractError(RuntimeError):
    """Raised when the checked-in capability contract is inconsistent."""


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


# The manifest is hand-maintained and machine-validated, not generated.
#
# Three one-shot in-place rewrites used to live here — `migrate_bootstrap`,
# `enrich_family_evidence`, and `tighten_operation_evidence`, about 180 of this
# file's 715 lines. All three were gated on `evidence_version` or on a
# `tables["python"]` key the checked-in manifest no longer has, so all three
# returned immediately and could never run again. They were also where the
# stale `reference` strings that finding 20.4 caught came from, which made the
# mental model "the manifest is generated, so fix the generator" wrong for
# exactly the field that was incorrect: re-running the generator could not
# correct them. Deleting them leaves `validate_schema`, `validate_exports`,
# `validate_nodes`, `validate_reference_versions`, and the emitters, which is
# what this file actually does.


def validation_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("validations", [])
    result: dict[str, dict[str, Any]] = {}
    required = {"id", "node_id", "oracle", "compared_outputs", "atol", "rtol", "reference"}
    for record in records:
        missing = required - set(record)
        if missing:
            raise ContractError(f"validation record missing {sorted(missing)}: {record!r}")
        record_id = record["id"]
        if record_id in result:
            raise ContractError(f"duplicate validation id: {record_id}")
        if not isinstance(record["compared_outputs"], list) or not record["compared_outputs"]:
            raise ContractError(f"validation {record_id} has no compared outputs")
        result[record_id] = record
    return result


def iter_support(manifest: dict[str, Any]):
    for table_name in ("families", "learners"):
        for row in manifest["tables"][table_name]:
            for frontend, support in row.get("support", {}).items():
                yield table_name, str(row["name"]), frontend, support
    for frontend, rows in manifest["tables"]["methods"].items():
        for row in rows:
            yield "methods", f"{row['model_class']}.{row['method']}", frontend, row.get("support", {}).get(frontend, {})


def validate_schema(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ContractError("capability manifest schema_version must be 1")
    if set(manifest.get("status_values", [])) != VALID_STATUSES:
        raise ContractError("capability manifest status_values do not match the supported vocabulary")
    if set(manifest.get("tables", {})) != {"families", "learners", "methods"}:
        raise ContractError("capability manifest must contain family, learner, and method tables")
    validations = validation_map(manifest)
    seen_entities: set[tuple[str, str]] = set()
    for table_name, name, frontend, support in iter_support(manifest):
        key = (f"{table_name}:{frontend}", name)
        if key in seen_entities:
            raise ContractError(f"duplicate capability entity: {key}")
        seen_entities.add(key)
        if table_name == "methods":
            statuses = [support.get("status")]
        else:
            operations = support.get("operations", {})
            if set(operations) != set(OPERATIONS):
                raise ContractError(f"{table_name} {name}/{frontend} lacks the five operation statuses")
            statuses = list(operations.values())
        bad = set(statuses) - VALID_STATUSES
        if bad:
            raise ContractError(f"{table_name} {name}/{frontend} uses invalid statuses: {sorted(bad)}")
        validation_ids = support.get("validation_ids", [])
        unknown = set(validation_ids) - set(validations)
        if unknown:
            raise ContractError(f"{table_name} {name}/{frontend} references unknown validations: {sorted(unknown)}")
        advertised = any(status in {"verified", "smoke", "intentional-deviation"} for status in statuses)
        if advertised and not validation_ids:
            raise ContractError(f"advertised capability has no validation: {table_name} {name}/{frontend}")
        if "intentional-deviation" in statuses:
            deviation_records = [validations[item] for item in validation_ids if item.startswith("deviation.")]
            if not deviation_records or any(not item.get("rationale") for item in deviation_records):
                raise ContractError(f"intentional deviation lacks a tested rationale: {table_name} {name}/{frontend}")
        if "verified" in statuses and not any(item.startswith(f"parity.{frontend}.") for item in validation_ids):
            raise ContractError(
                f"verified capability lacks frontend-specific parity evidence: {table_name} {name}/{frontend}"
            )
    validate_upstream_names(manifest)
    validate_node_capabilities(manifest)


def validate_upstream_names(manifest: dict[str, Any]) -> None:
    """One row per upstream name, so the ledger cannot contradict itself.

    The duplicate-entity check above is keyed on the *mboostrs* name, so two
    rows could claim the same `r_name` and disagree about it. They did: `Brad`
    recorded a verified Matern(3/2) radial basis against upstream `brad`,
    while a second row named `brad` recorded "R base-learner not implemented"
    for the same learner. The second was stale -- written before `Brad`
    existed and never removed -- and it reached users, because the README's
    generated summary counts these rows.

    A row for something upstream has and mboostrs does not is the point of
    the `unsupported` status and stays (`bss`, `blackboost`); what cannot
    stand is two rows for one upstream name.
    """
    for table_name in ("families", "learners"):
        seen: dict[str, str] = {}
        for row in manifest["tables"].get(table_name, []):
            upstream = row.get("r_name")
            if not upstream:
                continue
            if upstream in seen:
                raise ContractError(
                    f"{table_name}: {seen[upstream]!r} and {row['name']!r} "
                    f"both claim upstream {upstream!r}. One upstream name has "
                    f"one row: delete the stale one, or give the learner that "
                    f"is genuinely different its own r_name."
                )
            seen[upstream] = row["name"]


def collect_python_node_ids(manifest: dict[str, Any]) -> set[str]:
    files = sorted(
        {
            record["node_id"].split("::", 1)[0]
            for record in manifest["validations"]
            if record["node_id"].startswith("tests/")
        }
    )
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", *files]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode:
        raise ContractError(f"pytest collection failed:\n{result.stdout}{result.stderr}")
    return {line.strip() for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line}



R_VALIDATES_RE = re.compile(r"^#@validates\s+(\S+)\s*$", re.MULTILINE)


def collect_r_validation_markers() -> dict[str, set[str]]:
    """Validation ids claimed by each R test script, from its `#@validates`
    comments.

    Python validation ids name a test *function*, so a stale rename fails the
    contract. R ids used to name only a file, checked with `Path.is_file()`, so
    they proved nothing about what the file asserts — ten separate capability
    claims rested on `test_gamboostlss_mixed.R` existing, and deleting any
    assertion inside it broke no contract.

    A `#@validates <id>` comment placed at the head of the block that
    establishes a claim gives R ids the same resolution Python's have, without
    rewriting 32 plain scripts into `testthat`. The marker is a comment, so it
    costs the scripts nothing at run time.
    """
    markers: dict[str, set[str]] = {}
    for path in sorted((ROOT / "mboostrs/tests").glob("*.R")):
        key = str(path.relative_to(ROOT))
        markers[key] = set(R_VALIDATES_RE.findall(path.read_text(encoding="utf-8")))
    return markers


def validate_nodes(manifest: dict[str, Any]) -> None:
    collected = collect_python_node_ids(manifest)
    r_markers = collect_r_validation_markers()
    declared_ids = {record["id"] for record in manifest["validations"]}
    for path, ids in r_markers.items():
        for orphan in sorted(ids - declared_ids):
            raise ContractError(
                f"{path} carries `#@validates {orphan}`, which is not a validation id"
            )
    for record in manifest["validations"]:
        node_id = record["node_id"]
        if node_id.startswith("tests/"):
            if node_id not in collected:
                raise ContractError(f"stale Python validation node id: {node_id}")
        elif node_id.startswith("crates/"):
            # Rust evidence, as `<file>::<test module>`. The core crate is
            # published on its own, so a contract it enforces is only checked
            # where the contract lives: a Python test cannot show that
            # `QuantReg::try_new` rejects a tau, because the Python
            # constructor rejected it first.
            source, _, module = node_id.partition("::")
            path = ROOT / source
            if not module:
                raise ContractError(
                    f"Rust validation node id must name a test module as "
                    f"`<file>::<module>`: {node_id}"
                )
            if not path.is_file():
                raise ContractError(f"stale Rust validation source: {source}")
            if f"mod {module} {{" not in path.read_text(encoding="utf-8"):
                raise ContractError(
                    f"Rust validation {record['id']!r} names {node_id}, and "
                    f"{source} has no `mod {module}`"
                )
        elif node_id.startswith("mboostrs/tests/"):
            if not (ROOT / node_id).is_file():
                raise ContractError(f"stale R validation script id: {node_id}")
            claimed = r_markers.get(node_id, set())
            if record["id"] not in claimed:
                raise ContractError(
                    f"R validation {record['id']!r} names {node_id} but that file has "
                    f"no `#@validates {record['id']}` marker; add one to the block "
                    f"that establishes the claim"
                )
        else:
            raise ContractError(f"validation node id uses an unsupported test namespace: {node_id}")


def parse_python_all() -> set[str]:
    tree = ast.parse((ROOT / "python/mboostrs/__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise ContractError("python/mboostrs/__init__.py has no literal __all__")


def module_classes(path: Path, *, suffix: str | None = None) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef) and not node.name.startswith("_")}
    return {name for name in names if suffix is None or name.endswith(suffix)}


def parse_r_exports() -> set[str]:
    exports: set[str] = set()
    for raw in (ROOT / "mboostrs/NAMESPACE").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export(") and line.endswith(")"):
            exports.add(line[7:-1].strip('"'))
    return exports



def parse_r_s3_methods() -> set[tuple[str, str]]:
    """`(class, generic)` pairs registered as S3 methods in NAMESPACE.

    NAMESPACE has 138 `S3method(...)` lines against 104 `export(...)` lines, and
    `parse_r_exports` reads only the `export(` form, so the entire S3 surface —
    which is what the R half of the method table describes — was invisible to
    this tooling.

    Both `S3method(generic, class)` and the quoted `S3method("mstop<-", class)`
    form are handled; the replacement generics need the quotes because `<-` is
    a parse error in NAMESPACE otherwise.
    """
    pairs: set[tuple[str, str]] = set()
    for raw in (ROOT / "mboostrs/NAMESPACE").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not (line.startswith("S3method(") and line.endswith(")")):
            continue
        parts = [part.strip().strip('"') for part in line[9:-1].split(",")]
        if len(parts) >= 2:
            pairs.add((parts[1], parts[0]))
    return pairs


def python_class_members() -> dict[str, set[str]]:
    """Public attribute names defined on each documented Python model class.

    Parsed with `ast` rather than by importing, matching how `module_classes`
    and `parse_python_all` already work. Covers methods, properties, and
    class-level assignments, plus anything a class picks up from a mixin
    defined in the same file or in `_model_api.py`.
    """
    sources = [
        ROOT / "python/mboostrs/glmboost.py",
        ROOT / "python/mboostrs/gamboost.py",
        ROOT / "python/mboostrs/gamlss.py",
        ROOT / "python/mboostrs/cvrisk.py",
        ROOT / "python/mboostrs/stabsel.py",
        ROOT / "python/mboostrs/_model_api.py",
    ]
    bases: dict[str, list[str]] = {}
    own: dict[str, set[str]] = {}
    for path in sources:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            members = own.setdefault(node.name, set())
            bases.setdefault(node.name, []).extend(
                base.id for base in node.bases if isinstance(base, ast.Name)
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.Assign):
                    members.update(
                        target.id for target in item.targets if isinstance(target, ast.Name)
                    )
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)

    def resolved(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
        if name in seen:
            return set()
        members = set(own.get(name, ()))
        for base in bases.get(name, ()):
            members |= resolved(base, seen | {name})
        return members

    return {name: resolved(name) for name in own}


def validate_methods(manifest: dict[str, Any]) -> None:
    """Require every `(model_class, method)` row to name something real.

    The method table is 152 of the 213 contract rows — 71% — and nothing
    checked it. A fabricated method name passed `--check`, was published to the
    generated R file and twice into the book appendix, and one regeneration
    converted the fabrication into the new baseline. Families and learners
    already get this treatment through `validate_exports`.

    `model_class` values that are not Python classes (the `diagnostics` module
    rows) are checked against that module's public functions instead.
    """
    members = python_class_members()
    diagnostics_tree = ast.parse(
        (ROOT / "python/mboostrs/diagnostics.py").read_text(encoding="utf-8")
    )
    members["diagnostics"] = {
        node.name
        for node in diagnostics_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }

    unknown: list[str] = []
    for row in manifest["tables"]["methods"]["python"]:
        model_class, method = row["model_class"], row["method"]
        if model_class not in members:
            unknown.append(f"python row names unknown model_class {model_class!r}")
        elif method not in members[model_class]:
            unknown.append(f"python {model_class}.{method} is not defined in the source")

    s3 = parse_r_s3_methods()
    r_classes = {cls for cls, _ in s3}
    for row in manifest["tables"]["methods"]["r"]:
        model_class, method = row["model_class"], row["method"]
        if model_class not in r_classes:
            unknown.append(f"r row names unregistered class {model_class!r}")
        elif (model_class, method) not in s3:
            unknown.append(
                f"r {method}.{model_class} is not registered in NAMESPACE"
            )
    if unknown:
        raise ContractError(
            "capability method table names members that do not exist: "
            + "; ".join(sorted(unknown))
        )


def capture_node_capabilities(manifest: dict[str, Any]) -> bool:
    """Refresh the derived node list. Return whether the manifest changed.

    Family, learner, or method rows can be added after the list first exists.
    Treating presence as freshness left those new rows out of the node view,
    even though that view is documented as auto-derived.
    """
    current = manifest.get("node_capabilities")
    rebuilt = build_node_capabilities({**manifest, "node_capabilities": []})
    if current == rebuilt:
        return False
    manifest["node_capabilities"] = rebuilt
    return True


def capture_public_exports(manifest: dict[str, Any]) -> bool:
    """Record the complete public frontend surfaces once, then check exactly."""
    if "public_exports" in manifest:
        return False
    manifest["public_exports"] = {
        "python": sorted(parse_python_all()),
        "r": sorted(parse_r_exports()),
    }
    return True


def validate_exports(manifest: dict[str, Any]) -> None:
    family_rows = manifest["tables"]["families"]
    family_names = {row["name"] for row in family_rows if row["kind"] != "standard_missing"}
    learner_names = {row["name"] for row in manifest["tables"]["learners"] if row["name"][0].isupper()}
    source_families = module_classes(ROOT / "python/mboostrs/families.py")
    source_lss = module_classes(ROOT / "python/mboostrs/gamlss.py", suffix="LSS")
    source_learners = module_classes(ROOT / "python/mboostrs/learners.py")
    if family_names != source_families | source_lss:
        raise ContractError(
            f"Python family exports differ from manifest: manifest-only={sorted(family_names - source_families - source_lss)}, source-only={sorted((source_families | source_lss) - family_names)}"
        )
    if learner_names != source_learners:
        raise ContractError(
            f"Python learner exports differ from manifest: manifest-only={sorted(learner_names - source_learners)}, source-only={sorted(source_learners - learner_names)}"
        )
    python_exports = parse_python_all()
    recorded_python = set(manifest.get("public_exports", {}).get("python", []))
    if python_exports != recorded_python:
        raise ContractError(
            "Python public exports differ from manifest: "
            f"manifest-only={sorted(recorded_python - python_exports)}, "
            f"source-only={sorted(python_exports - recorded_python)}"
        )
    missing_python = (family_names | learner_names) - python_exports
    if missing_python:
        raise ContractError(f"manifest capability constructors missing from Python __all__: {sorted(missing_python)}")
    r_exports = parse_r_exports()
    recorded_r = set(manifest.get("public_exports", {}).get("r", []))
    if r_exports != recorded_r:
        raise ContractError(
            "R public exports differ from manifest: "
            f"manifest-only={sorted(recorded_r - r_exports)}, "
            f"source-only={sorted(r_exports - recorded_r)}"
        )
    # A family with no R frontend (r_name empty, R operations all
    # "unsupported") is a Python-only contribution and cannot be missing
    # from the R NAMESPACE; only families claiming an R name must be there.
    r_families = {row["name"] for row in manifest["tables"]["families"]
                  if row["kind"] != "standard_missing" and row.get("r_name")}
    missing_r = r_families - r_exports
    if missing_r:
        raise ContractError(f"manifest family constructors missing from R NAMESPACE: {sorted(missing_r)}")


# A `reference` is an attestation ("checked against this exact oracle version"),
# not free text, when it is nothing but "PackageName version" -- optionally two
# of those joined by " / ". Prose that merely mentions a package/version in
# passing (e.g. "upstream gamboostLSS 2.1-0 removes bns") does not match this
# pattern and is intentionally left unchecked; it is a historical note, not a
# claim about what the suite ran against.
_REFERENCE_ATTESTATION_RE = re.compile(r"^([A-Za-z][\w.]*)\s+([0-9][\w.\-]*)$")


def _installed_r_package_versions() -> dict[str, str]:
    """Query installed R package versions, mirroring parity_ledger.py's
    _r_versions() so this check and the ledger agree on what "installed" means."""
    package_names = ("mboost", "gamboostLSS", "gamlss", "stabs", "survival", "partykit", "TH.data")
    expression = (
        f"for (p in c({', '.join(json.dumps(name) for name in package_names)})) "
        'cat(paste0(p, "=", '
        'if (requireNamespace(p, quietly=TRUE)) as.character(packageVersion(p)) else "missing", "\\n"))'
    )
    try:
        result = subprocess.run(
            ["Rscript", "-e", expression], capture_output=True, text=True, timeout=60, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    versions: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator and value != "missing":
            versions[name] = value
    return versions


def _normalize_r_version(version: str) -> str:
    return version.replace("-", ".")


def validate_reference_versions(manifest: dict[str, Any]) -> None:
    """A `reference` names the oracle version a validation was checked against.
    Nothing previously compared that claim to what is actually installed, so it
    could -- and did -- silently outlive its oracle (28 records named
    gamboostLSS 2.1-0 while CI's `any::gamboostLSS` had long since moved to
    2.2.0). Skips silently if R/Rscript is unavailable in this environment."""
    installed = _installed_r_package_versions()
    if not installed:
        return
    by_claim: dict[tuple[str, str], list[str]] = {}
    for record in manifest.get("validations", []):
        for piece in record["reference"].split(" / "):
            match = _REFERENCE_ATTESTATION_RE.match(piece.strip())
            if match:
                by_claim.setdefault((match.group(1), match.group(2)), []).append(record["id"])
    mismatches = []
    for (package, claimed_version), ids in sorted(by_claim.items()):
        actual = installed.get(package)
        if actual is None or _normalize_r_version(claimed_version) == _normalize_r_version(actual):
            continue
        mismatches.append(
            f"{package} {claimed_version} claimed, {actual} installed "
            f"({len(ids)} validation(s), e.g. {ids[0]})"
        )
    if mismatches:
        raise ContractError(
            "capability manifest references a package version that is not installed:\n  "
            + "\n  ".join(mismatches)
        )


def _parse_validation_role(validation_id: str) -> tuple[str, str]:
    """Infer (entity_type, entity_name) from a validation id prefix.

    Convention: ``parity.python.<family>`` → ``("family", "<family>")``,
    ``smoke.python.learner.<learner>`` → ``("learner", "<learner>")``, etc.
    Contract-level ids (``py.family.contract``, ``r.method.contract``) map to
    the generic "Family" / "Learner" / "Method" row.
    """
    parts = validation_id.split(".")
    if len(parts) >= 3:
        if parts[0] in ("parity", "smoke", "deviation"):
            return parts[1], parts[2]
        if parts[0] in ("py", "r"):
            role = parts[1]
            if role == "predict":
                role = "learner"
            return role, parts[2]
    return "", ""


def build_node_capabilities(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Auto-derive a flat node list from the existing tables.

    Each node is one ``(frontend, family|learner|method, operation)`` cross
    with the operation status, the validation ids that back it, and the
    first-class ``model_class`` / ``family`` / ``learner`` / ``operation``
    / ``weight_kind`` / ``oracle_version`` / ``last_success_artifact`` fields
    that the old nested support structure隐含 but never named.

    ``weight_kind`` is ``"unknown"`` here; the per-record precision requires a
    person-scale pass over each validation's test fixture (P1.1 scope).
    ``oracle_version`` is ``""`` here; the reference_packages table is the
    source but the per-validation attribution requires a person-scale pass.
    """
    nodes: list[dict[str, Any]] = []
    node_keys: set[tuple[str, str, str, str]] = set()

    for frontend in ("python", "r"):
        for row in manifest["tables"].get("families", []):
            name = row["name"]
            support = row.get("support", {}).get(frontend, {})
            ops = support.get("operations", {})
            v_ids = support.get("validation_ids", [])
            for op in OPERATIONS:
                status = ops.get(op, "unsupported")
                key = (frontend, "family", name, op)
                if key in node_keys:
                    continue
                node_keys.add(key)
                nodes.append({
                    "frontend": frontend,
                    "model_class": "",
                    "family": name,
                    "learner": "",
                    "operation": op,
                    "status": status,
                    "validation_ids": list(v_ids),
                    "weight_kind": "unknown",
                    "oracle_version": "",
                    "last_success_artifact": "",
                })

        for row in manifest["tables"].get("learners", []):
            name = row["name"]
            support = row.get("support", {}).get(frontend, {})
            ops = support.get("operations", {})
            v_ids = support.get("validation_ids", [])
            for op in OPERATIONS:
                status = ops.get(op, "unsupported")
                key = (frontend, "learner", name, op)
                if key in node_keys:
                    continue
                node_keys.add(key)
                nodes.append({
                    "frontend": frontend,
                    "model_class": "",
                    "family": "",
                    "learner": name,
                    "operation": op,
                    "status": status,
                    "validation_ids": list(v_ids),
                    "weight_kind": "unknown",
                    "oracle_version": "",
                    "last_success_artifact": "",
                })

    for frontend in ("python", "r"):
        method_rows = manifest["tables"].get("methods", {}).get(frontend, [])
        for row in method_rows:
            model_class = row["model_class"]
            method = row["method"]
            support = row.get("support", {}).get(frontend, {})
            status = support.get("status", "unsupported")
            v_ids = support.get("validation_ids", [])
            key = (frontend, "method", f"{model_class}.{method}", "")
            if key in node_keys:
                continue
            node_keys.add(key)
            nodes.append({
                "frontend": frontend,
                "model_class": model_class,
                "family": "",
                "learner": "",
                "operation": method,
                "status": status,
                "validation_ids": list(v_ids),
                "weight_kind": "unknown",
                "oracle_version": "",
                "last_success_artifact": "",
            })

    return nodes


def validate_node_capabilities(manifest: dict[str, Any]) -> None:
    """Check the auto-derived node_capabilities list for consistency."""
    nodes = manifest.get("node_capabilities", [])
    if not nodes:
        return
    valid_ids = {v["id"] for v in manifest.get("validations", [])}
    seen_keys: set[tuple[str, str, str, str]] = set()
    for node in nodes:
        key = (node["frontend"], node["family"] or node["learner"] or node["operation"],
               node["operation"], node.get("model_class", ""))
        if key in seen_keys:
            raise ContractError(f"duplicate node_capabilities entry: {key}")
        seen_keys.add(key)
        bad = set(node["validation_ids"]) - valid_ids
        if bad:
            raise ContractError(
                f"node_capabilities references unknown validations: {sorted(bad)}"
            )
        if node.get("weight_kind") not in ("unknown", "unit", "fractional",
                                             "integer-count", "bootstrap", "subsampling"):
            raise ContractError(
                f"node_capabilities has invalid weight_kind: {node.get('weight_kind')}"
            )


def flatten_row(row: dict[str, Any], columns: tuple[str, ...], *, method_frontend: str | None = None) -> dict[str, str]:
    result = {column: str(row.get(column, "")) for column in columns}
    support = row.get("support", {})
    validation_ids: set[str] = set()
    if method_frontend is not None:
        details = support.get(method_frontend, {})
        result["status"] = str(details.get("status", "unsupported"))
        validation_ids.update(details.get("validation_ids", []))
    else:
        for frontend in ("python", "r"):
            details = support.get(frontend, {})
            operations = details.get("operations", {})
            for operation in OPERATIONS:
                result[f"{frontend}_{operation}_status"] = str(operations.get(operation, "unsupported"))
            validation_ids.update(details.get("validation_ids", []))
    result["validation_ids"] = "; ".join(sorted(validation_ids))
    return result


def r_string(value: str) -> str:
    """Render a Python string as a portable (ASCII-only) R string literal.

    R packages must contain only ASCII in their R code (``R CMD check``
    raises a WARNING otherwise). Any character outside the printable ASCII
    range is emitted as an R `\\uXXXX`/`\\UXXXXXXXX` escape instead of a raw
    UTF-8 byte sequence, so generated output is always portable regardless
    of what non-ASCII prose (arrows, "not identical to", em dashes, ...)
    ends up in the manifest.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    out = []
    for ch in escaped:
        codepoint = ord(ch)
        if 0x20 <= codepoint <= 0x7E:
            out.append(ch)
        elif codepoint <= 0xFFFF:
            out.append(f"\\u{codepoint:04x}")
        else:
            out.append(f"\\U{codepoint:08x}")
    return '"' + "".join(out) + '"'


def r_rows(name: str, rows: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for row in rows:
        fields = ", ".join(f"{key} = {r_string(value)}" for key, value in row.items())
        rendered.append(f"  list({fields})")
    return f"{name} <- list(\n" + ",\n".join(rendered) + "\n)\n"


def render_r(manifest: dict[str, Any]) -> str:
    families = [flatten_row(row, FAMILY_COLUMNS) for row in manifest["tables"]["families"]]
    learners = [flatten_row(row, LEARNER_COLUMNS) for row in manifest["tables"]["learners"]]
    methods = [flatten_row(row, METHOD_COLUMNS, method_frontend="r") for row in manifest["tables"]["methods"]["r"]]
    validations = []
    for record in manifest["validations"]:
        validations.append(
            {
                "id": str(record["id"]),
                "node_id": str(record["node_id"]),
                "oracle": str(record["oracle"]),
                "compared_outputs": "; ".join(record["compared_outputs"]),
                "atol": str(record["atol"]),
                "rtol": str(record["rtol"]),
                "reference": str(record["reference"]),
                "rationale": str(record.get("rationale", "")),
            }
        )
    digest_note = f"# Generated from python/mboostrs/capability_manifest.json (schema {manifest['schema_version']}).\n# Run python tools/generate_capabilities.py; do not edit this file.\n\n"
    return (
        digest_note
        + r_rows(".capability_family_rows", families)
        + "\n"
        + r_rows(".capability_learner_rows", learners)
        + "\n"
        + r_rows(".capability_method_rows", methods)
        + "\n"
        + r_rows(".capability_validation_rows", validations)
        + "\n"
        + r_rows(".parity_note_rows", parity_note_rows())
    )


def parity_note_rows() -> list[dict[str, str]]:
    """The parity notes, flattened for the R frontend.

    They live in `mboostrs._parity_notes`, which is the registry
    `model.parity_notes()` reads in Python. Emitting them here rather than
    hand-copying them into R keeps one registry: the alternative is the
    failure this project has paid for repeatedly -- a family listed in five
    places that do not check each other.
    """
    from mboostrs._parity_notes import FAMILY_NOTES, GLOBAL_NOTES, LEARNER_NOTES

    rows = []
    for kind, registry in (("family", FAMILY_NOTES), ("learner", LEARNER_NOTES)):
        for owner, notes in sorted(registry.items()):
            for note in notes:
                rows.append(
                    {
                        "kind": kind,
                        "owner": owner,
                        "name": note.name,
                        "mechanism": note.mechanism,
                        "consequence": note.consequence,
                        "shared_with_upstream": "yes" if note.shared_with_upstream else "no",
                        "evidence": note.evidence,
                    }
                )
    for note in GLOBAL_NOTES:
        rows.append(
            {
                "kind": "global",
                "owner": "",
                "name": note.name,
                "mechanism": note.mechanism,
                "consequence": note.consequence,
                "shared_with_upstream": "yes" if note.shared_with_upstream else "no",
                "evidence": note.evidence,
            }
        )
    return rows


def status_counts(rows: list[dict[str, Any]], frontend: str, operation: str) -> Counter[str]:
    return Counter(row["support"][frontend]["operations"][operation] for row in rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def method_matrix_rows(methods: dict[str, list[dict[str, Any]]]) -> list[list[str]]:
    """Merge the per-frontend method tables into one matrix keyed by
    (model_class, method), with each frontend's status side by side.

    Methods exposed by only one frontend show ``-`` for the other.
    """
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for frontend in ("python", "r"):
        for row in methods.get(frontend, []):
            key = (row["model_class"], row["method"])
            entry = merged.setdefault(
                key,
                {"implemented": "no", "partial": "no", "python": "-", "r": "-"},
            )
            entry[frontend] = row["support"][frontend]["status"]
            # A method counts as implemented/partial if any frontend says so.
            if row.get("implemented") == "yes":
                entry["implemented"] = "yes"
            if row.get("partial") == "yes":
                entry["partial"] = "yes"
    return [
        [model_class, method, entry["implemented"], entry["partial"], entry["python"], entry["r"]]
        for (model_class, method), entry in sorted(merged.items())
    ]


def render_book_markdown(manifest: dict[str, Any]) -> str:
    families = manifest["tables"]["families"]
    learners = manifest["tables"]["learners"]
    family_rows = []
    for row in families:
        py = row["support"]["python"]["operations"]
        rr = row["support"]["r"]["operations"]
        family_rows.append(
            [
                row["name"],
                row["kind"],
                py["fit"],
                py["predict"],
                py["diagnostics"],
                rr["fit"],
                rr["predict"],
                rr["diagnostics"],
            ]
        )
    learner_rows = []
    for row in learners:
        py = row["support"]["python"]["operations"]
        rr = row["support"]["r"]["operations"]
        learner_rows.append(
            [row["name"], py["fit"], py["predict"], py["cv"], py["stability_selection"], rr["fit"], rr["predict"]]
        )
    method_rows = method_matrix_rows(manifest["tables"]["methods"])
    validation_rows = [
        [
            record["id"],
            record["oracle"],
            ", ".join(record["compared_outputs"]),
            str(record["atol"]),
            str(record["rtol"]),
            record["node_id"],
        ]
        for record in manifest["validations"]
    ]
    return (
        "\n\n".join(
            [
                "# Capabilities and validation",
                "This page is generated from `python/mboostrs/capability_manifest.json`. Each operation lists its frontend support, validation status, and associated tests. Numerical comparisons identify the reference implementation and tolerance.",
                f"Manifest schema: **{manifest['schema_version']}**. Last reviewed: **{manifest['last_reviewed']}**. Families/helpers recorded: **{len(families)}**. Learners recorded: **{len(learners)}**. Validation records: **{len(manifest['validations'])}**.",
                "## Status vocabulary\n\n- `verified`: compared with the named oracle at the recorded tolerance.\n- `smoke`: executable behavior is covered, without a full external numerical oracle.\n- `intentional-deviation`: tested behavior differs deliberately; the validation record links its rationale.\n- `unsupported`: the frontend rejects or does not expose the operation.\n- `unverified`: code may exist, but the manifest does not make a correctness claim.",
                "## Family matrix\n\n"
                + markdown_table(
                    ["Family", "Kind", "Py fit", "Py predict", "Py diagnostics", "R fit", "R predict", "R diagnostics"],
                    family_rows,
                ),
                "## Learner matrix\n\n"
                + markdown_table(
                    ["Learner", "Py fit", "Py predict", "Py CV", "Py stabsel", "R fit", "R predict"], learner_rows
                ),
                "## Method matrix\n\n"
                "Model-class methods and helpers, merged across frontends. A `-` means the "
                "frontend does not expose that method.\n\n"
                + markdown_table(
                    ["Model class", "Method", "Implemented", "Partial", "Py status", "R status"], method_rows
                ),
                "## Validation ledger\n\n"
                + markdown_table(["ID", "Oracle", "Outputs", "atol", "rtol", "Test node"], validation_rows),
                "The parity workflow publishes a JSON record for each test. Read that run's result for the package versions, platform, commit, observed error, and tolerance. The manifest above lists which tests are intended to cover each operation.",
            ]
        )
        + "\n"
    )


def render_notebook(manifest: dict[str, Any]) -> str:
    markdown = render_book_markdown(manifest)
    notebook = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"


def render_readme_summary(manifest: dict[str, Any]) -> str:
    families = manifest["tables"]["families"]
    learners = manifest["tables"]["learners"]
    py_verified = status_counts(families, "python", "fit")["verified"]
    r_verified = status_counts(families, "r", "fit")["verified"]
    unsupported = status_counts(families, "python", "fit")["unsupported"]
    return (
        f"{README_START}\n"
        f"The generated capability contract records **{len(families)} family/helper rows**, "
        f"**{len(learners)} learner rows**, and **{len(manifest['validations'])} explicit validation records**. "
        f"For fitting, {py_verified} Python rows and {r_verified} R rows are externally verified; "
        f"{unsupported} Python rows are explicitly unsupported rather than inferred from missing code. "
        "See the [generated capability and validation matrix](docs/book/appendix/capabilities.ipynb).\n"
        f"{README_END}"
    )


def replace_readme_summary(source: str, replacement: str) -> str:
    if README_START not in source or README_END not in source:
        raise ContractError("README capability summary markers are missing")
    before, rest = source.split(README_START, 1)
    _, after = rest.split(README_END, 1)
    return before + replacement + after


def check_or_write(path: Path, expected: str, *, check: bool) -> None:
    actual = path.read_text(encoding="utf-8") if path.exists() else None
    if check:
        if actual != expected:
            raise ContractError(f"generated capability file is stale: {path.relative_to(ROOT)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files differ")
    parser.add_argument(
        "--skip-node-validation", action="store_true", help="skip pytest collection (for docs-only environments)"
    )
    args = parser.parse_args()
    manifest = load_manifest()
    captured_exports = capture_public_exports(manifest)
    captured_nodes = capture_node_capabilities(manifest)
    validate_schema(manifest)
    validate_exports(manifest)
    validate_methods(manifest)
    if not args.skip_node_validation:
        validate_nodes(manifest)
    if args.check:
        validate_reference_versions(manifest)
    if captured_exports or captured_nodes:
        if args.check:
            raise ContractError("capability manifest public exports or node_capabilities are stale; regenerate it")
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    check_or_write(R_OUTPUT, render_r(manifest), check=args.check)
    check_or_write(BOOK_OUTPUT, render_notebook(manifest), check=args.check)
    readme = README.read_text(encoding="utf-8")
    expected_readme = replace_readme_summary(readme, render_readme_summary(manifest))
    check_or_write(README, expected_readme, check=args.check)
    verb = "validated" if args.check else "generated"
    print(
        f"Capability contract {verb}: {len(manifest['tables']['families'])} families, {len(manifest['tables']['learners'])} learners, {len(manifest['validations'])} validations"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"capability contract error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
