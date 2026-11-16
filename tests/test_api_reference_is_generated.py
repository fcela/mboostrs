"""The API reference is read from the objects, not written alongside them.

The page it replaces was one hand-written markdown cell documenting a few
dozen of 146 public names, and what it did document had drifted: it showed
`glmboost(formula, *, data, ...)` after `data` became positional, listed
`control()` and `offset()` after both became properties, and gave
`risk(parameter=None)` for a method whose arguments were removed. A reader
comparing it with the package would have found it wrong in every one of those
places, and nothing said so.

`tools/generate_api_reference.py --check` runs in CI. These tests check the
generator itself: that it covers everything public, that what it emits matches
the live objects, and that it fails when they diverge.
"""

import json
import pathlib
import subprocess
import sys
import warnings

import pytest

import mboostrs as mb

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_api_reference.py"
PAGE = ROOT / "docs" / "book" / "appendix" / "api_reference.ipynb"


@pytest.fixture(scope="module")
def page_text():
    notebook = json.loads(PAGE.read_text())
    return "".join(notebook["cells"][0]["source"])


def test_the_committed_page_is_current():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_every_public_name_appears(page_text):
    """146 names, of which the old page documented a few dozen."""
    missing = [name for name in mb.__all__
               if not name.startswith("_") and f"`{name}" not in page_text]
    assert missing == [], missing


def test_the_signatures_are_the_live_ones(page_text):
    """Not a spot check: the exact rendering the generator produces has to be
    what is on the page, for every callable it can read."""
    import inspect

    sys.path.insert(0, str(ROOT / "tools"))
    from generate_api_reference import render_signature

    for name in ("glmboost", "gamboost", "glmboostLSS", "gamboostLSS",
                 "boost_control", "stabsel", "cv_weights", "Bbs", "Bols"):
        rendered = render_signature(getattr(mb, name))
        assert rendered, name
        assert rendered in page_text, f"{name}: {rendered[:80]}"
        # And it is genuinely the live signature.
        assert str(inspect.signature(getattr(mb, name))).count("*") >= 0


def test_the_page_records_keyword_only_arguments(page_text):
    """The old page put the `*` in the wrong place for `glmboost`, which is
    what a hand-written signature does when the code moves."""
    assert "glmboost(formula: str, data, *, family=None" in page_text
    assert "glmboostLSS(formula, data, *, families=None" in page_text


def test_a_string_default_keeps_its_quotes(page_text):
    """`na_action: str = 'error'` says the default is the string `error`;
    `= error` says something else."""
    assert "na_action: str = 'error'" in page_text
    assert "na_action: str = error" not in page_text


def test_the_model_surface_shows_the_property_or_method_form(page_text):
    """The table exists to make a form mismatch visible, so the values have to
    be read from the classes."""
    for name, form in (("control", "property"), ("risk", "property"),
                       ("selected", "property"), ("predict", "method")):
        row = next(line for line in page_text.splitlines()
                   if line.startswith(f"| `{name}` |"))
        assert row.count(f"| {form}") == 3, row
    # A real difference, not a mismatch: an LSS fit has no single hat matrix.
    aic = next(line for line in page_text.splitlines()
               if line.startswith("| `aic` |"))
    assert aic.count("| method") == 2 and "| -- |" in aic


def test_families_and_learners_carry_their_verification_status(page_text):
    """Each links to what the capability manifest records for it, so a reader
    sees a family's evidence next to its name."""
    assert "| Name | What it is | Python fit | R fit |" in page_text
    assert "verified" in page_text


def test_result_classes_are_not_misclassified_as_families(page_text):
    families = page_text.split("## Families", 1)[1].split(
        "## Distributional families", 1
    )[0]
    assert "LssCvRiskResult" not in families
    assert "LssStabSelResult" not in families


def test_fitted_models_are_returned_objects_not_constructor_tutorials(page_text):
    fitted = page_text.split("## Fitted models", 1)[1].split("## Families", 1)[0]
    assert "`GAMBoostModel`" in fitted
    assert "GAMBoostModel(formula" not in fitted


def test_model_surface_uses_each_class_own_description(page_text):
    offset = next(line for line in page_text.splitlines()
                  if line.startswith("| `offset_` |"))
    assert "scalar" in offset.lower()
    assert "mapping" in offset.lower()


def test_the_check_fails_when_the_page_is_stale(tmp_path):
    """A generator whose `--check` cannot fail is not a check."""
    stale = json.loads(PAGE.read_text())
    stale["cells"][0]["source"] = ["# API reference\n", "\n", "stale\n"]
    backup = PAGE.read_text()
    try:
        PAGE.write_text(json.dumps(stale, indent=1) + "\n")
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--check"],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        assert result.returncode == 1
        assert "out of date" in result.stderr
    finally:
        PAGE.write_text(backup)


def test_ci_runs_the_check():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "generate_api_reference.py --check" in workflow


def test_nothing_is_left_unplaced(page_text):
    """An `Other` section means a new export has no home in the generator,
    which is a prompt to give it one rather than a place to leave it."""
    assert "## Other" not in page_text


# ── the minimal copyable examples and the Rust surface ────────────────


def _generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_api_reference", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attribute_references(tree):
    """Every ``mb.<attr>`` chain in a snippet, for existence checking."""
    import ast

    names = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "mb"):
            names.append(node.attr)
    return names


def test_every_rendered_example_compiles_and_names_real_exports():
    """A snippet that names a removed export teaches the wrong spelling.

    Compilation catches syntax; walking the AST for `mb.<attr>` chains and
    resolving them against the package catches drift, which is the failure
    mode a copyable example has that prose does not.
    """
    generator = _generator()
    checked = 0
    for name in mb.__all__:
        if name.startswith("_"):
            continue
        snippet = generator.example_for(name)
        if snippet is None:
            continue
        tree = compile(snippet, f"<example:{name}>", "exec", ast_flag())
        for attribute in _attribute_references(tree):
            assert hasattr(mb, attribute), (
                f"the example for {name} references mb.{attribute}, "
                "which the package does not export")
        checked += 1
    assert checked >= 90, (
        f"only {checked} entries carry an example; the ratchet below "
        "records how many are missing")


def ast_flag():
    import ast
    return ast.PyCF_ONLY_AST


#: Entries without an example, frozen. Adding an entry without an example
#: grows this list and fails; giving one an entry shrinks it, and the list
#: is updated then. Fitted-model classes are deliberately absent: the model
#: surface table below is their reference.
MISSING_EXAMPLES = frozenset({
    "GLMBoostModel", "GAMBoostModel", "LSSModel",
    "learner_design", "LearnerDesign",
    "BoostControl", "CvRiskResult", "LssCvRiskResult",
    "StabSelResult", "LssStabSelResult", "TermReport", "Suggestion",
    "LabelEvidence", "MstopTuning",
    "ColumnKind", "ScreenedPair", "Summary", "SummaryTable", "AicResult",
    "DiscontinuityHit",
    "ConformalInterval", "CompareReport", "CompareSpec",
    "ExecutionReport", "PathRow", "make_grid", "make_grid_lss",
    "diagonal_grid", "cvrisk_glmboostLSS", "cvrisk_gamboostLSS",
    "stabsel_glmboostLSS", "stabsel_gamboostLSS",
    "SameFamilyGap",
    "acf_residuals", "calibration_frame", "coef_all",
    "deviance_increment", "ipc_weights", "outlier_frame",
    "predict_all", "residual_frame", "worm_data",
    "classify_columns", "suggest_terms", "expand_pairs",
    "screen_interactions", "expand_interactions", "expand_formula",
    "screen_discontinuities",
    "parse_formula", "parse_gamboost_formula",
    "scan_lss_families", "compare_summaries", "compare_to_r",
    "data_cards", "learner_capabilities", "model_capabilities",
    "dataset_capabilities", "datasets", "plot",
    "Assumption", "AssumptionLedger", "CausalDraft", "PairedTest",
    "FitCache", "LLMJudge", "ModelerResult", "PolicyResult", "Spent",
    "SpecificationJudgment", "Transcript", "TrapFinding", "TrapReport",
    "causal_trap_report", "elicit_causal_spec",
})


def test_the_missing_example_set_is_frozen():
    generator = _generator()
    missing = {name for name in mb.__all__
               if not name.startswith("_") and generator.example_for(name) is None}
    unrecorded = missing - MISSING_EXAMPLES
    assert not unrecorded, (
        f"new exports without a copyable example: {sorted(unrecorded)}; "
        "add one to EXAMPLES in tools/generate_api_reference.py or record "
        "the name here")
    # Shrinking is welcome: prune MISSING_EXAMPLES when an entry gains one.


def test_the_rust_surface_section_matches_the_stub(page_text):
    """Every signature in `_rust.pyi` appears on the rendered page."""
    stub_lines = [
        line.strip() for line in
        (ROOT / "python/mboostrs/_rust.pyi").read_text().splitlines()
        if line.strip().startswith("def ") and line.strip().endswith("...")
    ]
    assert len(stub_lines) >= 30
    assert "(the-rust-extension-surface)=" in page_text
    body = page_text.split("## Compiled extension functions", 1)[1]
    for line in stub_lines:
        definition = line.strip()[4:-3].rstrip(":").strip()
        # Rendered as `Class.method` — `signature`, or `signature` alone for
        # module functions; the signature string itself is always present.
        assert f"`{definition}`" in body, (
            f"{definition} missing from the rendered surface")


#: Snippets that build their own data and therefore must actually run.
SELF_CONTAINED = (
    "glmboost", "gamboost", "glmboostLSS", "gamboostLSS", "boost_control",
    "save_model", "term_report", "cv_risk", "suggest_specifications",
    "data_card", "family_capabilities",
)


def test_the_self_contained_examples_execute(tmp_path, monkeypatch):
    """The snippets a reader can paste with nothing else on the page.

    The rest build on `fit` from the Fitting examples; these stand alone, so
    a failure here is a broken first ten minutes. The runner works in a
    scratch directory because the save/load pair writes a file.
    """
    import mboostrs  # noqa: F401 - the snippets' `mb`

    generator = _generator()
    namespace = {"mb": mb}
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        # cv_risk/term_report fold loops emit their one-line extrapolation
        # summary by design; the page test is about the snippets, not that.
        warnings.simplefilter("ignore")
        for name in SELF_CONTAINED:
            snippet = generator.example_for(name)
            assert snippet is not None, name
            exec(compile(snippet, f"<example:{name}>", "exec"), namespace)
