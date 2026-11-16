"""`_rust.pyi` describes the module it is a stub for.

A type stub that drifts from its extension is worse than no stub: a checker
reports errors for calls that work and accepts calls that fail at run time.
The hand-written one had drifted both ways -- 22 of 36 registered symbols had
no stub, and 8 stub entries named symbols the module does not export -- because
nothing compared the two.

It is generated now, from the compiled module's own `__text_signature__`. This
file is the guard: it re-runs the comparison, so a new `#[pyfunction]` that
lands without regenerating fails here rather than at whatever type check first
notices.
"""

import ast
import pathlib
import subprocess
import sys

import mboostrs._rust as rust
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUB = ROOT / "python/mboostrs/_rust.pyi"
GENERATOR = ROOT / "tools/generate_rust_stub.py"


def _registered() -> set[str]:
    return {name for name in dir(rust) if not name.startswith("__")}


def _declared() -> set[str]:
    tree = ast.parse(STUB.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_every_registered_symbol_has_a_stub():
    missing = sorted(_registered() - _declared())
    assert not missing, (
        f"{len(missing)} registered symbols have no stub: {missing}. "
        f"Run python tools/generate_rust_stub.py"
    )


def test_no_stub_names_a_symbol_the_module_does_not_export():
    """The other direction, which a checker cannot warn about.

    A stub entry for an absent symbol makes a call that will fail at import
    look well typed.
    """
    phantom = sorted(_declared() - _registered())
    assert not phantom, (
        f"{len(phantom)} stub entries name absent symbols: {phantom}. "
        f"Run python tools/generate_rust_stub.py"
    )


def test_the_stub_is_valid_python():
    ast.parse(STUB.read_text())


def test_the_generator_reports_the_stub_as_current():
    """The end-to-end check CI runs, so a drift fails here first."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_stub_says_it_is_generated():
    """A reader who edits it by hand should be told the edit will be lost."""
    header = STUB.read_text()[:600]
    assert "generate_rust_stub.py" in header
    assert "Do not edit by hand" in header


@pytest.mark.parametrize(
    "name",
    ["fit_gamboost_specs", "family_diagnostics", "stabsel_params",
     "prepare_learner_store", "LearnerStore"],
)
def test_the_symbols_the_package_actually_calls_are_declared(name):
    """A spot check on the ones that were missing, so the count is not the
    only thing holding."""
    assert name in _declared()
    assert hasattr(rust, name)


def test_the_signature_matches_the_extension_for_a_defaulted_function():
    """Arity and defaults are the part a caller gets wrong.

    Argument *types* are `Any` -- PyO3 does not surface them -- so the stub
    pins the names, the count and the default values instead.
    """
    tree = ast.parse(STUB.read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "stabsel_params"
    )
    names = [argument.arg for argument in node.args.args]
    assert names == ["p", "cutoff", "q", "pfer", "b", "assumption", "sampling_type"]
    # Five of the seven carry defaults, matching the extension's own signature.
    assert len(node.args.defaults) == 6
