"""pandas must stay an optional dependency.

`pyproject.toml` declares numpy, polars, pyarrow, altair, and vl-convert-python
as the runtime dependencies — pandas is not among them. Until 2026-07-23 both
`mboostrs.datasets` and `mboostrs.plot` imported pandas at module scope, and
`mboostrs/__init__.py` imports both, so `import mboostrs` raised
ModuleNotFoundError on any install that honored the declared dependency set.

This test is static rather than environmental so it runs in the ordinary suite
(where pandas *is* installed) and still catches a reintroduced top-level import.
"""

import ast
import pathlib

import pytest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "python" / "mboostrs"
_OPTIONAL = {"pandas", "sklearn", "scipy", "rpy2", "matplotlib"}

# Dedicated adapter modules may import the library they adapt at module scope,
# because reaching them requires an explicit `from mboostrs.<name> import ...`.
# The companion test below pins that `import mboostrs` does not pull them in.
_ADAPTER_EXEMPT = {"sklearn.py": {"sklearn"}}


def _module_paths():
    return sorted(p for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _toplevel_imports(tree: ast.Module) -> set[str]:
    """Module names imported at module scope (not inside a function or class)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            # `try: import x / except ImportError:` is a guarded optional import.
            continue
    return names


@pytest.mark.parametrize("path", _module_paths(), ids=lambda p: p.name)
def test_no_toplevel_optional_dependency_imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    allowed = _ADAPTER_EXEMPT.get(path.name, set())
    offenders = (_toplevel_imports(tree) & _OPTIONAL) - allowed
    assert not offenders, (
        f"{path.relative_to(_PACKAGE_ROOT)} imports {sorted(offenders)} at module "
        f"scope, but they are not declared runtime dependencies. Import inside "
        f"the function that needs them instead."
    )


def test_package_init_does_not_import_adapter_modules():
    """`import mboostrs` must not reach a module that needs an optional extra."""
    init = _PACKAGE_ROOT / "__init__.py"
    tree = ast.parse(init.read_text(), filename=str(init))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "mboostrs":
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[-1] for alias in node.names)

    adapters = {name.removesuffix(".py") for name in _ADAPTER_EXEMPT}
    assert not (imported & adapters), (
        f"mboostrs/__init__.py imports {sorted(imported & adapters)}, which "
        f"requires an optional extra; `import mboostrs` would fail without it."
    )
