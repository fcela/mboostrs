"""`import mboostrs` should cost what fitting a model needs, and nothing else.

It cost **559 ms**, and **509 of them were `mboostrs.plot`** -- 453 inside
Altair, which pulls `jsonschema` and, through it, `rfc3987_syntax`, whose
module body alone took 349 ms (measured with `python -X importtime`). Every
caller paid that to fit a model, and a caller who never drew a chart paid it
for nothing. It is now about 50 ms.

Importing also flipped Altair's **process-wide** renderer from `default` to
`svg`. That changes how every Altair chart in the process renders, including
ones from libraries with nothing to do with mboostrs; a library import that
silently reconfigures a shared dependency is a surprise its caller cannot see.
The renderer is now a function to call, or an environment variable the
operator sets -- which is what the documentation book does, alongside the two
Polars formatting variables already in its build script.
"""

import ast
import pathlib
import subprocess
import sys

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "python" / "mboostrs"


def _run(code, env=None):
    import os

    environment = dict(os.environ)
    environment.pop("MBOOSTRS_SVG_RENDERER", None)
    environment.update(env or {})
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=environment, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# ─── plotting is loaded on demand ──────────────────────────────────


def test_importing_the_package_does_not_import_altair():
    loaded = _run(
        "import sys, mboostrs; "
        "print('altair' in sys.modules, 'vl_convert' in sys.modules)"
    )
    assert loaded == "False False"


def test_the_plot_module_still_resolves_as_an_attribute():
    assert _run(
        "import mboostrs; print(mboostrs.plot.__name__)"
    ) == "mboostrs.plot"


def test_from_import_still_works():
    assert _run(
        "from mboostrs import plot; print(plot.__name__)"
    ) == "mboostrs.plot"


def test_plot_is_listed_in_dir():
    assert _run("import mboostrs; print('plot' in dir(mboostrs))") == "True"


def test_an_unknown_attribute_still_raises():
    assert _run(
        "import mboostrs\n"
        "try:\n"
        "    mboostrs.not_a_thing\n"
        "except AttributeError as e:\n"
        "    print('raised')\n"
    ) == "raised"


def test_a_model_plot_method_pulls_it_in():
    """Laziness must not mean unavailability."""
    assert _run(
        "import numpy as np, mboostrs as mb\n"
        "rng = np.random.default_rng(0)\n"
        "d = {'x': rng.normal(size=60)}\n"
        "d['y'] = 2 * d['x'] + rng.normal(size=60)\n"
        "m = mb.glmboost('y ~ x', data=d, control=mb.boost_control(10, 0.1))\n"
        "chart = m.plot_risk()\n"
        "import sys; print('altair' in sys.modules)\n"
    ) == "True"


def test_importing_does_not_change_the_altair_renderer():
    assert _run(
        "import altair as alt, mboostrs, mboostrs.plot; print(alt.renderers.active)"
    ) == "default"


def test_the_renderer_can_be_enabled_explicitly():
    assert _run(
        "import altair as alt\n"
        "from mboostrs.plot import use_svg_renderer\n"
        "print(use_svg_renderer(), alt.renderers.active)\n"
    ) == "True svg"


def test_the_environment_variable_opts_in():
    """What the book's build script sets, next to the Polars formatting
    variables already there."""
    assert _run(
        "import altair as alt, mboostrs.plot; print(alt.renderers.active)",
        env={"MBOOSTRS_SVG_RENDERER": "1"},
    ) == "svg"


def test_a_falsy_environment_variable_does_not():
    for value in ("0", "false", "False", ""):
        assert _run(
            "import altair as alt, mboostrs.plot; print(alt.renderers.active)",
            env={"MBOOSTRS_SVG_RENDERER": value},
        ) == "default", value


def test_the_book_build_scripts_set_it():
    """Otherwise the book would lose its SVG output silently."""
    root = pathlib.Path(__file__).resolve().parents[1]
    book = root / "docs" / "book"
    assert "export MBOOSTRS_SVG_RENDERER=1" in (book / "kernel-env.sh").read_text()
    for script in ("build-html.sh", "build-pdf.sh"):
        text = (book / script).read_text()
        assert "source ./kernel-env.sh" in text, script


# ─── the declared dependency set ───────────────────────────────────


def test_altair_is_not_a_base_dependency():
    text = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    base = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "altair" not in base
    assert "vl-convert" not in base
    assert "pyarrow" not in base
    assert "polars" in base and "numpy" in base


def test_the_plot_extra_declares_what_plotting_needs():
    text = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    extra = text.split("plot = [", 1)[1].split("]", 1)[0]
    assert "altair" in extra and "vl-convert-python" in extra


def test_no_package_module_imports_altair_at_module_scope_except_plot():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "plot.py" or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            if "altair" in names:
                offenders.append(path.name)
    assert offenders == []


# ─── generally tagged wheels use a generally compatible baseline ───


def test_release_configuration_does_not_force_a_cpu_microarchitecture():
    """A wheel tag has no place to advertise x86-64-v3 requirements."""
    root = pathlib.Path(__file__).resolve().parents[1]
    cargo_config = (root / ".cargo" / "config.toml").read_text()
    release = (root / ".github" / "workflows" / "release.yml").read_text()
    assert "target-cpu=x86-64-v3" not in cargo_config
    assert "target-cpu=x86-64-v3" not in release
    assert "RUSTFLAGS:" not in release
    assert "RUSTFLAGS=" not in release


def test_import_does_not_apply_a_wheel_assumption_to_source_builds():
    """Python cannot infer how a locally compiled extension was targeted."""
    source = (PACKAGE / "__init__.py").read_text()
    assert "check_cpu_baseline" not in source
    assert not (PACKAGE / "_cpu.py").exists()
