"""Publishing is gated by the exact revision and all version sources agree."""

import pathlib
import re

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_publish_jobs_depend_on_release_verification():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    jobs = workflow["jobs"]
    verify = jobs["verify"]
    commands = "\n".join(
        step.get("run", "") for step in verify["steps"] if isinstance(step, dict)
    )
    for command in ("cargo clippy", "cargo test --workspace", "pytest",
                    "generate_rust_stub.py --check", "generate_capabilities.py --check",
                    "check_front_door.py"):
        assert command in commands
    for job in ("build-wheels", "build-sdist", "crates-io"):
        assert "verify" in ([jobs[job]["needs"]] if isinstance(jobs[job]["needs"], str)
                            else jobs[job]["needs"])
    assert set(jobs["pypi"]["needs"]) == {"build-wheels", "build-sdist", "test-sdist"}
    assert jobs["github-release"]["needs"] == ["pypi", "crates-io"]


def test_manual_verification_cannot_publish():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    assert "workflow_dispatch" in workflow[True]
    for name in ("crates-io", "pypi", "github-release"):
        assert workflow["jobs"][name]["if"] == "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"


def test_verification_installs_test_dependencies_from_the_built_artifact():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    commands = "\n".join(step.get("run", "") for step in jobs["verify"]["steps"])
    assert '${wheel}[dev,sklearn]' in commands
    source = jobs["test-sdist"]
    assert source["needs"] == "build-sdist"
    assert not any(step.get("uses", "").startswith("actions/checkout") for step in source["steps"])
    commands = "\n".join(step.get("run", "") for step in source["steps"])
    for text in ("pip install dist/*.tar.gz", "mb.gamboost", "cvrisk", "save_model", "load_model"):
        assert text in commands


def test_each_wheel_is_installed_and_exercised_before_upload():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    commands = "\n".join(step.get("run", "") for step in jobs["build-wheels"]["steps"]
                          if isinstance(step, dict))
    # Wheels are built for several interpreters, so the smoke test lets pip
    # pick the one that matches the runner's Python from the built directory.
    assert "pip install --force-reinstall --find-links dist mboostrs" in commands
    assert "mb.glmboost" in commands and "m.predict" in commands


def test_parity_triggers_cover_statistical_helper_boundaries():
    paths = yaml.safe_load((ROOT / ".github/workflows/parity.yml").read_text())[True]["pull_request"]["paths"]
    for path in (
        "python/mboostrs/folds.py", "python/mboostrs/_weights.py",
        "python/mboostrs/_refit.py", "python/mboostrs/transforms.py",
        "python/mboostrs/_criteria.py", "python/mboostrs/_likelihood.py",
        "python/mboostrs/diagnostics.py", "python/mboostrs/model_selection.py",
        "mboostrs/R/**",
    ):
        assert path in paths


def test_all_four_release_version_sources_agree():
    cargo = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    python = re.search(r'^__version__ = "([^"]+)"',
                       (ROOT / "python/mboostrs/__init__.py").read_text(), re.M).group(1)
    r_version = re.search(r'^Version: (\S+)', (ROOT / "mboostrs/DESCRIPTION").read_text(), re.M).group(1)
    assert cargo == pyproject == python == r_version
