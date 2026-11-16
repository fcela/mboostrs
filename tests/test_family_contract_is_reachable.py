"""Every family the contract says fits must actually fit — and predict.

Route B turned up the same defect twice, in two different registries:

* nine LSS families had an R constructor and a manifest row saying
  ``fit = yes`` while the Rust R binding had no match arm, so every one failed
  with "Unknown LSS family";
* three more could be fitted from R but not predicted from, because
  ``.lss_response`` — a per-family link map in R, separate from the Rust
  ``response()`` implementations — had no entry for them.

Both passed every check the project had, because those checks verify that a
*name* is exported, not that calling it does anything. A family must currently
be listed in five places that do not check each other: the Rust match arms
(eight sites), the R link map, the Python class, the R constructor, and the
manifest.

This test closes that by construction. It reads the manifest, and for each
family claiming ``fit = yes`` it fits a real model and predicts on both the
link and response scales.

The response domain is *discovered*, not hardcoded: each family is offered a
series of candidate responses and must accept at least one. A mapping of
family-to-domain would be a sixth registry to keep in step, which is the
problem this test exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mboostrs as mb

N = 160


def _candidate_responses():
    """Response vectors spanning the domains the families support.

    Ordered cheapest-first; a family only has to accept one.
    """
    rng = np.random.default_rng(11)
    x = rng.normal(size=N)
    linear = 0.5 + 0.4 * x
    counts = rng.poisson(np.exp(0.6 + 0.3 * x)).astype(float)
    return x, {
        "real": linear + rng.normal(0, 0.8, N),
        "positive": np.abs(rng.gamma(4.0, 0.5, N)) + 0.3,
        "count": counts,
        "count_with_zeros": np.where(rng.random(N) < 0.3, 0.0, counts + 1.0),
        "semicontinuous": np.where(
            rng.random(N) < 0.3, 0.0, rng.gamma(3.0, 0.7, N) + 0.2
        ),
        "unit_open": rng.beta(2.0, 5.0, N).clip(1e-3, 1 - 1e-3),
        "binary": (rng.random(N) < 0.4).astype(float),
    }


def _manifest_lss_families():
    manifest = json.loads(
        (Path(mb.__file__).parent / "capability_manifest.json").read_text()
    )
    return [
        row["name"]
        for row in manifest["tables"]["families"]
        if row["name"].endswith("LSS") and row["fit"] == "yes"
    ]


LSS_FAMILIES = _manifest_lss_families()
# Multivariate response; it takes a cbind() formula rather than a plain one, so
# it is covered by its own tests rather than this sweep.
# DirichletLSS: multivariate response plumbing the sweep's candidate
# responses don't cover. GaussianKnownVarLSS: an mboostrs family with no
# upstream gamboostLSS counterpart; exposed in both frontends, and its
# documentation contract is pinned by its own test below.
SKIP = {"DirichletLSS", "GaussianKnownVarLSS"}


def test_gaussianknownvar_docstates_what_is_boosted():
    """Both submodels are boosted, and the missing counterpart is upstream.

    The docstring once said "Only eta_sigma is boosted" (the mu predictor is
    boosted too) and "No R counterpart" (ambiguous: this package's own R
    frontend exposes the family; what is missing is an upstream gamboostLSS
    one). The claims are checked against the fit and against the R help page,
    not only against prose.
    """
    rng = np.random.default_rng(1)
    n = 120
    x = rng.normal(size=n)
    s2 = rng.uniform(0.05, 0.5, n)
    y = rng.normal(1 + x, np.sqrt(s2 + 0.25))
    fit = mb.glmboostLSS("cbind(y, s2) ~ x", data={"y": y, "s2": s2, "x": x},
                         families=mb.GaussianKnownVarLSS(),
                         control=mb.boost_control(40, 0.1))
    assert len(fit.selected["mu"]) > 0, "the mu submodel must select learners"

    doc = mb.GaussianKnownVarLSS.__doc__ or ""
    assert "Only ``eta_sigma`` is boosted" not in doc
    assert "Both submodels are boosted" in doc
    assert "No R counterpart" not in doc
    assert "gamboostLSS" in doc

    rd = (Path(__file__).resolve().parents[1]
          / "mboostrs" / "man" / "GaussianKnownVarLSS.Rd").read_text()
    assert "Only \\code{eta_sigma} is boosted" not in rd
    assert "Both submodels are boosted" in rd
    assert "No \\code{gamboostLSS} counterpart" in rd or "gamboostLSS counterpart" in rd


def test_the_manifest_lists_the_families_it_should():
    """A guard on the guard: if this drops to a handful, the sweep below has
    stopped covering anything and would pass vacuously."""
    assert len(LSS_FAMILIES) >= 20, LSS_FAMILIES


@pytest.mark.parametrize("name", sorted(set(LSS_FAMILIES) - SKIP))
def test_family_fits_and_predicts_in_python(name):
    family = getattr(mb, name)()
    x, candidates = _candidate_responses()
    failures = {}

    # The survival LSS families take `Surv(time, status)` rather than a plain
    # response, so the probe offers that shape too. Discovering the formula
    # shape the same way as the domain keeps this test free of a family list.
    rng = np.random.default_rng(5)
    survival_data = {
        "x": x,
        "time": np.abs(rng.gamma(3.0, 0.8, N)) + 0.2,
        "status": (rng.random(N) < 0.7).astype(float),
    }
    shapes = [("y ~ x", {"x": x, "y": y}, label) for label, y in candidates.items()]
    shapes.append(("Surv(time, status) ~ x", survival_data, "survival"))

    for formula, data, label in shapes:
        try:
            fit = mb.glmboostLSS(
                {p: formula for p in family.param_names},
                data=data, families=getattr(mb, name)(),
                control=mb.boost_control(12, 0.1),
            )
            link = fit.predict(type="link")
            response = fit.predict(type="response")
        except Exception as exc:  # noqa: BLE001 - the domain probe is the point
            failures[label] = f"{type(exc).__name__}: {str(exc)[:70]}"
            continue

        assert set(link) == set(family.param_names)
        for parameter in family.param_names:
            assert np.all(np.isfinite(np.asarray(link[parameter]))), (
                f"{name}/{parameter} link-scale prediction is not finite"
            )
            assert np.all(np.isfinite(np.asarray(response[parameter]))), (
                f"{name}/{parameter} response-scale prediction is not finite"
            )
        return

    pytest.fail(
        f"{name} claims fit=yes in the capability manifest but accepted none of "
        f"the candidate responses:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in failures.items())
    )


@pytest.mark.parametrize("name", sorted(set(LSS_FAMILIES) - SKIP))
def test_family_is_reachable_from_every_registry(name):
    """The five places a family has to be listed, checked directly.

    Fitting exercises the Rust match arms and the Python class; this checks the
    two registries that are easy to forget because nothing dereferences them
    until a user does.
    """
    family = getattr(mb, name)()
    rust_name = family._rust_name

    # The R constructor.
    r_source = Path("mboostrs/R/mboostrs_gamlss.R").read_text(encoding="utf-8")
    # The LSS constructors take `stabilization =`, so match the name only.
    assert f"{name} <- function(" in r_source, f"no R constructor for {name}"

    # The R link map, checked *inside* `.lss_response` rather than anywhere in
    # the file — the constructor also mentions the rust name, so a file-wide
    # search passes even when the link entry is missing. (It did: this
    # assertion was written that way first and failed to catch a deleted
    # entry when tested against the real defect.)
    start = r_source.index(".lss_response <- function(")
    depth, i = 0, start
    while True:
        if r_source[i] == "{":
            depth += 1
        elif r_source[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    link_map = r_source[start : i + 1]
    assert f'"{rust_name}"' in link_map, (
        f"{name} has no `{rust_name}` branch in .lss_response, so "
        f"predict(type='response') fails in R with "
        f"'unknown LSS family/parameter: {rust_name}/...'"
    )

    # Every Rust R-binding match site must know it, not just the first.
    r_binding = Path("crates/mboostrs-r/src/lib.rs").read_text(encoding="utf-8")
    arms = r_binding.count(f'"{rust_name}" =>')
    sites = r_binding.count('"gaussianlss" =>')
    assert arms == sites, (
        f"{name} appears in {arms} of the R binding's {sites} family match "
        f"sites; a fit path that goes through a site without it fails with "
        f"'Unknown LSS family: {rust_name}'"
    )

    # And the R NAMESPACE, or the constructor is not callable.
    namespace = Path("mboostrs/NAMESPACE").read_text(encoding="utf-8")
    assert f"export({name})" in namespace, f"{name} is not exported from R"
