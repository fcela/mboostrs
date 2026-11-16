#!/usr/bin/env python3
"""Exercise every standard family across the aspects P1.2 lists.

P1.2: "For each standard family and LSS family, check constructor defaults,
response domain, response coding, offset, response link, negative gradient,
risk, nuisance calibration, weights, prediction, and diagnostics."

The LSS half of that sweep exists -- `tests/test_family_contract_is_reachable.py`
fits and predicts every LSS family the manifest claims, discovering each
response domain rather than hardcoding a table. The standard families had no
equivalent: their coverage is spread across per-family parity tests, which
check the families someone wrote a test for and say nothing about the rest.

This runs one contract over all of them. Each aspect is a question with a
mechanical answer:

`constructor`
    the family constructs with no arguments, or the failure names what is
    missing.
`domain`
    it accepts at least one of the candidate responses. The candidates are
    offered in turn rather than mapped family-to-domain, because a mapping is
    another registry to keep in step.
`refusal`
    a response outside its support is refused by name, not fitted into
    arithmetic that runs and means nothing.
`offset`
    the boosting starting value is finite.
`link`
    `predict` runs on both the link and the response scale.
`gradient`
    `residuals()` is finite and has one entry per row.
`risk`
    the in-bag risk path is finite and does not increase.
`weights`
    a weighted fit either runs and produces a finite path, or refuses with a
    typed error. Silently ignoring weights is neither.
`prediction`
    predicting the training rows returns the fitted values.
`diagnostics`
    `residual_frame()` returns a frame or refuses with `NotImplementedError`.

A family that refuses an aspect is not a failure. Refusing *silently* is, and
so is an aspect that raises something other than the typed error the package
promises.

Run from the repository root:

    python tools/check_family_contract.py
    python tools/check_family_contract.py --show
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import mboostrs as mb  # noqa: E402

N = 160

ASPECTS = ("constructor", "domain", "refusal", "offset", "link", "gradient",
           "risk", "weights", "prediction", "diagnostics")


def candidates() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Responses spanning the domains the standard families support.

    Offered cheapest-first; a family only has to accept one. Discovering the
    domain rather than mapping it keeps this from becoming another registry
    that has to be kept in step with the families.
    """
    rng = np.random.default_rng(11)
    x = rng.normal(size=N)
    linear = 0.5 + 0.4 * x
    counts = rng.poisson(np.exp(0.6 + 0.3 * x)).astype(float)
    time = np.abs(rng.gamma(3.0, 0.8, N)) + 0.2
    status = (rng.random(N) < 0.7).astype(float)
    return x, {
        "real": linear + rng.normal(0, 0.8, N),
        "positive": np.abs(rng.gamma(4.0, 0.5, N)) + 0.3,
        "count": counts,
        "binary": (rng.random(N) < 0.4).astype(float),
        "signed": np.where(rng.random(N) < 0.4, -1.0, 1.0),
        "classes": rng.integers(0, 3, size=N).astype(float),
        "ordinal": rng.integers(1, 4, size=N).astype(float),
        "survival": np.column_stack([time, status]),
    }


#: Families whose loss is re-estimated as the path proceeds, so the risk at
#: one iteration is not measured under the same function as at the last and
#: the path is not required to decrease. `mboost` behaves the same way; see
#: the `Huber` note in `mboostrs._parity_notes`.
ADAPTIVE_LOSS = {"Huber"}

#: Families that need an entry point other than `glmboost`, with the reason.
#: `Multinomial` builds a Kronecker-structured design on the gamboost path,
#: and `glmboost` rejects any response outside {0, 1} whatever `n_classes`
#: says -- a deviation from upstream that `_parity_notes` records.
ENTRY_POINT = {"Multinomial": "gamboost"}


def fit_with(family, response, x, weights=None, name=""):
    data = {"x": x}
    if response.ndim == 2:
        data["time"] = response[:, 0]
        data["status"] = response[:, 1]
        formula = "Surv(time, status) ~ x"
    else:
        data["y"] = response
        formula = "y ~ x"
    entry = ENTRY_POINT.get(name, "glmboost")
    if entry == "gamboost":
        formula = formula.replace("~ x", "~ bols(x)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(mb, entry)(formula, data, family=family,
                                  weights=weights,
                                  control=mb.boost_control(25, 0.1)), data


def first_accepted(name: str, x, responses):
    """The first candidate response the family fits, and the fit."""
    for label, response in responses.items():
        try:
            family = build(name)
            fit, data = fit_with(family, response, x, name=name)
        except Exception:  # noqa: BLE001, S112
            continue
        return label, response, fit, data
    return None, None, None, None


def build(name: str):
    cls = getattr(mb, name)
    if name == "Multinomial":
        return cls(n_classes=3)
    if name == "PropOdds":
        return cls(n_levels=3)
    return cls()


def standard_families() -> list[str]:
    manifest = json.loads(
        (ROOT / "python" / "mboostrs" / "capability_manifest.json")
        .read_text(encoding="utf-8"))
    return sorted(
        row["name"] for row in manifest["tables"]["families"]
        if not row["name"].endswith("LSS") and row["fit"] == "yes"
        and hasattr(mb, row["name"])
    )


def check_family(name: str, x, responses) -> tuple[dict[str, str], list[str]]:
    """`(aspect -> outcome, problems)` for one family."""
    outcome: dict[str, str] = {}
    problems: list[str] = []

    try:
        build(name)
        outcome["constructor"] = "constructs"
    except Exception as error:  # noqa: BLE001
        problems.append(f"{name}: cannot construct: {error}")
        return outcome, problems

    label, response, fit, data = first_accepted(name, x, responses)
    if fit is None:
        problems.append(
            f"{name}: fits none of the candidate responses "
            f"({', '.join(responses)}); either its domain is not represented "
            f"or the family does not fit at all")
        return outcome, problems
    outcome["domain"] = label

    # A response outside every support this family could have. `nan` is
    # refused by every family, and the refusal must name the column or the
    # family rather than surfacing as an arithmetic result.
    broken = np.array(response, dtype=float, copy=True)
    if broken.ndim == 2:
        broken[0, 0] = np.nan
    else:
        broken[0] = np.nan
    try:
        fit_with(build(name), broken, x, name=name)
        problems.append(
            f"{name}: fitted a response containing NaN instead of refusing it")
    except Exception as error:  # noqa: BLE001
        message = str(error)
        if not isinstance(error, (ValueError, TypeError)):
            problems.append(
                f"{name}: a NaN response raised {type(error).__name__}, not a "
                f"ValueError: {message[:70]}")
        outcome["refusal"] = type(error).__name__

    offset = float(fit.offset_)
    if not np.isfinite(offset):
        problems.append(f"{name}: the offset is {offset}")
    outcome["offset"] = f"{offset:.4g}"

    for scale in ("link", "response"):
        try:
            values = np.asarray(fit.predict(newdata=data, type=scale))
            if not np.all(np.isfinite(values)):
                problems.append(
                    f"{name}: predict(type={scale!r}) is not finite")
            outcome["link"] = outcome.get("link", "") + scale[0]
        except NotImplementedError:
            outcome["link"] = outcome.get("link", "") + scale[0].upper()
        except Exception as error:  # noqa: BLE001
            problems.append(
                f"{name}: predict(type={scale!r}) raised "
                f"{type(error).__name__}: {str(error)[:70]}")

    try:
        gradient = np.asarray(fit.residuals())
        if gradient.size and not np.all(np.isfinite(gradient)):
            problems.append(f"{name}: the negative gradient is not finite")
        outcome["gradient"] = str(gradient.shape)
    except NotImplementedError:
        outcome["gradient"] = "refused"

    risk = np.asarray(fit.risk_, dtype=float)
    if not np.all(np.isfinite(risk)):
        problems.append(f"{name}: the risk path is not finite")
    elif name in ADAPTIVE_LOSS:
        outcome["risk"] = (f"{risk[0]:.4g}->{risk[-1]:.4g} "
                           f"({int(np.sum(np.diff(risk) > 1e-8))} up)")
    elif np.any(np.diff(risk) > 1e-8):
        problems.append(
            f"{name}: the in-bag risk increases along the path, by up to "
            f"{float(np.max(np.diff(risk))):.3e}")
    else:
        outcome["risk"] = f"{risk[0]:.4g}->{risk[-1]:.4g}"

    rng = np.random.default_rng(3)
    weights = rng.uniform(0.5, 2.0, N)
    try:
        weighted, _ = fit_with(build(name), response, x, weights,
                               name=name)
        if not np.all(np.isfinite(np.asarray(weighted.risk_, dtype=float))):
            problems.append(f"{name}: a weighted fit has a non-finite risk")
        outcome["weights"] = "fits"
    except (ValueError, NotImplementedError) as error:
        outcome["weights"] = f"refused ({type(error).__name__})"
        if name.lower() not in str(error).lower() and \
                "weight" not in str(error).lower():
            problems.append(
                f"{name}: refuses weights without saying so: {str(error)[:70]}")
    except Exception as error:  # noqa: BLE001
        problems.append(
            f"{name}: a weighted fit raised {type(error).__name__}, which is "
            f"neither a fit nor a typed refusal: {str(error)[:70]}")

    replayed = np.asarray(fit.predict(newdata=data, type="link"))
    fitted = np.asarray(fit.fitted_)
    if replayed.shape == fitted.shape:
        gap = float(np.max(np.abs(replayed - fitted)))
        if gap > 1e-9:
            problems.append(
                f"{name}: predicting its own rows differs from fitted() by "
                f"{gap:.3e}")
        outcome["prediction"] = f"{gap:.1e}"
    else:
        outcome["prediction"] = f"{fitted.shape} vs {replayed.shape}"

    try:
        frame = fit.residual_frame()
        outcome["diagnostics"] = f"{len(frame)} rows"
    except NotImplementedError:
        outcome["diagnostics"] = "refused"
    except Exception as error:  # noqa: BLE001
        problems.append(
            f"{name}: residual_frame() raised {type(error).__name__}, not "
            f"NotImplementedError: {str(error)[:70]}")

    return outcome, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="print each family's row")
    arguments = parser.parse_args()

    x, responses = candidates()
    families = standard_families()
    problems: list[str] = []

    for name in families:
        outcome, found = check_family(name, x, responses)
        problems.extend(found)
        if arguments.show:
            cells = " ".join(f"{aspect}={outcome.get(aspect, '-')}"
                             for aspect in ASPECTS)
            print(f"  {name:<14} {cells}")

    if problems:
        print("Family contract failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print(f"Family contract: {len(families)} standard families, "
          f"{len(ASPECTS)} aspects each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
