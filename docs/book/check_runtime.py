#!/usr/bin/env python3
"""Verify that the book interpreter has a source-compatible native extension."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import numpy as np

BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parents[1]


def fail(reason: str) -> int:
    interpreter = Path(sys.executable).resolve()
    print("Documentation runtime validation failed:", file=sys.stderr)
    print(f"- interpreter: {interpreter}", file=sys.stderr)
    print(f"- {reason}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Rebuild mboostrs for this exact interpreter from the repository root:", file=sys.stderr)
    print(f"  maturin build --interpreter {interpreter} --release", file=sys.stderr)
    print(
        f"  {interpreter} -m pip install --force-reinstall "
        "target/wheels/<wheel-built-above> --no-deps",
        file=sys.stderr,
    )
    return 1


def source_version() -> str | None:
    init = (ROOT / "python/mboostrs/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    try:
        import mboostrs as mb

        native = importlib.import_module("mboostrs._rust")
    except Exception as exc:  # import errors need the same rebuild guidance
        return fail(f"cannot import mboostrs and its native extension: {exc}")

    expected = source_version()
    installed = getattr(mb, "__version__", None)
    if expected is None or installed != expected:
        return fail(f"source version is {expected!r}, but the imported package is {installed!r}")

    # Exercise the smallest complete Python -> PyO3 -> Rust -> Python fit. This
    # detects native/Python return-contract drift even when both halves carry
    # the same release version (common after changing a tuple returned by Rust).
    data = {
        "x": np.array([0.0, 1.0, 2.0, 3.0]),
        "y": np.array([0.0, 1.0, 1.5, 3.0]),
    }
    try:
        model = mb.glmboost("y ~ x", data=data, control=(1, 0.1))
        report = model.execution_report()
    except Exception as exc:
        return fail(
            "the installed Python sources and native extension failed their fit "
            f"contract probe ({type(exc).__name__}: {exc})"
        )
    if report is None:
        return fail("the fit contract probe returned no execution report")

    # The book documents the generic heterogeneous gamboostLSS adapter. A stale
    # gamlss.py can still pass the scalar glmboost probe while rejecting every
    # learner except Bbs, so exercise one block and one scalar learner here.
    lss_data = {
        "group": np.array([0.0, 1.0, 0.0, 1.0]),
        "x": np.array([-1.0, 0.5, 1.0, -0.5]),
        "y": np.array([-0.8, 0.6, 1.1, -0.3]),
    }
    try:
        lss_model = mb.gamboostLSS(
            {
                "mu": "y ~ brandom(group, df=1)",
                "sigma": "y ~ bols(x)",
            },
            data=lss_data,
            families=mb.GaussianLSS(),
            control=(1, 0.1),
        )
        lss_model.predict(parameter="mu", type="link")
    except Exception as exc:
        return fail(
            "the installed Python sources and native extension failed their mixed "
            f"gamboostLSS contract probe ({type(exc).__name__}: {exc})"
        )

    print(
        "Documentation runtime valid: "
        f"mboostrs {installed}; Python {sys.version_info.major}.{sys.version_info.minor}; "
        f"native extension {Path(native.__file__).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
