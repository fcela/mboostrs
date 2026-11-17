#!/usr/bin/env python3
"""Copy the installed compiled module into the source tree.

Several checks in `tools/` and `docs/book/` import `mboostrs` from `python/`
(they put that directory first on `sys.path` so they read the checked-out
sources), and `python/mboostrs/_rust*.so` exists there only after a
`maturin develop`. A CI job that installs a wheel has the module in
site-packages instead. This script finds the installed copy and places it
next to the sources, so those checks run against the same compiled code the
wheel carries.

Run it after the wheel is installed and before any such check:

    python tools/install_extension_in_source.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = ROOT / "python" / "mboostrs"


def main() -> int:
    # Running a script puts its own directory (tools/) first on sys.path, not
    # the repository root, so this import resolves to the installed package.
    import mboostrs._rust as module

    installed = Path(module.__file__).resolve()
    if SOURCE_PACKAGE in installed.parents:
        print(f"{installed} is already in the source tree")
        return 0
    target = SOURCE_PACKAGE / installed.name
    shutil.copy2(installed, target)
    print(f"copied {installed} -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
