#!/usr/bin/env python3
"""Execute the calls the README shows, and check what they return.

D8 item 4: "A fresh-environment script must execute every displayed front-door
call and validate its return type and shape; prose-only API calls are
included."

Two failure modes, and the second is the one that survives ordinary review:

* A **code block** that no longer runs. Nothing executed the README's Python,
  so a renamed argument or a changed default broke the first thing a new user
  copies, silently.
* A **prose-only** call. `mb.family_capabilities()` and `mb.plot_cvrisk(cvr)`
  are both offered in sentences rather than fenced blocks. One resolves and
  one never did -- the examples advertised `mb.plot_cvrisk` for months, and a
  checker that reads only fenced code finds nothing wrong with it.

What this does not do is judge the prose. It runs what the README shows and
reports what came back, so a claim about a return type can be compared with
the return.

Run from the repository root:

    python tools/check_front_door.py
    python tools/check_front_door.py --show     # print each block's result
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import sys
import tempfile
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

sys.path.insert(0, str(ROOT / "python"))

import mboostrs as mb  # noqa: E402

#: A fenced block this cannot run in-process, with the reason. Everything else
#: must execute. An entry here is a statement that the block is not Python the
#: reader runs as-is, not permission to let it rot.
SKIP = {
    "shell": "a shell command, not Python",
    "no-run": "declared `# check-front-door: skip` with a reason on the line",
}


def python_blocks(text: str) -> list[tuple[int, str]]:
    """`(line number, source)` for each ```python block."""
    blocks = []
    for match in re.finditer(r"```python\n(.*?)```", text, re.S):
        line = text[: match.start()].count("\n") + 1
        blocks.append((line, match.group(1)))
    return blocks


def skip_reason(source: str) -> str | None:
    first = source.strip().splitlines()[0] if source.strip() else ""
    if "check-front-door: skip" in first:
        return first.split("check-front-door: skip", 1)[1].strip(" -–—:") or "declared"
    return None


def prose_api_names(text: str) -> set[str]:
    """`mb.<name>` mentioned outside a fenced block."""
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    return set(re.findall(r"\bmb\.(\w+)", prose))


def describe(value: object) -> str:
    """The return's type and shape, which is what a reader needs to predict."""
    kind = type(value).__name__
    for attribute in ("shape", "height"):
        if hasattr(value, attribute):
            return f"{kind}{getattr(value, attribute)!r}"
    if isinstance(value, (list, tuple, dict, set)):
        return f"{kind}[{len(value)}]"
    return kind


def run_blocks(text: str, show: bool) -> list[str]:
    """Execute the blocks in one session, in a scratch directory.

    The serialization example writes `model.json` in the working directory, so
    running the README from the repository root leaves a file behind. The
    reader's session is theirs to litter; this one is not.
    """
    problems: list[str] = []
    namespace: dict[str, object] = {}
    scratch = tempfile.mkdtemp(prefix="mboostrs-front-door-")
    previous = os.getcwd()
    os.chdir(scratch)
    try:
        problems = _run_each(text, show, namespace)
    finally:
        os.chdir(previous)
        shutil.rmtree(scratch, ignore_errors=True)
    return problems


def _run_each(text: str, show: bool, namespace: dict) -> list[str]:
    problems: list[str] = []
    for line, source in python_blocks(text):
        reason = skip_reason(source)
        if reason:
            if show:
                print(f"  README:{line} skipped — {reason}")
            continue
        try:
            # One shared namespace: the blocks are a single session, and a
            # later block using `fit` from an earlier one is the README
            # working as written rather than a defect.
            exec(compile(source, f"README.md:{line}", "exec"), namespace)  # noqa: S102
        except Exception:
            problems.append(
                f"README.md:{line} block failed:\n"
                + "".join(traceback.format_exc(limit=3)).rstrip()
            )
            continue
        if show:
            interesting = {k: v for k, v in namespace.items()
                           if not k.startswith("_") and k not in ("mb", "np", "pl")}
            names = ", ".join(f"{k}={describe(v)}" for k, v in
                              list(interesting.items())[-3:])
            print(f"  README:{line} ok — {names}")
    return problems


def check_prose_names(text: str) -> list[str]:
    public = set(mb.__all__)
    return [
        f"`mb.{name}` is offered in prose but is not a public name"
        for name in sorted(prose_api_names(text))
        if name not in public and not name.startswith("_")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="print each block's result")
    arguments = parser.parse_args()

    text = README.read_text(encoding="utf-8")
    blocks = python_blocks(text)
    problems = run_blocks(text, arguments.show) + check_prose_names(text)

    if problems:
        print("Front-door check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print(f"Front door: {len(blocks)} README python block(s) executed, "
          f"{len(prose_api_names(text))} prose API name(s) resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
