"""Unified CLI for listing, checking, rerunning, and updating mboostrs benchmarks.

All benchmark definitions live in ``benches/registry.json``. Common commands::

    python benches/run.py list
    python benches/run.py list --tier headline
    python benches/run.py status
    python benches/run.py run distributional_comparison --update-book
    python benches/run.py run realdata_autospec --workers 8 --update-book
    python benches/run.py update-book distributional_shootout
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "benches/registry.json"
VALID_TIERS = ("headline", "domain", "workflow", "calibration", "ablation")


def load_registry() -> list[dict]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return list(data["studies"])


def find_study(studies: list[dict], study_id: str) -> dict:
    for study in studies:
        if study["id"] == study_id:
            return study
    known = ", ".join(s["id"] for s in studies)
    raise SystemExit(f"Unknown benchmark id {study_id!r}. Known ids: {known}")


def _artifact_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    for candidate in (
        (doc.get("refresh") or {}).get("commit") if isinstance(doc.get("refresh"), dict) else None,
        (doc.get("provenance") or {}).get("commit") if isinstance(doc.get("provenance"), dict) else None,
        doc.get("commit"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:12]
    return None


def _head_commit() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _study_freshness(study: dict, head: str | None) -> str:
    artifacts = [ROOT / a for a in study.get("artifacts", ())]
    if not artifacts:
        return "inline/no-json"
    missing = [a for a in artifacts if not a.exists()]
    if missing:
        return f"missing ({len(missing)}/{len(artifacts)})"
    commits = [_artifact_commit(a) for a in artifacts]
    if any(c is None for c in commits):
        return "present (no commit)"
    if head and all(c and (head.startswith(c) or c.startswith(head)) for c in commits):
        return f"current ({commits[0][:7]})"
    return f"recorded ({commits[0][:7]})"


def _format_cmd(template: list[str], *, python: str, workers: int) -> list[str]:
    return [tok.format(python=python, workers=workers) for tok in template]


def _bench_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        env.setdefault(key, "1")
    venv_share = ROOT / ".venv/share/jupyter"
    if venv_share.exists():
        env.setdefault("JUPYTER_PATH", str(venv_share))
    return env


def _run_cmd(cmd: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {rendered}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def cmd_list(args: argparse.Namespace) -> int:
    studies = load_registry()
    if args.tier:
        studies = [s for s in studies if s["tier"] == args.tier]
    head = _head_commit()
    header = f"{'ID':28s} {'TIER':12s} {'MODE':17s} {'ARTIFACTS':20s} {'PAGES'}"
    print(header)
    print("-" * len(header))
    for s in studies:
        fresh = _study_freshness(s, head)
        pages = ", ".join(Path(p).name for p in s.get("pages", ()))
        print(f"{s['id']:28s} {s['tier']:12s} {s['refresh_mode']:17s} {fresh:20s} {pages}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    studies = load_registry()
    head = _head_commit()
    print(f"Repository HEAD: {head or 'unknown'}")
    print(f"Registered studies: {len(studies)}\n")
    for tier in VALID_TIERS:
        group = [s for s in studies if s["tier"] == tier]
        if not group:
            continue
        print(f"[{tier}] ({len(group)})")
        for s in group:
            fresh = _study_freshness(s, head)
            runner = s.get("runner_script") or "(recorded external)"
            print(f"  - {s['id']:28s} {fresh:22s} runner={runner}")
        print()
    return 0


def _update_book_pages(study: dict, *, python: str, dry_run: bool, env: dict[str, str]) -> None:
    nbconvert = str(Path(python).with_name("jupyter"))
    if not Path(nbconvert).exists():
        nbconvert = "jupyter"
    notebooks = [p for p in study.get("pages", ()) if p.endswith(".ipynb")]
    if not notebooks:
        print(f"No executable notebook pages registered for {study['id']}.")
        return
    for page in notebooks:
        cmd = [
            nbconvert,
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            page,
        ]
        _run_cmd(cmd, dry_run=dry_run, env=env)
    _run_cmd([python, "docs/book/check_register.py"], dry_run=dry_run, env=env)


def cmd_run(args: argparse.Namespace) -> int:
    studies = load_registry()
    study = find_study(studies, args.id)
    if not study.get("command"):
        raise SystemExit(
            f"Study {study['id']!r} is marked {study['refresh_mode']!r} and has no in-repo runner."
        )
    env = _bench_env()
    python = args.python or sys.executable
    main_cmd = _format_cmd(study["command"], python=python, workers=args.workers)
    _run_cmd(main_cmd, dry_run=args.dry_run, env=env)
    for post in study.get("post_commands", ()):
        _run_cmd(_format_cmd(post, python=python, workers=args.workers), dry_run=args.dry_run, env=env)
    if args.update_book:
        _update_book_pages(study, python=python, dry_run=args.dry_run, env=env)
    return 0


def cmd_update_book(args: argparse.Namespace) -> int:
    studies = load_registry()
    study = find_study(studies, args.id)
    env = _bench_env()
    python = args.python or sys.executable
    _update_book_pages(study, python=python, dry_run=args.dry_run, env=env)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, inspect, rerun, and update mboostrs benchmark suites."
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_list = sub.add_parser("list", help="List registered benchmark studies")
    p_list.add_argument("--tier", choices=VALID_TIERS, default=None)
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Show artifact freshness and runners by tier")
    p_status.set_defaults(func=cmd_status)

    p_run = sub.add_parser("run", help="Rerun a benchmark study and its post-analysis")
    p_run.add_argument("id", help="Study id from benches/registry.json")
    p_run.add_argument("--workers", type=int, default=4, help="Parallel worker processes (default: 4)")
    p_run.add_argument("--python", default=None, help="Python executable (default: current interpreter)")
    p_run.add_argument("--update-book", action="store_true", help="Re-execute dependent book notebooks")
    p_run.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p_run.set_defaults(func=cmd_run)

    p_book = sub.add_parser("update-book", help="Re-execute book notebooks for a study")
    p_book.add_argument("id", help="Study id from benches/registry.json")
    p_book.add_argument("--python", default=None, help="Python executable (default: current interpreter)")
    p_book.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p_book.set_defaults(func=cmd_update_book)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
