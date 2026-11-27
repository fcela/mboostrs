"""Command-line entry: ``python -m mboostrs <command>``.

    python -m mboostrs guide [TOPIC]            print the agent guide or a topic
    python -m mboostrs sig NAME                 print an API card for NAME
    python -m mboostrs skill install [--dest DIR] [--overwrite]
    python -m mboostrs skill path               print where the shipped skill is
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    from mboostrs.agent_docs import TOPICS, agent_guide, install_skill

    parser = argparse.ArgumentParser(prog="python -m mboostrs")
    sub = parser.add_subparsers(dest="command", required=True)
    guide = sub.add_parser("guide", help="print the agent guide, or one topic of it")
    guide.add_argument("topic", nargs="?", default=None,
                       help=f"'index', 'example:<name>', or one of: {', '.join(TOPICS)}")
    sig = sub.add_parser("sig", help="print a signature/return/example card for one name")
    sig.add_argument("name",
                     help="a top-level name ('tune_mstop') or a dotted path "
                          "('GAMBoostModel.predict')")
    skill = sub.add_parser("skill", help="install the agent skill into a skills directory")
    skill_sub = skill.add_subparsers(dest="action", required=True)
    install = skill_sub.add_parser("install", help="copy the skill and its topics")
    install.add_argument("--dest", default=".claude/skills",
                         help="skills directory (default: .claude/skills)")
    install.add_argument("--overwrite", action="store_true",
                         help="replace an existing copy")
    skill_sub.add_parser("path", help="print the shipped SKILL.md path")
    repro = sub.add_parser("reproduce", help="reproduce a fit and discovery report from a trail JSON and data file")
    repro.add_argument("trail", help="path to a saved decision trail JSON file")
    repro.add_argument("data", help="path to the dataset (.parquet or .csv)")
    repro.add_argument("--format", default="markdown", choices=("markdown", "text", "json"),
                       help="output format for the reproduced report (default: markdown)")
    repro.add_argument("--output", default=None,
                       help="optional file path to write the reproduced report")
    args = parser.parse_args(argv)

    if args.command == "reproduce":
        from pathlib import Path
        import mboostrs as mb

        data_path = Path(args.data)
        if not data_path.is_file():
            parser.error(f"data file not found: {args.data!r}")
        suffix = data_path.suffix.lower()
        if suffix == ".parquet":
            try:
                import polars as pl
                frame = pl.read_parquet(data_path)
            except ImportError:
                import pandas as pd
                frame = pd.read_parquet(data_path)
        elif suffix == ".csv":
            try:
                import polars as pl
                frame = pl.read_csv(data_path)
            except ImportError:
                import pandas as pd
                frame = pd.read_csv(data_path)
        else:
            parser.error(f"unsupported data file extension {suffix!r}; use .parquet or .csv")
        res = mb.reproduce(args.trail, frame, report_format=args.format)
        if args.output:
            Path(args.output).write_text(res.report, encoding="utf-8")
        else:
            sys.stdout.write(res.report + "\n")
        return 0 if res.matched else 2

    if args.command == "guide":
        try:
            sys.stdout.write(agent_guide(args.topic))
        except ValueError as exc:
            parser.error(str(exc))
        return 0
    if args.command == "sig":
        from mboostrs.api_card import api_card

        try:
            print(api_card(args.name))
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            parser.error(f"cannot resolve {args.name!r}: {exc}")
        return 0
    if args.action == "install":
        try:
            target = install_skill(args.dest, overwrite=args.overwrite)
        except FileExistsError as exc:
            parser.error(str(exc))
        print(target)
        return 0
    print(agent_guide(path_only=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
