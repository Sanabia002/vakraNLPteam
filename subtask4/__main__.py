"""Team 9 Subtask 4 command line. Run from anywhere; paths are resolved against the repository root.

    python -m subtask4 run subtask4/configs/p1.yaml --split dev   run one configuration (--final for the test split)
    python -m subtask4 eval <run_id> --judge local         official evaluator, cached judge
    python -m subtask4 score <run_id>                      scoreboard of one run
    python -m subtask4 compare <run_a> <run_b>             paired comparison, refusal-checked
    python -m subtask4 labels                              rebuild topic_labels.json from tune gold
    python -m subtask4 topics-check --base-url URL --model NAME   topic classifier accuracy on tune
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from . import data, scoreboard


def _run_rows(run_id: str, judge: str, final: bool):
    from .run import OUTPUT
    return scoreboard.rows(OUTPUT / run_id / f"eval_{judge}.json", OUTPUT / run_id, final)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m subtask4", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    mcp = dict(default="benchmark/mcp_connection_config.yaml", help="MCP server launch config")

    p = sub.add_parser("run")
    p.add_argument("config", type=Path)
    p.add_argument("--split", choices=data.SPLIT_NAMES, help="run the config on this split instead of its own")
    p.add_argument("--final", action="store_true", help="the one run on the frozen test split")
    p.add_argument("--allow-dirty", action="store_true", help="smoke runs only: uncommitted changes allowed")
    p.add_argument("--resume", action="store_true", help="finish an interrupted run (completed domains are kept)")
    p.add_argument("--mcp-config", **mcp)
    for name in ("eval", "score"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        p.add_argument("--judge", default="local", choices=("local", "groq"))
        p.add_argument("--final", action="store_true")
        if name == "eval":
            p.add_argument("--mcp-config", **mcp)
    p = sub.add_parser("compare")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--judge", default="local", choices=("local", "groq"))
    sub.add_parser("labels")
    p = sub.add_parser("topics-check")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible server, e.g. http://localhost:11434/v1")
    p.add_argument("--model", required=True)
    args = ap.parse_args()

    config = args.config.resolve() if args.command == "run" else None
    os.chdir(data.REPO)  # VAKRA's runner and evaluator use paths relative to the repository root

    if args.command == "run":
        from .config import RunConfig
        from .run import run
        print(run(RunConfig.load(config, split=args.split), args.final, args.allow_dirty, args.resume, args.mcp_config))
    elif args.command == "eval":
        from .evaluate import evaluate
        print(evaluate(args.run_id, args.judge, args.mcp_config, args.final))
    elif args.command == "score":
        from .run import OUTPUT
        table = _run_rows(args.run_id, args.judge, args.final)
        scoreboard.write(table, OUTPUT / args.run_id / f"score_{args.judge}.json")
        print(scoreboard.format_summary(args.run_id, scoreboard.summary(table)))
    elif args.command == "compare":
        result = scoreboard.compare(_run_rows(args.run_a, args.judge, False), _run_rows(args.run_b, args.judge, False))
        lo, hi = result["interval"]
        print(f"{args.run_b} vs {args.run_a}, {result['dialogues']} dialogues, refusal-checked: "
              f"{100 * result['difference']:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}], p = {result['p']:.3f}; "
              f"gained {len(result['gained'])}, lost {len(result['lost'])}")
    elif args.command == "labels":
        from .topics import write_labels
        print(f"{len(write_labels())} labelled tune domains")
    elif args.command == "topics-check":
        from langchain_openai import ChatOpenAI
        from .topics import check, load_labels
        llm = ChatOpenAI(model=args.model, base_url=args.base_url, api_key="local", temperature=0)
        right, misses = asyncio.run(check(llm, load_labels()))
        print(f"{right}/{right + len(misses)} tune domains right")
        for domain, label, predicted in misses:
            print(f"  {domain}: label {label}, predicted {predicted}")


if __name__ == "__main__":
    main()
