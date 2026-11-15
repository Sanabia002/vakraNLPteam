"""Answer Check: does the Evidence Check answer documents-only questions that the documents CAN answer?

Our data never asks such a question in the scored turn, so the Evidence Check has only ever been seen to refuse.
These questions come from the reference answers themselves: every document search inside a tune or dev reference
answer is a question the domain's own documents answer, and the value the reference passes to its next API call is
the answer (kept only when that value appears in the retrieved passage). Each question goes through the Evidence
Check alone, with the same tools and model as a scored run.

    python -m subtask4.answer_check build                       writes subtask4/answer_check.json
    python -m subtask4.answer_check run [--config FILE]         writes output/subtask4/answer_check-<run id>.json

Outcome per question: correct (verified, and the answer contains the known value), wrong (verified, other value),
refused (nothing verified). Course-test domains are never read.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
from pathlib import Path

from . import data
from .rules import is_retriever

ITEMS = Path(__file__).with_name("answer_check.json")
OUTPUT = data.REPO / "output" / "subtask4"


def build() -> list[dict]:
    items = []
    for split in ("tune", "dev"):
        for domain in data.domains(split):
            for uuid, g in data.gold(domain).items():
                for turn in g["output"]:
                    calls = turn["sequence"].get("tool_call") or []
                    responses = turn["sequence"].get("tool_response") or []
                    for i, (call, response) in enumerate(zip(calls, responses)):
                        if not is_retriever(call["name"]) or i + 1 >= len(calls):
                            continue
                        text = " ".join(x.get("text", "") for x in response) if isinstance(response, list) else str(response)
                        known = [str(v) for v in calls[i + 1]["arguments"].values()
                                 if isinstance(v, (str, int)) and len(str(v)) > 2 and str(v).lower() in text.lower()]
                        if known:
                            items.append({"split": split, "domain": domain, "uuid": uuid,
                                          "question": " ".join(str(call["arguments"].get("query", "")).split()),
                                          "known": known})
    ITEMS.write_text(json.dumps(items, indent=1) + "\n")
    return items


def outcome(verdict: str | None, answer: str | None, known: list[str]) -> str:
    if verdict != "verified":
        return "refused"
    return "correct" if any(k.lower() in str(answer).lower() for k in known) else "wrong"


async def run(config_path: str | None = None) -> dict:
    from .config import RunConfig
    from .system.graph import initial_state
    from .system.studio import _agent
    items = json.loads(ITEMS.read_text())
    data.check_access({x["domain"] for x in items})  # tune and dev only
    config = RunConfig.load(config_path or data.REPO / "subtask4" / "configs" / "decision_graph.yaml")
    results = []
    for item in items:
        agent = await _agent(item["domain"], str(config_path) if config_path else None)
        system = agent._system
        state = {**initial_state(question=item["question"], history=[], rule_text="", messages=[]),
                 "ev": system._ev_start("rule", item["question"])}
        final = await system.build_evidence_check().ainvoke(state, config={"recursion_limit": 60})
        ev = final["ev"]
        results.append({**item, "answer": ev.get("answer"), "verdict": ev.get("verdict"), "rounds": ev.get("round"),
                        "outcome": outcome(ev.get("verdict"), ev.get("answer"), item["known"])})
    summary = dict(collections.Counter(r["outcome"] for r in results))
    out = OUTPUT / f"answer_check-{config.run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": config.run_id, "summary": summary, "results": results}, indent=1) + "\n")
    return {"summary": summary, "file": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m subtask4.answer_check")
    parser.add_argument("command", choices=("build", "run"))
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.command == "build":
        print(f"{len(build())} questions with a known answer -> {ITEMS}")
    else:
        print(asyncio.run(run(args.config)))


if __name__ == "__main__":
    main()
