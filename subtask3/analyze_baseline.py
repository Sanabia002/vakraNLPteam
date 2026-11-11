"""Summarize VAKRA capability-3 runner outputs"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = ["uuid", "domain", "query", "status", "tool_calls", "answer", "error", "duration_s"]


def read_records(folder: Path):
    rows = []
    for file in sorted(folder.glob("*.json")):
        if file.stem.endswith("_tools"):
            continue
        with file.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, list):
            continue
        for record in data:
            if not isinstance(record, dict) or "output" not in record:
                continue
            for turn in record.get("output", []):
                if not isinstance(turn, dict):
                    continue
                sequence = turn.get("sequence") or {}
                calls = sequence.get("tool_call", []) if isinstance(sequence, dict) else []
                rows.append({
                    "uuid": record.get("uuid", ""),
                    "domain": record.get("domain", file.stem),
                    "query": turn.get("query", ""),
                    "status": record.get("status", ""),
                    "tool_calls": len(calls) if isinstance(calls, list) else 0,
                    "answer": turn.get("answer", ""),
                    "error": record.get("error", ""),
                    "duration_s": record.get("duration_s", ""),
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/subtask3/baseline")
    parser.add_argument("--output", default="results/subtask3/baseline_summary.csv")
    args = parser.parse_args()
    rows = read_records(Path(args.input))
    if not rows:
        raise SystemExit(f"No VAKRA result records found in {args.input}")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    statuses = Counter(row["status"] for row in rows)
    calls = Counter(row["tool_calls"] for row in rows)
    print(f"Total examples: {len(rows)}")
    print(f"Execution completed: {statuses['success']}")
    print(f"Execution errors: {statuses['error']}")
    print("Executed tool-call counts (not gold hop counts):")
    for hops, count in sorted(calls.items()):
        print(f"  {hops} calls: {count}")
    print("NOTE: Execution success is not answer accuracy.")
    print(f"CSV saved: {destination}")


if __name__ == "__main__":
    main()
