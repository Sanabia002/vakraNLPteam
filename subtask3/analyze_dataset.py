"""Analyze VAKRA Subtask 3 train JSON files without requiring API servers.

Usage (from repository root):
  python subtask3/analyze_dataset.py
  python subtask3/analyze_dataset.py --dataset data/train/capability_3_multihop_reasoning

Outputs CSVs and summary.json in results/subtask3/dataset_analysis/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path('data/train/capability_3_multihop_reasoning')
DEFAULT_OUTPUT = Path('results/subtask3/dataset_analysis')


def read_json_records(folder: Path) -> list[dict[str, Any]]:
    """Read either a list of records or a single record from each JSON file."""
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing directory: {folder}')
    files = sorted(folder.rglob('*.json'))
    if not files:
        raise FileNotFoundError(f'No JSON files found in {folder}')
    records: list[dict[str, Any]] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Cannot read {path}: {exc}') from exc
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f'Expected an object in {path}, got {type(item).__name__}')
            records.append(item)
    return records


def get_query(record: dict[str, Any]) -> str:
    dialogue = record.get('dialogue') or {}
    turns = dialogue.get('turns') or []
    if turns and isinstance(turns[0], dict):
        return str(turns[0].get('query', ''))
    return ''


def get_gold_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    turns = record.get('output') or []
    return [turn for turn in turns if isinstance(turn, dict)]


def normalize(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value)).strip().casefold()


def response_contains(value: Any, responses: list[Any]) -> bool:
    """Only check exact scalar provenance, not fuzzy substring inference."""
    expected = normalize(value)
    def visit(obj: Any) -> bool:
        if isinstance(obj, dict):
            return any(visit(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(visit(v) for v in obj)
        return normalize(obj) == expected
    return any(visit(r) for r in responses)


def analyze(input_records: list[dict[str, Any]], gold_records: list[dict[str, Any]]):
    gold_by_uuid: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in gold_records:
        uid = str(record.get('uuid', ''))
        if uid in gold_by_uuid:
            duplicates.append(uid)
        gold_by_uuid[uid] = record

    rows: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    for item in input_records:
        uid = str(item.get('uuid', ''))
        domain = str(item.get('domain', ''))
        query = get_query(item)
        if uid in seen_inputs:
            flags.append({'uuid': uid, 'domain': domain, 'type': 'duplicate_input_uuid', 'details': 'Duplicate input UUID'})
        seen_inputs.add(uid)

        gold = gold_by_uuid.get(uid)
        if gold is None:
            flags.append({'uuid': uid, 'domain': domain, 'type': 'missing_gold', 'details': 'No matching gold output UUID'})
            rows.append({'uuid': uid, 'domain': domain, 'query': query, 'gold_answer': '', 'gold_hops': '', 'tool_names': '', 'has_gold': False})
            continue
        gold_turns = get_gold_turns(gold)
        if not gold_turns:
            flags.append({'uuid': uid, 'domain': domain, 'type': 'missing_gold_turn', 'details': 'No gold output turns'})
        for turn in gold_turns:
            seq = turn.get('sequence') or {}
            calls = seq.get('tool_call') or []
            responses = seq.get('tool_response') or []
            if not isinstance(calls, list):
                calls = []
            if not isinstance(responses, list):
                responses = []
            turn_query = str(turn.get('query') or query)
            if turn_query and query and normalize(turn_query) != normalize(query):
                flags.append({'uuid': uid, 'domain': domain, 'type': 'query_mismatch', 'details': 'Input and gold queries differ'})
            if len(responses) != len(calls):
                flags.append({'uuid': uid, 'domain': domain, 'type': 'response_count_mismatch', 'details': f'{len(calls)} calls vs {len(responses)} responses'})
            for i, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                args = call.get('arguments') or {}
                if not isinstance(args, dict):
                    continue
                # Heuristic, limited to affiliation parameters: these usually come
                # from the question or an earlier API result. Other valid sources
                # may exist, so a flag is NOT evidence of an incorrect label.
                affiliation = args.get('affiliation')
                if isinstance(affiliation, str) and affiliation.strip():
                    in_query = normalize(affiliation) in normalize(turn_query)
                    in_previous = response_contains(affiliation, responses[:i])
                    if not in_query and not in_previous:
                        flags.append({
                            'uuid': uid, 'domain': domain,
                            'type': 'affiliation_needs_review',
                            'details': f'Call {i + 1} ({call.get("name", "unknown")}): affiliation={affiliation!r} not found in query or earlier tool responses',
                        })
            rows.append({
                'uuid': uid, 'domain': domain, 'query': query,
                'gold_answer': json.dumps(turn.get('answer'), ensure_ascii=False),
                'gold_hops': len(calls),
                'tool_names': ' -> '.join(str(c.get('name', '?')) for c in calls if isinstance(c, dict)),
                'has_gold': True,
            })
    for uid in duplicates:
        flags.append({'uuid': uid, 'domain': str(gold_by_uuid[uid].get('domain', '')), 'type': 'duplicate_gold_uuid', 'details': 'Duplicate gold UUID'})
    for uid in sorted(set(gold_by_uuid) - seen_inputs):
        flags.append({'uuid': uid, 'domain': str(gold_by_uuid[uid].get('domain', '')), 'type': 'gold_without_input', 'details': 'Gold output without input'})

    distribution = Counter(row['gold_hops'] for row in rows if row['has_gold'])
    domains = Counter(row['domain'] for row in rows)
    summary = {
        'input_records': len(input_records),
        'gold_records': len(gold_records),
        'matched_rows': sum(bool(r['has_gold']) for r in rows),
        'unique_domains': len(domains),
        'examples_by_domain': dict(sorted(domains.items())),
        'gold_hop_distribution': {str(k): v for k, v in sorted(distribution.items())},
        'review_flags': dict(sorted(Counter(f['type'] for f in flags).items())),
        'note': 'Hop counts describe reference tool calls, not agent accuracy. Review flags are heuristics, not confirmed dataset errors.',
    }
    return rows, flags, summary


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_records = read_json_records(args.dataset / 'input')
    gold_records = read_json_records(args.dataset / 'output')
    rows, flags, summary = analyze(input_records, gold_records)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'examples.csv', rows, ['uuid', 'domain', 'query', 'gold_answer', 'gold_hops', 'tool_names', 'has_gold'])
    write_csv(args.output / 'review_flags.csv', flags, ['uuid', 'domain', 'type', 'details'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('VAKRA Subtask 3 dataset analysis')
    print(f"Input records: {summary['input_records']}")
    print(f"Gold records: {summary['gold_records']}")
    print(f"Matched rows: {summary['matched_rows']}")
    print(f"Domains: {summary['unique_domains']}")
    print('Gold hop distribution:')
    for hops, count in summary['gold_hop_distribution'].items():
        print(f'  {hops} hop(s): {count}')
    print(f'Potential review flags: {len(flags)} (not verified errors)')
    print(f'Files saved under: {args.output}')


if __name__ == '__main__':
    main()
