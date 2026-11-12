"""Conservatively patch plan-execute agent, preserving other features.

Run from repository root:
    python subtask3/apply_missing_call_patch.py --dry-run
    python subtask3/apply_missing_call_patch.py --apply

Refuses to patch unexpected source or a file that doesn't parse afterwards.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import textwrap


IMPORT_LINE = "from subtask3.missing_tool_recovery import recover_missing_tool_call\n"
REPLACEMENT = '''\
outcome = await recover_missing_tool_call(
    initial_response=ai,
    allowed_tool_names=names,
    offered_tools=offered,
    question=question,
    plan=steps,
    current_step=step,
    step_index=si,
    history_text=self._history_text(step_records),
    parse_text_call=parse_text_tool_call,
    invoke_retry=lambda prompt: self._call_llm(
        step_llm,
        [SystemMessage(content=STEP_SYSTEM), HumanMessage(content=prompt)],
        deadline,
        self._final_reserve_s,
    ),
    enabled=_flag("MISSING_CALL_RECOVERY"),
)
call = outcome.call
if outcome.attempted:
    trajectory.append({
        "type": "AIMessage",
        "content": "Missing tool-call recovery " + outcome.reason,
        "step_index": si,
        "missing_call_recovery": True,
        "recovered": outcome.recovered,
        "tool_calls": [],
    })
    if outcome.recovered:
        ai_text = strip_think(getattr(outcome.response, "content", ""))
'''


def patch_source(source: str) -> str:
    if "recover_missing_tool_call(" in source:
        raise ValueError("Recovery integration appears already present; not applying twice")

    lines = source.splitlines(keepends=True)
    matches = []
    for i, line in enumerate(lines):
        if re.fullmatch(r"[ \t]+call = None\s*", line):
            upper = min(i + 40, len(lines))
            segment = "".join(lines[i:upper])
            if 'getattr(ai, "tool_calls"' not in segment or "parse_text_tool_call" not in segment:
                continue
            ends = [j for j in range(i + 1, upper) if lines[j].lstrip().startswith("# step header:")]
            if ends:
                matches.append((i, ends[0]))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Melissa call-selection block, found {len(matches)}")
    start, end = matches[0]
    indent = re.match(r"[ \t]*", lines[start]).group()
    patched = "".join(lines[:start]) + textwrap.indent(REPLACEMENT, indent) + "".join(lines[end:])

    if IMPORT_LINE.strip() not in patched:
        anchor = re.search(r"(?m)^from agents\.agent_interface import .*\n", patched)
        if anchor is None:
            raise ValueError("Could not locate agent_interface import; source not recognized")
        patched = patched[:anchor.end()] + IMPORT_LINE + patched[anchor.end():]

    ast.parse(patched)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes after validation")
    parser.add_argument("--dry-run", action="store_true", help="Validate only (default)")
    parser.add_argument("--path", type=Path, default=Path("agents/plan_execute_agent.py"))
    args = parser.parse_args()
    src = args.path.read_text(encoding="utf-8")
    output = patch_source(src)
    if args.apply:
        args.path.write_text(output, encoding="utf-8")
        print(f"Integrated optional recovery into {args.path}")
    else:
        print(f"Patch compatible with {args.path}; no changes written. Use --apply to integrate.")


if __name__ == "__main__":
    main()
