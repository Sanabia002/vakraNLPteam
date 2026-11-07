#!/usr/bin/env python3
"""Show one or more items from a benchmark run next to the gold answer.

  python show_item.py RUN UUID [UUID ...]      RUN = output/<name> or <name>

Reads output/RUN/<domain>.json, <domain>_tools.json and, if the run used SAVE_TRAJECTORY=1,
<domain>_trajectory.json. Gold comes from data/train/capability_3_multihop_reasoning/output/<domain>.json.
Never reads *.log. Frozen test domains are refused (their files are never opened); dev domains
are refused unless --allow-dev.
"""
import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import analysis_guard as guard

GOLD_DIR = "data/train/capability_3_multihop_reasoning/output"
CUT = 200


def cut(x: Any, n: int = CUT) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n})"


def run_dir(run: str) -> str:
    return run if os.path.isdir(run) else os.path.join("output", run)


def domain_files(rdir: str) -> List[str]:
    """Domain names in a run dir, from <domain>.json (not the _tools/_trajectory side files)."""
    out = []
    for f in sorted(glob.glob(os.path.join(rdir, "*.json"))):
        name = os.path.basename(f)[:-5]
        if not name.endswith(("_tools", "_trajectory")):
            out.append(name)
    return out


def by_uuid(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    return {r["uuid"]: r for r in json.load(open(path))}


def find_items(rdir: str, uuids: List[str], dev: set, allow_dev: bool) -> Dict[str, tuple]:
    """uuid -> (domain, record). Frozen/dev-disallowed domain files are skipped without opening."""
    want, found, skipped = set(uuids), {}, []
    for dom in domain_files(rdir):
        if not guard.domain_allowed(dom, dev, allow_dev):
            skipped.append(dom)
            continue
        for u, rec in by_uuid(os.path.join(rdir, dom + ".json")).items():
            if u in want:
                found[u] = (dom, rec)
    if skipped:
        print(f"(skipped unopened domain files: {', '.join(skipped)})")
    return found


def offered_from_trajectory(traj: List[dict]) -> List[str]:
    out: List[str] = []
    for e in traj:
        for n in e.get("offered_tools") or []:
            if n not in out:
                out.append(n)
    return out


def format_trajectory(traj: List[dict]) -> List[str]:
    lines = []
    plan = next((e for e in traj if "plan" in e), None)
    if plan is not None:
        lines.append("PLAN:")
        lines += [f"  {i + 1}. {s}" for i, s in enumerate(plan["plan"])]
    calls = {}  # tool_call_id -> args, from AIMessage tool_calls
    for e in traj:
        if e.get("type") == "AIMessage" and "offered_tools" in e:
            lines.append(f"STEP {e.get('step_index', 0) + 1}: {e.get('step', '')}")
            lines.append(f"  offered: {', '.join(e['offered_tools'])}")
            if e.get("no_tool_call"):
                lines.append(f"  no tool call: {cut(e.get('content', ''))}")
        elif e.get("type") == "AIMessage" and e.get("tool_calls"):
            for tc in e["tool_calls"]:
                calls[tc.get("id")] = tc.get("args", {})
                if "attempt" not in e:  # non-plan agents: show the call here
                    lines.append(f"  CALL {tc.get('name')}({cut(tc.get('args', {}))})")
        elif e.get("type") == "ToolMessage":
            res = e.get("result", e.get("content", ""))
            if "attempt" in e:
                mark = " KEPT" if e.get("kept") else ""
                stop = f" retry_stopped={e['retry_stopped']}" if e.get("retry_stopped") else ""
                lines.append(f"  [{e['attempt']}] {e.get('tool_name')}({cut(calls.get(e.get('tool_call_id'), {}))})"
                             f" verdict={e.get('verdict') or 'ok'}{mark}{stop}")
                lines.append(f"      -> {cut(res)}")
            else:
                lines.append(f"      -> {e.get('tool_name')}: {cut(res)}")
        elif e.get("final"):
            lines.append(f"FINAL: stop_reason={e.get('stop_reason')} fallback={e.get('final_fallback')}"
                         f" executions={e.get('n_executions')} elapsed_s={e.get('elapsed_s')}")
    return lines


def format_item(uuid: str, dom: str, rec: dict, tools: Optional[dict], traj: Optional[dict],
                gold: Optional[dict]) -> str:
    out = rec["output"][0]
    L = [f"=== {uuid}  [{dom}]",
         f"Q: {out.get('query', '')}",
         f"status: {rec.get('status')}" + (f"  error: {cut(rec['error'])}" if rec.get("error") else "")
         + f"  duration_s: {rec.get('duration_s')}",
         f"ANSWER: {cut(out.get('answer', ''), 500)}"]

    if traj and traj.get("trajectory"):
        L += format_trajectory(traj["trajectory"])
        offered = offered_from_trajectory(traj["trajectory"])
        offered_src = "per-step offered tools (trajectory)"
    else:
        L.append("TOOL CALLS (results/plan/verdicts not saved for this run; rerun with SAVE_TRAJECTORY=1):")
        for tc in out.get("sequence", {}).get("tool_call", []):
            L.append(f"  {tc.get('name')}({cut(tc.get('arguments', {}))})")
        offered = []
        offered_src = "none"
    if not offered and tools:
        offered = tools.get("shortlisted_tools") or tools.get("all_tools") or []
        offered_src = ("union of offered tools (_tools.json shortlisted_tools)" if tools.get("shortlisted_tools")
                       else "all_tools (_tools.json; shortlisted_tools empty)")

    if gold is None:
        L.append("GOLD: not found")
        return "\n".join(L)
    g = gold["output"][0]
    gcalls = g.get("sequence", {}).get("tool_call", [])
    resp = g.get("sequence", {}).get("tool_response", [])
    L.append(f"GOLD ANSWER: {cut(g.get('answer'), 500)}")
    L.append(f"GOLD CALLS ({len(gcalls)}):")
    for i, tc in enumerate(gcalls):
        L.append(f"  {tc.get('name')}({cut(tc.get('arguments', {}))})"
                 + (f"  -> {cut(resp[i])}" if i < len(resp) else ""))
    called = {tc.get("name") for tc in out.get("sequence", {}).get("tool_call", [])}
    gold_names = list(dict.fromkeys(tc.get("name") for tc in gcalls))
    missing = [n for n in gold_names if n not in offered]
    L.append(f"GOLD TOOLS NOT OFFERED [{offered_src}]: {', '.join(missing) if missing else '(none)'}")
    L.append("GOLD TOOLS CALLED: " + ", ".join(f"{n}={'yes' if n in called else 'no'}" for n in gold_names))
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("uuids", nargs="+")
    p.add_argument("--gold-dir", default=GOLD_DIR)
    p.add_argument("--split", default=guard.DEFAULT_SPLIT)
    p.add_argument("--allow-dev", action="store_true", help="only when explicitly asked")
    a = p.parse_args(argv)

    guard.check([], a.uuids, set(), True)  # seen uuids
    dev = guard.dev_domains(a.split)
    rdir = run_dir(a.run)
    if not os.path.isdir(rdir):
        sys.exit(f"no run dir {rdir}")
    found = find_items(rdir, a.uuids, dev, a.allow_dev)
    for u in a.uuids:
        if u not in found:
            print(f"=== {u}: not found in {rdir} (or in a refused domain)\n")
            continue
        dom, rec = found[u]
        guard.check([dom], [u], dev, a.allow_dev)
        tools = by_uuid(os.path.join(rdir, f"{dom}_tools.json")).get(u)
        traj = by_uuid(os.path.join(rdir, f"{dom}_trajectory.json")).get(u)
        gold = by_uuid(os.path.join(a.gold_dir, f"{dom}.json")).get(u)
        print(format_item(u, dom, rec, tools, traj, gold) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
