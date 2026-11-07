#!/usr/bin/env python3
"""Paired flip table between two evaluator JSONs.

  python compare_runs.py EVAL_A EVAL_B [--threshold 1.0] [--allow-dev]

Pairs dialogues by uuid. An item is excluded if its score_explanation contains "Judge Error" in
either file. Correct = dialogue score >= threshold. Tables: overall, by gold-call count (gt_steps from
the evaluator metadata), by domain. Then lists flipped uuids with a query prefix.
Frozen test domains and seen uuids are refused; dev domains are refused unless --allow-dev.
"""
import argparse
import collections
import json
import sys
from typing import Dict, List

import analysis_guard as guard

PREFIX = 80


def load(path: str, dev: set = frozenset(), allow_dev: bool = False) -> Dict[str, dict]:
    """uuid -> {domain, score, gt_steps, query, judge_error}. Domains are checked before dialogues are read."""
    d = json.load(open(path))
    guard.check(d.get("domains", {}).keys(), [], dev, allow_dev)
    out = {}
    for dom, dm in d.get("domains", {}).items():
        for dl in dm.get("dialogues", []):
            turns = dl.get("details", {}).get("per_turn", []) or [{}]
            t0 = turns[0]
            je = any("Judge Error" in json.dumps(t.get("metadata", {}).get("score_explanation", ""), default=str)
                     for t in turns)
            out[dl["uuid"]] = {"domain": dom, "score": float(dl.get("score") or 0.0),
                               "gt_steps": t0.get("metadata", {}).get("gt_steps"),
                               "query": str(t0.get("query", "")), "judge_error": je}
    return out


def cell(a_ok: bool, b_ok: bool) -> str:
    return {(True, True): "both", (True, False): "A_only", (False, True): "B_only", (False, False): "neither"}[
        (a_ok, b_ok)]


def compare(A: Dict[str, dict], B: Dict[str, dict], threshold: float = 1.0) -> dict:
    paired = sorted(set(A) & set(B))
    excluded = [u for u in paired if A[u]["judge_error"] or B[u]["judge_error"]]
    kept = [u for u in paired if u not in set(excluded)]
    rows = []
    for u in kept:
        rows.append({"uuid": u, "domain": A[u]["domain"],
                     "gt_steps": A[u]["gt_steps"] if A[u]["gt_steps"] is not None else B[u]["gt_steps"],
                     "query": A[u]["query"] or B[u]["query"],
                     "cell": cell(A[u]["score"] >= threshold, B[u]["score"] >= threshold)})

    def table(key):
        t = collections.defaultdict(collections.Counter)
        for r in rows:
            t[key(r)][r["cell"]] += 1
        return dict(t)

    return {"rows": rows, "excluded": excluded,
            "only_a": sorted(set(A) - set(B)), "only_b": sorted(set(B) - set(A)),
            "overall": table(lambda r: "all").get("all", collections.Counter()),
            "by_gt_steps": table(lambda r: r["gt_steps"]), "by_domain": table(lambda r: r["domain"])}


def fmt_table(title: str, t: Dict, key_name: str) -> List[str]:
    cols = ["n", "both", "A_only", "B_only", "neither", "A_acc", "B_acc", "net(B-A)"]
    L = [f"\n{title}", f"  {key_name:<28}" + "".join(f"{c:>9}" for c in cols)]
    for k in sorted(t, key=lambda x: (x is None, str(x) if not isinstance(x, int) else f"{x:06d}")):
        c = t[k]
        n = sum(c.values())
        a, b = c["both"] + c["A_only"], c["both"] + c["B_only"]
        vals = [n, c["both"], c["A_only"], c["B_only"], c["neither"],
                f"{a / n:.3f}" if n else "-", f"{b / n:.3f}" if n else "-", f"{b - a:+d}"]
        L.append(f"  {str(k):<28}" + "".join(f"{v:>9}" for v in vals))
    return L


def report(res: dict, name_a: str, name_b: str) -> str:
    L = [f"A = {name_a}", f"B = {name_b}",
         f"paired kept: {len(res['rows'])}  excluded (Judge Error in A or B): {len(res['excluded'])}"
         f"  only in A: {len(res['only_a'])}  only in B: {len(res['only_b'])}"]
    L += fmt_table("OVERALL", {"all": res["overall"]}, "")
    L += fmt_table("BY GOLD-CALL COUNT (gt_steps)", res["by_gt_steps"], "gt_steps")
    L += fmt_table("BY DOMAIN", res["by_domain"], "domain")
    for label, c in (("A right -> B wrong (A_only)", "A_only"), ("A wrong -> B right (B_only)", "B_only")):
        flips = [r for r in res["rows"] if r["cell"] == c]
        L.append(f"\n{label}: {len(flips)}")
        L += [f"  {r['uuid']}  [{r['domain']}, gt={r['gt_steps']}]  {r['query'][:PREFIX]}" for r in flips]
    for label, us in (("EXCLUDED (Judge Error)", res["excluded"]), ("ONLY IN A", res["only_a"]),
                      ("ONLY IN B", res["only_b"])):
        if us:
            L.append(f"\n{label}: " + ", ".join(us))
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_a")
    p.add_argument("eval_b")
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--split", default=guard.DEFAULT_SPLIT)
    p.add_argument("--allow-dev", action="store_true", help="only when explicitly asked")
    a = p.parse_args(argv)
    dev = guard.dev_domains(a.split)
    A, B = load(a.eval_a, dev, a.allow_dev), load(a.eval_b, dev, a.allow_dev)
    for X in (A, B):
        guard.check([], X.keys(), dev, a.allow_dev)  # seen uuids
    print(report(compare(A, B, a.threshold), a.eval_a, a.eval_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
