#!/usr/bin/env python3
"""python failure_sheet.py RUN EVAL_JSON -> results/failures_RUN.csv (tune data only; no model, no containers)."""
import csv, glob, json, os, sys
run, evp = sys.argv[1], sys.argv[2]
ev = json.load(open(evp)); info = {}
for dom, v in ev["domains"].items():
    for dl in v["dialogues"]:
        m = dl["details"]["per_turn"][0]["metadata"]
        info[dl["uuid"]] = {"domain": dom, "score": dl["score"], "em": m.get("exactmatch_score"), "ans": m.get("answer_score"),
                            "gr": m.get("groundedness_score"), "query": dl["details"]["per_turn"][0]["query"]}
gold = {}
for f in glob.glob("tmp_gt_tune/*.json"):
    for it in json.load(open(f)):
        t = it["output"][0]; gold[it["uuid"]] = ([c["name"] for c in t["sequence"]["tool_call"]], t["answer"])
rows = []
for f in sorted(glob.glob(f"output/{run}/*.json")):
    if f.endswith("_tools.json"): continue
    tl = {t["uuid"]: t for t in json.load(open(f[:-5] + "_tools.json"))}
    for r in json.load(open(f)):
        u = r["uuid"]
        if u not in info or info[u]["score"] >= 0.99: continue
        o = r["output"][0]; calls = [c["name"] for c in o["sequence"]["tool_call"]]
        gtools, ganswer = gold[u]; offered = tl.get(u, {}).get("shortlisted_tools", [])
        missing = [g for g in dict.fromkeys(gtools) if g not in offered]
        err = str(r.get("error", "")); i = info[u]
        tag = ("step_limit" if "Recursion" in err else "timeout" if "timed out" in err else "no_tools_used" if not calls
               else "answer_ungrounded" if i["gr"] == 0.0 and (i["em"] == 1.0 or i["ans"] == 1.0)
               else "retrieval_missing_tool" if missing else "")
        rows.append({"uuid": u, "domain": i["domain"], "gold_calls": len(gtools), "agent_calls": len(calls), "status": r["status"],
                     "gold_tools_missing_from_offered": ";".join(missing), "auto_tag": tag, "tag": tag, "notes": "",
                     "query": i["query"][:110], "agent_answer": str(o.get("answer"))[:140].replace("\n", " "), "gold_answer": json.dumps(ganswer)[:100]})
os.makedirs("results", exist_ok=True)
out = f"results/failures_{run}.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from collections import Counter
print(f"{len(rows)} wrong items -> {out}")
print("auto tags:", dict(Counter(r["auto_tag"] or "(needs reading)" for r in rows)))
print("wrong items with a gold tool missing from the offered tools:", sum(bool(r["gold_tools_missing_from_offered"]) for r in rows))
print("\nthree-call questions that were wrong:")
for r in rows:
    if r["gold_calls"] >= 3: print(" ", r["domain"], r["uuid"], "| agent calls", r["agent_calls"], "| missing:", r["gold_tools_missing_from_offered"] or "-", "|", r["query"][:70])
