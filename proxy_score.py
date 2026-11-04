#!/usr/bin/env python3
"""Rough PROXY scorer for capability 3 runs. NOT the VAKRA metric."""
import argparse, collections, csv, glob, json, math, os, re

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--show", type=int, default=8, help="how many non-matching items to print in detail")
a = ap.parse_args()

GOLD = {}
for f in glob.glob("data/train/capability_3_multihop_reasoning/output/*.json"):
    for item in json.load(open(f)):
        GOLD[item["uuid"]] = item

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def flatten(x):
    if isinstance(x, (list, tuple)):
        for y in x: yield from flatten(y)
    elif isinstance(x, dict):
        for y in x.values(): yield from flatten(y)
    elif x is not None:
        yield x

def as_float(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        try: return float(v.replace(",", ""))
        except ValueError: return None
    return None

def text_numbers(t):
    out = []
    for m in NUM.findall(t):
        try: out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError: pass
    return out

def norm(s): return re.sub(r"\s+", " ", str(s).lower()).strip()

def found(v, text, nums):
    f = as_float(v)
    if f is not None:
        return any(math.isclose(f, n, rel_tol=0.005, abs_tol=0.01) for n in nums)
    return norm(v) in norm(text)

rows = []
for f in sorted(glob.glob(os.path.join(a.run_dir, "*.json"))):
    if f.endswith("_tools.json"): continue
    for r in json.load(open(f)):
        g = GOLD.get(r["uuid"])
        o = r["output"][0]
        calls = [c["name"] for c in o["sequence"]["tool_call"]]
        text = o.get("answer") or ""
        if not isinstance(text, str): text = json.dumps(text)
        gcalls = [c["name"] for c in g["output"][0]["sequence"]["tool_call"]] if g else []
        gvals = list(flatten(g["output"][0]["answer"])) if g else []
        nums = text_numbers(text)
        hits = [found(v, text, nums) for v in gvals]
        err = str(r.get("error", ""))
        if r["status"] != "success":
            cat = "step_limit" if "Recursion" in err else ("timeout" if "timed out" in err else "error_other")
        elif g is None: cat = "no_gold_entry"
        elif not calls: cat = "answered_without_tools"
        elif gvals and all(hits): cat = "proxy_match"
        elif any(hits): cat = "proxy_partial"
        else: cat = "proxy_mismatch"
        weak = bool(gvals) and all((as_float(v) is not None and abs(as_float(v)) <= 10) for v in gvals)
        rows.append({"domain": r["domain"], "gold_calls": len(gcalls), "agent_calls": len(calls),
                     "category": cat, "weak_match": weak, "gold_vals": len(gvals), "gold_hits": sum(hits),
                     "same_tool_sequence": calls == gcalls, "duration_s": round(r.get("duration_s", 0), 1),
                     "uuid": r["uuid"], "query": o["query"][:90], "agent_answer": text[:160].replace("\n", " "),
                     "gold_values": json.dumps(gvals[:6])[:120]})

if not rows: raise SystemExit("no items found in " + a.run_dir)
os.makedirs("results", exist_ok=True)
out = f"results/{os.path.basename(a.run_dir.rstrip('/'))}_proxy.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def table(title, key):
    print(f"\n{title}")
    c = collections.defaultdict(collections.Counter)
    for x in rows: c[x[key]][x["category"]] += 1
    for k in sorted(c, key=str):
        n = sum(c[k].values())
        print(f"  {k}: n={n}  " + ", ".join(f"{cat}={v}" for cat, v in sorted(c[k].items())))

print(f"{len(rows)} items from {a.run_dir}  (PROXY, not the official metric)")
print("by category:", dict(collections.Counter(x["category"] for x in rows)))
table("by domain", "domain")
table("by gold tool-call count", "gold_calls")
ok = [x for x in rows if x["category"] == "proxy_match"]
print(f"\nproxy_match: {len(ok)}/{len(rows)}  (weak matches among them: {sum(x['weak_match'] for x in ok)})")
bad = [x for x in rows if x["category"] not in ("proxy_match",)]
print(f"\nfirst {min(a.show, len(bad))} non-matching items:")
for x in bad[:a.show]:
    print(f"- [{x['category']}] {x['domain']} gold_calls={x['gold_calls']} agent_calls={x['agent_calls']}")
    print("   Q:", x["query"]); print("   gold:", x["gold_values"]); print("   agent:", x["agent_answer"])
print("\nwrote", out)
