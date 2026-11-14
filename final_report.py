#!/usr/bin/env python3
"""python final_report.py dev|test  -> baseline vs F on one evaluation set, from results/eval_final_<set>_{baseline,F}.json
(merged files after re-judging are used when present). Needs tmp_gt_<set>/ for the gold-call counts."""
import glob, json, os, sys
S = sys.argv[1]
def path(r):
    m = f"results/eval_{r}_merged.json"
    return m if os.path.exists(m) else f"results/eval_{r}.json"
def load(r):
    o = {}
    for dom, v in json.load(open(path(r)))["domains"].items():
        for dl in v["dialogues"]:
            ex = dl["details"]["per_turn"][0]["metadata"].get("score_explanation") or {}
            bad = any("Judge Error" in str(ex.get(k) or "") for k in ("answer", "groundedness"))
            o[dl["uuid"]] = (dl["score"] >= 0.99, bad, dom)
    return o
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))  # clamp float noise (0/10 gave -2.8e-15 -> "-0")
hops = {it["uuid"]: len(it["output"][0]["sequence"]["tool_call"])
        for f in glob.glob(f"tmp_gt_{S}/*.json") for it in json.load(open(f))}
R = {n: load(f"final_{S}_{n}") for n in ("baseline", "F")}
for n, d in R.items():
    k = sum(c for c, b, _ in d.values()); bad = sum(b for c, b, _ in d.values())
    if not d:
        print(f"{S} {n:9s} 0/0: no questions in the results file"); continue
    lo, hi = wilson(k, len(d))
    print(f"{S} {n:9s} {k}/{len(d)} = {100*k/len(d):.1f}% (95% interval {lo:.0f}-{hi:.0f}%) unjudged {bad}" + ("   <-- NOT USABLE: re-judge first" if bad else ""))
B, F = R["baseline"], R["F"]
common = [u for u in B if u in F]  # paired numbers and breakdowns only over questions both runs have
if set(B) != set(F):
    print("WARNING: the two runs do not cover the same questions:", len(set(B) ^ set(F)), "differ;",
          f"paired numbers below use the {len(common)} questions in both")
both = sum(B[u][0] and F[u][0] for u in common)
print(f"paired: both {both} | only baseline {sum(B[u][0] and not F[u][0] for u in common)}"
      f" | only F {sum(F[u][0] and not B[u][0] for u in common)}")
for h in sorted(set(hops.values())):
    us = [u for u in common if hops.get(u) == h]
    print(f"  gold calls = {h}: n={len(us)} baseline {sum(B[u][0] for u in us)} F {sum(F[u][0] for u in us)}")
print("by domain (baseline / F / n):")
for dom in sorted({B[u][2] for u in common}):
    us = [u for u in common if B[u][2] == dom]
    print(f"  {dom:26s} {sum(B[u][0] for u in us):2d} / {sum(F[u][0] for u in us):2d} / {len(us)}")
