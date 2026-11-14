import json, os, itertools
def path(run):
    m = f"results/eval_{run}_merged.json"; return m if os.path.exists(m) else f"results/eval_{run}.json"
RUNS = [("baseline k=20", "results/eval_tune_base.json"), ("baseline + grounded answer", path("tune68_baseline_k20_ground")),
        ("full agent run 1", "results/eval_tune_plan_merged.json"), ("full agent run 2", path("tune68_plan_rep2")), ("full agent run 3", path("tune68_plan_rep3")),
        ("hybrid ReAct", path("tune68_hyb_base")), ("hybrid ReAct + grounded", path("tune68_hyb_base_ground")),
        ("hybrid full agent", path("tune68_hyb_plan")), ("hybrid full agent, STEP_RETRIEVAL=0", path("tune68_hyb_plan_nostep")),
        ("abl: VERIFY=0", path("tune68_abl_noverify")), ("abl: STEP_RETRIEVAL=0", path("tune68_abl_nostepretr")),
        ("abl: GROUND_PROMPT=0", path("tune68_abl_noground")), ("abl: RECORD_FAILED_RETRIES=0", path("tune68_abl_norecord"))]
D = {}
def load(p):
    ev = json.load(open(p)); out = {}
    for v in ev["domains"].values():
        for dl in v["dialogues"]:
            ex = dl["details"]["per_turn"][0]["metadata"].get("score_explanation") or {}
            bad = any("Judge Error" in str(ex.get(k) or "") for k in ("answer", "groundedness"))
            out[dl["uuid"]] = (dl["score"] >= 0.99, bad)
    return out
print(f"{'run':40s} correct  unjudged")
for name, p in RUNS:
    if not os.path.exists(p): print(f"{name:40s} (not scored yet)"); continue
    D[name] = load(p); d = D[name]
    print(f"{name:40s} {sum(c for c, b in d.values()):3d}/{len(d)}   {sum(b for c, b in d.values())}" + ("   <- do not use" if any(b for c, b in d.values()) else ""))
def pair(a, b):
    if a not in D or b not in D: return
    s = [u for u in D[a] if u in D[b] and not D[a][u][1] and not D[b][u][1]]
    x, y = sum(D[a][u][0] for u in s), sum(D[b][u][0] for u in s); both = sum(D[a][u][0] and D[b][u][0] for u in s)
    print(f"{a:34s} vs {b:36s} n={len(s)}: {x:2d} vs {y:2d} | both {both:2d}, only first {x-both:2d}, only second {y-both:2d}")
print("\nPaired comparisons (only questions judged in both):")
for a, b in [("baseline k=20", "hybrid ReAct"), ("baseline + grounded answer", "hybrid ReAct + grounded"), ("baseline k=20", "baseline + grounded answer"),
             ("baseline + grounded answer", "full agent run 1"), ("full agent run 1", "hybrid full agent"), ("full agent run 2", "hybrid full agent"),
             ("full agent run 3", "hybrid full agent"), ("hybrid full agent", "hybrid full agent, STEP_RETRIEVAL=0"), ("hybrid ReAct + grounded", "hybrid full agent")]: pair(a, b)
print("\nNoise among the three identical full-agent runs:")
for a, b in itertools.combinations(["full agent run 1", "full agent run 2", "full agent run 3"], 2): pair(a, b)
tot = [sum(c for c, b in D[n].values()) for n in ("full agent run 1", "full agent run 2", "full agent run 3") if n in D]
if len(tot) == 3: print("full agent totals:", tot, "mean", round(sum(tot) / 3, 1), "range", min(tot), "-", max(tot))
