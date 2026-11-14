import json, glob, os
C = [("A: ReAct+hybrid+grounded k=30", "pool191_hyb_base_ground_k30"),
     ("B: ReAct+hybrid+grounded k=20", "pool191_hyb_base_ground_k20"),
     ("C: hybrid plan agent", "pool191_hyb_plan")]
RUN = dict(C)
def path(r):
    m = f"results/eval_{r}_merged.json"
    return m if os.path.exists(m) else f"results/eval_{r}.json"
def load(r):
    ev = json.load(open(path(r))); o = {}
    for dom, v in ev["domains"].items():
        for dl in v["dialogues"]:
            ex = dl["details"]["per_turn"][0]["metadata"].get("score_explanation") or {}
            bad = any("Judge Error" in str(ex.get(k) or "") for k in ("answer", "groundedness"))
            o[dl["uuid"]] = (dl["score"] >= 0.99, bad, dom)
    return o
def secs(r):
    d = [x["duration_s"] for f in glob.glob(f"output/{r}/*.json") if not f.endswith(("_tools.json", "_trajectory.json")) for x in json.load(open(f))]
    return sum(d) / len(d)
D = {}
for n, r in C:
    if not os.path.exists(path(r)): print(f"{n}: NOT SCORED"); continue
    D[n] = load(r); d = D[n]; bad = sum(b for c, b, _ in d.values())
    print(f"{n:34s} correct {sum(c for c, b, _ in d.values()):3d}/{len(d)}  unjudged {bad}  mean {secs(r):5.1f} s/question" + ("   <- NOT USABLE" if bad else ""))
usable = [n for n in D if not any(b for c, b, _ in D[n].values())]
print("\nPaired flips (questions right in both / only first / only second):")
for i, a in enumerate(usable):
    for b in usable[i + 1:]:
        both = sum(D[a][u][0] and D[b][u][0] for u in D[a]); x = sum(D[a][u][0] for u in D[a]); y = sum(D[b][u][0] for u in D[a])
        print(f"  {a[:1]} vs {b[:1]}: both {both:3d} | only {a[:1]} {x - both:3d} | only {b[:1]} {y - both:3d} | either {x + y - both}")
if usable:
    tot = {n: sum(c for c, b, _ in D[n].values()) for n in usable}; best = max(tot.values())
    near = [n for n in usable if best - tot[n] <= 3]
    pick = min(near, key=lambda n: secs(RUN[n]))
    print(f"\nBest score: {best}. Within 3 of the best: {near}")
    print("PRE-DECLARED RULE PICKS:", pick)
    print("\nBy domain (correct / questions) for each usable candidate:")
    doms = sorted({v[2] for v in D[usable[0]].values()})
    print(f"  {'domain':30s}" + "".join(f"{n[:1]:>9s}" for n in usable))
    for dom in doms:
        row = []
        for n in usable:
            qs = [u for u, v in D[n].items() if v[2] == dom]; row.append(f"{sum(D[n][u][0] for u in qs)}/{len(qs)}")
        print(f"  {dom:30s}" + "".join(f"{x:>9s}" for x in row))
