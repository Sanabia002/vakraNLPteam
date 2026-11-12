import json, glob, os
C = [("A: ReAct+hybrid+grounded k=30, qwen3.5-4b", "pool191_hyb_base_ground_k30"), ("D: same, qwen3.8-27b on Groq", "pool191_groq27"),
     ("E: A with plan-agent fallback on errors", "pool191_E"), ("F: D with plan-agent fallback on errors", "pool191_F")]
RUN = dict(C); ORDER = [n for n, _ in C]                    # tie order: A, D, E, F
def path(r):
    m = f"results/eval_{r}_merged.json"; return m if os.path.exists(m) else f"results/eval_{r}.json"
def load(r):
    o = {}
    for dom, v in json.load(open(path(r)))["domains"].items():
        for dl in v["dialogues"]:
            ex = dl["details"]["per_turn"][0]["metadata"].get("score_explanation") or {}
            o[dl["uuid"]] = (dl["score"] >= 0.99, any("Judge Error" in str(ex.get(k) or "") for k in ("answer", "groundedness")), dom)
    return o
def secs(r):
    d = [x["duration_s"] for f in glob.glob(f"output/{r}/*.json") if not f.endswith(("_tools.json", "_trajectory.json")) for x in json.load(open(f))]
    return sum(d) / len(d)
D = {}
for n, r in C:
    if not os.path.exists(path(r)): print(f"{n}: NOT SCORED"); continue
    D[n] = load(r); d = D[n]; bad = sum(b for c, b, _ in d.values())
    print(f"{n:44s} correct {sum(c for c, b, _ in d.values()):3d}/{len(d)}  unjudged {bad}  mean {secs(r):5.1f} s/question" + ("   <- NOT USABLE" if bad or len(d) != 191 else ""))
ok = [n for n in ORDER if n in D and not any(b for c, b, _ in D[n].values()) and len(D[n]) == 191]
print("\nPaired flips:")
for i, a in enumerate(ok):
    for b in ok[i + 1:]:
        both = sum(D[a][u][0] and D[b][u][0] for u in D[a]); x = sum(D[a][u][0] for u in D[a]); y = sum(D[b][u][0] for u in D[a])
        print(f"  {a[:1]} vs {b[:1]}: both {both:3d} | only {a[:1]} {x-both:3d} | only {b[:1]} {y-both:3d}")
if ok:
    tot = {n: sum(c for c, b, _ in D[n].values()) for n in ok}; best = max(tot.values())
    near = [n for n in ok if best - tot[n] <= 3]
    print(f"\nBest {best}. Within 3 of the best: {[n[:1] for n in near]}\nPRE-DECLARED RULE PICKS: {near[0]}")
