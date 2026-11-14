#!/usr/bin/env python3
"""Do planner plans contain values the question never mentions? Tune sample only (tmp_gt_tune + a saved-trajectory run)."""
import collections, glob, json, os, re, sys
RUN = sys.argv[1] if len(sys.argv) > 1 else "tune68_hyb_plan"
FROZEN = {"bike_share_1","computer_student","disney","olympics","restaurant","student_loan","hockey"}
gold = {}
for f in glob.glob("tmp_gt_tune/*.json"):
    for it in json.load(open(f)):
        t = it["output"][0]; seq = t["sequence"]
        gold[it["uuid"]] = {"q": t["query"], "n": len(seq["tool_call"]), "chain": json.dumps([seq["tool_call"], seq.get("tool_response", [])])}
plans = {}
for f in glob.glob(f"output/{RUN}/*_trajectory.json"):
    for item in json.load(open(f)):
        if item["domain"] in FROZEN: continue
        for e in item["trajectory"]:
            if isinstance(e, dict) and "plan" in e: plans[item["uuid"]] = e["plan"]; break
DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
def extract(text):
    vals = [("date", m.group(0), (int(m.group(1)), int(m.group(2)), int(m.group(3)))) for m in DATE.finditer(text)]
    rest = DATE.sub(" ", text)
    for m in re.finditer(r'"([^"]{2,60})"', rest): vals.append(("quoted", m.group(1), None))
    rest = re.sub(r'"[^"]*"', " ", rest)
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", rest):
        s = m.group(1); v = float(s)
        vals.append(("year" if re.fullmatch(r"(19|20)\d{2}", s) else "small number" if v <= 10 else "number", s, v))
    return vals
def in_question(kind, s, v, q):
    nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", q)}
    if kind == "date": return all(float(x) in nums for x in v)
    if kind == "quoted": return s.lower() in q.lower()
    return v in nums
def in_chain(kind, s, v, chain):
    if kind == "date": return s in chain or f"{v[0]}-{v[1]:02d}-{v[2]:02d}" in chain
    if kind == "quoted": return s.lower() in chain.lower()
    return re.search(r"(?<![\d.])" + re.escape(s) + r"(?!\d)", chain) is not None
cat, ex, flag = collections.Counter(), [], {}
for u, plan in plans.items():
    if u not in gold: continue
    text = " ".join(plan) if isinstance(plan, list) else str(plan); q = gold[u]["q"]; bad = False
    for kind, s, v in extract(text):
        c = "in question" if in_question(kind, s, v, q) else "not in question, but in gold chain" if in_chain(kind, s, v, gold[u]["chain"]) else "NOT in question or gold chain"
        cat[(kind, c)] += 1
        if c.startswith("NOT") and kind != "small number":
            bad = True
            if len(ex) < 12: ex.append((kind, s, q[:75]))
    flag[u] = bad
print(f"plans read: {len(plans)} (run {RUN}); questions with a gold entry: {len(flag)}")
for kind in ("date", "year", "number", "small number", "quoted"):
    row = {c: n for (k, c), n in cat.items() if k == kind}
    if row: print(f"  {kind:13s} {row}")
n_bad = sum(flag.values())
print(f"\nplans with at least one likely-invented value (excluding numbers <= 10): {n_bad} of {len(flag)}")
print("examples:", *ex, sep="\n   ")
evp = f"results/eval_{RUN}.json"
if os.path.exists(evp):
    sc = {dl["uuid"]: dl["score"] >= 0.99 for v in json.load(open(evp))["domains"].values() for dl in v["dialogues"]}
    for lab, want in (("with a likely-invented value", True), ("without", False)):
        us = [u for u in flag if flag[u] == want and u in sc]
        print(f"correct among plans {lab}: {sum(sc[u] for u in us)}/{len(us)}")
pl = collections.Counter((gold[u]["n"], len(plans[u]) if isinstance(plans[u], list) else 0) for u in flag)
print("\n(gold calls, planned steps): count ->", dict(sorted(pl.items())))
