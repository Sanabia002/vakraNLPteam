#!/usr/bin/env python3
"""Numbers for the report. Usage: python report_numbers.py [results/eval_X.json]"""
import csv, json, sys, glob, collections
rec = list(csv.DictReader(open("results/shortlist_recall_cap3.csv")))
dev = set(json.load(open("splits/dev_split.json"))["capability_3"]["dev_domains"])
def allgold(r, k): return all(float(g.rsplit(":", 1)[1]) <= k for g in r["gold_ranks"].split(";"))
out = []
def p(s=""): print(s); out.append(s)

p("RETRIEVAL: questions with ALL gold tools in the baseline top-k (training-set questions, gold-tool recall)")
for name, rows in (("all", rec), ("dev domains", [r for r in rec if r["domain"] in dev])):
    p(f"  {name} (n={len(rows)}): " + "  ".join(f"k={k}: {sum(allgold(r, k) for r in rows)}/{len(rows)}" for k in (5, 10, 20, 40, 80)))
by = collections.defaultdict(list)
for r in rec: by[r["hops"]].append(r)
for h in sorted(by):
    p(f"  gold calls={h} (n={len(by[h])}): " + "  ".join(f"k={k}: {sum(allgold(r, k) for r in by[h])}/{len(by[h])}" for k in (10, 20, 40)))

p("\nVALUE RECOVERABILITY: gold string arguments, by where the value comes from")
c = collections.Counter(); ex = collections.defaultdict(list)
for f in glob.glob("data/train/capability_3_multihop_reasoning/output/*.json"):
    for it in json.load(open(f)):
        for t in it["output"]:
            q, seq = t["query"], t["sequence"]
            for i, call in enumerate(seq["tool_call"]):
                prior = json.dumps(seq.get("tool_response", [])[:i])
                for k, v in call["arguments"].items():
                    if not isinstance(v, str) or v.isdigit(): continue
                    kind = ("verbatim in question" if v in q else "from an earlier tool result" if v in prior
                            else "differs from question only by case" if v.lower() in q.lower()
                            else "not recoverable from question or earlier results")
                    c[kind] += 1
                    if len(ex[kind]) < 5: ex[kind].append(v)
tot = sum(c.values())
for k, n in c.most_common(): p(f"  {n} ({100*n/tot:.0f}%) {k}  e.g. {ex[k]}")

if len(sys.argv) > 1:
    ev = json.load(open(sys.argv[1]))
    p(f"\nEVALUATOR: {sys.argv[1]}")
    p("  summary: " + json.dumps(ev["summary"]))
    sc = {}
    for d, v in ev["domains"].items():
        s = v["summary"]
        p(f"  {d}: correct {s['num_correct']}/{s['num_samples']}  missing predictions: {len(v['missing_prediction_uuids'])}")
        for dl in v["dialogues"]: sc[dl["uuid"]] = dl["score"]
    p("\nRETRIEVAL vs MODEL (k=20): evaluator-correct by whether ALL gold tools were in the top 20")
    byu = {r["uuid"]: r for r in rec}
    tab = collections.Counter()
    for u, s in sc.items():
        if u in byu: tab[(allgold(byu[u], 20), s >= 0.99)] += 1
    for gold in (True, False):
        n = tab[(gold, True)] + tab[(gold, False)]
        p(f"  all gold in top-20 = {gold}: correct {tab[(gold, True)]}/{n}")
open("results/report_numbers.txt", "w").write("\n".join(out) + "\n")
print("\nwrote results/report_numbers.txt")
