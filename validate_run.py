import json, glob, sys
run, n = sys.argv[1], int(sys.argv[2])
frozen = {"bike_share_1","computer_student","disney","olympics","restaurant","student_loan","hockey"}
sample = set(json.load(open("results/tune_sample_uuids.json")))
uu, doms, bad, ok_items, empty, nocall = set(), set(), 0, 0, 0, 0
for f in glob.glob(f"output/{run}/*.json"):
    if f.endswith("_tools.json") or f.endswith("_trajectory.json"): continue
    for r in json.load(open(f)):
        uu.add(r["uuid"]); doms.add(r["domain"]); e = str(r.get("error", ""))
        if r["status"] != "success":
            if "Recursion" not in e and "timed out" not in e: bad += 1
            continue
        o = r["output"][0]; ok_items += 1
        empty += not str(o.get("answer") or "").strip(); nocall += not o["sequence"]["tool_call"]
p = []
if len(uu) != n or uu != sample: p.append(f"items {len(uu)} or sample mismatch")
if doms & frozen: p.append(f"frozen domains {sorted(doms & frozen)}")
if bad > 3: p.append(f"{bad} errors other than step limit/timeout")
if ok_items and empty / ok_items > 0.3: p.append(f"{empty}/{ok_items} successful items with empty answers")
if ok_items and nocall / ok_items > 0.3: p.append(f"{nocall}/{ok_items} successful items with no tool calls")
print("OK" if not p else "PROBLEM: " + "; ".join(p))
