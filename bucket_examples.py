import json, glob, os, sys, random, collections
S = sys.argv[1]; random.seed(0)
def path(r): m = f"results/eval_{r}_merged.json"; return m if os.path.exists(m) else f"results/eval_{r}.json"
ev = json.load(open(path(f"final_{S}_F"))); ok, meta = {}, {}
for v in ev["domains"].values():
    for dl in v["dialogues"]: ok[dl["uuid"]] = dl["score"] >= 0.99; meta[dl["uuid"]] = dl["details"]["per_turn"][0]["metadata"]
gold = {it["uuid"]: it["output"][0] for f in glob.glob(f"tmp_gt_{S}/*.json") for it in json.load(open(f))}
fb = set(json.load(open(f"results/merge_log_final_{S}_F.json"))["used_fallback"]); B = collections.defaultdict(list)
for f in glob.glob(f"output/final_{S}_F/*.json"):
    if f.endswith(("_tools.json", "_trajectory.json")): continue
    for r in json.load(open(f)):
        u = r["uuid"]
        if ok.get(u): continue
        g = list(dict.fromkeys(c["name"] for c in gold[u]["sequence"]["tool_call"])); calls = [c["name"] for c in r["output"][0]["sequence"]["tool_call"]]; m = meta[u]; called = [x for x in g if x in calls]
        k = ("groundedness artifact" if (m.get("exactmatch_score") == 1.0 or m.get("answer_score") == 1.0) and m.get("groundedness_score") == 0.0 else "no tool calls" if not calls
             else "every gold tool called" if len(called) == len(g) else "some gold tools" if called else "no gold tool")
        B[("fallback C" if u in fb else "D", k)].append((r["domain"], u, r["output"][0]["query"][:110], g, calls, str(r["output"][0].get("answer"))[:60], json.dumps(gold[u]["answer"])[:60]))
for key, items in sorted(B.items(), key=lambda x: -len(x[1])):
    random.shuffle(items); seen = set(); shown = 0
    print(f"\n== {key[0]} / {key[1]} ({len(items)})")
    for it in items:
        if it[0] in seen and shown < len(items) - 1: continue
        seen.add(it[0]); shown += 1; print(f"  {it[0]} {it[1]}\n    Q: {it[2]}\n    gold: {it[3]}\n    called: {it[4]}\n    answer: {it[5]} | gold answer: {it[6]}")
        if shown == 2: break
