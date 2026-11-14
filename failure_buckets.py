import json, glob, os, sys, collections
S = sys.argv[1]
def path(r): m = f"results/eval_{r}_merged.json"; return m if os.path.exists(m) else f"results/eval_{r}.json"
ev = json.load(open(path(f"final_{S}_F"))); meta = {}; ok = {}
for v in ev["domains"].values():
    for dl in v["dialogues"]:
        ok[dl["uuid"]] = dl["score"] >= 0.99; meta[dl["uuid"]] = dl["details"]["per_turn"][0]["metadata"]
gold = {it["uuid"]: it["output"][0] for f in glob.glob(f"tmp_gt_{S}/*.json") for it in json.load(open(f))}
fb = set(json.load(open(f"results/merge_log_final_{S}_F.json"))["used_fallback"])
cat = collections.Counter(); ex = collections.defaultdict(list)
for f in glob.glob(f"output/final_{S}_F/*.json"):
    if f.endswith(("_tools.json", "_trajectory.json")): continue
    for r in json.load(open(f)):
        u = r["uuid"]
        if ok.get(u): continue
        g = list(dict.fromkeys(c["name"] for c in gold[u]["sequence"]["tool_call"]))
        calls = [c["name"] for c in r["output"][0]["sequence"]["tool_call"]]; called = [x for x in g if x in calls]; m = meta[u]
        src = "answer from fallback C" if u in fb else "answer from D"
        k = ("correct content, failed the groundedness check" if (m.get("exactmatch_score") == 1.0 or m.get("answer_score") == 1.0) and m.get("groundedness_score") == 0.0
             else "no tool calls" if not calls else "called every gold tool, still wrong" if len(called) == len(g)
             else "called some gold tools" if called else "called none of the gold tools")
        cat[(src, k)] += 1
        if len(ex[(src, k)]) < 2: ex[(src, k)].append((r["domain"], u))
print(f"{S}: F wrong on {sum(cat.values())} of {len(ok)}")
for (src, k), n in sorted(cat.items(), key=lambda x: -x[1]): print(f"  {n:3d}  {src:24s} {k:48s} e.g. {ex[(src, k)][:1]}")
