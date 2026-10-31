#!/usr/bin/env python3
"""Gold-tool recall of the baseline ToolShortlister on capability 3 (CPU only)."""
import collections, csv, glob, json, os, re, sys
sys.path.insert(0, os.getcwd())
from agents.components.tool_shortlister import ToolShortlister

CAP = "capability_3_multihop_reasoning"
KS = [5, 10, 20, 40, 80]
INF = float("inf")

def load_tools(path):
    lines = open(path).read().splitlines()
    tools = []
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*\[\s*\d+\]\s+(\S+)\s*$", ln)
        if m:
            desc = lines[i + 1].strip() if i + 1 < len(lines) else ""
            tools.append({"name": m.group(1), "description": desc})
    return tools

sl = ToolShortlister(top_k=1)
stats = collections.defaultdict(lambda: collections.defaultdict(list))
rows, skipped, unknown_gold = [], [], 0

for gf in sorted(glob.glob(f"data/train/{CAP}/output/*.json")):
    domain = os.path.basename(gf)[:-5]
    tf = f"tool_dumps/{domain}.txt"
    tools = load_tools(tf) if os.path.exists(tf) else []
    if not tools:
        skipped.append(domain)
        continue
    names = {t["name"] for t in tools}
    n = len(tools)
    sl.encode_tools(tools)
    sl.top_k = max(n - 1, 1)  # forces ranking; the one unreturned tool gets rank n
    for item in json.load(open(gf)):
        for turn in item["output"]:
            gold = [c["name"] for c in turn["sequence"]["tool_call"]]
            if not gold:
                continue
            ranked = [t["name"] for t in sl.shortlist(turn["query"], tools)]
            rank = {nm: i + 1 for i, nm in enumerate(ranked)}
            ugold = list(dict.fromkeys(gold))
            ranks = []
            for g in ugold:
                if g not in names:
                    unknown_gold += 1
                    ranks.append(INF)
                else:
                    ranks.append(rank.get(g, n))
            hops = len(gold)
            for k in KS:
                hit = [r <= k for r in ranks]
                stats[hops][k].append((sum(hit) / len(hit), all(hit)))
                stats["all"][k].append((sum(hit) / len(hit), all(hit)))
            rows.append({"uuid": item["uuid"], "domain": domain, "hops": hops, "n_tools": n,
                         "gold_ranks": ";".join(f"{g}:{r}" for g, r in zip(ugold, ranks)),
                         "query": turn["query"][:100]})

if not rows:
    sys.exit(f"No questions processed. skipped domains: {skipped}")
os.makedirs("results", exist_ok=True)
with open("results/shortlist_recall_cap3.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"questions: {len(rows)}; skipped domains: {skipped}; gold tools not in tool list: {unknown_gold}")
print("cell = mean per-tool recall % / % of questions with ALL gold tools present")
for h in sorted([x for x in stats if x != "all"]) + ["all"]:
    cells = []
    for k in KS:
        v = stats[h][k]
        cells.append(f"k={k}: {100*sum(a for a,_ in v)/len(v):.0f}/{100*sum(b for _,b in v)/len(v):.0f}")
    print(f"hops={h} (n={len(stats[h][KS[0]])}): " + "  ".join(cells))
