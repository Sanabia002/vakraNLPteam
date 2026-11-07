#!/usr/bin/env python3
"""Offline: does LLM-planned, per-step tool retrieval cover more gold tools than one-shot retrieval
at the SAME tool budget? TUNE domains only (non-dev). Needs llama-server running and .env loaded."""
import argparse, csv, glob, json, os, random, re
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from frozen_split import FROZEN_TEST, SEEN_UUIDS, assert_no_frozen

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["q", "qnames"], default="q", help="qnames also shows the planner the operation names")
ap.add_argument("--n2", type=int, default=60, help="number of 2-call questions to sample")
ap.add_argument("--m", type=int, default=5, help="tools retrieved per step")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="qwen3.5-4b")
a = ap.parse_args()

CAP = "capability_3_multihop_reasoning"
dev = set(json.load(open("splits/dev_split.json"))["capability_3"]["dev_domains"])
EXCLUDED = dev | FROZEN_TEST | {"hockey"}  # hockey: analysis only, not tuned

def load_tools(domain):
    lines = open(f"tool_dumps/{domain}.txt").read().splitlines()
    out = []
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*\[\s*\d+\]\s+(\S+)\s*$", ln)
        if m: out.append({"name": m.group(1), "description": lines[i + 1].strip() if i + 1 < len(lines) else ""})
    return out

questions = []
for f in sorted(glob.glob(f"data/train/{CAP}/output/*.json")):
    dom = os.path.basename(f)[:-5]
    if dom in EXCLUDED: continue
    for it in json.load(open(f)):
        if it["uuid"] in SEEN_UUIDS: continue
        for t in it["output"]:
            calls = [c["name"] for c in t["sequence"]["tool_call"]]
            if len(calls) >= 2:
                questions.append({"uuid": it["uuid"], "domain": dom, "q": t["query"],
                                  "gold": list(dict.fromkeys(calls)), "hops": len(calls)})
rng = random.Random(a.seed)
three = [x for x in questions if x["hops"] >= 3]
two = [x for x in questions if x["hops"] == 2]
sample = three + rng.sample(two, min(a.n2, len(two)))
assert_no_frozen([x["uuid"] for x in sample], {x["uuid"]: x["domain"] for x in sample})
os.makedirs("results", exist_ok=True)
json.dump([x["uuid"] for x in sample], open("results/tune_sample_uuids.json", "w"))
print(f"tune sample: {len(three)} three-call + {len(sample) - len(three)} two-call questions")

client = OpenAI(base_url=os.environ["LITELLM_BASE_URL"], api_key=os.environ["LITELLM_API_KEY"])
def words(n): return re.sub(r"^get_", "", n).replace("_", " ")
SYSTEM = ("You split a question into the minimal ordered list of lookups against a database API. "
          "Each step is one short sentence saying what to look up, and which value from an earlier step it needs. "
          "Use as few steps as possible (usually 2 or 3, never more than 5). Answer with ONLY a JSON list of strings.")
EX_Q = "How many albums did the artist with the most awards release?"
EX_A = '["Find the artist with the most awards", "Count the albums released by that artist"]'

def ask(q, tools):
    user = f"Question: {q}"
    if a.variant == "qnames":
        user = "Available operations: " + "; ".join(words(t["name"]) for t in tools) + "\n\n" + user
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Question: " + EX_Q},
            {"role": "assistant", "content": EX_A}, {"role": "user", "content": user}]
    r = client.chat.completions.create(model=a.model, messages=msgs, temperature=0, max_tokens=300)
    return r.choices[0].message.content or ""

def parse(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            v = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
            if v: return v[:5]
        except Exception: pass
    lines = [re.sub(r"^[\-\d\.\)\s]+", "", l).strip() for l in txt.splitlines()]
    return [l for l in lines if len(l.split()) >= 3][:5]

tools_by_dom = {d: load_tools(d) for d in {x["domain"] for x in sample}}
cache_path = f"results/plans_{a.variant}.json"
cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
for i, x in enumerate(sample):
    if x["uuid"] in cache: continue
    cache[x["uuid"]] = ask(x["q"], tools_by_dom[x["domain"]])
    if i % 10 == 0:
        json.dump(cache, open(cache_path, "w")); print(f"planned {i + 1}/{len(sample)}", flush=True)
json.dump(cache, open(cache_path, "w"))

enc = SentenceTransformer("all-MiniLM-L6-v2")
def norm(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
emb = {d: norm(enc.encode([f"{t['name']}: {t['description']}" for t in ts], convert_to_numpy=True)) for d, ts in tools_by_dom.items()}

rows = []
for x in sample:
    ts, E = tools_by_dom[x["domain"]], emb[x["domain"]]
    steps = parse(cache[x["uuid"]])
    qv = norm(enc.encode([x["q"]], convert_to_numpy=True))[0]
    base = [int(i) for i in np.argsort(-(E @ qv))]
    union = []
    if steps:
        for sv in norm(enc.encode(steps, convert_to_numpy=True)):
            for i in np.argsort(-(E @ sv))[:a.m]:
                if int(i) not in union: union.append(int(i))
    plus = list(union) + [i for i in base[:a.m] if i not in union]
    gi = {j for j, t in enumerate(ts) if t["name"] in x["gold"]}
    k1, k2 = max(len(union), 1), len(plus)
    rows.append({"uuid": x["uuid"], "domain": x["domain"], "hops": x["hops"], "n_steps": len(steps),
                 "union_size": len(union), "plus_size": k2,
                 "plan_all": gi <= set(union), "base_same_k": gi <= set(base[:k1]),
                 "plan_plus_all": gi <= set(plus), "base_same_k_plus": gi <= set(base[:k2]),
                 "plan": json.dumps(steps), "gold": ";".join(x["gold"]), "query": x["q"][:110]})

with open(f"results/plan_recall_{a.variant}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def table(sel, title):
    n = len(sel)
    if not n: return
    f = lambda k: sum(r[k] for r in sel)
    print(f"\n{title}: n={n} | avg tools offered: plan {np.mean([r['union_size'] for r in sel]):.1f}, plan+question {np.mean([r['plus_size'] for r in sel]):.1f}")
    print(f"   steps only:       plan {f('plan_all')}/{n}   one-shot, same budget {f('base_same_k')}/{n}")
    print(f"   steps + question: plan {f('plan_plus_all')}/{n}   one-shot, same budget {f('base_same_k_plus')}/{n}")
print(f"\nvariant={a.variant} m={a.m}; empty plans: {sum(r['n_steps'] == 0 for r in rows)}")
table(rows, "all")
table([r for r in rows if r["hops"] == 2], "2 gold calls")
table([r for r in rows if r["hops"] >= 3], "3+ gold calls")
print("\nexample plans (3-call questions):")
for r in [r for r in rows if r["hops"] >= 3][:4]:
    print("-", r["query"]); print("   plan:", r["plan"]); print("   gold:", r["gold"], "| covered:", r["plan_all"])
print(f"\nwrote results/plan_recall_{a.variant}.csv")
