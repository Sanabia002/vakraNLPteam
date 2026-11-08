#!/usr/bin/env python3
"""Offline Tier-1 retrieval experiments, capability 3 (CPU only; no LLM, no containers, no judge).
Metric: questions whose gold tools are ALL in the top-k. Tune domains only unless --show-dev."""
import argparse, collections, csv, glob, json, os, re
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="skip the bge/mpnet downloads")
ap.add_argument("--finalists", nargs="+", default=None, help="variant names to show on dev")
ap.add_argument("--show-dev", action="store_true", help="also print dev rows (use once, for finalists)")
a = ap.parse_args()

CAP, KS = "capability_3_multihop_reasoning", [10, 20, 40]
try:
    from frozen_split import FROZEN_TEST
    FROZEN = set(FROZEN_TEST)
except Exception:
    FROZEN = {"bike_share_1", "computer_student", "disney", "olympics", "restaurant", "student_loan"}
EXCL = FROZEN | {"hockey"}                      # frozen test domains and analysis-only hockey: never read
dev = set(json.load(open("splits/dev_split.json"))["capability_3"]["dev_domains"])

api = json.load(open("tool_dumps/openapi_cap3.json"))
tools = collections.defaultdict(list)
for path, item in api["paths"].items():
    parts = path.strip("/").split("/")
    if len(parts) < 3 or "get" not in item or parts[1] in EXCL: continue
    op = item["get"]
    params = [(p["name"], p.get("description") or p.get("schema", {}).get("description") or "")
              for p in op.get("parameters", []) if p.get("in") == "query"]
    tools[parts[1]].append({"name": op["operationId"], "desc": op.get("summary") or op.get("description") or "", "params": params})

questions, skipped = [], 0
for f in sorted(glob.glob(f"data/train/{CAP}/output/*.json")):
    dom = os.path.basename(f)[:-5]
    if dom in EXCL: continue                    # skip BEFORE opening the file
    names = {t["name"] for t in tools[dom]}
    for it in json.load(open(f)):
        for t in it["output"]:
            gold = list(dict.fromkeys(c["name"] for c in t["sequence"]["tool_call"]))
            if not gold or any(g not in names for g in gold): skipped += 1; continue
            questions.append({"uuid": it["uuid"], "dom": dom, "q": t["query"], "gold": gold,
                              "hops": len(t["sequence"]["tool_call"]), "split": "dev" if dom in dev else "tune"})
needed = {x["dom"] for x in questions}
tools = {d: ts for d, ts in tools.items() if d in needed}
tune_qs = [x for x in questions if x["split"] == "tune"]; dev_qs = [x for x in questions if x["split"] == "dev"]
print(f"tune: {len(tune_qs)} questions in {len({x['dom'] for x in tune_qs})} domains | dev: {len(dev_qs)} questions (hidden unless --show-dev) | skipped: {skipped}")

def words(n): return re.sub(r"^get_", "", n).replace("_", " ")
def t_base(t):  return f"{t['name']}: {t['desc']}"
def t_words(t): return f"{words(t['name'])}: {t['desc']}"
def t_full(t):  return f"{words(t['name'])}: {t['desc']} Parameters: " + "; ".join(f"{n.replace('_', ' ')} {d}".strip() for n, d in t["params"])
def pieces(q):
    ps = [p.strip() for p in re.split(r"\b(?:that|who|which|whose|where|and)\b|[,;]", q) if len(p.split()) >= 3]
    return [q] + ps
STOP = set("the a an of in on at to for and or is was were are be by with that which who whom whose what how many much from as it its this these those has have had than then when where".split())
def toks(s): return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]
def norm(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

models = {}
def model(n):
    if n not in models: models[n] = SentenceTransformer(n)
    return models[n]
def dense(mname, text_fn, clauses=False, qprefix=""):
    m = model(mname)
    emb = {d: norm(m.encode([text_fn(t) for t in ts], convert_to_numpy=True, batch_size=32)) for d, ts in tools.items()}
    out = {}
    for x in questions:
        qs = pieces(x["q"]) if clauses else [x["q"]]
        out[x["uuid"]] = (norm(m.encode([qprefix + s for s in qs], convert_to_numpy=True)) @ emb[x["dom"]].T).max(axis=0)
    return out
def bm25(text_fn):
    idx = {d: BM25Okapi([toks(text_fn(t)) for t in ts]) for d, ts in tools.items()}
    return {x["uuid"]: np.array(idx[x["dom"]].get_scores(toks(x["q"]))) for x in questions}
def rrf(*sds, k=60):
    out = {}
    for u in sds[0]:
        f = np.zeros(len(sds[0][u]))
        for sd in sds:
            for r, i in enumerate(np.argsort(-sd[u], kind="stable")): f[i] += 1.0 / (k + r + 1)
        out[u] = f
    return out
def ranks(sd):
    out = {}
    for x in questions:
        ts = tools[x["dom"]]
        pos = {ts[i]["name"]: r + 1 for r, i in enumerate(np.argsort(-sd[x["uuid"]], kind="stable"))}
        out[x["uuid"]] = [pos[g] for g in x["gold"]]
    return out

MINI, MPNET, BGE = "all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"
BGEQ = "Represent this sentence for searching relevant passages: "
BASE = "base: MiniLM name+desc"
S = {}
def add(name, sd): S[name] = sd; print("done:", name, flush=True)
add(BASE, dense(MINI, t_base))
add("MiniLM words+desc", dense(MINI, t_words))
add("MiniLM words+desc+params", dense(MINI, t_full))
add("MiniLM name+desc, clause-max", dense(MINI, t_base, clauses=True))
add("MiniLM words+desc+params, clause-max", dense(MINI, t_full, clauses=True))
if BM25Okapi:
    add("BM25 words+desc+params", bm25(t_full))
    add("hybrid: MiniLM full + BM25", rrf(S["MiniLM words+desc+params"], S["BM25 words+desc+params"]))
    add("hybrid: MiniLM full clause-max + BM25", rrf(S["MiniLM words+desc+params, clause-max"], S["BM25 words+desc+params"]))
if not a.quick:
    add("bge-small words+desc+params", dense(BGE, t_full, qprefix=BGEQ))
    add("mpnet words+desc+params", dense(MPNET, t_full))
    if BM25Okapi: add("hybrid: bge-small full + BM25", rrf(S["bge-small words+desc+params"], S["BM25 words+desc+params"]))
R = {n: ranks(sd) for n, sd in S.items()}

def allk(rk, k): return all(r <= k for r in rk)
def table(title, qs, only=None):
    base, n3 = R[BASE], sum(x["hops"] >= 3 for x in qs)
    print(f"\n{title}: n={len(qs)} ({n3} with 3+ gold calls)")
    print(f"{'variant':42s} @10 @20 @40 | 3+call@20 | vs base @20")
    for name, rk in R.items():
        if only is not None and name not in only and name != BASE: continue
        c = {k: sum(allk(rk[x["uuid"]], k) for x in qs) for k in KS}
        h3 = sum(allk(rk[x["uuid"]], 20) for x in qs if x["hops"] >= 3)
        g = sum(allk(rk[x["uuid"]], 20) and not allk(base[x["uuid"]], 20) for x in qs)
        l = sum(allk(base[x["uuid"]], 20) and not allk(rk[x["uuid"]], 20) for x in qs)
        print(f"{name:42s} {c[10]:3d} {c[20]:3d} {c[40]:3d} | {h3:3d}/{n3:<3d}   | +{g} -{l}")
table("TUNE", tune_qs)
if a.show_dev:
    assert a.finalists, "pass --finalists with the pre-chosen variant names"
    table("DEV (finalists only; look once)", dev_qs, only=set(a.finalists))
else: print("\n(dev rows hidden; run with --show-dev once for the finalists)")

try:   # sanity: the base row should match the existing recall file
    rec = {r["uuid"]: r for r in csv.DictReader(open("results/shortlist_recall_cap3.csv"))}
    ok = lambda r, k: all(float(g.rsplit(":", 1)[1]) <= k for g in r["gold_ranks"].split(";"))
    print("\nsanity (base vs results/shortlist_recall_cap3.csv, tune questions): " + "  ".join(
        f"@{k}: {sum(allk(R[BASE][x['uuid']], k) for x in tune_qs)} vs {sum(ok(rec[x['uuid']], k) for x in tune_qs if x['uuid'] in rec)}" for k in KS))
except FileNotFoundError:
    print("sanity file not found")
os.makedirs("results", exist_ok=True)
with open("results/retrieval_variant_ranks.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["uuid", "domain", "split", "gold_calls", "variant", "gold_ranks"])
    for name, rk in R.items():
        for x in questions: w.writerow([x["uuid"], x["dom"], x["split"], x["hops"], name, ";".join(map(str, rk[x["uuid"]]))])
print("wrote results/retrieval_variant_ranks.csv")
