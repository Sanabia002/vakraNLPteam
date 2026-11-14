#!/usr/bin/env python3
"""Upper bound: does adding the (gold) previous tool result to the query help retrieve later-hop tools? TUNE domains only; CPU only."""
import collections, glob, json, os, re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from frozen_split import FROZEN_TEST
SKIP = set(FROZEN_TEST) | {"hockey"} | set(json.load(open("splits/dev_split.json"))["capability_3"]["dev_domains"])
CAP = "capability_3_multihop_reasoning"
api = json.load(open("tool_dumps/openapi_cap3.json")); tools = collections.defaultdict(list)
for path, item in api["paths"].items():
    parts = path.strip("/").split("/")
    if len(parts) < 3 or "get" not in item or parts[1] in SKIP: continue      # skipped domains are dropped immediately
    op = item["get"]
    tools[parts[1]].append({"name": op["operationId"], "desc": op.get("summary") or op.get("description") or "",
        "params": [(p["name"], p.get("description") or p.get("schema", {}).get("description") or "") for p in op.get("parameters", []) if p.get("in") == "query"]})
Q = []
for f in sorted(glob.glob(f"data/train/{CAP}/output/*.json")):
    dom = os.path.basename(f)[:-5]
    if dom in SKIP: continue
    for it in json.load(open(f)):
        t = it["output"][0]; seq = t["sequence"]; names = [c["name"] for c in seq["tool_call"]]
        if len(names) >= 2: Q.append((dom, t["query"], names, seq.get("tool_response", [])))
STOP = set("the a an of in on at to for and or is was were are be by with that which who whom whose what how many much from as it its this these those has have had than then when where".split())
toks = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]
words = lambda n: re.sub(r"^get_", "", n).replace("_", " ")
ttext = lambda t: f"{words(t['name'])}: {t['desc']} Parameters: " + "; ".join(f"{n.replace('_', ' ')} {d}".strip() for n, d in t["params"])
def pieces(q):
    return [q] + [p.strip() for p in re.split(r"\b(?:that|who|which|whose|where|and)\b|[,;]", q) if len(p.split()) >= 3]
norm = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
model = SentenceTransformer("all-MiniLM-L6-v2"); IDX = {}
def idx(dom):
    if dom not in IDX:
        txt = [ttext(t) for t in tools[dom]]
        IDX[dom] = (norm(model.encode(txt, convert_to_numpy=True, batch_size=32)), BM25Okapi([toks(x) for x in txt]))
    return IDX[dom]
def rank(dom, clauses, bmq):
    E, bm = idx(dom); dense = (norm(model.encode(clauses, convert_to_numpy=True)) @ E.T).max(axis=0); f = np.zeros(len(dense))
    for r, i in enumerate(np.argsort(-dense, kind="stable")): f[i] += 1.0 / (60 + r + 1)
    if toks(bmq):
        for r, i in enumerate(np.argsort(-np.array(bm.get_scores(toks(bmq))), kind="stable")): f[i] += 1.0 / (60 + r + 1)
    return [int(i) for i in np.argsort(-f, kind="stable")]
def flat(x):
    if isinstance(x, (list, tuple)):
        for y in x: yield from flat(y)
    elif isinstance(x, dict):
        for y in x.values(): yield from flat(y)
    elif x is not None: yield x
rows, skipped = [], 0
for dom, q, names, resps in Q:
    pos = {t["name"]: i for i, t in enumerate(tools[dom])}
    if any(n not in pos for n in names): skipped += 1; continue
    cl = pieces(q); base = rank(dom, cl, q)
    for h in range(1, len(names)):
        g = names[h]
        if g in names[:h] or h - 1 >= len(resps): continue
        prev = " ".join(str(v) for v in flat(resps[h - 1]))[:300]
        if not prev.strip(): continue
        qp, po = rank(dom, cl + [prev], q + " " + prev), rank(dom, [prev], prev); gi = pos[g]
        rows.append({"hop": h + 1, "base": base.index(gi) + 1, "qp": qp.index(gi) + 1, "po": po.index(gi) + 1,
                     "union": gi in (set(base[:10]) | set(qp[:10]))})
print(f"tune questions with >=2 gold calls: {len(Q)} (skipped for missing tools: {skipped}); later-hop gold tools measured: {len(rows)}")
for h in sorted({r["hop"] for r in rows}):
    s = [r for r in rows if r["hop"] == h]; n = len(s); c = lambda k, m: sum(r[k] <= m for r in s)
    print(f"\nhop {h} tool (n={n}):  in top 10 / top 20")
    print(f"   question only            {c('base', 10):3d} / {c('base', 20):3d}")
    print(f"   question + prev result   {c('qp', 10):3d} / {c('qp', 20):3d}   (vs question only at 20: +{sum(r['qp'] <= 20 and r['base'] > 20 for r in s)} -{sum(r['base'] <= 20 and r['qp'] > 20 for r in s)})")
    print(f"   prev result only         {c('po', 10):3d} / {c('po', 20):3d}")
    print(f"   union of top10 (question) + top10 (question+prev), at most 20 tools: {sum(r['union'] for r in s)}  vs question-only top 20: {c('base', 20)}")
