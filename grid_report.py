#!/usr/bin/env python3
"""python grid_report.py  -> results/grid_test.txt: supplementary uniform model-by-agent grid on the frozen test set.
Existing cells come from results/eval_final_test_*.json, new ones from results/eval_gridtest_final_test_*.json
(the _merged file after re-judging when present). Stops without writing if an existing number does not reproduce
or any file has unjudged items. Reads only result/output files; no model or judge calls."""
import glob, json, os, sys
EXPECTED = {"baseline": 16, "C": 23, "D": 35, "F": 36}
EXISTING = {"baseline": "final_test_baseline", "C": "final_test_C", "D": "final_test_D", "F": "final_test_F"}
NEW = {"baseline27": "final_test_baseline27", "C27": "final_test_C27", "D4": "final_test_D4",
       "E": "final_test_E", "F27": "final_test_F27"}
NEW_STAGES = ("D4", "C27", "baseline27")

def path(run, grid):
    base = f"results/eval_{'gridtest_' if grid else ''}{run}"
    return base + "_merged.json" if os.path.exists(base + "_merged.json") else base + ".json"
def load(run, grid):
    o = {}
    for v in json.load(open(path(run, grid)))["domains"].values():
        for dl in v["dialogues"]:
            ex = dl["details"]["per_turn"][0]["metadata"].get("score_explanation") or {}
            bad = any("Judge Error" in str(ex.get(k) or "") for k in ("answer", "groundedness"))
            o[dl["uuid"]] = (dl["score"] >= 0.99, bad)
    return o
def wilson(k, n, z=1.96):  # same as final_report.py
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))

N = len(json.load(open("results/test_uuids.json")))
R = {k: load(r, False) for k, r in EXISTING.items()} | {k: load(r, True) for k, r in NEW.items()}
stop = []
for k, d in R.items():
    if len(d) != N: stop.append(f"{k}: {len(d)} questions, expected {N}")
    if any(b for _, b in d.values()): stop.append(f"{k}: {sum(b for _, b in d.values())} unjudged items")
for k, want in EXPECTED.items():
    got = sum(c for c, _ in R[k].values())
    if got != want: stop.append(f"existing {k} = {got}, expected {want}")
if stop:
    sys.exit("STOP, results/grid_test.txt not written:\n  " + "\n  ".join(stop))

def cell(k):
    d = R[k]; c = sum(x for x, _ in d.values()); lo, hi = wilson(c, len(d))
    return f"{c}/{len(d)} = {100*c/len(d):4.1f}% [{lo:.0f}-{hi:.0f}%]"
L = ["Supplementary uniform grid, frozen test set (67 questions). Protocol: results/freeze_rule.txt, 'Supplementary uniform grid'.",
     "Not used to select or change the final system. Cells: correct/67, %, 95% Wilson interval.", "",
     f"{'':44s} {'Qwen3.5-4B (local)':30s} qwen3.8-27b (Groq)",
     f"{'Baseline (provided ReAct, k=20, no flags)':44s} {cell('baseline'):30s} {cell('baseline27')}",
     f"{'C (plan-execute, frozen flags)':44s} {cell('C'):30s} {cell('C27')}",
     f"{'D (ReAct, hybrid, k=30, grounded)':44s} {cell('D4'):30s} {cell('D')}",
     f"{'D then C (status fallback)':44s} {'E  ' + cell('E'):30s} {'F27 ' + cell('F27')}", "",
     f"Frozen system F (D 27B + C 4B, reported, unchanged): {cell('F')}", "",
     "Paired counts over the same 67 questions (both / only first / only second / neither):"]
for a, b in (("baseline", "baseline27"), ("C", "C27"), ("D4", "D"), ("E", "F27"), ("baseline27", "D")):
    A, B = R[a], R[b]
    L.append(f"  {a:10s} vs {b:10s}  {sum(A[u][0] and B[u][0] for u in A):2d} / {sum(A[u][0] and not B[u][0] for u in A):2d}"
             f" / {sum(B[u][0] and not A[u][0] for u in A):2d} / {sum(not A[u][0] and not B[u][0] for u in A):2d}")
L += ["", "New stages: run status (from output/<run>/; step limit = 'Recursion' in error, timeout = 'timed out' in error):"]
for k in NEW_STAGES:
    items = [r for f in glob.glob(f"output/{NEW[k]}/*.json") if not f.endswith(("_tools.json", "_trajectory.json"))
             for r in json.load(open(f))]
    ok = [r for r in items if r["status"] == "success"]; err = [r for r in items if r["status"] != "success"]
    step = sum("Recursion" in str(r.get("error", "")) for r in err); tmo = sum("timed out" in str(r.get("error", "")) for r in err)
    empty = sum(not str(r["output"][0].get("answer") or "").strip() for r in ok)
    L.append(f"  {k:10s} n={len(items)} success {len(ok)}, error {len(err)} (step limit {step}, timeout {tmo}, other {len(err)-step-tmo});"
             f" no answer {len(err) + empty} ({len(err)} non-success + {empty} empty successful)")
for k in ("E", "F27"):
    m = json.load(open(f"results/merge_log_{NEW[k]}.json"))
    L.append(f"  {k:10s} merge: fallback used {len(m['used_fallback'])}, fallback also failed {len(m['fallback_also_failed'])}")
L += ["", "Judge consistency: a merged run's non-fallback answers are byte-identical to its primary run's, but are judged again:"]
for k, prim, fbk in (("F", "D", "C"), ("E", "D4", "C"), ("F27", "D", "C27")):
    log = f"results/merge_log_{(EXISTING | NEW)[k]}.json"
    used = set(json.load(open(log))["used_fallback"]); M, P, Fb = R[k], R[prim], R[fbk]
    kept = [u for u in M if u not in used]
    L.append(f"  {k:4s} kept from {prim:3s} {len(kept)}: {sum(M[u][0] for u in kept)} correct here vs {sum(P[u][0] for u in kept)} in {prim}'s own scoring"
             f" ({sum(M[u][0] != P[u][0] for u in kept)} judged differently); fallback items {len(used)}: {sum(M[u][0] for u in used)} correct here,"
             f" {sum(Fb[u][0] for u in used)} in {fbk}'s own scoring")
L += ["", "Noise: re-judging the same outputs flipped about 1 question per set earlier (results/judgecheck_*), and up to 3 here (above),",
      "so differences of up to about 3 questions are within judge noise. F27 < D is judge noise on identical answers, not a merge effect.",
      "The 27B model is hosted (Groq preview) and not deterministic; each cell is a single run."]
open("results/grid_test.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
