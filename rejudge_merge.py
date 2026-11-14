#!/usr/bin/env python3
"""python rejudge_merge.py RUN -- re-judge only items whose judge call failed, merge into results/eval_RUN_merged.json.
Needs API_KEY + JUDGE_BACKEND exported, containers up, tmp_gt_tune/ present."""
import json, os, shutil, subprocess, sys
run = sys.argv[1]
orig = json.load(open(f"results/eval_{run}.json"))
def has_err(dl):
    ex = dl["details"]["per_turn"][0]["metadata"].get("score_explanation") or {}
    return any("Judge Error" in str(ex.get(k) or "") for k in ("answer", "groundedness"))
bad = {dl["uuid"]: dom for dom, v in orig["domains"].items() for dl in v["dialogues"] if has_err(dl)}
if not bad: sys.exit("no judge errors: nothing to do")
doms = sorted(set(bad.values()))
gt_dir, pr_dir = f"tmp_rejudge/{run}/gt", f"tmp_rejudge/{run}/pred"
for d in (gt_dir, pr_dir): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
for dom in doms:
    for src, dst in ((f"tmp_gt_tune/{dom}.json", gt_dir), (f"output/{run}/{dom}.json", pr_dir)):
        json.dump([x for x in json.load(open(src)) if x["uuid"] in bad], open(f"{dst}/{dom}.json", "w"))
out = f"results/eval_{run}_rejudge.json"
if os.path.exists(out): os.remove(out)
print(f"re-judging {len(bad)} items in {len(doms)} domains ...", flush=True)
with open(f"eval_{run}_rejudge.log", "w") as fh:
    subprocess.run([sys.executable, "evaluator/evaluator.py", "--capability_name", "capability_multihop_reasoning",
                    "--gt_root", gt_dir, "--pred_root", pr_dir, "--domains", *doms, "--output", out],
                   stdout=fh, stderr=subprocess.STDOUT)
rv = {dl["uuid"]: dl for v in json.load(open(out))["domains"].values() for dl in v["dialogues"]}
fixed, still = 0, []
for dom, v in orig["domains"].items():
    for i, dl in enumerate(v["dialogues"]):
        if dl["uuid"] in bad:
            new = rv.get(dl["uuid"])
            if new is not None and not has_err(new): v["dialogues"][i] = new; fixed += 1
            else: still.append(dl["uuid"])
    sc = [dl["score"] for dl in v["dialogues"]]
    v["summary"].update({"num_samples": len(sc), "num_correct": sum(sc), "mean_dialogue_score": sum(sc) / len(sc),
                         "min_dialogue_score": min(sc), "max_dialogue_score": max(sc)})
allsc = [dl["score"] for v in orig["domains"].values() for dl in v["dialogues"]]
orig["summary"].update({"n_samples": len(allsc), "n_correct": sum(allsc), "mean_dialogue_score": sum(allsc) / len(allsc)})
json.dump(orig, open(f"results/eval_{run}_merged.json", "w"), indent=1)
print(f"fixed {fixed}; still unjudged {len(still)}; correct {sum(allsc):.0f}/{len(allsc)}"
      + (f" (report as {sum(allsc):.0f} to {sum(allsc) + len(still):.0f})" if still else ""))
