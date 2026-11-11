#!/usr/bin/env python3
"""python merge_fallback.py PRIMARY_RUN FALLBACK_RUN OUT_RUN
Per question: PRIMARY's result unless its status != success and FALLBACK's is success. duration_s of a fallbacked item = both. No gold, no model."""
import glob, json, os, shutil, sys
prim, fb, out = sys.argv[1:4]
os.makedirs(f"output/{out}", exist_ok=True)
skip = ("_tools.json", "_trajectory.json")
fbitems = {r["uuid"]: r for f in glob.glob(f"output/{fb}/*.json") if not f.endswith(skip) for r in json.load(open(f))}
log = {"primary": prim, "fallback": fb, "used_fallback": [], "fallback_also_failed": []}
for f in sorted(glob.glob(f"output/{prim}/*.json")):
    if f.endswith(skip): continue
    merged = []
    for r in json.load(open(f)):
        if r["status"] != "success":
            alt = fbitems.get(r["uuid"])
            if alt is not None and alt["status"] == "success":
                alt = dict(alt); alt["duration_s"] = r.get("duration_s", 0) + alt.get("duration_s", 0)
                merged.append(alt); log["used_fallback"].append(r["uuid"]); continue
            log["fallback_also_failed"].append(r["uuid"])
        merged.append(r)
    json.dump(merged, open(f"output/{out}/{os.path.basename(f)}", "w"), indent=1)
for f in glob.glob(f"output/{prim}/*_tools.json") + glob.glob(f"output/{prim}/*_trajectory.json"):
    shutil.copy(f, f"output/{out}/")        # primary's provenance files, for show_item.py
os.makedirs("results", exist_ok=True); json.dump(log, open(f"results/merge_log_{out}.json", "w"), indent=1)
print(f"{out}: used fallback for {len(log['used_fallback'])} questions; fallback also failed on {len(log['fallback_also_failed'])}")
