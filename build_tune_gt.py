#!/usr/bin/env python3
"""Build an evaluator ground-truth dir restricted to a uuid sample (one <domain>.json per domain,
records copied verbatim from the capability-3 train output files).

  python build_tune_gt.py --uuids-file results/tune_sample_uuids.json --out tmp_gt_tune68
"""
import argparse, glob, json, os, sys
from frozen_split import assert_no_frozen, uuid_domains

ap = argparse.ArgumentParser()
ap.add_argument("--uuids-file", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--src", default="data/train/capability_3_multihop_reasoning/output")
a = ap.parse_args()

if os.path.exists(a.out):
    sys.exit(f"{a.out} already exists; remove it or choose a new --out")
wanted = json.load(open(a.uuids_file))
assert len(wanted) == len(set(wanted)), "duplicate uuids in sample"
assert_no_frozen(wanted, uuid_domains(os.path.dirname(a.src.rstrip("/"))))

by_domain, found = {}, set()
for f in sorted(glob.glob(os.path.join(a.src, "*.json"))):
    keep = [r for r in json.load(open(f)) if r["uuid"] in set(wanted)]
    if keep:
        by_domain[os.path.basename(f)[:-5]] = keep
        found |= {r["uuid"] for r in keep}
missing = set(wanted) - found
assert not missing, f"{len(missing)} uuids not found in {a.src}, e.g. {sorted(missing)[:3]}"

os.makedirs(a.out)
for d, recs in by_domain.items():
    json.dump(recs, open(os.path.join(a.out, f"{d}.json"), "w"), indent=2)
print(f"wrote {sum(map(len, by_domain.values()))} records in {len(by_domain)} domain files to {a.out}")
print("domains:", ",".join(sorted(by_domain)))
