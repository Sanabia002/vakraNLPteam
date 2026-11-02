#!/usr/bin/env python3
"""Domain-held-out dev split for capabilities 3 and 4 (reads data/train only)."""
import argparse, collections, glob, json, os, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("--n-shared", type=int, default=6, help="dev domains taken from domains present in both caps")
ap.add_argument("--n-cap4-only", type=int, default=1, help="extra dev domains only in cap 4")
ap.add_argument("--exclude", default="hockey", help="comma-separated domains kept out of dev")
ap.add_argument("--frac-min", type=float, default=0.15)
ap.add_argument("--frac-max", type=float, default=0.27)
ap.add_argument("--min-3hop", type=int, default=3)
ap.add_argument("--max-seed", type=int, default=20000)
ap.add_argument("--out", default="splits/dev_split.json")
a = ap.parse_args()

CAP = {3: "capability_3_multihop_reasoning", 4: "capability_4_multiturn"}

def question_counts(cap):
    out = {}
    for f in sorted(glob.glob(f"data/train/{CAP[cap]}/input/*.json")):
        out[os.path.basename(f)[:-5]] = len(json.load(open(f)))
    return out

def hop_counts_cap3():
    h = collections.defaultdict(collections.Counter)
    for f in glob.glob(f"data/train/{CAP[3]}/output/*.json"):
        dom = os.path.basename(f)[:-5]
        for item in json.load(open(f)):
            for turn in item["output"]:
                h[dom][len(turn["sequence"]["tool_call"])] += 1
    return h

c3, c4, hops = question_counts(3), question_counts(4), hop_counts_cap3()
if not c3 or not c4:
    sys.exit("no input files found under data/train; check the symlink and download")
excl = {x.strip() for x in a.exclude.split(",") if x.strip()}
shared = sorted((set(c3) & set(c4)) - excl)
only4 = sorted(set(c4) - set(c3) - excl)
tot3, tot4 = sum(c3.values()), sum(c4.values())

def evaluate(dev3, dev4):
    f3 = sum(c3[d] for d in dev3) / tot3
    f4 = sum(c4[d] for d in dev4) / tot4
    h3 = sum(hops[d][3] for d in dev3)
    ok = a.frac_min <= f3 <= a.frac_max and a.frac_min <= f4 <= a.frac_max and h3 >= a.min_3hop
    return ok, f3, f4, h3

chosen = None
for seed in range(a.max_seed):
    rng = random.Random(seed)
    sh = sorted(rng.sample(shared, a.n_shared))
    o4 = sorted(rng.sample(only4, a.n_cap4_only)) if a.n_cap4_only else []
    dev3, dev4 = sh, sorted(sh + o4)
    ok, f3, f4, h3 = evaluate(dev3, dev4)
    if ok:
        chosen = (seed, dev3, dev4, f3, f4, h3)
        break
if not chosen:
    sys.exit("no seed satisfied the constraints; loosen --frac-min/--frac-max or --min-3hop")
seed, dev3, dev4, f3, f4, h3 = chosen

def block(cap, counts, dev):
    train = sorted(set(counts) - set(dev))
    assert not set(dev) & set(train)
    return {"dev_domains": dev, "train_domains": train,
            "dev_questions": sum(counts[d] for d in dev),
            "train_questions": sum(counts[d] for d in train),
            "dev_per_domain": {d: counts[d] for d in dev}}

result = {"seed": seed, "exclude": sorted(excl),
          "criteria": {"n_shared": a.n_shared, "n_cap4_only": a.n_cap4_only,
                       "frac_range": [a.frac_min, a.frac_max], "min_3hop_dev_cap3": a.min_3hop},
          "capability_3": block(3, c3, dev3), "capability_4": block(4, c4, dev4)}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(result, open(a.out, "w"), indent=1)

print(f"seed {seed}; excluded from dev: {sorted(excl)}")
for cap, counts, dev, frac in ((3, c3, dev3, f3), (4, c4, dev4, f4)):
    n = sum(counts[d] for d in dev)
    print(f"\ncapability {cap}: {len(dev)} of {len(counts)} domains, {n} of {sum(counts.values())} questions ({100*frac:.0f}%)")
    for d in dev:
        extra = f"  hops {dict(sorted(hops[d].items()))}" if cap == 3 else ""
        print(f"   {d}: {counts[d]}{extra}")
dev_h = collections.Counter()
for d in dev3:
    dev_h.update(hops[d])
all_h = collections.Counter()
for d in c3:
    all_h.update(hops[d])
print("\ncap 3 gold tool-call counts, dev vs all:", dict(sorted(dev_h.items())), "vs", dict(sorted(all_h.items())))
print("wrote", a.out)
