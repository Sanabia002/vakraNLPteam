#!/usr/bin/env python3
"""Run benchmark_runner.py and append one row to runs_log.csv."""
import argparse, csv, collections, datetime, glob, json, os, subprocess, sys, time

LOG = "runs_log.csv"
FIELDS = ["timestamp", "output_dir", "capability", "domain", "model", "llama_cpp_build",
          "thinking", "top_k_tools", "max_iterations", "max_samples", "temperature",
          "items", "status_counts", "avg_duration_s", "wall_s", "return_code", "notes"]

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--cap", required=True)
p.add_argument("--domain", required=True)
p.add_argument("--model", required=True)
p.add_argument("--top-k", default="0")
p.add_argument("--max-iter", default="")
p.add_argument("--samples", default="")
p.add_argument("--temperature", default="")
p.add_argument("--thinking", default="")
p.add_argument("--build", default="")
p.add_argument("--notes", default="")
p.add_argument("--uuids-file", default="", help="passed through to benchmark_runner.py --uuids-file")
a = p.parse_args()

if os.path.exists(a.out):
    sys.exit(f"{a.out} already exists; choose a new --out name so runs stay separate")

cmd = [sys.executable, "benchmark_runner.py", "--capability_id", a.cap]
for _d in a.domain.split(","):
    cmd += ["--domain", _d.strip()]
cmd += ["--provider", "litellm", "--model", a.model, "--output", a.out]
if a.top_k != "0": cmd += ["--top-k-tools", a.top_k]
if a.max_iter: cmd += ["--max-iterations", a.max_iter]
if a.samples: cmd += ["--max-samples-per-domain", a.samples]
if a.temperature: cmd += ["--temperature", a.temperature]
if a.uuids_file: cmd += ["--uuids-file", a.uuids_file]

t0 = time.time()
rc = subprocess.run(cmd).returncode
wall = round(time.time() - t0, 1)

statuses, durs = collections.Counter(), []
for f in glob.glob(os.path.join(a.out, "*.json")):
    if f.endswith("_tools.json"): continue
    try:
        for it in json.load(open(f)):
            statuses[it.get("status", "?")] += 1
            if "duration_s" in it: durs.append(it["duration_s"])
    except Exception as e:
        statuses[f"unreadable:{type(e).__name__}"] += 1

row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
       "output_dir": a.out, "capability": a.cap, "domain": a.domain, "model": a.model,
       "llama_cpp_build": a.build, "thinking": a.thinking, "top_k_tools": a.top_k,
       "max_iterations": a.max_iter, "max_samples": a.samples, "temperature": a.temperature,
       "items": sum(statuses.values()), "status_counts": json.dumps(dict(statuses)),
       "avg_duration_s": round(sum(durs) / len(durs), 1) if durs else "",
       "wall_s": wall, "return_code": rc, "notes": a.notes}
new = not os.path.exists(LOG)
with open(LOG, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new: w.writeheader()
    w.writerow(row)
print("logged ->", LOG)
