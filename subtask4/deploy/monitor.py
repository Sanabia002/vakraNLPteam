"""Catch problems on the GPU machine before and during a queue of runs, from the repository root.

    python -m subtask4.deploy.monitor preflight      everything a queue needs; exits 1 on any failure
    python -m subtask4.deploy.monitor watch LOG      every minute: progress, health, behaviour; problems -> ALERT lines
    python -m subtask4.deploy.monitor status         one look at the current run

preflight
  clean commit; servers up; both models on the GPU (the judge loaded without mmap); no memory thrashing; disk space;
  the Subtask 4 unit tests; then a canary run of the full system on the dialogues most likely to break it: the
  longest histories in tune, one dialogue per rule path, and earlier regressions. Every canary must succeed within
  CANARY_LIMIT_S, with a decision and a non-empty answer.
watch
  the newest run: done / timeouts / errors, pace and ETA, slow dialogues; the judge's pace; Ollama (models loaded,
  on the GPU), memory pressure, GPU use; queue failures; and, from the trajectories so far, behaviour rates (empty
  answers, forced answers at the step limit, made-up tools, repeated identical calls). Thresholds are below.
"""
from __future__ import annotations

import collections
import glob
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..data import REPO

OUTPUT = REPO / "output" / "subtask4"
PY = str(REPO / ".venv" / "bin" / "python")
MODELS = ("gpt-oss:120b", "qwen3.5-4b")
CANARY_LIMIT_S = 150
SLOW_S = 150                      # one dialogue slower than this
PRESSURE_FULL = 20.0              # /proc/pressure/memory "full avg60" above this: thrashing
JUDGE_SLOW_S = 30.0               # judge seconds per dialogue
LIMITS = {"empty answers": 0.02, "forced at step limit": 0.05, "made-up tools per dialogue": 0.2,
          "3+ identical calls in a row": 0.06}
KNOWN_REGRESSIONS = [("books", "6e317bcd6839-b2a89bd88c27"), ("books", "6e317bcd6839-678ae5e673c2")]


def sh(cmd: str, timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=REPO).stdout
    except subprocess.TimeoutExpired:
        return ""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M")


# ------------------------------------------------------------------ health

def ollama_models() -> dict[str, str]:
    """model -> processor column of `ollama ps` (e.g. "100% GPU")."""
    return parse_ollama_ps(sh("ollama ps"))


def parse_ollama_ps(text: str) -> dict[str, str]:
    """`ollama ps` rows -> {model: processor}; the SIZE column ("64 GB") has a space, so match the processor."""
    out = {}
    for row in text.splitlines()[1:]:
        if row.strip():
            match = re.search(r"\d+%(?:/\d+%)? (?:CPU/GPU|GPU|CPU)", row)
            out[row.split()[0].removesuffix(":latest")] = match.group(0) if match else "?"
    return out


def memory_pressure() -> float:
    text = Path("/proc/pressure/memory").read_text() if Path("/proc/pressure/memory").exists() else ""
    match = re.search(r"full avg10=[\d.]+ avg60=([\d.]+)", text)
    return float(match.group(1)) if match else 0.0


def gpu_use() -> tuple[int, int]:
    out = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits").strip()
    util, mem = (int(x) for x in out.split(",")) if out else (0, 0)
    return util, mem


def health() -> list[str]:
    problems = []
    loaded = ollama_models()
    for model in MODELS:
        if model not in loaded:
            problems.append(f"model {model} not loaded")
        elif loaded[model] != "100% GPU":
            problems.append(f"model {model} not fully on the GPU ({loaded[model]})")
    if (p := memory_pressure()) > PRESSURE_FULL:
        problems.append(f"memory thrashing (pressure full avg60 = {p:.0f}%)")
    free_gb = int(sh("df -BG --output=avail / | tail -1").strip().rstrip("G") or 0)
    if free_gb < 20:
        problems.append(f"only {free_gb} GB disk left")
    return problems


# ------------------------------------------------------------------ runs

def newest_run() -> Path | None:
    runs = [Path(p) for p in glob.glob(str(OUTPUT / "*")) if Path(p).is_dir() and (Path(p) / "run.log").exists()]
    return max(runs, key=lambda p: (p / "run.log").stat().st_mtime) if runs else None


def run_progress(run: Path) -> dict:
    log = (run / "run.log").read_text(errors="replace")
    times = [float(t) for t in re.findall(r"Status: success .*?Time: ([\d.]+)s", log)]
    total = sum(int(n) for n in re.findall(r"# DOMAIN: \S+ \((\d+) items\)", log))
    record = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    stamps = re.findall(r"^\[(\d\d:\d\d:\d\d)", log, re.M)
    return {"run": run.name, "status": record.get("status"), "done": len(times),
            "timeouts": len(re.findall(r"Status: timeout", log)), "errors": len(re.findall(r"Status: (?:error|fail)", log)),
            "domains_started": len(re.findall(r"# DOMAIN:", log)), "items_started": total, "times": times,
            "slow": sum(t > SLOW_S for t in times), "last_stamp": stamps[-1] if stamps else None}


def judge_pace(run: Path) -> float | None:
    out = run.parent / f"{run.name}.eval.out"
    if not out.exists():
        return None
    rates = re.findall(r"([\d.]+)s/it", out.read_text(errors="replace")[-3000:])
    return float(rates[-1]) if rates else None


def behaviour(run: Path) -> dict:
    n, c = 0, collections.Counter()
    for path in glob.glob(str(run / "*_trajectory.json")):
        try:
            items = json.loads(Path(path).read_text())
        except ValueError:  # being written
            continue
        for item in items:
            trajectory = item.get("trajectory") or []
            if not trajectory or trajectory[0].get("type") != "Subtask4Decision":
                continue
            n += 1
            events = trajectory[-1].get("events", []) if trajectory[-1].get("type") == "Subtask4Log" else []
            c["forced at step limit"] += any(e.get("forced") for e in events)
            c["made-up tools per dialogue"] += sum(bool(e.get("unknown")) for e in events)
            c[f"path {trajectory[0].get('mode')}"] += 1
            calls = [(x.get("name"), json.dumps(x.get("args"), sort_keys=True)) for e in trajectory
                     if e.get("type") == "AIMessage" for x in e.get("tool_calls") or []]
            c["3+ identical calls in a row"] += any(calls[i] == calls[i + 1] == calls[i + 2] for i in range(len(calls) - 2))
        for prediction in json.loads(Path(path.replace("_trajectory", "")).read_text()) \
                if Path(path.replace("_trajectory", "")).exists() else []:
            c["empty answers"] += not str((prediction.get("output") or [{}])[-1].get("answer") or "").strip()
    return {"dialogues": n, **c}


# ------------------------------------------------------------------ preflight

def canaries(k_long: int = 3) -> list[tuple[str, str]]:
    """The longest histories in tune, one dialogue per rule path, and earlier regressions."""
    from benchmark.runner_helpers import BenchmarkItem

    from .. import data
    from ..rules import UnknownRule, parse_rule
    sizes, per_path = [], {}
    for domain in data.domains("tune"):
        for uuid, item in data.inputs(domain).items():
            b = BenchmarkItem.from_dict(item)
            sizes.append((sum(len(m.content) for m in (b.context or [])), domain, uuid))
            try:
                rule = parse_rule(item["additional_instructions"])
            except UnknownRule:
                continue
            path = "none" if rule is None else f"{rule.allowed.value}{'|topic' if rule.category else ''}"
            per_path.setdefault(path, (domain, uuid))
    longest = [(d, u) for _, d, u in sorted(sizes, reverse=True)[:k_long]]
    return list(dict.fromkeys(longest + list(per_path.values()) + KNOWN_REGRESSIONS))


def preflight() -> bool:
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{(': ' + detail) if detail else ''}", flush=True)

    print(f"[{now()}] preflight")
    dirty = sh("git status --porcelain --untracked-files=no").strip()
    check("clean commit", not dirty, sh("git log --oneline -1").strip())
    services = sh(f"{PY} -m subtask4.deploy.services check").strip()
    check("servers", "down" not in services, services)
    for model in MODELS:  # load both, then check they sit on the GPU together
        sh(f"curl -s localhost:11434/api/generate -d '{{\"model\":\"{model}\",\"prompt\":\"OK\",\"stream\":false,"
           f"\"keep_alive\":-1,\"think\":false}}'", timeout=600)
    check("judge loads without mmap", "use_mmap" in sh("ollama show gpt-oss:120b --parameters") and
          "false" in sh("ollama show gpt-oss:120b --parameters | grep use_mmap"))
    problems = health()
    check("health (models on GPU, memory, disk)", not problems, "; ".join(problems))
    tests = sh(f"{PY} -m pytest -q tests/unit -k s4 --ignore tests/unit/test_initialize_active_data.py", timeout=600)
    check("unit tests", " failed" not in tests and "passed" in tests, tests.strip().splitlines()[-1] if tests else "")

    picked = canaries()
    uuids = REPO / "output" / "subtask4" / "canary_uuids.json"
    uuids.parent.mkdir(parents=True, exist_ok=True)
    uuids.write_text(json.dumps([u for _, u in picked]))
    domains = sorted({d for d, _ in picked})
    config = REPO / "output" / "subtask4" / "canary.yaml"
    config.write_text(f"name: canary\nmax_iterations: 12\ndomains: [{', '.join(domains)}]\n"
                      f"uuids_file: {uuids}\nsystem: {{}}\n")
    sh(f"rm -rf {OUTPUT}/canary-*")
    start = time.time()
    sh(f"{PY} -m subtask4 run {config} --allow-dirty --mcp-config subtask4/deploy/mcp_native.yaml "
       f"> {OUTPUT}/canary.out 2>&1", timeout=CANARY_LIMIT_S * len(picked) + 300)
    runs = glob.glob(str(OUTPUT / "canary-*"))
    if not runs:
        check(f"canary run ({len(picked)} dialogues)", False, f"no output; see {OUTPUT}/canary.out")
        return False
    p, b = run_progress(Path(runs[0])), behaviour(Path(runs[0]))
    worst = max(p["times"], default=0)
    check(f"canary run ({len(picked)} dialogues: longest histories, each rule path, regressions)",
          p["done"] == len(picked) and not p["timeouts"] and not p["errors"] and worst <= CANARY_LIMIT_S
          and b.get("empty answers", 0) == 0 and b["dialogues"] == len(picked),
          f"{p['done']}/{len(picked)} done, {p['timeouts']} timeouts, {p['errors']} errors, slowest {worst:.0f}s, "
          f"{b.get('empty answers', 0)} empty answers, {time.time() - start:.0f}s in all")
    print(f"[{now()}] preflight {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


# ------------------------------------------------------------------ watch

def status_lines() -> tuple[list[str], list[str]]:
    lines, alerts = [], []
    run = newest_run()
    if run is not None:
        p = run_progress(run)
        pace = ""
        if len(p["times"]) >= 10:
            mean = sum(p["times"][-50:]) / len(p["times"][-50:])
            remaining = max(p["items_started"] - p["done"] - p["timeouts"] - p["errors"], 0)
            pace = f", ~{mean:.0f}s/dialogue, {remaining} left in started domains"
        lines.append(f"{p['run']}: {p['status']}, {p['done']} done, {p['timeouts']} timeouts, {p['errors']} errors, "
                     f"{p['slow']} slower than {SLOW_S}s{pace}")
        if p["timeouts"] or p["errors"]:
            alerts.append(f"{p['run']}: {p['timeouts']} timeouts, {p['errors']} errors")
        if p["slow"]:
            alerts.append(f"{p['run']}: {p['slow']} dialogues slower than {SLOW_S}s")
        if (rate := judge_pace(run)) is not None:
            lines.append(f"judge: {rate:.1f}s per dialogue")
            if rate > JUDGE_SLOW_S:
                alerts.append(f"judge slow: {rate:.0f}s per dialogue")
        b = behaviour(run)
        if b["dialogues"]:
            rates = {k: b.get(k, 0) / b["dialogues"] for k in LIMITS}
            lines.append("behaviour: " + ", ".join(f"{k} {v:.1%}" if k != "made-up tools per dialogue" else f"{k} {v:.2f}"
                                                   for k, v in rates.items())
                         + " | " + ", ".join(f"{k} {v}" for k, v in b.items() if k.startswith("path")))
            if b["dialogues"] >= 30:
                alerts += [f"{k} at {v:.1%} (limit {LIMITS[k]:.0%})" for k, v in rates.items() if v > LIMITS[k]]
    util, mem = gpu_use()
    lines.append(f"GPU {util}% busy, {mem // 1024} GB used; memory pressure {memory_pressure():.0f}%; "
                 f"ollama {ollama_models()}")
    alerts += health()
    for queue_log in glob.glob(str(Path.home() / "queue_*.log")):
        alerts += [f"{Path(queue_log).name}: {line.strip()}" for line in Path(queue_log).read_text().splitlines()
                   if "FAILED" in line]
    return lines, alerts


def watch(log: str, every_s: int = 60) -> None:
    seen: set[str] = set()
    while True:
        lines, alerts = status_lines()
        with open(log, "a") as out:
            out.write(f"[{now()}] " + " || ".join(lines) + "\n")
            for alert in alerts:
                key = re.sub(r"\d+(\.\d+)?", "#", alert)  # one alert per kind of problem, not per count change
                if key not in seen:
                    seen.add(key)
                    out.write(f"[{now()}] ALERT {alert}\n")
        time.sleep(every_s)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "preflight":
        sys.exit(0 if preflight() else 1)
    if command == "watch":
        watch(sys.argv[2] if len(sys.argv) > 2 else str(Path.home() / "monitor.log"))
    lines, alerts = status_lines()
    print("\n".join(lines + [f"ALERT {a}" for a in alerts]))


if __name__ == "__main__":
    main()
