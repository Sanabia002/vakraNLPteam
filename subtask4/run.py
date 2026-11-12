"""Run one configuration through VAKRA's benchmark runner, in-process, and keep a record of it.

Everything lands in output/subtask4/<run_id>/: the predictions ({domain}.json, the evaluator's input), the tool
shortlists, the trajectories (each starting with the agent's Subtask4Decision), run.log, and run.json: settings,
code version, environment, model server, split fingerprint, timings and item counts.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from . import data
from .config import RunConfig

OUTPUT = data.REPO / "output" / "subtask4"
PACKAGES = ("langchain-core", "langgraph", "langchain-litellm", "mcp", "sentence-transformers", "rank-bm25", "torch")


def load_record(run_id: str) -> dict:
    return json.loads((OUTPUT / run_id / "run.json").read_text())


def code_version() -> dict:
    """Commit, branch, and whether tracked files differ from it (with a hash of the difference)."""
    git = lambda *args: subprocess.run(["git", *args], cwd=data.REPO, capture_output=True, text=True).stdout.strip()
    diff = git("diff", "HEAD")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "dirty": bool(diff),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else None}


def _get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as reply:
            return json.load(reply)
    except Exception as error:  # the record says why, rather than failing the run
        return f"unavailable: {type(error).__name__}"


def environment() -> dict:
    """Python, packages, and the model server: the models it serves and, from Ollama, their digests."""
    base = os.environ["LITELLM_BASE_URL"].rstrip("/")
    served, tags = _get_json(f"{base}/models"), _get_json(f"{base.removesuffix('/v1')}/api/tags")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in PACKAGES},
            "model_server": {"base_url": base,
                             "models": [m["id"] for m in served["data"]] if isinstance(served, dict) else served,
                             "digests": {m["name"]: m["digest"] for m in tags["models"]} if isinstance(tags, dict)
                             else tags}}


def _topics(config: RunConfig, domains: list[str], llm):
    from .topics import OracleTopics, TopicClassifier, gold_labels
    if config.topics == "oracle":
        return OracleTopics(gold_labels(domains))
    return TopicClassifier(llm)


def run(config: RunConfig, final: bool = False, allow_dirty: bool = False, resume: bool = False,
        mcp_config: str = "benchmark/mcp_connection_config.yaml") -> Path:
    split_domains = data.domains(config.split)
    domains = list(config.domains) or split_domains
    if not set(domains) <= set(split_domains):
        raise SystemExit(f"domains {sorted(set(domains) - set(split_domains))} are not in the {config.split} split")
    if final != (config.split == "test"):
        raise SystemExit("--final is required for the test split, and only for it")
    data.check_access(domains, final)
    code = code_version()
    if code["dirty"] and not allow_dirty:
        raise SystemExit("tracked files differ from the last commit: commit first, or pass --allow-dirty for a smoke run")
    out = OUTPUT / config.run_id
    if out.exists() and not resume:
        raise SystemExit(f"{out} exists: change the config's name, or pass --resume to finish an interrupted run")

    import benchmark_runner
    from benchmark import runner_helpers
    from benchmark.mcp_client import MCPConnectionConfig, load_mcp_config

    from .agent import Subtask4Agent
    from .system.agent import GraphAgent

    topics = None

    def make_agent(llm, tools, top_k_tools, max_iterations, domain):
        nonlocal topics
        topics = topics or _topics(config, domains, llm)
        agent_class = GraphAgent if config.system else Subtask4Agent
        return agent_class(config=config, domain=domain, topics=topics, llm=llm, tools=tools,
                           top_k_tools=top_k_tools, max_iterations=max_iterations)

    from .deploy.services import apply_env
    apply_env()
    out.mkdir(parents=True, exist_ok=True)
    record = {"run_id": config.run_id, "settings": config.settings(), "notes": config.notes, "final": final,
              "domains": domains, "code": code, "environment": environment(), "mcp_config": mcp_config,
              "test_split_sha256": (data.SPLITS / "course_test_split.sha256").read_text().split()[0],
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "running"}
    (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")

    os.environ["SAVE_TRAJECTORY"] = "1"  # the runner then writes {domain}_trajectory.json
    runner_helpers.CAPABILITY_PATHS[4] = str(data.TRAIN)  # all our splits are domains of the public training release
    benchmark_runner.AGENT_FACTORY = make_agent
    start = time.perf_counter()
    try:
        results = asyncio.run(benchmark_runner.run_capability(
            capability_id=4, cfg=load_mcp_config(mcp_config).get(4, MCPConnectionConfig()), provider="litellm",
            model=config.model, max_samples_per_domain=config.max_per_domain, output_dir=str(out),
            domains=domains, top_k_tools=config.top_k, max_iterations=config.max_iterations,
            temperature=config.temperature, uuids_file=config.uuids_file))
        record.update(status="done" if results else "failed", items=len(results),
                      status_counts=dict(collections.Counter(r.status for r in results)))
    finally:
        benchmark_runner.AGENT_FACTORY = None
        if record["status"] == "running":
            record["status"] = "failed"
        record.update(wall_s=round(time.perf_counter() - start, 1),
                      finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")
    return out
