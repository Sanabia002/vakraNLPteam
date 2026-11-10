"""Score a run with VAKRA's official evaluator (policy judge off, as in public evaluation).

Judge replies are cached in output/subtask4/judge_cache.sqlite, keyed by the exact prompt and judge settings:
re-scoring a run gives the same verdicts, and a dialogue answered identically by two configurations is judged
identically, so judge noise does not enter paired comparisons.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumps, loads

from . import data
from .run import OUTPUT, load_record


@dataclass(frozen=True)
class Judge:
    model: str
    end_point: str        # OpenAI-compatible server, without /v1
    key_file: str | None  # read into API_KEY in-process; never printed


JUDGES = {
    "groq": Judge("openai/gpt-oss-120b", "https://api.groq.com/openai", "~/.groq_key"),
    "local": Judge("gpt-oss:120b", "http://localhost:11434", None),  # self-hosted with Ollama
}


class SQLiteCache(BaseCache):
    """A persistent LangChain LLM cache in one SQLite file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (prompt TEXT, llm TEXT, reply TEXT, PRIMARY KEY (prompt, llm))")

    def lookup(self, prompt, llm_string):
        row = self._db.execute("SELECT reply FROM cache WHERE prompt = ? AND llm = ?", (prompt, llm_string)).fetchone()
        return loads(row[0]) if row else None

    def update(self, prompt, llm_string, return_val):
        self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (prompt, llm_string, dumps(return_val)))
        self._db.commit()

    def clear(self, **kwargs):
        self._db.execute("DELETE FROM cache")
        self._db.commit()


def evaluate(run_id: str, judge_name: str = "local", mcp_config: str = "benchmark/mcp_connection_config.yaml",
             final: bool = False) -> Path:
    """Write output/subtask4/<run_id>/eval_<judge>.json and return its path."""
    record = load_record(run_id)
    data.check_access(record["domains"], final)
    judge = JUDGES[judge_name]
    from .deploy.services import apply_env
    apply_env()
    # The evaluator builds default judges when imported, from these variables.
    os.environ["JUDGE_BACKEND"] = "groq"  # VAKRA's OpenAI-compatible judge client, whatever the host
    os.environ["API_KEY"] = Path(judge.key_file).expanduser().read_text().strip() if judge.key_file else "local"
    sys.path.append(str(data.REPO / "evaluator"))  # the evaluator imports its siblings as top-level modules
    from benchmark.mcp_client import load_mcp_config
    from evaluator.evaluator import CapabilityPolicy, evaluate_capability
    from judge import CorrectnessJudge, ExactMatchJudge, GroundednessJudge

    set_llm_cache(SQLiteCache(OUTPUT / "judge_cache.sqlite"))
    settings = {"backend": "groq", "model_name": judge.model, "end_point": judge.end_point}
    policy = CapabilityPolicy(correctness_judge=CorrectnessJudge(config=settings),
                              groundedness_judge=GroundednessJudge(config=settings),
                              exactmatch_judge=ExactMatchJudge(config=settings))
    out = OUTPUT / run_id / f"eval_{judge_name}.json"
    evaluate_capability(capability_name="capability_multiturn", gt_dir=data.TRAIN / "output",
                        pred_dir=OUTPUT / run_id, out_path=out, registry={"capability_multiturn": policy},
                        mcp_config=load_mcp_config(mcp_config).get(4), selected_domains=set(record["domains"]))
    record.setdefault("evaluations", {})[judge_name] = {"model": judge.model, "file": out.name}
    (OUTPUT / run_id / "run.json").write_text(json.dumps(record, indent=1) + "\n")
    return out
