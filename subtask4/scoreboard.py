"""The Subtask 4 scoreboard: one row per dialogue, then the numbers we report.

Per run
  official          the evaluator's mean score, with a 95% Wilson interval
  refusal-checked   no credit on a should-refuse dialogue unless the answer refuses (the scorer does not check)
  rule breaks       dialogues whose applicable rule the agent broke (a forbidden tool called), of those with one
  refusals          should-refuse dialogues refused; answerable dialogues refused (false refusals)
  failures          one cause per lost dialogue (CAUSES, checked in order)
  topic decisions   the agent's "does this topic rule apply" against the gold-derived label
Between two runs, on the same dialogues: gained and lost dialogues, exact McNemar p, domain-bootstrap interval.

Whether a topic rule applies is taken from gold (topics.gold_labels), so rule breaks are only counted where
that label exists. Standard library apart from the package's own modules.
"""
from __future__ import annotations

import collections
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from . import data
from .rules import Source, UnknownRule, is_retriever, parse_rule
from .topics import gold_labels

REFUSAL_PHRASES = ("can not answer", "cannot answer", "can't answer", "unable to answer", "not able to answer")
CAUSES = ("correct", "credited, never refused", "should refuse, answered", "refused, answerable", "agent error",
          "answered without tools", "right data, judged ungrounded", "wrong tool", "partial chain",
          "right tools, wrong arguments", "other")


@dataclass
class Row:
    uuid: str
    domain: str
    rule: str | None          # rule kind, "unknown", or None
    applies: bool | None      # gold-derived; None when unknown
    should_refuse: bool
    refused: bool
    official: int
    checked: int              # refusal-checked
    broke_rule: bool | None   # None when no applicable rule is known
    decided_applies: bool | None
    cause: str


def _family(name: str) -> str:
    return "document search" if is_retriever(name) else name


def _cause(gold: dict, pred: dict, turn: dict, official: int, refused: bool) -> str:
    meta = turn.get("metadata") or {}
    should_refuse = data.should_refuse(gold)
    if official:
        return "credited, never refused" if should_refuse and not refused else "correct"
    if should_refuse:
        return "should refuse, answered"
    if refused:
        return "refused, answerable"
    if pred.get("status") != "success":
        return "agent error"
    used = {_family(c["name"]) for c in pred["output"][-1]["sequence"].get("tool_call") or []}
    needed = {_family(n) for n in data.gold_tool_names(gold)}
    if not used:
        return "answered without tools"
    if meta.get("exactmatch_score") == 1.0 and meta.get("groundedness_score") == 0.0:
        return "right data, judged ungrounded"
    if meta.get("exactmatch_score") == 0.0:
        shared = needed & used
        return "wrong tool" if not shared else "partial chain" if shared != needed else "right tools, wrong arguments"
    return "other"


def _by_uuid(path: Path) -> dict[str, dict]:
    return {item["uuid"]: item for item in json.loads(path.read_text())} if path.exists() else {}


def rows(eval_path: Path, pred_dir: Path, final: bool = False) -> list[Row]:
    evaluation = json.loads(Path(eval_path).read_text())
    domains = sorted(evaluation["domains"])
    labels = gold_labels(domains, final)
    pred_dir, out = Path(pred_dir), []
    for domain in domains:
        gold, inputs = data.gold(domain, final), data.inputs(domain, final)
        preds = _by_uuid(pred_dir / f"{domain}.json")
        trajectories = _by_uuid(pred_dir / f"{domain}_trajectory.json")
        for dialogue in evaluation["domains"][domain]["dialogues"]:
            uuid, turn = dialogue["uuid"], dialogue["details"]["per_turn"][-1]
            try:
                rule = parse_rule(inputs[uuid]["additional_instructions"])
            except UnknownRule:
                rule = None
            applies = None if rule is None else True if rule.category is None else (
                labels[domain] == rule.category if domain in labels else None)
            pred = preds[uuid]
            used = [c["name"] for c in pred["output"][-1]["sequence"].get("tool_call") or []]
            broke = None
            if applies:
                broke = any(is_retriever(n) != (rule.allowed is Source.DOCS) for n in used)
            decision = next((e for e in (trajectories.get(uuid) or {}).get("trajectory", [])
                             if e.get("type") == "Subtask4Decision"), {})
            refused = any(p in str(turn.get("pred_answer") or "").lower() for p in REFUSAL_PHRASES)
            official = int(dialogue["score"] == 1.0)
            should_refuse = data.should_refuse(gold[uuid])
            out.append(Row(uuid, domain, rule.kind if rule else None, applies, should_refuse, refused, official,
                           int(official and (refused or not should_refuse)), broke, decision.get("applies"),
                           _cause(gold[uuid], pred, turn, official, refused)))
    return out


# ---------- statistics ----------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def domain_bootstrap(diffs: dict[str, list[int]], reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean paired difference, resampling whole domains."""
    rng, names = random.Random(seed), list(diffs)
    means = sorted(sum(s := [d for name in rng.choices(names, k=len(names)) for d in diffs[name]]) / len(s)
                   for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


# ---------- summaries ----------

def summary(table: list[Row]) -> dict:
    n = len(table)
    k = sum(r.official for r in table)
    ruled = [r for r in table if r.broke_rule is not None]
    refuse = [r for r in table if r.should_refuse]
    decided = [r for r in table if r.decided_applies is not None and r.applies is not None]
    return {
        "dialogues": n,
        "official": k / n, "official_ci": wilson(k, n),
        "refusal_checked": sum(r.checked for r in table) / n,
        "rule_breaks": [sum(r.broke_rule for r in ruled), len(ruled)],
        "refused_when_should": [sum(r.refused for r in refuse), len(refuse)],
        "false_refusals": [sum(r.refused for r in table if not r.should_refuse), n - len(refuse)],
        "topic_decisions_right": [sum(r.decided_applies == r.applies for r in decided), len(decided)],
        "causes": {c: m for c in CAUSES if (m := sum(r.cause == c for r in table))},
    }


def compare(a: list[Row], b: list[Row]) -> dict:
    """b against a, refusal-checked, on the dialogues both runs have."""
    first, second = {r.uuid: r for r in a}, {r.uuid: r for r in b}
    shared = sorted(first.keys() & second.keys())
    gained = [u for u in shared if second[u].checked > first[u].checked]
    lost = [u for u in shared if second[u].checked < first[u].checked]
    diffs = collections.defaultdict(list)
    for u in shared:
        diffs[first[u].domain].append(second[u].checked - first[u].checked)
    return {"dialogues": len(shared), "difference": sum(sum(d) for d in diffs.values()) / len(shared),
            "interval": domain_bootstrap(diffs), "p": mcnemar_exact(len(lost), len(gained)),
            "gained": gained, "lost": lost}


def write(table: list[Row], path: Path) -> None:
    path.write_text(json.dumps({"summary": summary(table), "rows": [asdict(r) for r in table]}, indent=1) + "\n")


def format_summary(name: str, s: dict) -> str:
    lo, hi = s["official_ci"]
    pair = lambda key: f"{s[key][0]}/{s[key][1]}"
    lines = [f"{name}  ({s['dialogues']} dialogues)",
             f"  official          {100 * s['official']:5.1f}  [{100 * lo:.1f}, {100 * hi:.1f}]",
             f"  refusal-checked   {100 * s['refusal_checked']:5.1f}",
             f"  rule breaks       {pair('rule_breaks')}",
             f"  refused / should  {pair('refused_when_should')}    false refusals {pair('false_refusals')}",
             f"  topic decisions   {pair('topic_decisions_right')} right",
             "  causes            " + ", ".join(f"{c} {m}" for c, m in s["causes"].items())]
    return "\n".join(lines)
