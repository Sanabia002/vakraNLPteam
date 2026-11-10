"""Subtask 4 data: the tune / dev / course-test domain split, dialogue inputs, gold answers.

tune  every domain outside dev and the course test (hockey included: it is VAKRA's demo domain,
      kept out of dev and test only)
dev   splits/dev_split.json, used to compare configurations
test  splits/course_test_split.json, frozen 2026-11-05; checked against its SHA-256 on every use, and
      readable only by a run or score marked final
"""
from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "train" / "capability_4_multiturn"
SPLITS = REPO / "splits"
SPLIT_NAMES = ("tune", "dev", "test")
REFUSAL_MARK = "i can not answer"  # the gold answer of every should-refuse dialogue contains this


class FrozenSplitError(RuntimeError):
    """An attempt to touch the frozen course test set outside a final run."""


def _split_domains(name: str) -> frozenset[str]:
    return frozenset(json.loads((SPLITS / name).read_text())["capability_4"]["dev_domains"])


def test_domains() -> frozenset[str]:
    """The frozen course-test domains, after checking the split file against its recorded SHA-256."""
    path = SPLITS / "course_test_split.json"
    expected = (SPLITS / "course_test_split.sha256").read_text().split()[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise FrozenSplitError(f"{path} does not match its recorded SHA-256; the frozen split was changed")
    return _split_domains(path.name)


def all_domains() -> list[str]:
    return sorted(p.stem for p in (TRAIN / "input").glob("*.json"))


def domains(split: str) -> list[str]:
    """The domains of one split, sorted."""
    dev, test = _split_domains("dev_split.json"), test_domains()
    chosen = {"dev": dev, "test": test}.get(split)
    if split == "tune":
        chosen = set(all_domains()) - dev - test
    if chosen is None:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    return sorted(chosen)


def check_access(names: list[str] | set[str], final: bool = False) -> None:
    """Refuse any frozen test domain unless this is the final run (or the final run's scoring)."""
    frozen = sorted(set(names) & test_domains())
    if frozen and not final:
        raise FrozenSplitError(f"frozen course-test domain(s) {frozen}: only a final run may use them")


@cache
def _load(kind: str, domain: str) -> dict[str, dict]:
    return {item["uuid"]: item for item in json.loads((TRAIN / kind / f"{domain}.json").read_text())}


def inputs(domain: str, final: bool = False) -> dict[str, dict]:
    """uuid -> input dialogue (with its `additional_instructions` rule)."""
    check_access([domain], final)
    return _load("input", domain)


def gold(domain: str, final: bool = False) -> dict[str, dict]:
    """uuid -> gold dialogue (answers and tool calls for every turn)."""
    check_access([domain], final)
    return _load("output", domain)


def should_refuse(gold_item: dict) -> bool:
    return REFUSAL_MARK in str(gold_item["output"][-1].get("answer", "")).lower()


def gold_tool_names(gold_item: dict) -> list[str]:
    """Tool names of the scored (last) turn."""
    return [call["name"] for call in gold_item["output"][-1]["sequence"].get("tool_call") or []]
