"""One run = one YAML file. Every setting that changes the agent's behaviour lives here, so the run ID (the name
plus a hash of the settings) identifies what was run; the run record adds the code version.

The defaults are the intermediate report's "k=20 + policy fix" baseline.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from .data import SPLIT_NAMES


@dataclass(frozen=True)
class SystemConfig:
    """The new system (the LangGraph agent graph, design revision 6). Defaults are the full design; each switch
    names its component. With every new component off it reduces to the interim system (the parity run)."""
    rule_fallback: bool = True                       # C1: the model classifies a rule in new wording
    topic_check: Literal["domain", "question"] = "question"   # C2: once per domain (interim) or per question
    topic_context: bool = False                      # C2: show the domain's document-search description
    own_search: bool = True                          # C3: in the no-rule path, keep only the domain's own search
    evidence: Literal["refuse", "a", "b", "d"] = "d"  # C5 under a documents-only rule: refuse at once (interim),
    #   a = code check only, b = critic once + code check, d = critic with re-search + code check
    phrasings: Literal[1, 3] = 3                     # C5: searches per round (as asked, standalone, keywords)
    max_research: int = 2                            # C5 variant d: extra search rounds
    fuzzy: float = 0.9                               # C5 code check: word overlap a cited sentence needs
    scope_check: bool = True                         # C5 critic: database-record questions are out of scope
    loop_docs: Literal["evidence", "plain"] = "evidence"     # C5 second caller: document searches in the loop
    ranker: Literal["once", "stepwise"] = "stepwise"  # C7: one shortlist (interim) or one per step
    rewrite: bool = True                             # C7: standalone rewrite of back-referring questions
    api_first: Literal["off", "instruct", "hide", "fallback", "relevance"] = "relevance"  # C8 (no-rule path)
    arg_check: Literal["off", "warn"] = "warn"       # C10
    empty: Literal["off", "hint", "retry"] = "retry"  # C11: retry = case variants once, then the hint
    grounded: bool = True                            # C13: rewrite the final answer from the tool results


@dataclass(frozen=True)
class RunConfig:
    name: str
    split: Literal["tune", "dev", "test"] = "tune"
    domains: tuple[str, ...] = ()        # a subset of the split's domains (smoke runs); empty = all
    uuids_file: str | None = None        # a subset of dialogues; recorded by content hash
    max_per_domain: int | None = None

    model: str = "openai/qwen3.5-4b"
    temperature: float = 0.0
    max_iterations: int = 12
    top_k: int = 20

    deliver_rule: bool = True            # give the rule to the model for single-turn input too (the policy fix)
    router: bool = False                 # P1: hide the tools the rule forbids
    topics: Literal["model", "oracle"] = "model"   # P1: how to decide whether a topic rule applies
    gate: Literal["off", "refuse", "evidence"] = "off"  # P2, under an applicable documents-only rule:
    #   refuse    refuse at once, without tool calls
    #   evidence  search the domain's documents first; refuse only if they do not answer the question
    picker: Literal["base", "hybrid"] = "base"     # P3: tool shortlist (hybrid = Melissa's dense + BM25)
    system: SystemConfig | None = None   # the new system's switches; None = the interim agent (router, gate, picker)

    notes: str = field(default="", compare=False)

    def __post_init__(self):
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, not {self.split!r}")
        if self.split == "test" and self.topics == "oracle":
            raise ValueError("the oracle reads gold answers; it cannot be used on the test split")

    @classmethod
    def load(cls, path: str | Path, split: str | None = None) -> "RunConfig":
        """The config in a YAML file; `split` overrides the file's split."""
        settings = yaml.safe_load(Path(path).read_text())
        settings["domains"] = tuple(settings.get("domains") or ())
        if "system" in settings:
            settings["system"] = SystemConfig(**(settings["system"] or {}))
        if split:
            settings["split"] = split
        return cls(**settings)

    def settings(self) -> dict:
        """Every behaviour-changing setting (notes excluded), with a referenced uuids file by content hash."""
        values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.compare}
        values["domains"] = list(self.domains)
        if self.system is None:
            del values["system"]  # older run IDs keep their hashes
        else:
            values["system"] = dataclasses.asdict(self.system)
        if self.uuids_file:
            values["uuids_file"] = hashlib.sha256(Path(self.uuids_file).read_bytes()).hexdigest()[:12]
        return values

    @property
    def run_id(self) -> str:
        digest = hashlib.sha256(json.dumps(self.settings(), sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.name}-{digest}"
