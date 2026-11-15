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
    """The Decision Graph's switches, one per component, named after it. Defaults are the final system; switching
    every component off (configs/rebuild_check.yaml) gives back the Rule Gate.

    Read     rule_reader, topic_check, tool_filter
    Act      tool_ranker (+ tool_order), search_on_demand, value_check, empty_recovery, loop_guard, verified_lookup
             (the Rule Guard is always on; the Step Budget is RunConfig.max_iterations)
    Answer   evidence_check, grounded_answer
    """
    rule_reader: Literal["patterns", "patterns+model"] = "patterns+model"    # the model reads a rule in new wording
    topic_check: Literal["domain", "question"] = "question"                  # decide per domain (Rule Gate) or per question
    topic_check_context: bool = False                                        # show the domain's document description
    tool_filter: Literal["rule", "rule+own-search"] = "rule+own-search"      # no-rule path: only the domain's own search
    tool_ranker: Literal["once", "stepwise", "stepwise+rewrite"] = "stepwise+rewrite"
    tool_order: Literal["ranked", "shuffled"] = "ranked"                     # shuffled: the order test only
    search_on_demand: Literal["off", "relevance", "fallback", "instruct", "hide"] = "relevance"  # off: search pinned
    evidence_check: Literal["refuse", "code", "critic", "critic+research"] = "critic+research"   # refuse: Rule Gate
    evidence_phrasings: Literal[1, 3] = 3                                    # searches in the first round
    evidence_rounds: int = 2                                                 # extra search rounds (critic+research)
    evidence_match: float = 0.9                                              # word overlap a cited sentence needs
    evidence_scope: bool = True                                              # critic: database-record questions out of scope
    verified_lookup: bool = True                                             # a document search inside the loop is checked
    value_check: bool = True
    empty_recovery: Literal["off", "hint", "retry"] = "retry"                # retry: other forms of the values, then hint
    loop_guard: bool = True
    grounded_answer: bool = True


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
