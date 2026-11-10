"""Which topic category a domain belongs to; this decides whether a topic rule applies (P1).

The agent's own model is asked once per domain, zero-shot, given the domain name, some of its API tool names and
the twelve categories. (Few-shot prompts with the tune labels did worse with a 3B model: it copied the examples.)
The tune labels measure the classifier: in a tune domain, the category named by documents-only topic rules whose
gold answer is a refusal.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from . import data
from .rules import CATEGORIES, Source, is_retriever, parse_rule

LABELS = Path(__file__).with_name("topic_labels.json")
TOOLS_SHOWN = 15
_SERVERS = data.REPO / "environment" / "m3" / "rest" / "server"
_NAMES = list(CATEGORIES)


def domain_tools(domain: str) -> list[str]:
    """A domain's API tool names, read from its REST server source (the same names the MCP server lists)."""
    return re.findall(r'operation_id="([A-Za-z0-9_]+)"', (_SERVERS / f"{domain}.py").read_text())


def gold_labels(domains: list[str], final: bool = False) -> dict[str, str]:
    """domain -> category, for each domain with a documents-only topic rule whose gold answer is a refusal.
    Reads gold, so frozen domains need final=True; used for the tune labels, the oracle and scoring."""
    labels = {}
    for domain in domains:
        gold, votes = data.gold(domain, final), collections.Counter()
        for uuid, item in data.inputs(domain, final).items():
            rule = parse_rule(item["additional_instructions"])
            if rule and rule.category and rule.allowed is Source.DOCS and data.should_refuse(gold[uuid]):
                votes[rule.category] += 1
        if votes:
            labels[domain] = votes.most_common(1)[0][0]
    return labels


def write_labels(path: Path = LABELS) -> dict[str, str]:
    """Store the tune labels (the classifier's check; the agent never reads them)."""
    labels = gold_labels(data.domains("tune"))
    path.write_text(json.dumps(labels, indent=1) + "\n")
    return labels


def load_labels(path: Path = LABELS) -> dict[str, str]:
    return json.loads(path.read_text())


def build_prompt(domain: str, tool_names: list[str]) -> str:
    menu = "\n".join(f"{i}. {name}: {CATEGORIES[name]}" for i, name in enumerate(_NAMES, 1))
    tools = ", ".join(sorted(t for t in tool_names if not is_retriever(t))[:TOOLS_SHOWN])
    return (f"A database named '{domain}' has these API tools: {tools}.\n"
            f"Which ONE category below best describes what this database is about?\n{menu}\n"
            "Reply with the category number only.")


def parse_answer(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    match = re.search(r"\d+", text)
    return _NAMES[int(match.group()) - 1] if match and 1 <= int(match.group()) <= len(_NAMES) else None


class TopicClassifier:
    """Asks the model once per domain and remembers the answer."""

    def __init__(self, llm):
        self._llm, self._seen = llm, {}

    async def category(self, domain: str, tool_names: list[str]) -> str | None:
        if domain not in self._seen:
            reply = await self._llm.ainvoke([HumanMessage(content=build_prompt(domain, tool_names))])
            self._seen[domain] = parse_answer(str(reply.content))
        return self._seen[domain]


class OracleTopics:
    """Upper bound for analysis: gold_labels() of the run's domains. Never used in a final run."""

    def __init__(self, labels: dict[str, str]):
        self._labels = labels

    async def category(self, domain: str, tool_names: list[str]) -> str | None:
        return self._labels.get(domain)


async def check(llm, labels: dict[str, str]) -> tuple[int, list[tuple[str, str, str | None]]]:
    """How many labelled domains the classifier gets right, and the misses (domain, label, predicted)."""
    classifier, misses = TopicClassifier(llm), []
    for domain, label in labels.items():
        predicted = await classifier.category(domain, domain_tools(domain))
        if predicted != label:
            misses.append((domain, label, predicted))
    return len(labels) - len(misses), misses
