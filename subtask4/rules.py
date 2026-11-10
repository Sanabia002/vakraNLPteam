"""Subtask 4 tool-use rules.

Every rule in the training data is one of four templates:

    Use document retrievers to answer questions. Do not use any other type of tool.            docs only
    Do not use document retrievers to answer questions. Use other types of tools.              APIs only
    If a user's query pertains to <C>, which is/are about <D>, make sure you try answering
        them by only using document retrievers. Do not use other types of tools.               docs only, topic C
    If a user's query pertains to <C>, which is/are about <D>, make sure you do not use
        document retrievers to try answering those questions. Use other types of tools.        APIs only, topic C

<C> is one of the twelve CATEGORIES below. A topic rule sometimes names a category unrelated to the
domain; it then does not apply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Source(str, Enum):
    """The kind of tool a rule allows."""
    DOCS = "docs"
    APIS = "apis"


@dataclass(frozen=True)
class Rule:
    allowed: Source
    category: str | None = None  # None: the rule covers every question

    @property
    def kind(self) -> str:
        """Short label for tables: docs, apis, docs|topic, apis|topic."""
        return self.allowed.value + ("|topic" if self.category else "")


class UnknownRule(ValueError):
    """A non-empty rule that matches none of the four templates."""


CATEGORIES = {
    "Education & Students": "Topics related to schools, universities, student information, and education statistics",
    "Finance & Economics": "Topics that revolve around financial data, transactions, donations, and currencies",
    "Food, Restaurants & Beverage": "Topics concerning restaurants, food inspection, menu items, and beverages like beer",
    "Geography & Demographics": "Topics centered around population, regional information, countries, and life expectancy",
    "Government, Law & Public Services, Human Capital":
        "Topics concerning legislation, public databases, law enforcement, and civic institutions",
    "Health & Medicine": "Topics that pertain to health data, medical conditions, surveys, and synthetic patients",
    "Literature, Language & Publishing": "Topics focusing on books, authors, literary works, and linguistic corpora",
    "Movies, TV & Entertainment, Social Media": "Topics focused on movies, episodes, characters, and streaming platforms",
    "Retail, Sales & Commerce":
        "Topics related to retail operations, customer purchases, inventory, and product ordering",
    "Sports & Athletics": "Topics covering various sports, athletes, games, and statistics from different leagues",
    "Technology & Software":
        "Topics focusing on codebases, software platforms, applications, and user interactions in tech",
    "Transportation & Mobility": "Topics dealing with modes of transport such as trains, cars, bikes, and flights",
}

_UNCONDITIONAL = {
    "Use document retrievers to answer questions. Do not use any other type of tool.": Source.DOCS,
    "Do not use document retrievers to answer questions. Use other types of tools.": Source.APIS,
}
_TOPIC = re.compile(r"If a user's query pertains to (?P<category>.+?), which is/are about .+?, make sure you (?P<action>.+)")
_TOPIC_ACTIONS = {
    "try answering them by only using document retrievers. Do not use other types of tools.": Source.DOCS,
    "do not use document retrievers to try answering those questions. Use other types of tools.": Source.APIS,
}


def parse_rule(text: str | None) -> Rule | None:
    """The rule in `text`; None when there is no rule. Raises UnknownRule for an unrecognised one."""
    text = " ".join((text or "").split())
    if not text:
        return None
    if text in _UNCONDITIONAL:
        return Rule(_UNCONDITIONAL[text])
    match = _TOPIC.fullmatch(text)
    if match and match["action"] in _TOPIC_ACTIONS:
        return Rule(_TOPIC_ACTIONS[match["action"]], match["category"])
    raise UnknownRule(text)


def is_retriever(tool_name: str) -> bool:
    """Document search: `query_<domain>` on the live server, `retriever_*` in the gold files."""
    return tool_name.startswith(("query_", "retriever"))
