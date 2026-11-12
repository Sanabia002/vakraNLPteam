"""The new system behind VAKRA's agent interface, so the benchmark runner can run it like any other agent.

GraphAgent keeps what the interim agent already does right (rule delivery for single-turn input, the guard against
bare-JSON answers) and replaces its run(): the dialogue goes through the agent graph (graph.System) instead of
VAKRA's prebuilt ReAct agent. The decision (path, rule, topic) is the first trajectory entry and the full decision
log the last, so both are saved with the run.
"""
from __future__ import annotations

import json
from typing import List, Union

from agents.agent_interface import AgentResponse, Message

from ..agent import Subtask4Agent, _question
from ..data import REPO
from .graph import System, initial_state

_DESCRIPTIONS = REPO / "environment" / "retrievers" / "domain_description_dict.json"


def _description(domain: str) -> str:
    try:
        return json.loads(_DESCRIPTIONS.read_text()).get(domain, {}).get("description", "")
    except (OSError, ValueError):
        return ""


class GraphAgent(Subtask4Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._system = System(llm=self._llm, tools=self._tools, domain=self._domain, config=self._config.system,
                              topics=self._topics, shortlister=self._shortlister, top_k=self._config.top_k,
                              max_steps=self._config.max_iterations, parse_text_call=self._parse_json_tool_call,
                              domain_description=_description(self._domain))
        self._graph = self._system.build()

    def _build_agent(self, tools):
        return None  # the agent graph replaces VAKRA's prebuilt ReAct agent

    async def run(self, input: Union[str, List[Message]], additional_instructions: str = None) -> AgentResponse:
        history = [] if isinstance(input, str) else [{"role": m.role, "content": m.content} for m in input]
        if history and history[-1]["role"] == "user":
            history = history[:-1]
        state = initial_state(question=_question(input), history=history, rule_text=additional_instructions or "",
                              messages=self._input_messages(input, additional_instructions))
        final = await self._graph.ainvoke(state, config={"recursion_limit": self._system.recursion_limit()})
        decision = {"type": "Subtask4Decision", "engine": "graph", "rule": final.get("rule"),
                    "applies": final.get("applies"), "mode": final.get("mode"), "hops": final.get("hop", 0),
                    "evidence": {k: final["ev"].get(k) for k in ("caller", "verdict", "round", "queries")}
                    if final.get("ev") else None}
        shown = [e["shown"] for e in final["log"] if e["node"] == "rank"]
        return AgentResponse(content=final.get("answer") or "", tool_calls=final["tool_calls"],
                             trajectory=[decision, *_trajectory(final["messages"]),
                                         {"type": "Subtask4Log", "events": final["log"]}],
                             metadata={"subtask4": decision}, all_tools=[t.name for t in self._tools],
                             shortlisted_tools=shown[0] if shown else [])


def _trajectory(messages: list) -> list[dict]:
    """The model's conversation, in the shape the upstream agent saves."""
    out = []
    for m in messages:
        entry = {"type": m.__class__.__name__, "content": getattr(m, "content", "")}
        if getattr(m, "tool_calls", None):
            entry["tool_calls"] = [{"id": c.get("id"), "name": c.get("name"), "args": c.get("args")} for c in m.tool_calls]
        if entry["type"] == "ToolMessage":
            entry.update(tool_name=getattr(m, "name", None), tool_call_id=getattr(m, "tool_call_id", None),
                         result=m.content)
        out.append(entry)
    return out
