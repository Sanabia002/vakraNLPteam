"""LangGraph Studio entry point: watch the new system work through one training dialogue, node by node.

    LANGSMITH_TRACING=false LANGSMITH_API_KEY="$(cat ~/.langsmith_key)" \\
        .venv-studio/bin/langgraph dev --config subtask4/deploy/langgraph.json --no-browser

Then, with port 2024 forwarded from the GPU machine, open
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024 in Chrome and start a run with the input
{"domain": "books", "uuid": "<a dialogue uuid>"} (or just {"domain": "books"} for its first dialogue).

The graph Studio shows is the system's own graph (graph.System.build) behind one extra node, `load`, which reads the
dialogue from the training data and builds the starting state exactly as GraphAgent.run does. Only tune and dev
dialogues can be loaded: data.inputs refuses the frozen course-test domains. The run config is
subtask4/configs/system.yaml unless STUDIO_CONFIG names another one.
"""
from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # langgraph dev loads this file by path
    sys.path.insert(0, str(REPO))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from agents.agent_interface import Message  # noqa: E402
from subtask4 import data  # noqa: E402
from subtask4.agent import _question  # noqa: E402
from subtask4.config import RunConfig  # noqa: E402
from subtask4.system.graph import State, initial_state  # noqa: E402


class StudioState(State, total=False):
    domain: str
    uuid: Optional[str]


def dialogue(domain: str, uuid: Optional[str]) -> tuple[list[Message] | str, str]:
    """(the runner's input for this dialogue, its rule), built by the runner's own loader; tune and dev only."""
    from benchmark.runner_helpers import BenchmarkItem
    items = data.inputs(domain)  # raises FrozenSplitError for a course-test domain
    item = BenchmarkItem.from_dict(items[uuid] if uuid else next(iter(items.values())))
    return (item.context or item.query), item.additional_instructions or ""


def starting_state(agent, domain: str, uuid: Optional[str]) -> dict:
    input, rule = dialogue(domain, uuid)
    history = [] if isinstance(input, str) else [{"role": m.role, "content": m.content} for m in input]
    if history and history[-1]["role"] == "user":
        history = history[:-1]
    return initial_state(question=_question(input), history=history, rule_text=rule,
                         messages=agent._input_messages(input, rule))


_AGENTS: dict[str, object] = {}
_STACK = AsyncExitStack()  # keeps each domain's MCP session open for the server's lifetime


async def _agent(domain: str):
    if domain in _AGENTS:
        return _AGENTS[domain]
    from agents.llm import create_llm
    from benchmark.mcp_client import MCPConnectionConfig, create_client_and_connect, load_mcp_config
    from agents.mcp_tool_wrapper import MCPToolWrapper

    from subtask4.deploy.services import apply_env
    from subtask4.system.agent import GraphAgent
    from subtask4.topics import TopicClassifier
    apply_env()
    config = RunConfig.load(os.environ.get("STUDIO_CONFIG", REPO / "subtask4" / "configs" / "system.yaml"))
    llm = create_llm(provider="litellm", model=config.model, temperature=config.temperature)
    mcp = load_mcp_config(str(REPO / "subtask4" / "deploy" / "mcp_native.yaml")).get(4, MCPConnectionConfig())
    session = await _STACK.enter_async_context(create_client_and_connect(mcp, domain))
    tools = await MCPToolWrapper(session).get_tools()
    _AGENTS[domain] = GraphAgent(config=config, domain=domain, topics=TopicClassifier(llm), llm=llm, tools=tools,
                                 top_k_tools=config.top_k, max_iterations=config.max_iterations)
    return _AGENTS[domain]


async def make_graph(config: dict | None = None):
    """The graph Studio runs: `load`, then the system's graph for the requested domain."""
    domain = ((config or {}).get("configurable") or {}).get("domain") or os.environ.get("STUDIO_DOMAIN", "books")
    agent = await _agent(domain)

    async def load(state: StudioState) -> dict:
        target = await _agent(state.get("domain") or domain)
        return starting_state(target, state.get("domain") or domain, state.get("uuid"))

    outer = StateGraph(StudioState)
    outer.add_node("load", load)
    outer.add_node("system", agent._graph)
    outer.add_edge(START, "load")
    outer.add_edge("load", "system")
    outer.add_edge("system", END)
    return outer.compile()
