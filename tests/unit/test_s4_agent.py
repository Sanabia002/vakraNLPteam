"""subtask4.agent: the rule decision, P1 router, P2 gate, rule delivery, and the runner's agent-factory seam.
Fake model and graph; no servers."""
import asyncio

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool

import benchmark_runner
from agents.agent_interface import LangGraphReActAgent, Message
from subtask4.agent import REFUSAL, Subtask4Agent
from subtask4.config import RunConfig

DOCS = "Use document retrievers to answer questions. Do not use any other type of tool."
APIS = "Do not use document retrievers to answer questions. Use other types of tools."
DOCS_SPORTS = ("If a user's query pertains to Sports & Athletics, which is/are about Topics covering various sports, "
               "athletes, games, and statistics from different leagues, make sure you try answering them by only "
               "using document retrievers. Do not use other types of tools.")
NAMES = ["get_team_wins", "get_player_age", "query_hockey", "query_books"]


def tool(name, reply="passage"):
    def fn(query: str = "") -> str:
        return reply
    return StructuredTool.from_function(fn, name=name, description=name)


class FakeLLM:
    def __init__(self, reply="no"):
        self.reply, self.prompts = reply, []

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self.reply)


class FakeGraph:
    def __init__(self, tools):
        self.tools = [t.name for t in tools]

    async def ainvoke(self, inp, config=None):
        self.messages = inp["messages"]
        return {"messages": list(inp["messages"]) + [AIMessage(content="The answer is 42.")]}


class Topics:
    def __init__(self, category):
        self.category_, self.asked = category, 0

    async def category(self, domain, tool_names):
        self.asked += 1
        return self.category_


class Agent(Subtask4Agent):
    def _build_agent(self, tools):
        self.graph = FakeGraph(tools)
        return self.graph


def make(category="Sports & Athletics", llm_reply="no", search_reply="passage", **settings):
    tools = [tool(n, search_reply) for n in NAMES]
    topics = Topics(category)
    agent = Agent(config=RunConfig("t", **settings), domain="hockey", topics=topics, llm=FakeLLM(llm_reply),
                  tools=tools, top_k_tools=0)
    return agent, topics


def ask(agent, rule, question="Who won in 1990?"):
    return asyncio.run(agent.run(question, additional_instructions=rule))


# ---------- parts off ----------

def test_all_off_shows_every_tool_and_decides_nothing():
    agent, topics = make(deliver_rule=False)
    response = ask(agent, DOCS)
    assert agent.graph.tools == NAMES and response.content == "The answer is 42."
    assert response.trajectory[0] == {"type": "Subtask4Decision", "rule": "docs", "category": None,
                                      "domain_category": None, "applies": None, "gate": None}
    assert topics.asked == 0


def test_rule_delivery_for_single_turn_input():
    agent, _ = make(deliver_rule=True)
    ask(agent, DOCS)
    assert isinstance(agent.graph.messages[0], SystemMessage) and DOCS in agent.graph.messages[0].content
    agent, _ = make(deliver_rule=False)
    ask(agent, DOCS)
    assert not any(isinstance(m, SystemMessage) for m in agent.graph.messages)


def test_rule_delivery_is_not_doubled_for_dialogue_input():
    agent, _ = make(deliver_rule=True)
    asyncio.run(agent.run([Message("user", "hi"), Message("assistant", "hello"), Message("user", "who?")], DOCS))
    assert sum(isinstance(m, SystemMessage) for m in agent.graph.messages) == 1


# ---------- P1 router ----------

def test_router_docs_only_keeps_retrievers():
    agent, _ = make(router=True)
    ask(agent, DOCS)
    assert agent.graph.tools == ["query_hockey", "query_books"]


def test_router_apis_only_drops_retrievers():
    agent, _ = make(router=True)
    ask(agent, APIS)
    assert agent.graph.tools == ["get_team_wins", "get_player_age"]


def test_topic_rule_applies_only_to_its_category():
    agent, topics = make(category="Sports & Athletics", router=True)
    assert ask(agent, DOCS_SPORTS).trajectory[0]["applies"] is True
    assert agent.graph.tools == ["query_hockey", "query_books"]
    agent, topics = make(category="Health & Medicine", router=True)
    assert ask(agent, DOCS_SPORTS).trajectory[0]["applies"] is False
    assert agent.graph.tools == NAMES


def test_unknown_rule_falls_back_to_the_baseline():
    agent, _ = make(router=True, gate="refuse")
    response = ask(agent, "Never use tools on Tuesdays.")
    assert response.content == "The answer is 42." and agent.graph.tools == NAMES
    assert response.trajectory[0]["rule"] == "unknown"


# ---------- P2 gate ----------

def test_gate_refuse_answers_without_any_tool_call():
    agent, _ = make(gate="refuse")
    response = ask(agent, DOCS)
    assert response.content == REFUSAL and response.tool_calls == []
    assert response.trajectory[0]["gate"] == "refused"
    assert "i can not answer" in response.content.lower()


def test_gate_leaves_apis_rules_alone():
    agent, _ = make(gate="refuse")
    assert ask(agent, APIS).content == "The answer is 42."


def test_gate_evidence_refuses_when_documents_do_not_answer():
    agent, _ = make(gate="evidence", llm_reply="No.")
    response = ask(agent, DOCS)
    assert response.content == REFUSAL
    assert response.tool_calls == [{"tool_name": "query_hockey", "arguments": {"query": "Who won in 1990?"},
                                    "result": "passage"}]
    assert response.trajectory[0]["gate"] == "evidence-no"


def test_gate_evidence_carries_on_when_documents_answer():
    agent, _ = make(gate="evidence", llm_reply="Yes", router=True)
    response = ask(agent, DOCS)
    assert response.content == "The answer is 42." and response.trajectory[0]["gate"] == "evidence-yes"
    assert response.tool_calls[0]["tool_name"] == "query_hockey"  # the gate's search is reported, not hidden
    assert agent.graph.tools == ["query_hockey", "query_books"]


class NumberGraph(FakeGraph):
    async def ainvoke(self, inp, config=None):
        return {"messages": list(inp["messages"]) + [AIMessage(content="1990")]}


def test_a_bare_number_answer_is_not_an_agent_error():
    agent, _ = make()
    agent._agent = NumberGraph(agent._tools)
    assert ask(agent, None).content == "1990"


# ---------- runner seam ----------

def test_runner_uses_the_factory_only_when_set(monkeypatch):
    assert benchmark_runner.AGENT_FACTORY is None
    seen = {}
    monkeypatch.setattr(benchmark_runner, "AGENT_FACTORY", lambda **kw: seen.update(kw) or "agent")
    assert benchmark_runner._get_agent(4, "llm", [], 20, 12, domain="hockey") == "agent"
    assert seen == {"llm": "llm", "tools": [], "top_k_tools": 20, "max_iterations": 12, "domain": "hockey"}


def test_base_agent_hooks_keep_upstream_behaviour():
    agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
    agent._tools, agent._shortlister = [tool(n) for n in NAMES], None
    assert agent._select_tools("q", DOCS) is agent._tools
    assert [type(m).__name__ for m in agent._input_messages("q", DOCS)] == ["HumanMessage"]
    messages = agent._input_messages([Message("user", "q")], DOCS)
    assert [type(m).__name__ for m in messages] == ["SystemMessage", "HumanMessage"]
