"""BASELINE_GROUND_ANSWER on LangGraphReActAgent, with a fake LLM and a fake compiled ReAct graph."""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import agents.plan_execute_agent as pe
from agents.agent_interface import LangGraphReActAgent

Q = "How many albums did the top artist release?"


class FakeLLM:
    """Records every bind() kwargs and ainvoke() messages; bind_tools must never be used for the grounded call."""
    def __init__(self, reply="12 albums", exc=None):
        self.reply, self.exc = reply, exc
        self.binds, self.calls = [], []

    def bind(self, **kw):
        self.binds.append(kw)
        return self

    def bind_tools(self, *a, **kw):
        raise AssertionError("grounded call must be tool-free")

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return AIMessage(content=self.reply)


class FakeGraph:
    def __init__(self, messages):
        self.messages = messages

    async def ainvoke(self, inp, config=None):
        return {"messages": list(inp["messages"]) + self.messages}


def with_tools(result="Artist X"):
    return [AIMessage(content="", tool_calls=[{"name": "top_artist", "args": {"k": 1}, "id": "c1"}]),
            ToolMessage(content=result, tool_call_id="c1", name="top_artist"),
            AIMessage(content="original answer")]


def make_agent(llm, graph_msgs):
    agent = LangGraphReActAgent.__new__(LangGraphReActAgent)  # skip tool/graph building
    agent._llm, agent._tools, agent._max_iterations = llm, [], None
    agent._initial_data_handle = agent._initial_data_peek = None
    agent._shortlister, agent._agent = None, FakeGraph(graph_msgs)
    return agent


def run(agent, q=Q):
    return asyncio.run(agent.run(q))


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("BASELINE_GROUND_ANSWER", "1")
    monkeypatch.delenv("PLAN_RESULT_CHARS", raising=False)


def test_flag_off_no_extra_call(monkeypatch):
    monkeypatch.delenv("BASELINE_GROUND_ANSWER", raising=False)
    llm = FakeLLM()
    r = run(make_agent(llm, with_tools()))
    assert r.content == "original answer" and llm.calls == [] and r.metadata == {}


def test_flag_on_one_grounded_call_same_prompt_cap_and_cut(flag_on):
    llm = FakeLLM(reply="12 albums")
    long_result = "x" * 1000
    r = run(make_agent(llm, with_tools(long_result)))
    assert r.content == "12 albums"
    assert len(llm.calls) == 1 and llm.binds == [{"temperature": 0, "max_tokens": pe.FINAL_MAX_TOKENS}]
    sys_msg, user_msg = llm.calls[0]
    assert isinstance(sys_msg, SystemMessage) and sys_msg.content is pe.FINAL_SYSTEM_GROUNDED
    expected_results = pe.format_tool_results(
        [{"tool_name": "top_artist", "arguments": {"k": 1}, "result": long_result}], pe.RESULT_CHARS_DEFAULT)
    assert isinstance(user_msg, HumanMessage)
    assert user_msg.content == pe.FINAL_USER_TEMPLATE.format(question=Q, results=expected_results)
    assert "x" * 600 + "...(truncated)" in user_msg.content and "x" * 601 not in user_msg.content
    assert r.metadata == {"baseline_ground_answer": "used"}
    assert r.trajectory[-1]["grounded_answer"] and r.trajectory[-1]["original_content"] == "original answer"
    assert r.tool_calls == [{"tool_name": "top_artist", "arguments": {"k": 1}, "result": long_result}]


def test_plan_result_chars_env_respected(flag_on, monkeypatch):
    monkeypatch.setenv("PLAN_RESULT_CHARS", "10")
    llm = FakeLLM()
    run(make_agent(llm, with_tools("y" * 50)))
    assert "y" * 10 + "...(truncated)" in llm.calls[0][1].content


def test_flag_on_no_tool_calls_keeps_original(flag_on):
    llm = FakeLLM()
    r = run(make_agent(llm, [AIMessage(content="direct answer")]))
    assert r.content == "direct answer" and llm.calls == [] and r.metadata == {}


@pytest.mark.parametrize("llm,status", [(FakeLLM(reply=""), "empty"),
                                         (FakeLLM(reply="<think>hmm</think>"), "empty"),
                                         (FakeLLM(exc=RuntimeError("boom")), "error")])
def test_empty_or_error_keeps_original(flag_on, llm, status):
    r = run(make_agent(llm, with_tools()))
    assert r.content == "original answer" and r.metadata == {"baseline_ground_answer": status}
    assert not any(e.get("grounded_answer") for e in r.trajectory)


def test_think_stripped(flag_on):
    r = run(make_agent(FakeLLM(reply="<think>reasoning</think> 12"), with_tools()))
    assert r.content == "12"


def test_question_from_message_list(flag_on):
    from agents.agent_interface import Message
    llm = FakeLLM()
    agent = make_agent(llm, with_tools())
    asyncio.run(agent.run([Message(role="user", content="list question?")]))
    assert llm.calls[0][1].content.startswith("Question: list question?\nTool results:\n1. top_artist(")


def test_format_tool_results_matches_old_history_text():
    recs = [{"tool_name": "a", "arguments": {"s": "v"}, "result": "z" * 8}, {"tool_name": "b", "arguments": {},
                                                                            "result": [1]}]
    assert pe.format_tool_results(recs, 5) == '1. a({"s": "v"}) -> zzzzz...(truncated)\n2. b({}) -> [1]'
    assert pe.format_tool_results([], 5) == "(none yet)"
