"""BASELINE_ANSWER_ON_LIMIT on LangGraphReActAgent: fake graphs, fake LLMs, no model server."""
import asyncio
import logging
import os
import warnings
from typing import List, Union

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError

import agents.plan_execute_agent as pe
from agents.agent_interface import AgentResponse, LangGraphReActAgent, Message
from benchmark.runner_helpers import BenchmarkResult, log_message_history, log_trajectory

logger = logging.getLogger("test_answer_on_limit")

Q = "How many albums did the top artist release?"
FLAGS = ("BASELINE_ANSWER_ON_LIMIT", "BASELINE_GROUND_ANSWER", "PLAN_RESULT_CHARS")


class OriginalReAct(LangGraphReActAgent):
    """Reference: VERBATIM copy of LangGraphReActAgent.run from commit 553c756 (before this flag)."""

    async def run(
        self,
        input: Union[str, List[Message]],
        additional_instructions: str = None,
    ) -> AgentResponse:
        """Run the ReAct agent with given input."""
        query_preview = input if isinstance(input, str) else next(
            (m.content for m in reversed(input) if m.role == "user"), ""
        )
        logger.debug("run: starting agent run, query_preview=%.200s", query_preview)

        # Determine active tools (shortlisted or full set)
        if self._shortlister:
            if isinstance(input, str):
                query = input
            else:
                query = next(
                    (m.content for m in reversed(input) if m.role == "user"),
                    "",
                )
            active_tools = self._shortlister.shortlist(query, self._tools)
            logger.debug("run: shortlisted to %d tools: %s", len(active_tools), [t.name for t in active_tools])
            agent = self._build_agent(active_tools)
        else:
            active_tools = self._tools
            agent = self._agent
            logger.debug("run: using all %d tools (shortlisting disabled)", len(active_tools))

        all_tool_names = [t.name for t in self._tools]
        active_tool_names = [t.name for t in active_tools]

        # Build tool map for fallback manual execution
        tool_map = {t.name: t for t in active_tools}

        # Convert input to LangChain messages
        lc_messages=[]
        if isinstance(input, str):
            lc_messages.append(HumanMessage(content=input))
        else:
            if (additional_instructions) and (additional_instructions not in [""," "]):
                lc_messages.append(self._build_policy_guidance(additional_instructions=additional_instructions))
            lc_messages.extend(self._messages_to_langchain(input))

        # Inject system message when using server-side handle system
        if self._initial_data_handle is not None:
            has_system = any(m.__class__.__name__ == "SystemMessage" for m in lc_messages)
            if not has_system:
                logger.debug("run: injecting handle-system system message")
                lc_messages = [SystemMessage(content=self._build_system_message())] + lc_messages

        # Build invocation config
        config: RunnableConfig | None = None
        if self._max_iterations is not None:
            config = RunnableConfig(recursion_limit=self._max_iterations * 2 + 1)
            logger.debug("run: recursion_limit=%d (max_iterations=%d)", self._max_iterations * 2 + 1, self._max_iterations)

        # Run the agent
        logger.debug("run: invoking LangGraph agent with %d input messages", len(lc_messages))
        result = await agent.ainvoke({"messages": lc_messages}, config=config)
        logger.debug("run: agent invocation complete, result message count=%d", len(result.get("messages", [])))
        # Extract results
        response_messages = []
        tool_calls = []
        final_content = ""
        trajectory = []

        tool_call_args = {}  # Map tool_call_id → {name, args}
        # RETRIEVER=hybrid only: record which retriever ran (base output stays byte-identical)
        hybrid = getattr(self._shortlister, "_hybrid", None) if self._shortlister else None

        if result and "messages" in result:
            for msg in result["messages"]:
                msg_class = msg.__class__.__name__

                trajectory_entry = {
                    "type": msg_class,
                    "content": getattr(msg, "content", ""),
                }

                if msg_class == "HumanMessage":
                    response_messages.append(Message(role="user", content=msg.content))
                    if hybrid is not None and not any(e.get("type") == "HumanMessage" for e in trajectory):
                        trajectory_entry.update(retriever="hybrid", retriever_build_s=hybrid.build_s)
                    trajectory.append(trajectory_entry)

                elif msg_class == "AIMessage":
                    if msg.content:
                        final_content = msg.content
                    response_messages.append(Message(
                        role="assistant",
                        content=msg.content or msg.additional_kwargs.get("reasoning")
                        ))

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        if msg.additional_kwargs.get("reasoning"):
                            trajectory_entry["reasoning"] = msg.additional_kwargs.get("reasoning")
                        trajectory_entry["tool_calls"] = []
                        for tc in msg.tool_calls:
                            tc_id = tc.get("id", "") or tc.get("tool_call_id", "")
                            tool_call_args[tc_id] = {
                                "name": tc.get("name", "unknown"),
                                "args": tc.get("args", {}),
                            }
                            trajectory_entry["tool_calls"].append({
                                "id": tc_id,
                                "name": tc.get("name", "unknown"),
                                "args": tc.get("args", {}),
                            })
                    trajectory.append(trajectory_entry)

                elif msg_class == "ToolMessage":
                    tool_call_id = getattr(msg, "tool_call_id", "")
                    tool_info = tool_call_args.get(tool_call_id, {})
                    tool_name = getattr(msg, "name", None) or tool_info.get("name", "unknown")
                    tool_calls.append({
                        "tool_name": tool_name,
                        "arguments": tool_info.get("args", {}),
                        "result": msg.content,
                    })
                    trajectory_entry["tool_name"] = tool_name
                    trajectory_entry["tool_call_id"] = tool_call_id
                    trajectory_entry["result"] = msg.content
                    trajectory.append(trajectory_entry)

                elif msg_class == "SystemMessage":
                    trajectory.append(trajectory_entry)

        logger.debug(
            "run: extracted %d tool calls, final_content length=%d",
            len(tool_calls),
            len(final_content),
        )

        # BASELINE_GROUND_ANSWER=1 (default off): rewrite the final answer with ONE tool-free call that uses
        # PlanExecuteAgent's grounded prompt, cap and result cutting. Only when there were tool calls.
        metadata = {}
        if hybrid is not None:
            metadata.update(retriever="hybrid", retriever_build_s=hybrid.build_s)
        if tool_calls and os.environ.get("BASELINE_GROUND_ANSWER", "0").strip().lower() in ("1", "true", "yes", "on"):
            final_content, metadata["baseline_ground_answer"] = await self._ground_answer(
                query_preview, tool_calls, final_content, trajectory)

        # FALLBACK: If no tool calls captured but final_content looks like a JSON tool call,
        # manually parse and execute it (for models that output tool calls as text)
        if not tool_calls and final_content and self._initial_data_handle is None:
            parsed_call = self._parse_json_tool_call(final_content)
            if parsed_call:
                tool_name = parsed_call.get("name", "")
                tool_args = parsed_call.get("arguments", {})
                logger.debug("run: [FALLBACK] detected text tool call: %s(%s)", tool_name, tool_args)

                if tool_name in tool_map:
                    try:
                        tool_result = await tool_map[tool_name].ainvoke(tool_args)
                        tool_result_str = str(tool_result)
                        logger.debug("run: [FALLBACK] tool result: %.200s", tool_result_str)
                        tool_calls.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": tool_result_str,
                        })
                        final_content = tool_result_str
                    except Exception as e:
                        logger.warning("run: [FALLBACK] tool error for %s: %s", tool_name, e)
                        tool_calls.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": f"Error: {e}",
                        })
                else:
                    logger.warning("run: [FALLBACK] tool '%s' not found in available tools", tool_name)

        logger.debug("run: returning AgentResponse with %d tool calls", len(tool_calls))
        return AgentResponse(
            content=final_content,
            tool_calls=tool_calls,
            messages=response_messages,
            metadata=metadata,
            trajectory=trajectory,
            all_tools=all_tool_names,
            shortlisted_tools=active_tool_names,
        )


# ---------------------------------------------------------------- fakes

class FakeLLM:
    """Grounded-call fake: records bind() kwargs and ainvoke() messages."""
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
    """Compiled-agent fake. astream yields cumulative 'values' states, then raises `exc` if given."""
    def __init__(self, messages, exc=None, forbid_astream=False):
        self.messages, self.exc, self.forbid_astream = messages, exc, forbid_astream
        self.used = []

    async def ainvoke(self, inp, config=None):
        self.used.append("ainvoke")
        if self.exc:
            raise self.exc
        return {"messages": list(inp["messages"]) + self.messages}

    async def astream(self, inp, config=None, stream_mode=None):
        if self.forbid_astream:
            raise AssertionError("astream must not be used when the flag is off")
        assert stream_mode == "values"
        self.used.append("astream")
        msgs = list(inp["messages"])
        yield {"messages": list(msgs)}
        for m in self.messages:
            msgs.append(m)
            yield {"messages": list(msgs)}
        if self.exc:
            raise self.exc


class FakeHybrid:
    build_s = 0.123


class FakeShortlister:
    _hybrid = FakeHybrid()

    def shortlist(self, query, tools):
        return tools


def _tool(name="top_artist"):
    def fn(k: int = 1) -> str:
        """fake tool"""
        return f"{name} result {k}"
    return StructuredTool.from_function(fn, name=name)


def make_agent(llm, graph, cls=LangGraphReActAgent, tools=(), hybrid=False):
    agent = cls.__new__(cls)  # skip tool/graph building
    agent._llm, agent._tools, agent._max_iterations = llm, list(tools), None
    agent._initial_data_handle = agent._initial_data_peek = None
    agent._agent = graph
    agent._shortlister = FakeShortlister() if hybrid else None
    agent._build_agent = lambda tools: graph
    return agent


def call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def two_tools_then_limit():
    return [call("top_artist", {"k": 1}, "c1"), ToolMessage(content="Artist X", tool_call_id="c1", name="top_artist"),
            call("albums", {"artist": "Artist X"}, "c2"), ToolMessage(content="12", tool_call_id="c2", name="albums")]


def with_tools(result="Artist X"):
    return [call("top_artist", {"k": 1}, "c1"), ToolMessage(content=result, tool_call_id="c1", name="top_artist"),
            AIMessage(content="original answer")]


TWO_RECS = [{"tool_name": "top_artist", "arguments": {"k": 1}, "result": "Artist X"},
            {"tool_name": "albums", "arguments": {"artist": "Artist X"}, "result": "12"}]


def run(agent, q=Q):
    return asyncio.run(agent.run(q))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("BASELINE_ANSWER_ON_LIMIT", "1")


# ---------------------------------------------------------------- flag off: identical to the original

FALLBACK_TEXT = '{"name": "top_artist", "arguments": {"k": 3}}'
SCENARIOS = {
    "normal": (with_tools(), False),
    "no_tools": ([AIMessage(content="direct answer")], False),
    "text_fallback": ([AIMessage(content=FALLBACK_TEXT)], False),
    "normal_hybrid": (with_tools(), True),
    "no_tools_hybrid": ([AIMessage(content="direct answer")], True),
    "text_fallback_hybrid": ([AIMessage(content=FALLBACK_TEXT)], True),
}


@pytest.mark.parametrize("ground", [False, True])
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_flag_off_identical_to_original_run(monkeypatch, name, ground):
    if ground:
        monkeypatch.setenv("BASELINE_GROUND_ANSWER", "1")
    msgs, hybrid = SCENARIOS[name]
    new_g, old_g = FakeGraph(msgs, forbid_astream=True), FakeGraph(msgs, forbid_astream=True)
    new_llm, old_llm = FakeLLM(), FakeLLM()
    new = run(make_agent(new_llm, new_g, tools=[_tool()], hybrid=hybrid))
    old = run(make_agent(old_llm, old_g, cls=OriginalReAct, tools=[_tool()], hybrid=hybrid))
    assert new == old
    assert new_g.used == old_g.used == ["ainvoke"]
    assert new_llm.calls == old_llm.calls and new_llm.binds == old_llm.binds
    if name.startswith("text_fallback"):
        assert new.tool_calls == [{"tool_name": "top_artist", "arguments": {"k": 3}, "result": "top_artist result 3"}]
    if hybrid:
        assert new.metadata["retriever"] == "hybrid" and new.trajectory[0]["retriever"] == "hybrid"


def test_flag_off_recursion_error_propagates():
    err = GraphRecursionError("limit")
    g = FakeGraph(two_tools_then_limit(), exc=err, forbid_astream=True)
    llm = FakeLLM()
    with pytest.raises(GraphRecursionError) as ei:
        run(make_agent(llm, g))
    assert ei.value is err and g.used == ["ainvoke"] and llm.calls == []


@pytest.mark.parametrize("val", ["0", "", "false", "off", "no"])
def test_flag_falsy_values_use_ainvoke(monkeypatch, val):
    monkeypatch.setenv("BASELINE_ANSWER_ON_LIMIT", val)
    g = FakeGraph(with_tools(), forbid_astream=True)
    run(make_agent(FakeLLM(), g))
    assert g.used == ["ainvoke"]


# ---------------------------------------------------------------- flag on, normal finish

@pytest.mark.parametrize("ground", [False, True])
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_flag_on_normal_finish_same_as_flag_off(monkeypatch, name, ground):
    if ground:
        monkeypatch.setenv("BASELINE_GROUND_ANSWER", "1")
    msgs, hybrid = SCENARIOS[name]
    off_llm, on_llm = FakeLLM(), FakeLLM()
    off = run(make_agent(off_llm, FakeGraph(msgs), tools=[_tool()], hybrid=hybrid))
    monkeypatch.setenv("BASELINE_ANSWER_ON_LIMIT", "1")
    g = FakeGraph(msgs)
    on_ = run(make_agent(on_llm, g, tools=[_tool()], hybrid=hybrid))
    assert g.used == ["astream"]
    assert on_ == off
    assert on_llm.calls == off_llm.calls  # nothing extra on a normal finish
    assert "answered_on_limit" not in on_.metadata
    assert not any(e.get("type") == "AnswerOnLimit" for e in on_.trajectory)


def test_astream_last_state_extracts_like_ainvoke(on):
    agent = make_agent(FakeLLM(), None)
    g = FakeGraph(with_tools())
    inp = {"messages": [HumanMessage(content=Q)]}

    async def go():
        last = None
        async for s in g.astream(inp, stream_mode="values"):
            last = s
        return last, await g.ainvoke(inp)
    last, full = asyncio.run(go())
    assert agent._extract_result(last, None) == agent._extract_result(full, None)


# ---------------------------------------------------------------- flag on, limit hit

def test_limit_after_two_tool_calls_answers(on):
    llm = FakeLLM(reply="12 albums")
    r = run(make_agent(llm, FakeGraph(two_tools_then_limit(), exc=GraphRecursionError("limit 5"))))
    assert r.content == "12 albums"
    assert r.tool_calls == TWO_RECS
    assert r.metadata == {"answered_on_limit": True, "baseline_ground_answer": "used"}
    assert len(llm.calls) == 1 and llm.binds == [{"temperature": 0, "max_tokens": pe.FINAL_MAX_TOKENS}]
    sys_msg, user_msg = llm.calls[0]
    assert isinstance(sys_msg, SystemMessage) and sys_msg.content is pe.FINAL_SYSTEM_GROUNDED
    assert isinstance(user_msg, HumanMessage)
    assert user_msg.content == pe.FINAL_USER_TEMPLATE.format(
        question=Q, results=pe.format_tool_results(TWO_RECS, pe.RESULT_CHARS_DEFAULT))
    marker = [e for e in r.trajectory if e.get("type") == "AnswerOnLimit"]
    assert len(marker) == 1 and marker[0]["answered_on_limit"] is True and "limit 5" in marker[0]["error"]
    assert r.trajectory[-1]["grounded_answer"] and r.trajectory[-1]["content"] == "12 albums"
    assert r.trajectory.index(marker[0]) == len(r.trajectory) - 2


def test_limit_result_cut_applied(on, monkeypatch):
    llm = FakeLLM()
    msgs = [call("top_artist", {"k": 1}, "c1"), ToolMessage(content="x" * 1000, tool_call_id="c1", name="top_artist")]
    run(make_agent(llm, FakeGraph(msgs, exc=GraphRecursionError("l"))))
    assert "x" * 600 + "...(truncated)" in llm.calls[0][1].content and "x" * 601 not in llm.calls[0][1].content
    monkeypatch.setenv("PLAN_RESULT_CHARS", "10")
    llm2 = FakeLLM()
    run(make_agent(llm2, FakeGraph(msgs, exc=GraphRecursionError("l"))))
    assert "x" * 10 + "...(truncated)" in llm2.calls[0][1].content


def test_both_flags_one_grounded_call(on, monkeypatch):
    monkeypatch.setenv("BASELINE_GROUND_ANSWER", "1")
    llm = FakeLLM()
    r = run(make_agent(llm, FakeGraph(two_tools_then_limit(), exc=GraphRecursionError("l"))))
    assert len(llm.calls) == 1 and r.metadata["answered_on_limit"] is True


def test_limit_with_hybrid_keeps_retriever_metadata(on):
    r = run(make_agent(FakeLLM(), FakeGraph(two_tools_then_limit(), exc=GraphRecursionError("l")), hybrid=True))
    assert r.metadata == {"retriever": "hybrid", "retriever_build_s": 0.123,
                          "answered_on_limit": True, "baseline_ground_answer": "used"}


def test_limit_with_pending_tool_call_only_completed_results(on):
    msgs = two_tools_then_limit() + [call("tracks", {"album": "A"}, "c3")]  # c3 never answered
    llm = FakeLLM()
    r = run(make_agent(llm, FakeGraph(msgs, exc=GraphRecursionError("l"))))
    assert r.tool_calls == TWO_RECS
    assert "tracks" not in llm.calls[0][1].content
    assert llm.calls[0][1].content == pe.FINAL_USER_TEMPLATE.format(
        question=Q, results=pe.format_tool_results(TWO_RECS, pe.RESULT_CHARS_DEFAULT))
    assert any(tc["name"] == "tracks" for e in r.trajectory for tc in e.get("tool_calls", []))


@pytest.mark.parametrize("msgs", [[], [AIMessage(content="thinking")], [call("top_artist", {"k": 1}, "c1")]],
                         ids=["nothing", "text_only", "pending_call_only"])
def test_limit_without_completed_tool_message_raises(on, msgs):
    err = GraphRecursionError("limit")
    llm = FakeLLM()
    with pytest.raises(GraphRecursionError) as ei:
        run(make_agent(llm, FakeGraph(msgs, exc=err)))
    assert ei.value is err and llm.calls == []


class ContextWindowExceeded(Exception):
    pass


@pytest.mark.parametrize("err", [ContextWindowExceeded("context length exceeded"), asyncio.TimeoutError(),
                                 RuntimeError("other"), RecursionError("plain python recursion")],
                         ids=["context", "timeout", "runtime", "plain_recursion"])
def test_non_recursion_errors_propagate(on, err):
    llm = FakeLLM()
    with pytest.raises(type(err)) as ei:
        run(make_agent(llm, FakeGraph(two_tools_then_limit(), exc=err)))
    assert ei.value is err and llm.calls == []


@pytest.mark.parametrize("llm", [FakeLLM(reply=""), FakeLLM(reply="<think>hmm</think>"),
                                 FakeLLM(exc=RuntimeError("grounded boom"))], ids=["empty", "think_only", "raises"])
def test_grounded_failure_reraises_original(on, llm):
    err = GraphRecursionError("limit")
    with pytest.raises(GraphRecursionError) as ei:
        run(make_agent(llm, FakeGraph(two_tools_then_limit(), exc=err)))
    assert ei.value is err and len(llm.calls) == 1


def test_runner_logging_on_limit_response(on):
    r = run(make_agent(FakeLLM(), FakeGraph(two_tools_then_limit(), exc=GraphRecursionError("l"))))
    res = BenchmarkResult(uuid="u", domain="books", query=Q, answer=r.content, tool_calls=r.tool_calls,
                          trajectory=r.trajectory, status="success")
    lines = []
    log_trajectory(res, lines.append)
    log_message_history(res, lines.append)
    text = "\n".join(lines)
    assert "top_artist" in text and "12 albums" in text


# ---------------------------------------------------------------- real LangGraph graph, scripted fake chat model

class LoopingChatModel(BaseChatModel):
    """Always calls tool `t`; answers the grounded prompt with plain text. No server."""
    n: int = 0

    @property
    def _llm_type(self):
        return "looping-fake"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if isinstance(messages[0], SystemMessage) and messages[0].content == pe.FINAL_SYSTEM_GROUNDED:
            msg = AIMessage(content="grounded answer")
        else:
            self.n += 1
            msg = AIMessage(content="", tool_calls=[{"name": "t", "args": {"x": self.n}, "id": f"c{self.n}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _real_agent():
    def t(x: int) -> str:
        """fake tool"""
        return f"r{x}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LangGraphReActAgent(llm=LoopingChatModel(), tools=[StructuredTool.from_function(t)], max_iterations=3)


def test_real_graph_flag_off_raises():
    with pytest.raises(GraphRecursionError):
        run(_real_agent())


def test_real_graph_flag_on_answers_from_partial_state(on):
    r = run(_real_agent())
    assert r.content == "grounded answer" and r.metadata["answered_on_limit"] is True
    assert len(r.tool_calls) >= 1
    assert all(tc["tool_name"] == "t" and tc["result"] == f"r{tc['arguments']['x']}" for tc in r.tool_calls)
