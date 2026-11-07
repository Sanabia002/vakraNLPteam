"""Offline tests for PlanExecuteAgent: plan parsing, result checks, retry variants and the zero-count
rule, flags, limits, and compatibility of the returned trajectory with the runner's logging helpers.
Uses a scripted fake LLM and a fake sentence encoder inside a real ToolShortlister (no server, no MCP)."""
import asyncio
import hashlib
import json

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents import plan_execute_agent as pe
from agents.agent_interface import AgentResponse
from agents.components.tool_shortlister import ToolShortlister
from benchmark.runner_helpers import (BenchmarkResult, log_message_history, log_trajectory,
                                      save_results_ground_truth)


# ------------------------------------------------------------------ pure helpers
def test_parse_plan_json_list():
    assert pe.parse_plan('["Find the team", "Count its players"]') == ["Find the team", "Count its players"]


def test_parse_plan_json_in_think_and_prose():
    txt = '<think>hmm ["wrong"]</think>Sure:\n["Find the gene with most chromosomes", "Count non-essential genes there"]'
    assert pe.parse_plan(txt) == ["Find the gene with most chromosomes", "Count non-essential genes there"]


def test_parse_plan_line_fallback_and_cap():
    txt = "1. Find the oldest author\n2) Get the books of that author\nok\n- Count those books please"
    assert pe.parse_plan(txt) == ["Find the oldest author", "Get the books of that author", "Count those books please"]
    assert len(pe.parse_plan(json.dumps([f"step number {i}" for i in range(9)]))) == 5


def test_parse_plan_empty():
    assert pe.parse_plan("") == [] and pe.parse_plan(None) == []


@pytest.mark.parametrize("text,expected", [
    ("[]", "empty"), ("[[]]", "empty"), ("No result", "empty"), ("", "empty"), ("[[null]]", "empty"),
    ("[[0]]", "zero"), ("0", "zero"), ("[0]", "zero"),
    ('{"error": "no such column"}', "error"), ("Error: ValidationError: x", "error"),
    ('{"count": 0}', "zero"), ('{\n  "count": 0\n}', "zero"), ('{"end_station_id": []}', "empty"),
    ('{"a": [], "b": null}', "empty"), ('{"trip_details": []}', "empty"), ('{"count": 4}', None),
    ('{"city": "San Francisco"}', None), ('{"max_temperature": 102, "date": "8/29/2013"}', None),
    ('{"count": 0, "city": "X"}', None), ('{"error": "bad"}', "error"),
    ("[[1]]", None), ('[["FIBER"]]', None), ("[[0, 3]]", None), ('[["a"], ["b"]]', None), ("[[false]]", None),
])
def test_classify_result(text, expected):
    assert pe.classify_result(text) == expected


def test_variants_change_strings_only_and_are_deterministic():
    args = {"name": " new york ", "year": 2003, "flag": True}
    v1 = pe.argument_variants(args, limit=10)
    assert v1 == pe.argument_variants(args, limit=10)
    # capitalize and lower leave " new york " unchanged -> skipped as no-ops
    assert [l for l, _ in v1] == ["strip", "title", "upper"]
    for _, cand in v1:
        assert cand["year"] == 2003 and cand["flag"] is True and cand != args
    assert len({json.dumps(c, sort_keys=True) for _, c in v1}) == len(v1)
    assert dict(v1)["title"]["name"] == " New York "


def test_variants_limit_and_no_strings():
    assert len(pe.argument_variants({"a": "x y"})) == 2  # default limit is 2
    assert pe.argument_variants({"year": 1999}) == []
    assert pe.argument_variants({"a": "ABC"}, limit=10) == [("title", {"a": "Abc"}), ("lower", {"a": "abc"})]


def test_no_enum_handling():
    assert not hasattr(pe, "tool_enums")
    # values that look like enum candidates are only ever case/space transformed
    assert all(set(c) == {"status"} and c["status"].lower() == "active"
               for _, c in pe.argument_variants({"status": "ACTIVE"}, limit=10))


@pytest.mark.parametrize("verdict,replaces", [(None, True), ("zero", False), ("empty", False), ("error", False)])
def test_retry_replaces(verdict, replaces):
    assert pe.retry_replaces(verdict) is replaces


# ------------------------------------------------------------------ fakes
class FakeEncoder:
    """Bag-of-words hashing embedder; records every encode() batch."""
    def __init__(self):
        self.batches = []

    def encode(self, texts, convert_to_numpy=True):
        self.batches.append(list(texts))
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().replace("_", " ").replace(":", " ").split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out


def fake_retriever():
    r = ToolShortlister.__new__(ToolShortlister)  # real class, fake model (skips the MiniLM download)
    r.top_k, r.model, r._tool_embeddings, r._tools = 8, FakeEncoder(), None, None
    return r


class FakeLLM:
    """Scripted chat model. Records bound tool names and bind_tools kwargs for each call."""
    def __init__(self, replies, delay=0.0):
        self.replies, self.delay = list(replies), delay
        self.calls, self.bind_tools_kwargs = [], []

    def bind(self, **kw):
        return self

    def bind_tools(self, tools, **kw):
        self.bind_tools_kwargs.append(kw)
        return _Bound(self, [t.name for t in tools])

    async def ainvoke(self, messages, names=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.calls.append((names, messages))
        r = self.replies.pop(0)
        return r if isinstance(r, AIMessage) else AIMessage(content=r)


class _Bound:
    def __init__(self, base, names):
        self.base, self.names = base, names

    def bind(self, **kw):
        return self

    async def ainvoke(self, messages):
        return await self.base.ainvoke(messages, names=self.names)


def tc(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1"}])


class CityArgs(BaseModel):
    city: str = Field(..., description="City name")


class CountArgs(BaseModel):
    country: str = Field(..., description="Country name")


def make_tools(responses=None):
    """capital: Paris -> France, else []. count: driven by `responses` dict, default France -> 7 else 0."""
    responses = responses or {"France": "[[7]]"}

    async def capital(city):
        return '[["France"]]' if city == "Paris" else "[]"

    async def count(country):
        return responses.get(country, "[[0]]")

    async def other(**kw):
        return "[]"
    tools = [
        StructuredTool(name="get_country_by_capital_city", description="Get the country whose capital is a city",
                       args_schema=CityArgs, coroutine=capital, func=lambda **k: ""),
        StructuredTool(name="count_rivers_by_country", description="Count rivers flowing through a country",
                       args_schema=CountArgs, coroutine=count, func=lambda **k: ""),
    ]
    for w in ["album", "artist", "award", "player", "team", "season"]:
        tools.append(StructuredTool(name=f"get_{w}_info", description=f"Look up {w} records",
                                    args_schema=CityArgs, coroutine=other, func=lambda **k: ""))
    return tools


def agent(monkeypatch, replies, verify=True, step=True, ground=True, record_failed=None, delay=0.0,
          timeout_s=300, max_iter=12, responses=None, **env):
    for k, v in {"VERIFY": verify, "STEP_RETRIEVAL": step, "GROUND_PROMPT": ground}.items():
        monkeypatch.setenv(k, "1" if v else "0")
    if record_failed is None:
        monkeypatch.delenv("RECORD_FAILED_RETRIES", raising=False)
    else:
        monkeypatch.setenv("RECORD_FAILED_RETRIES", "1" if record_failed else "0")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    llm = FakeLLM(replies, delay=delay)
    a = pe.PlanExecuteAgent(llm=llm, tools=make_tools(responses), max_iterations=max_iter, timeout_s=timeout_s,
                            retriever=fake_retriever())
    return a, llm


def steps_of(r):
    return [t for t in r.trajectory if t.get("step") is not None]


def tool_msgs(r):
    return [t for t in r.trajectory if t["type"] == "ToolMessage"]


Q = "How many rivers flow through the country whose capital is paris?"
PLAN = '["Find the country whose capital city is paris", "Count rivers flowing through that country"]'
ONE_STEP_COUNT = '["Count rivers flowing through the country"]'


# ------------------------------------------------------------------ full runs
def test_full_run_with_verify_retry(monkeypatch):
    a, llm = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="paris"),
                                 tc("count_rivers_by_country", country="France"), "7"])
    r = asyncio.run(a.run(Q))
    assert r.content == "7"
    # paris -> [] (empty); "strip" is a no-op, so the first variant is title -> "Paris" -> good -> replaces
    assert [c["arguments"] for c in r.tool_calls] == [{"city": "paris"}, {"city": "Paris"}, {"country": "France"}]
    tm = tool_msgs(r)
    assert [(t["attempt"], t["verdict"], t["kept"]) for t in tm] == [
        ("llm", "empty", False), ("title", None, True), ("llm", None, True)]
    plan = [t for t in r.trajectory if "plan" in t][0]
    assert plan["type"] == "AIMessage" and len(plan["plan"]) == 2
    # shortlisted = union offered; each bound set <= m+n; step 1's own tool is offered
    bound = [set(n) for n, _ in llm.calls if n]
    assert set(r.shortlisted_tools) == set().union(*bound) and all(len(b) <= 8 for b in bound)
    assert "get_country_by_capital_city" in steps_of(r)[0]["offered_tools"]
    # step 2 sees the kept (retried) result
    assert '[["France"]]' in llm.calls[2][1][1].content
    # final call: grounded prompt, no tools bound; retries made no LLM calls
    assert llm.calls[-1][0] is None and "add nothing else" in llm.calls[-1][1][0].content
    assert len(llm.calls) == 4


def test_exactly_one_tool_call_requested_and_used(monkeypatch):
    two = AIMessage(content="", tool_calls=[{"name": "get_country_by_capital_city", "args": {"city": "Paris"}, "id": "a"},
                                            {"name": "count_rivers_by_country", "args": {"country": "France"}, "id": "b"}])
    a, llm = agent(monkeypatch, ['["Find the country whose capital is paris"]', two, "France"])
    r = asyncio.run(a.run(Q))
    assert llm.bind_tools_kwargs == [{"tool_choice": "required", "parallel_tool_calls": False}]
    assert [c["tool_name"] for c in r.tool_calls] == ["get_country_by_capital_city"]


def test_zero_count_kept_when_variants_also_zero(monkeypatch):
    a, llm = agent(monkeypatch, [ONE_STEP_COUNT, tc("count_rivers_by_country", country="germany"), "0"])
    r = asyncio.run(a.run(Q))
    assert [c["arguments"]["country"] for c in r.tool_calls] == ["germany", "Germany", "GERMANY"]
    assert [t["kept"] for t in tool_msgs(r)] == [True, False, False]
    assert "germany" in llm.calls[-1][1][1].content and "GERMANY" not in llm.calls[-1][1][1].content


def test_zero_count_replaced_when_variant_nonzero(monkeypatch):
    a, llm = agent(monkeypatch, [ONE_STEP_COUNT, tc("count_rivers_by_country", country="spain"), "3"],
                   responses={"Spain": "[[3]]"})
    r = asyncio.run(a.run(Q))
    assert [t["kept"] for t in tool_msgs(r)] == [False, True]
    assert '"Spain"' in llm.calls[-1][1][1].content


def test_empty_original_not_replaced_by_zero(monkeypatch):
    a, llm = agent(monkeypatch, [ONE_STEP_COUNT, tc("count_rivers_by_country", country="italy"), "?"],
                   responses={"italy": "[]", "Italy": "[[0]]", "ITALY": "[[0]]"})
    r = asyncio.run(a.run(Q))
    tm = tool_msgs(r)
    assert [t["verdict"] for t in tm] == ["empty", "zero", "zero"]
    assert [t["kept"] for t in tm] == [True, False, False]


def test_error_original_replaced_by_good(monkeypatch):
    a, llm = agent(monkeypatch, [ONE_STEP_COUNT, tc("count_rivers_by_country", country="peru"), "2"],
                   responses={"peru": '{"error": "not found"}', "Peru": "[[2]]"})
    r = asyncio.run(a.run(Q))
    assert [(t["verdict"], t["kept"]) for t in tool_msgs(r)] == [("error", False), (None, True)]


def test_record_failed_retries_off_lists_only_kept(monkeypatch):
    a, _ = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="paris"),
                               tc("count_rivers_by_country", country="France"), "7"], record_failed=False)
    r = asyncio.run(a.run(Q))
    assert [c["arguments"] for c in r.tool_calls] == [{"city": "Paris"}, {"country": "France"}]
    assert len(tool_msgs(r)) == 3  # trajectory still shows every attempt


def test_record_failed_retries_off_keeps_original_when_all_fail(monkeypatch):
    a, _ = agent(monkeypatch, [ONE_STEP_COUNT, tc("count_rivers_by_country", country="germany"), "0"],
                 record_failed=False)
    r = asyncio.run(a.run(Q))
    assert r.tool_calls == [{"tool_name": "count_rivers_by_country", "arguments": {"country": "germany"},
                             "result": "[[0]]"}]


def test_record_failed_retries_default_on(monkeypatch):
    a, _ = agent(monkeypatch, ["[]"])
    assert a.record_failed_retries is True


def test_verify_off_no_retry(monkeypatch):
    a, _ = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="paris"),
                               tc("count_rivers_by_country", country="?"), "unknown"], verify=False)
    r = asyncio.run(a.run(Q))
    assert len(r.tool_calls) == 2 and r.metadata["flags"]["VERIFY"] is False


def test_step_retrieval_off_uses_question_only(monkeypatch):
    a, llm = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="Paris"),
                                 tc("count_rivers_by_country", country="France"), "7"], step=False)
    asyncio.run(a.run(Q))
    bound = [n for n, _ in llm.calls if n]
    assert bound[0] == bound[1] and len(bound[0]) == 8


def test_tools_encoded_once_per_agent(monkeypatch):
    a, llm = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="Paris"),
                                 tc("count_rivers_by_country", country="France"), "7"] * 2)
    enc = a._retriever.model
    assert len(enc.batches) == 1 and len(enc.batches[0]) == 8  # encode_tools at build time
    asyncio.run(a.run(Q))
    asyncio.run(a.run(Q))
    # each question adds exactly one batch: [question] + plan steps; tool texts never re-encoded
    assert [len(b) for b in enc.batches[1:]] == [3, 3]
    tool_texts = set(enc.batches[0])
    assert not any(t in tool_texts for b in enc.batches[1:] for t in b)


def test_ground_off_plain_prompt(monkeypatch):
    a, llm = agent(monkeypatch, ['["Find the country whose capital is paris"]',
                                 tc("get_country_by_capital_city", city="Paris"), "France"], ground=False)
    asyncio.run(a.run(Q))
    assert llm.calls[-1][1][0].content == pe.FINAL_SYSTEM_PLAIN


def test_unoffered_or_missing_tool_call_is_skipped(monkeypatch):
    a, _ = agent(monkeypatch, [PLAN, tc("not_a_tool", x=1), AIMessage(content="I think France"), "dunno"])
    r = asyncio.run(a.run(Q))
    assert r.tool_calls == [] and r.content == "dunno"
    assert all(s["no_tool_call"] for s in steps_of(r))


def test_text_tool_call_fallback(monkeypatch):
    txt = '```json\n{"name": "get_country_by_capital_city", "arguments": {"city": "Paris"}}\n```'
    a, _ = agent(monkeypatch, ['["Find the country whose capital is paris"]', txt, "France"])
    r = asyncio.run(a.run(Q))
    assert r.tool_calls[0]["result"] == '[["France"]]'


def test_bad_plan_falls_back_to_question(monkeypatch):
    a, _ = agent(monkeypatch, ["ok", tc("get_country_by_capital_city", city="Paris"), "France"])
    r = asyncio.run(a.run(Q))
    assert [t for t in r.trajectory if "plan" in t][0]["plan"] == [Q]


def test_max_iterations_returns_partial(monkeypatch):
    a, _ = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="paris"), "France"], max_iter=1)
    r = asyncio.run(a.run(Q))
    assert len(r.tool_calls) == 1
    assert r.trajectory[-1]["stop_reason"] == "max_iterations" and r.content == "France"


def test_time_budget_returns_partial_without_raising(monkeypatch):
    # budget 3s, 1s reserved for the final call, each LLM call 0.6s: plan + step 1 fit, step 2 does not.
    a, _ = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="Paris"), "France"],
                 delay=0.6, PLAN_FINAL_RESERVE_S="1.0")
    a._budget_s = 3.0
    r = asyncio.run(a.run(Q))
    assert r.trajectory[-1]["stop_reason"] == "time_budget"
    assert len(r.tool_calls) == 1 and r.content == "France" and r.trajectory[-1]["elapsed_s"] < 3.0


def test_final_timeout_falls_back_to_last_good_result(monkeypatch):
    a, llm = agent(monkeypatch, ['["Find the country whose capital is paris"]',
                                 tc("get_country_by_capital_city", city="Paris"), "late"], PLAN_FINAL_RESERVE_S="0.5")
    a._budget_s = 2.0
    orig = llm.ainvoke

    async def slow_final(messages, names=None):
        if len(llm.calls) == 2:
            await asyncio.sleep(5)
        return await orig(messages, names)
    llm.ainvoke = slow_final
    r = asyncio.run(a.run(Q))
    assert r.content == '[["France"]]'
    assert r.trajectory[-1]["stop_reason"] == "time_budget" and r.trajectory[-1]["final_fallback"] is True


# ------------------------------------------------------------------ runner logging compatibility
def _scenarios(monkeypatch):
    out = []
    a, _ = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="paris"),
                               tc("count_rivers_by_country", country="France"), "7"])
    out.append(asyncio.run(a.run(Q)))
    a, _ = agent(monkeypatch, [PLAN, tc("not_a_tool"), AIMessage(content=""), ""])  # no calls, empty final
    out.append(asyncio.run(a.run(Q)))
    a, _ = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="paris"), "x"], max_iter=1)
    out.append(asyncio.run(a.run(Q)))
    return out


def _as_result(resp: AgentResponse) -> BenchmarkResult:
    r = BenchmarkResult(uuid="u1", domain="fake", query=Q, all_tools=resp.all_tools)
    r.answer, r.tool_calls, r.trajectory, r.status = resp.content, resp.tool_calls, resp.trajectory, "success"
    r.shortlisted_tools = resp.shortlisted_tools
    return r


def test_trajectory_matches_runner_logging_helpers(monkeypatch, tmp_path):
    for resp in _scenarios(monkeypatch):
        for e in resp.trajectory:
            assert e["type"] in ("HumanMessage", "AIMessage", "ToolMessage")
            assert isinstance(e["content"], str)
            if e["type"] == "AIMessage":
                assert isinstance(e["tool_calls"], list)
                assert all(set(tc) >= {"id", "name", "args"} and isinstance(tc["args"], dict) for tc in e["tool_calls"])
            if e["type"] == "ToolMessage":
                assert {"tool_name", "tool_call_id", "result"} <= set(e)
        lines = []
        res = _as_result(resp)
        log_trajectory(res, lines.append)       # must not raise
        log_message_history(res, lines.append)  # must not raise
        text = "\n".join(lines)
        assert "[FINAL ANSWER]" in text and "User: How many rivers" in text
        for c in resp.tool_calls:
            assert f"Tool ({c['tool_name']})" in text and f"[TOOL CALL] {c['tool_name']}(" in text
        save_results_ground_truth([res], tmp_path)  # tool_calls format accepted by the saver
        saved = json.load(open(tmp_path / "fake.json"))[0]["output"][0]["sequence"]["tool_call"]
        assert [s["name"] for s in saved] == [c["tool_name"] for c in resp.tool_calls]


def test_logging_helpers_on_handmade_response():
    resp = AgentResponse(content="7", tool_calls=[{"tool_name": "t", "arguments": {"a": "b"}, "result": "[[7]]"}],
                         trajectory=[{"type": "HumanMessage", "content": "q"},
                                     {"type": "AIMessage", "content": "", "tool_calls": [{"id": "1", "name": "t", "args": {"a": "b"}}]},
                                     {"type": "ToolMessage", "content": "[[7]]", "tool_name": "t", "tool_call_id": "1", "result": "[[7]]"},
                                     {"type": "AIMessage", "content": "7", "tool_calls": [], "final": True}])
    lines = []
    log_trajectory(_as_result(resp), lines.append)
    log_message_history(_as_result(resp), lines.append)
    assert any("Tool (t): [[7]]" in l for l in lines)
