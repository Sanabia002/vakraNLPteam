"""LLM_EXTRA_KWARGS for the litellm provider: fakes only, no network, no model server."""
import asyncio
import os
import socket
from typing import Any, Dict

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # litellm must not fetch its price map

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

import agents.llm as llm_mod
from agents.llm import create_llm

NESTED = '{"extra_body": {"reasoning_effort": "none"}, "max_tokens": 1024}'


@pytest.fixture(autouse=True)
def no_network_clean_env(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("network access attempted in a unit test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for k in ("LLM_EXTRA_KWARGS", "LITELLM_BASE_URL", "LITELLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)


class Recorder:
    """Stands in for ChatLiteLLM and records constructor kwargs."""
    calls: list = []

    def __init__(self, **kw):
        Recorder.calls.append(kw)


@pytest.fixture
def recorder(monkeypatch):
    import langchain_litellm
    Recorder.calls = []
    monkeypatch.setattr(langchain_litellm, "ChatLiteLLM", Recorder)
    return Recorder



def old_litellm_path(provider="litellm", model=None, api_key=None, temperature=0, **kwargs):
    """VERBATIM copy of the litellm path of create_llm in agents/llm.py at commit 6c16fbf (before LLM_EXTRA_KWARGS)."""
    if provider == "watsonx":
        pass
    elif provider == "litellm":
        resolved_base_url = os.environ.get("LITELLM_BASE_URL")
        if resolved_base_url:
            kwargs.setdefault("api_base", resolved_base_url)
        if not api_key:
            api_key = os.environ.get("LITELLM_API_KEY")
    if provider == "rits":
        pass
    elif provider == "litellm":
        try:
            from langchain_litellm import ChatLiteLLM
            # Uncomment to debug litellm connection
            # import litellm
            # litellm._turn_on_debug()
        except ImportError:
            raise ImportError(
                "langchain-litellm is required for LiteLLM support. "
                "Install with: pip install langchain-litellm"
            )

        # This is the cheapest model in our allowed set: 
        # ['aws/claude-opus-4-5', 'claude-opus-4-5-20251101', 
        # 'claude-sonnet-4-5-20250929', 'aws/claude-sonnet-4-5', 
        # 'Azure/gpt-5.1-2025-11-13', 'GCP/gemini-2.5-flash', 
        # 'GCP/gemini-2.5-flash-lite', 'GCP/gemini-2.0-flash', 
        # 'gcp/gemini-3-flash-preview']
        model_name = model or "GCP/gemini-2.0-flash"
        params: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
        }
        if api_key:
            params["api_key"] = api_key
        if "api_base" in kwargs:
            params["api_base"] = kwargs["api_base"]
        # This parameter is critical. Without it, the client attempts
        # to infer the provider name from the base of the model name, 
        # and there is no way to satisfy the check for the model being in 
        # the allow-list [e.g. GCP/gemini-2.0-flash] and the check for an
        # existing provider (GCP isn't an existing provider)
        params["custom_llm_provider"] = "openai"

        return ChatLiteLLM(**params)



CASES = [dict(), dict(model="openai/qwen3.5-4b"), dict(model="m", api_key="k", temperature=0.3),
         dict(model="m", api_base="http://localhost:8080/v1")]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("val", [None, "", "   "])
def test_unset_or_empty_identical_constructor_kwargs(recorder, monkeypatch, case, val):
    if val is not None:
        monkeypatch.setenv("LLM_EXTRA_KWARGS", val)
    monkeypatch.setenv("LITELLM_BASE_URL", "http://fake-base/v1")
    create_llm(provider="litellm", **case)
    old_litellm_path(**case)
    new_kw, old_kw = recorder.calls
    assert new_kw == old_kw and "model_kwargs" not in new_kw
    assert list(new_kw) == list(old_kw)  # same keys, same order


def test_unset_identical_real_chatlitellm():
    a = create_llm(provider="litellm", model="openai/qwen3.5-4b", api_key="k", api_base="http://x/v1")
    b = old_litellm_path(model="openai/qwen3.5-4b", api_key="k", api_base="http://x/v1")
    assert a.model_dump() == b.model_dump() and a.model_fields_set == b.model_fields_set
    assert a._client_params == b._client_params


def test_set_merges_into_model_kwargs(recorder, monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_KWARGS", NESTED)
    create_llm(provider="litellm", model="m")
    old_litellm_path(model="m")
    new_kw, old_kw = recorder.calls
    assert new_kw.pop("model_kwargs") == {"extra_body": {"reasoning_effort": "none"}, "max_tokens": 1024}
    assert new_kw == old_kw


@pytest.mark.parametrize("raw,msg", [("{not json", "could not parse"), ("[1, 2]", "got list"),
                                     ('"text"', "got str"), ("3", "got int"), ("null", "got NoneType")])
def test_bad_values_raise(monkeypatch, raw, msg):
    monkeypatch.setenv("LLM_EXTRA_KWARGS", raw)
    with pytest.raises(ValueError, match="LLM_EXTRA_KWARGS must be a JSON object") as ei:
        create_llm(provider="litellm", model="m")
    assert msg in str(ei.value)


@pytest.mark.parametrize("key", llm_mod.LLM_EXTRA_KWARGS_DENY)
def test_denied_keys_raise(monkeypatch, key):
    monkeypatch.setenv("LLM_EXTRA_KWARGS", f'{{"{key}": "x", "top_p": 0.9}}')
    with pytest.raises(ValueError, match="may not set") as ei:
        create_llm(provider="litellm", model="m")
    assert repr(key) in str(ei.value)


def test_other_providers_ignore_it(monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_KWARGS", "{not json")  # would raise if the litellm path read it
    with pytest.raises(ValueError, match="Unknown provider"):
        create_llm(provider="nope")


def test_logs_key_names_only(monkeypatch, caplog):
    monkeypatch.setenv("LLM_EXTRA_KWARGS", '{"extra_headers": {"Authorization": "SECRET-VALUE"}}')
    with caplog.at_level("INFO", logger="agents.llm"):
        create_llm(provider="litellm", model="m")
    assert "extra_headers" in caplog.text and "SECRET-VALUE" not in caplog.text


# ---------------------------------------------------------------- real ChatLiteLLM, fake completion call

@pytest.fixture
def sent(monkeypatch):
    """Patch ChatLiteLLM.acompletion_with_retry: record the final params litellm would receive."""
    import litellm
    from langchain_litellm import ChatLiteLLM
    record: list = []

    async def fake(self, run_manager=None, **kwargs):
        record.append(kwargs)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"},
                                               "finish_reason": "stop", "index": 0}])
    monkeypatch.setattr(ChatLiteLLM, "acompletion_with_retry", fake)
    return record


MSGS = [SystemMessage(content="sys"), HumanMessage(content="q")]


def _tool():
    def lookup(k: int) -> str:
        """fake tool"""
        return "r"
    return StructuredTool.from_function(lookup)


def test_nested_value_reaches_call_params_all_paths(monkeypatch, sent):
    monkeypatch.setenv("LLM_EXTRA_KWARGS", NESTED)
    llm = create_llm(provider="litellm", model="openai/qwen3.5-4b", api_base="http://x/v1")

    r = asyncio.run(llm.ainvoke(MSGS))
    assert r.content == "ok"
    asyncio.run(llm.bind_tools([_tool()]).ainvoke(MSGS))
    asyncio.run(llm.bind(temperature=0, max_tokens=200).ainvoke(MSGS))   # the grounded final-answer call

    plain, tools, grounded = sent
    for p in (plain, tools, grounded):
        assert p["extra_body"] == {"reasoning_effort": "none"}
        assert p["model"] == "openai/qwen3.5-4b" and p["custom_llm_provider"] == "openai"
        assert p["api_base"] == "http://x/v1"
    assert plain["max_tokens"] == 1024 and tools["max_tokens"] == 1024
    assert tools["tools"][0]["function"]["name"] == "lookup"
    assert grounded["max_tokens"] == 200 and grounded["temperature"] == 0   # bind wins
    assert llm.model_kwargs == {"extra_body": {"reasoning_effort": "none"}, "max_tokens": 1024}  # not mutated


def test_grounded_bind_without_extra_kwargs_unchanged(sent):
    llm = create_llm(provider="litellm", model="m")
    old = old_litellm_path(model="m")
    asyncio.run(llm.bind(temperature=0, max_tokens=200).ainvoke(MSGS))
    asyncio.run(old.bind(temperature=0, max_tokens=200).ainvoke(MSGS))
    assert sent[0] == sent[1] and "extra_body" not in sent[0] and sent[0]["max_tokens"] == 200


def test_grounded_answer_path_through_react_agent(monkeypatch, sent):
    """_ground_answer (BASELINE_ANSWER_ON_LIMIT / BASELINE_GROUND_ANSWER) on a real ChatLiteLLM keeps the extra kwargs."""
    import agents.plan_execute_agent as pe
    from agents.agent_interface import LangGraphReActAgent
    monkeypatch.setenv("LLM_EXTRA_KWARGS", NESTED)
    agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
    agent._llm = create_llm(provider="litellm", model="m")
    recs = [{"tool_name": "t", "arguments": {}, "result": "r"}]
    answer, status = asyncio.run(agent._ground_answer("q?", recs, "orig", []))
    assert (answer, status) == ("ok", "used")
    (p,) = sent
    assert p["extra_body"] == {"reasoning_effort": "none"} and p["max_tokens"] == pe.FINAL_MAX_TOKENS == 200
    assert p["temperature"] == 0 and p["messages"][0]["content"] == pe.FINAL_SYSTEM_GROUNDED
