"""HybridRetriever (RETRIEVER=hybrid) and the default-identity guarantee. Fake tools and fake embedders only."""
import asyncio
import hashlib

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import agents.components.hybrid_retriever as hr
import agents.components.tool_shortlister as ts_mod
from agents import plan_execute_agent as pe
from agents.agent_interface import LangGraphReActAgent
from agents.components.tool_shortlister import ToolShortlister
from tests.unit.test_plan_execute_agent import PLAN, Q, FakeEncoder, agent, fake_retriever, make_tools, tc


# ----------------------------------------------------------------------------- fakes
class HashEmbedder:
    """Deterministic bag-of-words embedder; counts encode calls."""
    def __init__(self, *a, dim=64, **k):
        self.dim, self.batches = dim, []

    def encode(self, texts, convert_to_numpy=True, **kw):
        self.batches.append(list(texts))
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for w in t.lower().replace("_", " ").replace(":", " ").split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return out


class TableEmbedder(HashEmbedder):
    """Exact-text -> vector table; anything else falls back to a fixed far-away vector."""
    def __init__(self, table, dim=4):
        super().__init__(dim=dim)
        self.table = {k: np.asarray(v, dtype=float) for k, v in table.items()}

    def encode(self, texts, convert_to_numpy=True, **kw):
        self.batches.append(list(texts))
        return np.stack([self.table.get(t, np.eye(self.dim)[-1]) for t in texts])


class P(BaseModel):
    city_name: str = Field(..., description="Name of the city")
    min_age: int = Field(0)


class PConfig(BaseModel):
    team: str = Field(..., description="Team name")
    config: str = Field("", description="ignored")
    args: str = Field("", description="ignored")


def tool(name, desc, schema=P):
    return StructuredTool(name=name, description=desc, args_schema=schema, func=lambda **k: "")


def catalog(n=12, pinned=("query_sql", "query_peek")):
    words = ["album", "artist", "award", "player", "team", "season", "river", "city", "country", "match", "venue",
             "price", "coin", "book", "order", "author"]
    ts = [tool(f"get_{w}_count", f"Count {w} records by city") for w in words[:n]]
    for i, p in enumerate(pinned):
        ts.insert(2 + 3 * i, tool(p, "Run a generic query"))
    return ts


@pytest.fixture(autouse=True)
def no_model_download(monkeypatch):
    monkeypatch.setattr(hr, "SentenceTransformer", HashEmbedder)
    monkeypatch.setattr(ts_mod, "SentenceTransformer", HashEmbedder)
    monkeypatch.delenv("RETRIEVER", raising=False)


@pytest.fixture
def no_hybrid(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("HybridRetriever must not be built in base mode")
    for m in (ts_mod, pe):
        monkeypatch.setattr(m, "HybridRetriever", boom)


# ----------------------------------------------------------------------------- reference copies of the base code
def ref_shortlist(sl, query, tools):
    """Verbatim copy of ToolShortlister.shortlist before RETRIEVER was added (commit 73dacce)."""
    if len(tools) <= sl.top_k:
        return tools
    pinned = [t for t in tools if sl._tool_name(t).startswith("query_")]
    candidates = [t for t in tools if not sl._tool_name(t).startswith("query_")]
    remaining_slots = sl.top_k - len(pinned)
    if remaining_slots <= 0 or not candidates:
        return pinned
    if remaining_slots >= len(candidates):
        return pinned + candidates
    if sl._tool_embeddings is not None and sl._tools is not None and len(sl._tools) == len(tools):
        candidate_indices = [i for i, t in enumerate(tools) if not sl._tool_name(t).startswith("query_")]
        tool_embs = sl._tool_embeddings[candidate_indices]
    else:
        tool_embs = sl.model.encode([sl._tool_text(t) for t in candidates], convert_to_numpy=True)
    query_emb = sl.model.encode([query], convert_to_numpy=True)
    tool_norms = np.linalg.norm(tool_embs, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
    similarities = ((tool_embs / np.where(tool_norms == 0, 1, tool_norms))
                    @ (query_emb / np.where(query_norm == 0, 1, query_norm)).T).squeeze()
    return pinned + [candidates[i] for i in np.argsort(similarities)[::-1][:remaining_slots]]


def ref_offered(a, question, steps):
    """Base PlanExecuteAgent offered lists (STEP_RETRIEVAL=1), computed from the base dense ranking."""
    E = pe.PlanExecuteAgent._norm(FakeEncoder().encode([ToolShortlister._tool_text(t) for t in a._tools]))
    V = pe.PlanExecuteAgent._norm(FakeEncoder().encode([question] + steps))
    top = [[a._tools[int(i)].name for i in np.argsort(-(E @ v))[:max(a.m, a.n)]] for v in V]
    return [list(dict.fromkeys(top[1 + i][:a.m] + top[0][:a.n])) for i in range(len(steps))]


QUERIES = ["How many albums did the top artist release in the city of Paris?", "count team season", "",
           "the of and", "Which venue hosted the most matches and who owned it?"]


# ----------------------------------------------------------------------------- mode parsing
@pytest.mark.parametrize("v,mode", [(None, "base"), ("base", "base"), (" BASE ", "base"), ("", "base"),
                                    ("hybrid", "hybrid"), ("  Hybrid\n", "hybrid")])
def test_retriever_mode(monkeypatch, v, mode):
    if v is not None:
        monkeypatch.setenv("RETRIEVER", v)
    assert hr.retriever_mode() == mode


@pytest.mark.parametrize("v", ["bm25", "dense", "hybrid2"])
def test_bad_retriever_value_raises(monkeypatch, v):
    monkeypatch.setenv("RETRIEVER", v)
    with pytest.raises(ValueError, match="RETRIEVER"):
        ToolShortlister(top_k=3)


# ----------------------------------------------------------------------------- default identity
@pytest.mark.parametrize("env", [None, "base", " Base "])
@pytest.mark.parametrize("k", [1, 2, 3, 5, 13, 14, 20])
def test_base_shortlist_identical_to_pre_change(monkeypatch, no_hybrid, env, k):
    if env is not None:
        monkeypatch.setenv("RETRIEVER", env)
    tools = catalog()
    sl = ToolShortlister(top_k=k)
    sl.encode_tools(tools)
    for q in QUERIES:
        got = sl.shortlist(q, tools)
        assert [t.name for t in got] == [t.name for t in ref_shortlist(sl, q, tools)]
        if len(tools) <= k:
            assert got is tools
    assert sl._hybrid is None and sl._tool_embeddings is not None


def test_base_shortlist_on_the_fly_path_identical(monkeypatch, no_hybrid):
    tools = catalog()
    sl = ToolShortlister(top_k=5)  # encode_tools never called -> fallback branch
    assert [t.name for t in sl.shortlist(QUERIES[0], tools)] == [t.name for t in ref_shortlist(sl, QUERIES[0], tools)]


def _plan_run(monkeypatch, env):
    if env is None:
        monkeypatch.delenv("RETRIEVER", raising=False)
    else:
        monkeypatch.setenv("RETRIEVER", env)
    a, llm = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="Paris"),
                                 tc("count_rivers_by_country", country="France"), "7"])
    r = asyncio.run(a.run(Q))
    for e in r.trajectory:
        e.pop("elapsed_s", None)  # wall-clock
    return a, r


def test_base_plan_agent_identical_unset_vs_base(monkeypatch, no_hybrid):
    a0, r0 = _plan_run(monkeypatch, None)
    a1, r1 = _plan_run(monkeypatch, "base")
    assert r0.trajectory == r1.trajectory and r0.metadata == r1.metadata and r0.tool_calls == r1.tool_calls
    assert r0.shortlisted_tools == r1.shortlisted_tools
    assert "retriever" not in r0.trajectory[0] and "RETRIEVER" not in r0.metadata["flags"]
    assert "retriever_build_s" not in r0.metadata
    offered = [e["offered_tools"] for e in r0.trajectory if "offered_tools" in e]
    steps = next(e["plan"] for e in r0.trajectory if "plan" in e)
    assert offered == ref_offered(a0, Q, steps)


def test_base_react_metadata_unchanged(monkeypatch, no_hybrid):
    from tests.unit.test_baseline_ground_answer import FakeGraph, FakeLLM, make_agent, with_tools
    r = asyncio.run(make_agent(FakeLLM(), with_tools()).run("q"))
    assert r.metadata == {} and "retriever" not in r.trajectory[0]


# ----------------------------------------------------------------------------- tool text
def test_tool_text_variant_b():
    assert hr.tool_text(tool("get_count_players_by_city", "Count players.")) == \
        "count players by city: Count players. Parameters: city name Name of the city; min age"


def test_tool_text_skips_generic_params_and_handles_dict_schema():
    assert hr.tool_text(tool("team_info", "Team.", PConfig)) == "team info: Team. Parameters: team Team name"
    d = {"name": "get_x", "description": "D", "args_schema": {"properties": {"a_b": {"description": "AB"},
                                                                             "config": {}}}}
    assert hr.tool_text(d) == "x: D Parameters: a b AB"
    assert hr.tool_text({"name": "get_y", "description": "E"}) == "y: E Parameters: "


# ----------------------------------------------------------------------------- ranking
def fusion_setup():
    tools = [tool("get_alpha", "Alpha lookup", PConfig), tool("get_beta", "Beta lookup", PConfig),
             tool("get_near", "Near lookup", PConfig), tool("get_zebra", "Zebra stripes lookup", PConfig)]
    q = "zebra stripes count"
    texts = [hr.tool_text(t) for t in tools]
    table = {texts[0]: [0, 0, 1, 0], texts[1]: [0, 0, 0.9, 0.1], texts[2]: [1, 0, 0, 0],   # near: dense twin of q
             texts[3]: [0.8, 0.6, 0, 0], q: [1, 0, 0, 0]}
    return hr.HybridRetriever(tools, model=TableEmbedder(table)), q


def test_keyword_match_outranks_dense_near_miss_when_fused():
    r, q = fusion_setup()
    dense_only = hr.HybridRetriever.__new__(hr.HybridRetriever)
    dense_only.__dict__.update(r.__dict__, _bm25=None)
    assert dense_only.rank(q)[0] == 2           # dense alone: the near miss wins
    assert r.rank(q)[:2] == [3, 2]              # fused: the keyword match (zebra) wins


def test_clause_max_finds_tool_whole_question_misses():
    tools = [tool("get_first", "First", PConfig), tool("get_second", "Second", PConfig)]
    q = "Find the team whose coach won the title in 2010"
    clause = "coach won the title in 2010"
    assert clause in hr.pieces(q)
    texts = [hr.tool_text(t) for t in tools]
    table = {texts[0]: [1, 0, 0, 0], texts[1]: [0, 1, 0, 0], q: [0.9, 0.1, 0, 0], clause: [0, 1, 0, 0],
             "Find the team": [0, 0, 1, 0]}
    r = hr.HybridRetriever(tools, model=TableEmbedder(table))
    r._bm25 = None                              # isolate the dense component
    assert r.rank(q, clauses=False) == [0, 1]
    assert r.rank(q, clauses=True) == [1, 0]


def test_deterministic_and_ties_keep_original_order():
    tools = catalog()
    r = hr.HybridRetriever(tools, model=HashEmbedder())
    for q in QUERIES:
        assert r.rank(q) == r.rank(q) and r.shortlist(q, 5) == r.shortlist(q, 5)
    const = TableEmbedder({})                   # every text -> the same vector: all dense scores tie
    rc = hr.HybridRetriever(tools, model=const)
    assert rc.rank("the of and") == list(range(len(tools)))      # no BM25 tokens -> dense only, all ties
    assert rc.rank("xylophone quartz") == list(range(len(tools)))  # BM25 all zero too


# ----------------------------------------------------------------------------- build-once / encode counts
def test_embeddings_and_bm25_built_once(monkeypatch):
    built = []
    real = hr.BM25Okapi
    monkeypatch.setattr(hr, "BM25Okapi", lambda corpus: (built.append(1), real(corpus))[1])
    tools = catalog()
    emb = HashEmbedder()
    r = hr.HybridRetriever(tools, model=emb)
    assert len(emb.batches) == 1 and len(emb.batches[0]) == len(tools) and built == [1]
    for q in QUERIES:
        r.rank(q)
        r.shortlist(q, 5)
    assert built == [1]
    tool_texts = set(emb.batches[0])
    for b in emb.batches[1:]:
        assert not tool_texts & set(b)
    assert emb.batches[1] == hr.pieces(QUERIES[0])
    assert r.build_s >= 0


# ----------------------------------------------------------------------------- shortlist edges
def test_shortlist_pinned_and_k_edges():
    tools = catalog()
    r = hr.HybridRetriever(tools, model=HashEmbedder())
    got = [t.name for t in r.shortlist(QUERIES[0], 5)]
    assert got[:2] == ["query_sql", "query_peek"] and len(got) == 5
    assert r.shortlist(QUERIES[0], len(tools)) == tools and r.shortlist("x", 100) == tools
    sl = ToolShortlister(top_k=len(tools))
    assert sl.shortlist("x", tools) is tools
    assert [t.name for t in r.shortlist("x", 2)] == ["query_sql", "query_peek"]           # no slots left
    assert [t.name for t in r.shortlist("x", len(tools) - 1)][2:] == \
        [t.name for t in tools if not t.name.startswith("query_")][:len(tools) - 3]


@pytest.mark.parametrize("q", ["", "the of and", "   "])
def test_empty_or_stopword_query_is_dense_only(q):
    tools = catalog()
    r = hr.HybridRetriever(tools, model=HashEmbedder())
    dense_only = hr.HybridRetriever.__new__(hr.HybridRetriever)
    dense_only.__dict__.update(r.__dict__, _bm25=None)
    assert r.rank(q) == dense_only.rank(q)
    assert len(r.shortlist(q, 5)) == 5


def test_empty_bm25_corpus_and_empty_catalog(monkeypatch):
    monkeypatch.setattr(hr, "tool_text", lambda t: "")
    r = hr.HybridRetriever(catalog(3, pinned=()), model=HashEmbedder())
    assert r._bm25 is None and sorted(r.rank("albums")) == [0, 1, 2]
    e = hr.HybridRetriever([], model=HashEmbedder())
    assert e.rank("q") == [] and e.shortlist("q", 3) == []


# ----------------------------------------------------------------------------- wiring: ToolShortlister / runner path
def test_runner_call_path_react_hybrid(monkeypatch):
    """benchmark_runner.py: agent._shortlister.shortlist(item.query, agent._tools) with RETRIEVER=hybrid."""
    from tests.unit.test_baseline_ground_answer import FakeGraph, FakeLLM
    monkeypatch.setenv("RETRIEVER", "hybrid")
    monkeypatch.setattr(LangGraphReActAgent, "_build_agent",
                        lambda self, tools: FakeGraph([AIMessage(content="done")]))
    tools = catalog()
    ag = LangGraphReActAgent(llm=FakeLLM(), tools=tools, top_k_tools=5)
    sl = ag._shortlister
    assert isinstance(sl._hybrid, hr.HybridRetriever) and sl._tool_embeddings is None
    assert len(sl.model.batches) == 1                                   # tools encoded once (hybrid text)
    got = sl.shortlist(QUERIES[0], ag._tools)                           # the runner's call
    big = ToolShortlister(top_k=100)
    big.retriever_mode = "hybrid"
    assert big.shortlist("x", tools) is tools                           # k >= n: caller's list, like base
    ref = hr.HybridRetriever(tools, model=HashEmbedder()).shortlist(QUERIES[0], 5)
    assert [t.name for t in got] == [t.name for t in ref]
    r = asyncio.run(ag.run(QUERIES[0]))
    assert r.shortlisted_tools == [t.name for t in ref]
    assert r.metadata["retriever"] == "hybrid" and isinstance(r.metadata["retriever_build_s"], float)
    assert r.trajectory[0]["retriever"] == "hybrid"


def test_shortlister_hybrid_rebuilds_on_catalog_change(monkeypatch):
    monkeypatch.setenv("RETRIEVER", "hybrid")
    sl = ToolShortlister(top_k=3)
    sl.encode_tools(catalog())
    other = catalog(8, pinned=())
    got = sl.shortlist("count albums", other)
    assert sl._hybrid.names == [t.name for t in other] and all(t in other for t in got)


# ----------------------------------------------------------------------------- wiring: PlanExecuteAgent
def test_plan_agent_hybrid_steps_and_question(monkeypatch):
    monkeypatch.setenv("RETRIEVER", "hybrid")
    a, llm = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="Paris"),
                                 tc("count_rivers_by_country", country="France"), "7"], PLAN_QUESTION_N=2, PLAN_STEP_M=2)
    enc = a._retriever.model
    assert a._hybrid is not None and len(enc.batches) == 1             # built once, base encode skipped
    r = asyncio.run(a.run(Q))
    steps = next(e["plan"] for e in r.trajectory if "plan" in e)
    assert enc.batches[1] == hr.pieces(Q) + steps                      # one batch: question clauses + raw steps
    ref = hr.HybridRetriever(a._tools, model=FakeEncoder())
    names = lambda idx: [a._tools[i].name for i in idx]
    q_top = names(ref.rank(Q, clauses=True))[:2]
    expect = [list(dict.fromkeys(names(ref.rank(s, clauses=False))[:2] + q_top)) for s in steps]
    assert [e["offered_tools"] for e in r.trajectory if "offered_tools" in e] == expect
    assert r.trajectory[0]["retriever"] == "hybrid" and isinstance(r.trajectory[0]["retriever_build_s"], float)
    assert r.metadata["flags"]["RETRIEVER"] == "hybrid" and "retriever_build_s" in r.metadata


def test_plan_agent_hybrid_step_retrieval_off(monkeypatch):
    monkeypatch.setenv("RETRIEVER", "hybrid")
    a, llm = agent(monkeypatch, [PLAN, tc("get_country_by_capital_city", city="Paris"),
                                 tc("count_rivers_by_country", country="France"), "7"], step=False,
                   PLAN_QUESTION_N=2, PLAN_STEP_M=3)
    r = asyncio.run(a.run(Q))
    ref = hr.HybridRetriever(a._tools, model=FakeEncoder())
    expect = [a._tools[i].name for i in ref.rank(Q, clauses=True)][:5]
    assert all(e["offered_tools"] == expect for e in r.trajectory if "offered_tools" in e)
    assert a._retriever.model.batches[1] == hr.pieces(Q)               # question only, with clauses
