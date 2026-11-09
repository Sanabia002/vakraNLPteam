"""Hybrid tool retriever: dense (MiniLM, clause-max) + BM25, fused by reciprocal rank. Used only when RETRIEVER=hybrid.

Variant B from retrieval_experiments.py ("hybrid: MiniLM full clause-max + BM25"):
  tool text = words(name) + ": " + description + " Parameters: " + "; ".join(f"{param words} {param description}")
  dense     = max cosine over pieces(question) (the question plus its clauses); clauses=False uses the text alone
  bm25      = BM25Okapi over toks(tool text), query toks(question)
  fused     = RRF, k=60, 1-based ranks, stable sorts (ties keep the original tool order)
Queries with no BM25 tokens (empty or all stopwords) fall back to dense only.
Must not import agents.agent_interface (import cycle).
"""
import os
import re
import time
from typing import Any, List, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60
SKIP_PARAMS = {"config", "args", "kwargs"}
MODES = ("base", "hybrid")

# ---- copied from retrieval_experiments.py
STOP = set("the a an of in on at to for and or is was were are be by with that which who whom whose what how many much "
           "from as it its this these those has have had than then when where".split())


def words(n: str) -> str:
    return re.sub(r"^get_", "", n).replace("_", " ")


def pieces(q: str) -> List[str]:
    ps = [p.strip() for p in re.split(r"\b(?:that|who|which|whose|where|and)\b|[,;]", q) if len(p.split()) >= 3]
    return [q] + ps


def toks(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def norm(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
# ----


def retriever_mode() -> str:
    """RETRIEVER env var: unset/'base' -> 'base', 'hybrid' -> 'hybrid' (case-insensitive, stripped); else ValueError."""
    v = os.environ.get("RETRIEVER", "base").strip().lower() or "base"
    if v not in MODES:
        raise ValueError(f"RETRIEVER must be one of {MODES}, got {os.environ.get('RETRIEVER')!r}")
    return v


def _tool_name(tool: Any) -> str:
    return tool.get("name", "") if isinstance(tool, dict) else getattr(tool, "name", "")


def tool_params(tool: Any) -> List[tuple]:
    """(name, description) for each tool argument, from a pydantic args_schema or a JSON-schema dict."""
    schema = tool.get("args_schema") if isinstance(tool, dict) else getattr(tool, "args_schema", None)
    out = []
    if schema is None:
        return out
    if isinstance(schema, dict):
        for n, info in (schema.get("properties") or {}).items():
            out.append((n, (info or {}).get("description") or ""))
    elif hasattr(schema, "model_fields"):
        for n, f in schema.model_fields.items():
            out.append((n, f.description or ""))
    return [(n, d) for n, d in out if n not in SKIP_PARAMS]


def tool_text(tool: Any) -> str:
    desc = tool.get("description", "") if isinstance(tool, dict) else getattr(tool, "description", "")
    return (f"{words(_tool_name(tool))}: {desc or ''} Parameters: "
            + "; ".join(f"{n.replace('_', ' ')} {d}".strip() for n, d in tool_params(tool)))


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


class HybridRetriever:
    def __init__(self, tools: Sequence[Any], model: Optional[Any] = None):
        t0 = time.perf_counter()
        self.tools = list(tools)
        self.names = [_tool_name(t) for t in self.tools]
        self.model = model if model is not None else SentenceTransformer(MODEL_NAME)
        self.texts = [tool_text(t) for t in self.tools]
        self._E = norm(self.model.encode(self.texts, convert_to_numpy=True)) if self.tools else np.zeros((0, 1))
        corpus = [toks(t) for t in self.texts]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        self.build_s = round(time.perf_counter() - t0, 3)

    def _fuse(self, dense: np.ndarray, query: str) -> List[int]:
        q = toks(query)
        lists = [dense]
        if self._bm25 is not None and q:
            lists.append(np.asarray(self._bm25.get_scores(q), dtype=float))
        if len(lists) == 1:
            return [int(i) for i in _ranks_desc(dense)]
        fused = np.zeros(len(self.tools))
        for s in lists:
            for r, i in enumerate(_ranks_desc(s)):
                fused[i] += 1.0 / (RRF_K + r + 1)
        return [int(i) for i in _ranks_desc(fused)]

    def rank_many(self, texts: Sequence[str], clauses: Sequence[bool]) -> List[List[int]]:
        """Tool indices best-first for each text; all query pieces are encoded in ONE batch."""
        if not self.tools:
            return [[] for _ in texts]
        groups = [pieces(t) if c else [t] for t, c in zip(texts, clauses)]
        flat = [p for g in groups for p in g]
        if not flat:
            return []
        V = norm(self.model.encode(flat, convert_to_numpy=True))
        out, j = [], 0
        for t, g in zip(texts, groups):
            dense = (V[j:j + len(g)] @ self._E.T).max(axis=0)
            j += len(g)
            out.append(self._fuse(dense, t))
        return out

    def rank(self, query: str, clauses: bool = True) -> List[int]:
        return self.rank_many([query], [clauses])[0]

    def shortlist(self, query: str, k: int) -> list:
        """Same rules as ToolShortlister.shortlist: query_* tools pinned first, rest by hybrid rank."""
        tools = self.tools
        if len(tools) <= k:
            return tools
        pinned = [t for t in tools if _tool_name(t).startswith("query_")]
        candidates = [t for t in tools if not _tool_name(t).startswith("query_")]
        slots = k - len(pinned)
        if slots <= 0 or not candidates:
            return pinned
        if slots >= len(candidates):
            return pinned + candidates
        order = [i for i in self.rank(query) if not self.names[i].startswith("query_")]
        return pinned + [tools[i] for i in order[:slots]]
