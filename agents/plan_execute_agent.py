"""Plan-then-execute agent for capability 3 (multi-hop). Enabled only with PLAN_AGENT=1.

Flow per question (LLM calls: 1 plan + 1 per step + 1 final; retries make no LLM calls):
  1. PLAN: one LLM call, no tools -> JSON list of short steps (prompt from plan_recall.py).
  2. For each step: offer a small tool set and request exactly ONE tool call, then execute it.
     STEP_RETRIEVAL=1 -> top-m tools for the step text + top-n tools for the question.
     STEP_RETRIEVAL=0 -> top-(m+n) tools for the question (same budget, no per-step retrieval).
     Tool embeddings come from ToolShortlister.encode_tools, computed once per domain (agent build).
  3. VERIFY=1: if a result is an error / empty / zero count, retry up to 2 times changing only
     string arguments (strip, title, capitalize, lower, upper). A retry replaces the original only
     when its result is good (not error/empty/zero); otherwise the original is kept.
     RECORD_FAILED_RETRIES=1 (default ON) lists every attempt in tool_calls; 0 lists only the kept call.
  4. Final answer: one LLM call, no tools. GROUND_PROMPT=1 uses the strict grounded prompt.
  5. Own limits: max_iterations tool executions and a wall-clock budget below AGENT_TIMEOUT_SECONDS;
     on a limit the agent returns what it has instead of raising.

Trajectory entries use the LangGraph message format read by benchmark.runner_helpers.log_trajectory /
log_message_history: type HumanMessage / AIMessage / ToolMessage, string content, AIMessage tool_calls
as [{id, name, args}], ToolMessage tool_name / tool_call_id / result. Extra keys (plan, step,
offered_tools, attempt, verdict, kept, stop_reason, ...) are ignored by those helpers.

Tunables (env): PLAN_STEP_M (5), PLAN_QUESTION_N (3), PLAN_MAX_STEPS (5), PLAN_TIME_MARGIN_S (30),
PLAN_FINAL_RESERVE_S (45), PLAN_RESULT_CHARS (600).
"""
import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from agents.agent_interface import AgentInterface, AgentResponse, Message
from agents.components.tool_shortlister import ToolShortlister

logger = logging.getLogger(__name__)


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------- prompts (planner prompt as in plan_recall.py)
PLAN_SYSTEM = ("You split a question into the minimal ordered list of lookups against a database API. "
               "Each step is one short sentence saying what to look up, and which value from an earlier step it needs. "
               "Use as few steps as possible (usually 2 or 3, never more than 5). Answer with ONLY a JSON list of strings.")
PLAN_EX_Q = "How many albums did the artist with the most awards release?"
PLAN_EX_A = '["Find the artist with the most awards", "Count the albums released by that artist"]'

STEP_SYSTEM = ("You execute one step of a plan by calling exactly ONE of the offered tools. "
               "Fill the arguments from the question and from earlier results (copy values exactly). "
               "Do not answer the question; only make the tool call.")

FINAL_SYSTEM_GROUNDED = ("Answer the question only from the tool results below. Reuse the question's wording. "
                         "Give only the answer; add nothing else.")
FINAL_SYSTEM_PLAIN = "Answer the question using the tool results below."

FINAL_USER_TEMPLATE = "Question: {question}\nTool results:\n{results}\n\nAnswer:"

PLAN_MAX_TOKENS, STEP_MAX_TOKENS, FINAL_MAX_TOKENS = 300, 256, 200  # server generation cap is 1024
RESULT_CHARS_DEFAULT = 600  # PLAN_RESULT_CHARS


def format_tool_results(records: List[dict], result_chars: int) -> str:
    """Numbered 'tool(args) -> result' lines, each result cut to result_chars. Shared with the baseline agent."""
    if not records:
        return "(none yet)"
    lines = []
    for i, r in enumerate(records, 1):
        res = str(r["result"])
        if len(res) > result_chars:
            res = res[:result_chars] + "...(truncated)"
        lines.append(f"{i}. {r['tool_name']}({json.dumps(r['arguments'], default=str)}) -> {res}")
    return "\n".join(lines)


def strip_think(text: Any) -> str:
    s = text if isinstance(text, str) else ("" if text is None else str(text))
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


def parse_plan(txt: str, max_steps: int = 5) -> List[str]:
    """Same parsing as plan_recall.py: first JSON list, else non-trivial lines."""
    txt = strip_think(txt)
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            v = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
            if v:
                return v[:max_steps]
        except Exception:
            pass
    lines = [re.sub(r"^[\-\d\.\)\s]+", "", l).strip() for l in txt.splitlines()]
    return [l for l in lines if len(l.split()) >= 3][:max_steps]


# ---------------------------------------------------------------- result checking and retry variants
def classify_result(text: Any) -> Optional[str]:
    """Return 'error', 'empty', 'zero' for a bad tool result, else None (good)."""
    if text is None:
        return "empty"
    s = str(text).strip()
    if not s or s in ("No result", "None", "null", "[]", "{}", '""'):
        return "empty"
    low = s.lower()
    if low.startswith(("error", "exception", "traceback", "tool error")):
        return "error"
    try:
        v = json.loads(s)
    except Exception:
        return "error" if re.match(r"^[\w\s]*error\b", low) else None
    if isinstance(v, dict) and ("error" in v or ("detail" in v and len(v) == 1)):
        return "error"

    def leaves(x):  # leaf values of nested lists / dicts, e.g. {"count": 0} -> [0], {"ids": []} -> []
        if isinstance(x, dict):
            for y in x.values():
                yield from leaves(y)
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from leaves(y)
        else:
            yield x
    if isinstance(v, (dict, list, tuple)):
        vals = list(leaves(v))
        if not vals or all(x is None for x in vals):
            return "empty"
        if len(vals) == 1 and isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool) and vals[0] == 0:
            return "zero"
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
        return "zero"
    return None


STRING_TRANSFORMS = [("strip", str.strip), ("title", str.title), ("capitalize", str.capitalize),
                     ("lower", str.lower), ("upper", str.upper)]


def argument_variants(args: Dict[str, Any], limit: int = 2) -> List[Tuple[str, Dict[str, Any]]]:
    """Deterministic retry candidates: one generic transform applied to every string argument.
    Non-string arguments are never changed; no-op and duplicate candidates are skipped."""
    str_keys = [k for k, v in args.items() if isinstance(v, str)]
    seen = {json.dumps(args, sort_keys=True, default=str)}
    out: List[Tuple[str, Dict[str, Any]]] = []
    for label, fn in STRING_TRANSFORMS:
        if len(out) >= limit:
            break
        cand = {k: (fn(v) if k in str_keys else v) for k, v in args.items()}
        key = json.dumps(cand, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append((label, cand))
    return out


def retry_replaces(new_verdict: Optional[str]) -> bool:
    """Zero-count rule. Retries only run when the original was error / empty / zero, and a retry
    replaces it only with a good result: error or empty -> non-empty, or a count of 0 -> nonzero.
    If every variant is also error / empty / zero, the original is kept."""
    return new_verdict is None


def parse_text_tool_call(content: Any) -> Optional[dict]:
    """Fallback for a tool call written as text: {"name": ..., "arguments": {...}}."""
    content = strip_think(content)
    for pat in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"(\{.*\})"):
        m = re.search(pat, content, re.S)
        if m:
            try:
                d = json.loads(m.group(1))
                if isinstance(d, dict) and "name" in d:
                    args = d.get("arguments") or d.get("args") or {}
                    return {"name": d["name"], "args": args if isinstance(args, dict) else {}}
            except Exception:
                continue
    return None


class _Budget(Exception):
    pass


class PlanExecuteAgent(AgentInterface):
    def __init__(self, llm, tools: List[StructuredTool] | None = None, max_iterations: int | None = None,
                 timeout_s: float | None = None, retriever: ToolShortlister | None = None, **kwargs):
        self._llm = llm
        self._tools = list(tools or [])
        self._tool_map = {t.name: t for t in self._tools}
        self._max_iterations = max_iterations or 12
        total = timeout_s if timeout_s is not None else float(os.environ.get("AGENT_TIMEOUT_SECONDS", "300"))
        self._budget_s = max(10.0, total - float(os.environ.get("PLAN_TIME_MARGIN_S", "30")))
        self._final_reserve_s = float(os.environ.get("PLAN_FINAL_RESERVE_S", "45"))
        self.step_retrieval = _flag("STEP_RETRIEVAL")
        self.verify = _flag("VERIFY")
        self.ground_prompt = _flag("GROUND_PROMPT")
        self.record_failed_retries = _flag("RECORD_FAILED_RETRIES", "1")
        self.m = _int_env("PLAN_STEP_M", 5)
        self.n = _int_env("PLAN_QUESTION_N", 3)
        self.max_steps = _int_env("PLAN_MAX_STEPS", 5)
        self.result_chars = _int_env("PLAN_RESULT_CHARS", RESULT_CHARS_DEFAULT)
        # The runner precomputes a shortlist when agent._shortlister is set; keep it None.
        # The tools actually offered are returned in AgentResponse.shortlisted_tools.
        self._shortlister = None
        self._initial_data_handle = None
        self._initial_data_peek = None
        # Tool embeddings via ToolShortlister.encode_tools, once per agent (= once per domain).
        self._retriever = retriever or ToolShortlister(top_k=max(1, self.m + self.n))
        if self._tools:
            self._retriever.encode_tools(self._tools)
            self._E = self._norm(self._retriever._tool_embeddings)
        logger.info("PlanExecuteAgent: tools=%d step_retrieval=%s verify=%s ground=%s record_failed=%s "
                    "m=%d n=%d budget=%.0fs", len(self._tools), self.step_retrieval, self.verify,
                    self.ground_prompt, self.record_failed_retries, self.m, self.n, self._budget_s)

    @staticmethod
    def _norm(x):
        x = np.asarray(x, dtype=float)
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.where(n == 0, 1, n)

    def _rank(self, texts: List[str], k: int) -> List[List[str]]:
        """Top-k tool names per text. Texts are encoded in one batch; tools are never re-encoded."""
        if k <= 0 or not self._tools or not texts:
            return [[] for _ in texts]
        V = self._norm(self._retriever.model.encode(texts, convert_to_numpy=True))
        return [[self._tools[int(i)].name for i in np.argsort(-(self._E @ v))[:k]] for v in V]

    async def _call_llm(self, llm, messages, deadline: float, reserve: float = 0.0):
        remaining = deadline - time.monotonic() - reserve
        if remaining <= 1:
            raise _Budget()
        try:
            return await asyncio.wait_for(llm.ainvoke(messages), timeout=remaining)
        except asyncio.TimeoutError:
            raise _Budget()

    async def _execute(self, tool, args) -> str:
        try:
            return str(await tool.ainvoke(args))
        except Exception as e:  # validation errors and tool failures become results
            return f"Error: {type(e).__name__}: {e}"

    def _history_text(self, records: List[dict]) -> str:
        return format_tool_results(records, self.result_chars)

    async def run(self, input: Union[str, List[Message]], additional_instructions: str = None) -> AgentResponse:
        question = input if isinstance(input, str) else next(
            (m.content for m in reversed(input) if m.role == "user"), "")
        t0 = time.monotonic()
        deadline = t0 + self._budget_s
        trajectory: List[dict] = [{"type": "HumanMessage", "content": question}]
        tool_calls: List[dict] = []    # returned to the runner (all attempts, or kept only)
        n_exec = 0                     # every execution, retries included, counts toward max_iterations
        step_records: List[dict] = []  # kept result per step, shown to later steps and the final answer
        offered_union: List[str] = []
        stop_reason = "completed"
        content = ""

        def record(si, label, tool_name, args, result, verdict, call_id, ai_content=""):
            trajectory.append({"type": "AIMessage", "content": ai_content, "step_index": si, "attempt": label,
                               "tool_calls": [{"id": call_id, "name": tool_name, "args": args}]})
            msg = {"type": "ToolMessage", "content": result, "tool_name": tool_name, "tool_call_id": call_id,
                   "result": result, "step_index": si, "attempt": label, "verdict": verdict, "kept": False}
            trajectory.append(msg)
            return msg

        try:
            # 1. PLAN
            msgs = [SystemMessage(content=PLAN_SYSTEM), HumanMessage(content="Question: " + PLAN_EX_Q),
                    AIMessage(content=PLAN_EX_A), HumanMessage(content="Question: " + question)]
            plan_llm = self._llm.bind(temperature=0, max_tokens=PLAN_MAX_TOKENS)
            plan_raw = strip_think((await self._call_llm(plan_llm, msgs, deadline, self._final_reserve_s)).content)
            steps = parse_plan(plan_raw, self.max_steps) or [question]
            trajectory.append({"type": "AIMessage", "content": plan_raw, "plan": steps, "tool_calls": []})

            if self.step_retrieval:
                ranked = self._rank([question] + steps, max(self.m, self.n))
                q_top, step_tops = ranked[0][: self.n], [r[: self.m] for r in ranked[1:]]
            else:
                q_top, step_tops = self._rank([question], self.m + self.n)[0], [[] for _ in steps]

            # 2. EXECUTE
            for si, step in enumerate(steps):
                if n_exec >= self._max_iterations:
                    stop_reason = "max_iterations"
                    break
                names = list(dict.fromkeys(step_tops[si] + q_top))
                offered_union.extend(n for n in names if n not in offered_union)
                offered = [self._tool_map[n] for n in names]

                user = (f"Question: {question}\nPlan: {json.dumps(steps)}\n"
                        f"Results so far:\n{self._history_text(step_records)}\n\n"
                        f"Current step {si + 1}: {step}\nCall ONE tool for this step.")
                step_llm = self._llm.bind_tools(offered, tool_choice="required",
                                                parallel_tool_calls=False).bind(max_tokens=STEP_MAX_TOKENS)
                ai = await self._call_llm(step_llm, [SystemMessage(content=STEP_SYSTEM), HumanMessage(content=user)],
                                          deadline, self._final_reserve_s)
                ai_text = strip_think(getattr(ai, "content", ""))
                call = None
                for tc in getattr(ai, "tool_calls", None) or []:  # only the first offered call is used
                    if tc.get("name") in names:
                        a = tc.get("args")
                        call = {"name": tc["name"], "args": a if isinstance(a, dict) else {},
                                "id": tc.get("id") or f"s{si}"}
                        break
                if call is None:
                    parsed = parse_text_tool_call(ai_text)
                    if parsed and parsed["name"] in names:
                        call = {**parsed, "id": f"s{si}"}
                # step header: carries the step text and the offered tools
                trajectory.append({"type": "AIMessage", "content": f"Step {si + 1}: {step}" + (
                                       "" if call else f" | no tool call: {ai_text[:300]}"),
                                   "step_index": si, "step": step, "offered_tools": names,
                                   "no_tool_call": call is None, "tool_calls": []})
                if call is None:
                    continue

                tool = self._tool_map[call["name"]]
                args = dict(call["args"])
                result = await self._execute(tool, args)
                n_exec += 1
                verdict = classify_result(result)
                msgs_for_step = [record(si, "llm", tool.name, args, result, verdict, call["id"], ai_text)]
                attempts = [{"tool_name": tool.name, "arguments": args, "result": result}]
                kept_i = 0

                # 3. VERIFY: string-only deterministic retries
                if self.verify and verdict is not None:
                    for j, (label, cand) in enumerate(argument_variants(args, limit=2), 1):
                        if n_exec >= self._max_iterations or time.monotonic() > deadline - self._final_reserve_s:
                            msgs_for_step[-1]["retry_stopped"] = "limit"
                            break
                        r2 = await self._execute(tool, cand)
                        n_exec += 1
                        v2 = classify_result(r2)
                        msgs_for_step.append(record(si, label, tool.name, cand, r2, v2, f"{call['id']}_r{j}"))
                        attempts.append({"tool_name": tool.name, "arguments": cand, "result": r2})
                        if retry_replaces(v2):
                            kept_i = len(attempts) - 1
                            break
                msgs_for_step[kept_i]["kept"] = True
                tool_calls.extend(attempts if self.record_failed_retries else [attempts[kept_i]])
                step_records.append(attempts[kept_i])
        except _Budget:
            stop_reason = "time_budget"
        except Exception as e:  # never raise out of run(); return partial results
            logger.exception("PlanExecuteAgent step failure")
            stop_reason = f"error: {type(e).__name__}: {e}"

        # 4. FINAL ANSWER (step calls kept the reserve free for this one)
        fallback = False
        try:
            system = FINAL_SYSTEM_GROUNDED if self.ground_prompt else FINAL_SYSTEM_PLAIN
            user = FINAL_USER_TEMPLATE.format(question=question, results=self._history_text(step_records))
            final_llm = self._llm.bind(temperature=0, max_tokens=FINAL_MAX_TOKENS)
            ai = await self._call_llm(final_llm, [SystemMessage(content=system), HumanMessage(content=user)], deadline)
            content = strip_think(ai.content)
        except _Budget:
            stop_reason = stop_reason if stop_reason != "completed" else "time_budget"
        except Exception as e:
            logger.exception("PlanExecuteAgent final answer failure")
            stop_reason = f"final error: {type(e).__name__}: {e}"
        if not content:
            good = [r for r in step_records if classify_result(r["result"]) is None]
            content = str(good[-1]["result"]) if good else ""
            fallback = True

        trajectory.append({"type": "AIMessage", "content": content, "final": True, "tool_calls": [],
                           "stop_reason": stop_reason, "final_fallback": fallback,
                           "elapsed_s": round(time.monotonic() - t0, 1), "n_executions": n_exec})
        return AgentResponse(content=content, tool_calls=tool_calls,
                             messages=[Message(role="user", content=question),
                                       Message(role="assistant", content=content)],
                             metadata={"stop_reason": stop_reason,
                                       "flags": {"STEP_RETRIEVAL": self.step_retrieval, "VERIFY": self.verify,
                                                 "GROUND_PROMPT": self.ground_prompt,
                                                 "RECORD_FAILED_RETRIES": self.record_failed_retries}},
                             trajectory=trajectory, all_tools=[t.name for t in self._tools],
                             shortlisted_tools=offered_union)

    def restart(self):
        self._initial_data_handle = None
        self._initial_data_peek = None
