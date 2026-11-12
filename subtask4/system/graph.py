"""The new Subtask 4 system as one LangGraph state graph (design revision 6, Figure 1).

Every component is a node with the name it has in spec.NODES; every decision diamond is a conditional edge; the
evidence procedure (C5) is a sub-graph used by both of its callers. tests/unit/test_s4_graph_structure.py checks the
built graph's edges against spec.EDGES.

    before the loop   rule_reader (C1) -> topic_check (C2) -> [which rule applies?] -> tool_filter (C3, C4)
    the loop          subgoal -> rank (C7, C8) -> model (C12) -> guard (C9) -> [API or documents?]
                        API:        arg_check (C10) -> run_api -> [empty?] -> hint (C11) -> merge -> subgoal
                        documents:  evidence (the verified document lookup) -> merge
    after the loop    grounded (C13); a documents-only rule ends in answer_docs or refuse instead
    evidence          retriever -> code_check -> critic -> [supported?] -> verified | nothing; critic -> retriever

State values are replaced, never appended by a reducer: the evidence sub-graph returns the whole state, and an
appending reducer would add its lists twice. Each node returns new lists. Every decision is logged (C15).
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional, TypedDict

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from ..config import SystemConfig
from ..rules import CATEGORIES, Rule, Source, UnknownRule, is_retriever, parse_rule
from ..topics import parse_answer
from . import checks, prompts

REFUSAL = ("I can not answer this question: I may only use document retrievers here, "
           "and the documents do not contain the answer.")
PASSAGE_CHARS = 1200   # per passage shown to the retriever and the critic
PASSAGES_SHOWN = 6
_MENU = "\n".join(f"{i}. {name}: {text}" for i, (name, text) in enumerate(CATEGORIES.items(), 1))
_CATEGORY = list(CATEGORIES)


class State(TypedDict, total=False):
    question: str                 # the last user turn
    history: list[dict]           # earlier turns: {"role", "content"}
    rule_text: str
    rule: Optional[dict]          # {"allowed": "docs" | "apis", "category": str | None, "source": "pattern" | "model"}
    applies: Optional[bool]
    mode: str                     # docs | apis | none: the path taken
    allowed: list[str]            # tool names the rule allows (C4)
    messages: list                # the model's conversation
    standalone: Optional[str]     # the standalone rewrite of a back-referring question (C7, C5)
    hop: int                      # model steps taken (C12)
    subgoal: str
    shortlist: list[str]
    pending: Optional[dict]       # the call being handled: {"id", "name", "args"}
    warned: list[str]             # calls already warned about (C10)
    attempt: int                  # 0 before the call runs, 1 after, 2 after the case retry (C11)
    matched: Optional[dict]       # the case variant that returned rows
    result: str                   # the pending API call's result
    hint: str
    hints: int
    revealed: bool                # C8 fallback: document search revealed in the no-rule path
    tool_calls: list[dict]        # every call made, for scoring: {"tool_name", "arguments", "result"}
    ev: dict                      # the evidence procedure's working state
    draft: str
    answer: str
    log: list[dict]


def _schema(tool) -> dict:
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema
    return schema.model_json_schema() if schema is not None else {}


def text_field(tool) -> str:
    """A search tool's query argument: its first required string field."""
    props = (schema := _schema(tool)).get("properties", {})
    return next((n for n in schema.get("required", props) if props.get(n, {}).get("type") == "string"), "query")


def _content(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


class System:
    """The nodes and edges of the agent graph for one domain."""

    def __init__(self, *, llm, tools: list, domain: str, config: SystemConfig, topics, shortlister=None,
                 top_k: int = 20, max_steps: int = 20, parse_text_call: Callable[[str], Optional[dict]] = None,
                 domain_description: str = ""):
        self.llm, self.cfg, self.domain, self.topics = llm, config, domain, topics
        self.tools = {t.name: t for t in tools}
        self.own_search = f"query_{domain}"
        self.shortlister, self.top_k, self.max_steps = shortlister, top_k, max_steps
        self.parse_text_call = parse_text_call or (lambda content: None)
        self.domain_description = domain_description
        self._embeddings: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _log(state: State, node: str, **event) -> list[dict]:
        return [*state.get("log", []), {"node": node, **event}]

    async def _ask(self, prompt: str) -> str:
        reply = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return re.sub(r"<think>.*?</think>", "", _content(reply), flags=re.S).strip()

    async def _call(self, name: str, args: dict, call_id: str) -> str:
        """Run one tool as LangGraph's ToolNode would, so the recorded result is the same text."""
        try:
            message = await self.tools[name].ainvoke({"name": name, "args": args, "id": call_id, "type": "tool_call"})
            return _content(message)
        except Exception as error:  # a failing tool is a result the model sees, as in the baseline
            return f"Error: {error}"

    def _history_text(self, state: State) -> str:
        return "\n".join(f"{t['role']}: {t['content']}" for t in state.get("history", []))

    async def _standalone(self, state: State) -> Optional[str]:
        if state.get("standalone") is not None or not state.get("history"):
            return state.get("standalone")
        return await self._ask(prompts.STANDALONE.format(history=self._history_text(state), question=state["question"]))

    def _embed(self, names: list[str]) -> np.ndarray:
        new = [n for n in names if n not in self._embeddings]
        if new:
            texts = [f"{n}: {self.tools[n].description}" for n in new]
            for n, e in zip(new, self.shortlister.model.encode(texts, convert_to_numpy=True)):
                self._embeddings[n] = e / (np.linalg.norm(e) or 1)
        return np.stack([self._embeddings[n] for n in names])

    def _scores(self, query: str, names: list[str]) -> np.ndarray:
        q = self.shortlister.model.encode([query], convert_to_numpy=True)[0]
        return self._embed(names) @ (q / (np.linalg.norm(q) or 1))

    # ------------------------------------------------------------------ before the loop

    async def rule_reader(self, state: State) -> State:
        """C1: the rule's pattern; the model classifies a rule in new wording (if switched on)."""
        text = state.get("rule_text") or ""
        try:
            rule, source = parse_rule(text), "pattern"
        except UnknownRule:
            rule, source = (await self._classify_rule(text), "model") if self.cfg.rule_fallback else (None, "unknown")
        found = None if rule is None else {"allowed": rule.allowed.value, "category": rule.category, "source": source}
        return {"rule": found, "log": self._log(state, "rule_reader", rule=found, source=source if text.strip() else None)}

    async def _classify_rule(self, text: str) -> Optional[Rule]:
        reply = (await self._ask(prompts.RULE_TYPES.format(rule=text, menu=_MENU))).lower()
        match = re.search(r"\b(docs|apis)\b(?:\s+(\d+))?", reply)
        if not match:
            return None
        n = int(match.group(2)) if match.group(2) else None
        category = _CATEGORY[n - 1] if n and 1 <= n <= len(_CATEGORY) else None
        return Rule(Source(match.group(1)), category)

    async def topic_check(self, state: State) -> State:
        """C2: does the rule apply to this question? Picks the path, and starts the evidence procedure for a
        documents-only rule."""
        rule, update = state.get("rule"), {}
        if rule is None:
            applies, category = None, None
        elif rule["category"] is None:
            applies, category = True, None
        else:
            if self.cfg.topic_check == "domain":
                category = await self.topics.category(self.domain, [n for n in self.tools if not is_retriever(n)])
            else:
                context = f"(The database's documents: {self.domain_description})\n" if self.cfg.topic_context \
                    and self.domain_description else ""
                category = parse_answer(await self._ask(prompts.QUESTION_TOPIC.format(
                    domain=self.domain, question=state["question"], context=context, menu=_MENU)))
            applies = category == rule["category"]
        mode = rule["allowed"] if applies else "none"
        if mode == "docs":
            update["ev"] = self._ev_start("rule", state["question"])
        return {**update, "applies": applies, "mode": mode,
                "log": self._log(state, "topic_check", category=category, applies=applies, mode=mode)}

    def route_rule(self, state: State) -> str:
        return "evidence" if state["mode"] == "docs" else "tool_filter"

    def tool_filter(self, state: State) -> State:
        """C3, C4: the tools the path allows. A forbidden tool is never shown, and the guard blocks it."""
        names = list(self.tools)
        if state["mode"] == "apis" or (state["mode"] == "none" and self.cfg.api_first == "hide"):
            allowed = [n for n in names if not is_retriever(n)]
        elif self.cfg.own_search:
            allowed = [n for n in names if not is_retriever(n) or n == self.own_search]
        else:
            allowed = names
        return {"allowed": allowed, "log": self._log(state, "tool_filter", allowed=len(allowed),
                                                     search=[n for n in allowed if is_retriever(n)])}

    # ------------------------------------------------------------------ the controlled ReAct loop

    async def subgoal(self, state: State) -> State:
        """What this hop is for: the question (or its standalone rewrite) and what the last hop found."""
        update, standalone = {}, state.get("standalone")
        if state["hop"] == 0 and self.cfg.rewrite and checks.back_refers(state["question"]):
            standalone = await self._standalone(state)
            update["standalone"] = standalone
        goal = standalone or state["question"]
        calls = state.get("tool_calls", [])
        if self.cfg.ranker == "stepwise" and calls and state["hop"] > 0:
            goal = f"{goal} {calls[-1]['result'][:300]}"
        return {**update, "subgoal": goal, "log": self._log(state, "subgoal", hop=state["hop"], subgoal=goal[:300])}

    def rank(self, state: State) -> State:
        """C7, C8: the tools shown for this hop."""
        if self.cfg.ranker == "once" and state.get("shortlist"):
            return {}
        names, revealed = list(state["allowed"]), state.get("revealed", False)
        none_path = state["mode"] == "none"
        if none_path and self.cfg.api_first == "fallback" and not revealed and self.shortlister:
            apis = [n for n in names if not is_retriever(n)]
            search = [n for n in names if is_retriever(n)]
            if search and apis:
                scores = self._scores(state["subgoal"], search + apis)
                revealed = bool(scores[:len(search)].max() > scores[len(search):].max())
            if not revealed:
                names = apis
        if self.shortlister is None or len(names) <= self.top_k:
            shown = names
        elif none_path and self.cfg.api_first in ("relevance", "fallback"):
            scores = self._scores(state["subgoal"], names)  # search competes for the slots, unpinned
            shown = [names[i] for i in np.argsort(-scores)[:self.top_k]]
        else:
            shown = [t.name for t in self.shortlister.shortlist(state["subgoal"], [self.tools[n] for n in names])]
        if self.cfg.ranker == "stepwise":  # tools already used stay available
            used = [c["tool_name"] for c in state.get("tool_calls", []) if c["tool_name"] in names]
            for name in dict.fromkeys(used):
                if name not in shown:
                    shown = shown[:self.top_k - 1] + [name] if len(shown) >= self.top_k else shown + [name]
        return {"shortlist": shown, "revealed": revealed,
                "log": self._log(state, "rank", hop=state["hop"], shown=shown, revealed=revealed)}

    def _model_input(self, state: State) -> list:
        messages = list(state["messages"])
        if state["mode"] == "none" and self.cfg.api_first == "instruct":
            if messages and isinstance(messages[0], SystemMessage):
                messages[0] = SystemMessage(content=f"{messages[0].content}\n{prompts.API_FIRST}")
            else:
                messages.insert(0, SystemMessage(content=prompts.API_FIRST))
        return messages

    async def model(self, state: State) -> State:
        """C12: one think-act step; at the step limit the model must answer from what it has."""
        hop, messages = state["hop"], self._model_input(state)
        forced = hop >= self.max_steps
        if forced:
            reply = await self.llm.ainvoke(messages + [HumanMessage(content=prompts.BUDGET)])
        else:
            reply = await self.llm.bind_tools([self.tools[n] for n in state["shortlist"]]).ainvoke(messages)
        calls = [] if forced else list(getattr(reply, "tool_calls", None) or [])
        if not calls and not forced:  # a call written as text (the baseline's fallback, but at every step)
            parsed = self.parse_text_call(_content(reply))
            if parsed and parsed.get("name") in self.tools and isinstance(parsed.get("arguments"), dict):
                calls = [{"name": parsed["name"], "args": parsed["arguments"], "id": f"text-{hop}"}]
        if calls:
            call = {"name": calls[0]["name"], "args": calls[0].get("args") or {}, "id": calls[0].get("id") or f"call-{hop}"}
            message = AIMessage(content=_content(reply), tool_calls=[{**call, "type": "tool_call"}])
            return {"messages": [*state["messages"], message], "pending": call, "hop": hop + 1,
                    "log": self._log(state, "model", hop=hop, call=call["name"], dropped=len(calls) - 1)}
        draft = re.sub(r"<think>.*?</think>", "", _content(reply), flags=re.S).strip()
        return {"messages": [*state["messages"], reply], "pending": None, "draft": draft, "hop": hop + 1,
                "log": self._log(state, "model", hop=hop, answer=True, forced=forced)}

    def route_model(self, state: State) -> str:
        return "guard" if state.get("pending") else "grounded"

    def guard(self, state: State) -> State:
        """C9: a forbidden or unknown tool is blocked before it runs; a document search goes to the evidence
        procedure."""
        call = state["pending"]
        if call["name"] not in state["allowed"]:
            message = ToolMessage(content=prompts.BLOCKED.format(name=call["name"]), tool_call_id=call["id"],
                                  name=call["name"])
            return {"messages": [*state["messages"], message], "pending": None,
                    "log": self._log(state, "guard", blocked=call["name"])}
        if is_retriever(call["name"]):
            field = text_field(self.tools[call["name"]])
            question = str(call["args"].get(field) or next((v for v in call["args"].values() if isinstance(v, str)),
                                                           state["question"]))
            return {"ev": self._ev_start("loop", question), "log": self._log(state, "guard", allowed=call["name"])}
        return {"log": self._log(state, "guard", allowed=call["name"])}

    def route_guard(self, state: State) -> str:
        if not state.get("pending"):
            return "model"
        return "evidence" if is_retriever(state["pending"]["name"]) else "arg_check"

    def _seen(self, state: State) -> str:
        return " ".join([state["question"], self._history_text(state),
                         *(c["result"] for c in state.get("tool_calls", []))])

    def arg_check(self, state: State) -> State:
        """C10: required values and types; an untraceable text value gets one warning, never a block."""
        if self.cfg.arg_check == "off":
            return {"attempt": 0}
        call = state["pending"]
        args, problems = checks.schema_problems(call["args"], _schema(self.tools[call["name"]]))
        if problems:
            message = ToolMessage(content=prompts.ARG_PROBLEMS.format(name=call["name"], problems="; ".join(problems)),
                                  tool_call_id=call["id"], name=call["name"])
            return {"messages": [*state["messages"], message], "pending": None,
                    "log": self._log(state, "arg_check", problems=problems)}
        signature = json.dumps([call["name"], args], sort_keys=True, default=str)
        missing = checks.untraceable(args, self._seen(state))
        if missing and signature not in state.get("warned", []):
            message = ToolMessage(content=prompts.ARG_WARNING.format(name=call["name"], values=", ".join(map(repr, missing))),
                                  tool_call_id=call["id"], name=call["name"])
            return {"messages": [*state["messages"], message], "pending": None,
                    "warned": [*state.get("warned", []), signature],
                    "log": self._log(state, "arg_check", warned=missing)}
        return {"pending": {**call, "args": args}, "attempt": 0,
                "log": self._log(state, "arg_check", ok=True, fixed=args != call["args"])}

    def route_arg_check(self, state: State) -> str:
        return "run_api" if state.get("pending") else "model"

    async def run_api(self, state: State) -> State:
        """Runs the call; on the case retry (C11), runs each case variant until one returns rows."""
        call, calls = state["pending"], list(state.get("tool_calls", []))
        if state.get("attempt", 0) == 0:
            result = await self._call(call["name"], call["args"], call["id"])
            calls.append({"tool_name": call["name"], "arguments": call["args"], "result": result})
            return {"result": result, "tool_calls": calls, "attempt": 1, "matched": None,
                    "log": self._log(state, "run_api", call=call["name"], empty=checks.is_empty(result))}
        result, matched = state["result"], None
        for variant in checks.case_variants(call["args"]):
            found = await self._call(call["name"], variant, call["id"])
            calls.append({"tool_name": call["name"], "arguments": variant, "result": found})
            if not checks.is_empty(found):
                result, matched = found, variant
                break
        return {"result": result, "tool_calls": calls, "attempt": 2, "matched": matched,
                "log": self._log(state, "run_api", retry=True, matched=matched)}

    def route_empty(self, state: State) -> str:
        if not checks.is_empty(state["result"]):
            return "merge"
        if self.cfg.empty == "retry" and state["attempt"] == 1 and checks.case_variants(state["pending"]["args"]):
            return "run_api"
        if self.cfg.empty != "off" and state.get("hints", 0) < 2:
            return "hint"
        return "merge"

    def hint(self, state: State) -> State:
        """C11: the empty-result hint (at most two per dialogue)."""
        return {"hint": prompts.EMPTY_HINT, "hints": state.get("hints", 0) + 1, "log": self._log(state, "hint")}

    def merge(self, state: State) -> State:
        """The hop's result joins what is known, as the tool's reply to the model."""
        call, update = state["pending"], {}
        if is_retriever(call["name"]):
            ev = state["ev"]
            content = {"verified": lambda: prompts.VERIFIED.format(answer=ev["answer"], citation=(ev["citations"] or [""])[0]),
                       "nothing": lambda: prompts.NOT_FOUND,
                       "plain": lambda: ev["raw"]}[ev["verdict"]]()
        else:
            content = state["result"]
            if state.get("matched"):
                content = f"(Found after changing letter case: {json.dumps(state['matched'])})\n{content}"
            if state.get("hint"):
                content = f"{content}\n\n{state['hint']}"
            if state["mode"] == "none" and self.cfg.api_first == "fallback" and checks.is_empty(state["result"]):
                update["revealed"] = True
        message = ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])
        return {**update, "messages": [*state["messages"], message], "pending": None, "attempt": 0, "hint": "",
                "result": "", "matched": None, "log": self._log(state, "merge", call=call["name"])}

    # ------------------------------------------------------------------ after the loop

    async def grounded(self, state: State) -> State:
        """C13: the final answer rewritten from the tool results only, keeping every list item."""
        draft, calls = state.get("draft", ""), state.get("tool_calls", [])
        if not (self.cfg.grounded and any(not checks.is_empty(c["result"]) for c in calls)):
            return {"answer": draft, "log": self._log(state, "grounded", used=False)}
        from agents.plan_execute_agent import (FINAL_MAX_TOKENS, FINAL_SYSTEM_GROUNDED, FINAL_USER_TEMPLATE,
                                               RESULT_CHARS_DEFAULT, format_tool_results, strip_think)
        user = FINAL_USER_TEMPLATE.format(question=state["question"], results=format_tool_results(calls, RESULT_CHARS_DEFAULT))
        try:
            reply = await self.llm.bind(temperature=0, max_tokens=FINAL_MAX_TOKENS).ainvoke(
                [SystemMessage(content=f"{FINAL_SYSTEM_GROUNDED}\n{prompts.GROUND_LIST}"), HumanMessage(content=user)])
            answer = strip_think(_content(reply))
        except Exception as error:
            return {"answer": draft, "log": self._log(state, "grounded", used=False, error=str(error)[:200])}
        items = [i for c in calls if not is_retriever(c["tool_name"]) for i in checks.list_items(c["result"])]
        count = lambda text: sum(i.lower() in text.lower() for i in items)
        if not answer or (items and count(answer) < count(draft)):
            return {"answer": draft, "log": self._log(state, "grounded", used=False, kept_draft=True)}
        return {"answer": answer, "log": self._log(state, "grounded", used=True)}

    def answer_docs(self, state: State) -> State:
        return {"answer": state["ev"]["answer"], "log": self._log(state, "answer_docs")}

    def refuse(self, state: State) -> State:
        return {"answer": REFUSAL, "log": self._log(state, "refuse")}

    # ------------------------------------------------------------------ the evidence procedure (C5)

    def _ev_start(self, caller: str, question: str) -> dict:
        return {"caller": caller, "question": question, "round": 0, "queries": [], "passages": [], "answer": None,
                "citations": [], "code_ok": False, "supported": False, "missing": "", "research": False,
                "verdict": None, "skip": False, "raw": ""}

    def _ev_mode(self, ev: dict) -> str:
        if ev["caller"] == "rule":
            return "refuse" if self.cfg.evidence == "refuse" else "verify"
        return "plain" if self.cfg.loop_docs == "plain" else "verify"

    def _ev_variant(self) -> str:
        return "a" if self.cfg.evidence == "refuse" else self.cfg.evidence

    async def retriever(self, state: State) -> State:
        """Searches (1-3 phrasings, or one new query when the critic named something missing), then proposes an
        answer with up to 3 sentences copied from the passages."""
        ev, calls = dict(state["ev"]), list(state.get("tool_calls", []))
        mode = self._ev_mode(ev)
        tool = state["pending"]["name"] if ev["caller"] == "loop" else self.own_search
        if mode == "refuse" or tool not in self.tools:
            ev.update(skip=True, verdict="nothing")
            return {"ev": ev, "log": self._log(state, "retriever", mode=mode, searched=False)}
        field = text_field(self.tools[tool])
        if mode == "plain":
            call_id = state["pending"]["id"]
            raw = await self._call(tool, {**state["pending"]["args"]}, call_id)
            calls.append({"tool_name": tool, "arguments": state["pending"]["args"], "result": raw})
            ev.update(skip=True, verdict="plain", raw=raw)
            return {"ev": ev, "tool_calls": calls, "log": self._log(state, "retriever", mode=mode)}
        update = {}
        if ev["round"] == 0:
            queries = [ev["question"]]
            if self.cfg.phrasings == 3:
                if ev["caller"] == "rule" and state.get("history"):
                    standalone = await self._standalone(state)
                    update["standalone"] = standalone
                    queries.append(standalone)
                queries.append(await self._ask(prompts.KEYWORDS.format(question=ev["question"])))
        else:
            queries = [await self._ask(prompts.RESEARCH.format(question=ev["question"], missing=ev["missing"]))]
        passages = list(ev["passages"])
        for query in dict.fromkeys(q.strip().strip('"') for q in queries if q and q.strip()):
            if query in ev["queries"]:
                continue
            args = {field: query}
            result = await self._call(tool, args, f"search-{len(calls)}")
            calls.append({"tool_name": tool, "arguments": args, "result": result})
            ev["queries"] = [*ev["queries"], query]
            passages += [p for p in checks.passages(result) if p not in passages]
        shown = "\n\n".join(f"[{i}] {p[:PASSAGE_CHARS]}" for i, p in enumerate(passages[:PASSAGES_SHOWN], 1))
        history = f"(Earlier turns of the dialogue:\n{self._history_text(state)})\n" if state.get("history") else ""
        reply = checks.parse_json(await self._ask(prompts.RETRIEVER.format(
            question=ev["question"], history=history, passages=shown or "(none)"))) or {}
        answer = reply.get("answer")
        answer = None if answer in (None, "", "null", "None") else str(answer)
        citations = [str(c) for c in (reply.get("citations") or []) if str(c).strip()][:3]
        ev.update(round=ev["round"] + 1, passages=passages, answer=answer, citations=citations)
        return {**update, "ev": ev, "tool_calls": calls,
                "log": self._log(state, "retriever", round=ev["round"], queries=ev["queries"][-len(queries):],
                                 answer=answer, citations=len(citations))}

    def code_check(self, state: State) -> State:
        """Each cited sentence is in the passages (fuzzy), and the answer's names and numbers are in the citations."""
        ev = dict(state["ev"])
        if ev["skip"]:
            return {}
        found = [checks.citation_found(c, ev["passages"], self.cfg.fuzzy) for c in ev["citations"]]
        ev["code_ok"] = bool(ev["answer"] and ev["citations"] and all(found) and checks.answer_in(ev["answer"], ev["citations"]))
        return {"ev": ev, "log": self._log(state, "code_check", ok=ev["code_ok"], citations_found=found)}

    async def critic(self, state: State) -> State:
        """The same model in a fresh conversation: does the evidence support the answer, as asked? If not, what is
        missing (variant d searches again with it)."""
        ev = dict(state["ev"])
        if ev["skip"]:
            return {}
        variant = self._ev_variant()
        if variant == "a":
            ev.update(supported=ev["code_ok"], research=False)
            return {"ev": ev, "log": self._log(state, "critic", variant="a", supported=ev["supported"])}
        if ev["code_ok"]:
            shown = "\n\n".join(f"[{i}] {p[:PASSAGE_CHARS]}" for i, p in enumerate(ev["passages"][:PASSAGES_SHOWN], 1))
            reply = checks.parse_json(await self._ask(prompts.CRITIC.format(
                question=ev["question"], passages=shown, answer=ev["answer"],
                citations="\n".join(f"- {c}" for c in ev["citations"]),
                scope=prompts.SCOPE if self.cfg.scope_check else ""))) or {}
            supported = reply.get("supported") is True or str(reply.get("supported")).lower() == "true"
            missing = str(reply.get("missing") or "")
        else:
            supported = False
            missing = ("the passages did not contain the answer" if not ev["answer"]
                       else "the cited sentences were not found in the passages or do not contain the answer")
        research = variant == "d" and not supported and ev["round"] <= self.cfg.max_research
        ev.update(supported=supported, missing=missing, research=research)
        return {"ev": ev, "log": self._log(state, "critic", variant=variant, supported=supported,
                                           missing=missing[:200], research=research)}

    def route_supported(self, state: State) -> str:
        ev = state["ev"]
        if ev["skip"]:
            return "verified" if ev["verdict"] == "plain" else "nothing"
        if ev["research"]:
            return "retriever"
        return "verified" if ev["code_ok"] and ev["supported"] else "nothing"

    def verified(self, state: State) -> State:
        ev = {**state["ev"], "verdict": state["ev"]["verdict"] if state["ev"]["skip"] else "verified"}
        return {"ev": ev, "log": self._log(state, "verified", verdict=ev["verdict"])}

    def nothing(self, state: State) -> State:
        return {"ev": {**state["ev"], "verdict": "nothing"}, "log": self._log(state, "nothing")}

    def route_caller(self, state: State) -> str:
        ev = state["ev"]
        if ev["caller"] == "loop":
            return "merge"
        return "answer_docs" if ev["verdict"] == "verified" else "refuse"

    # ------------------------------------------------------------------ wiring

    def build(self):
        ev = StateGraph(State)
        for name in ("retriever", "code_check", "critic", "verified", "nothing"):
            ev.add_node(name, getattr(self, name))
        ev.add_edge(START, "retriever")
        ev.add_edge("retriever", "code_check")
        ev.add_edge("code_check", "critic")
        ev.add_conditional_edges("critic", self.route_supported, ["retriever", "verified", "nothing"])
        ev.add_edge("verified", END)
        ev.add_edge("nothing", END)

        g = StateGraph(State)
        for name in ("rule_reader", "topic_check", "tool_filter", "subgoal", "rank", "model", "guard", "arg_check",
                     "run_api", "hint", "merge", "grounded", "answer_docs", "refuse"):
            g.add_node(name, getattr(self, name))
        g.add_node("evidence", ev.compile())
        g.add_edge(START, "rule_reader")
        g.add_edge("rule_reader", "topic_check")
        g.add_conditional_edges("topic_check", self.route_rule, ["evidence", "tool_filter"])
        g.add_edge("tool_filter", "subgoal")
        g.add_edge("subgoal", "rank")
        g.add_edge("rank", "model")
        g.add_conditional_edges("model", self.route_model, ["guard", "grounded"])
        g.add_conditional_edges("guard", self.route_guard, ["model", "evidence", "arg_check"])
        g.add_conditional_edges("arg_check", self.route_arg_check, ["model", "run_api"])
        g.add_conditional_edges("run_api", self.route_empty, ["run_api", "hint", "merge"])
        g.add_edge("hint", "merge")
        g.add_edge("merge", "subgoal")
        g.add_conditional_edges("evidence", self.route_caller, ["answer_docs", "refuse", "merge"])
        for name in ("answer_docs", "refuse", "grounded"):
            g.add_edge(name, END)
        return g.compile()

    def recursion_limit(self) -> int:
        """Enough graph steps for the step budget: each hop visits at most ~8 nodes, plus evidence rounds."""
        return 12 * (self.max_steps + 1) + 60


def initial_state(*, question: str, history: list[dict], rule_text: str, messages: list) -> State:
    return {"question": question, "history": history, "rule_text": rule_text or "", "messages": messages,
            "hop": 0, "shortlist": [], "warned": [], "attempt": 0, "hints": 0, "revealed": False,
            "tool_calls": [], "log": [], "pending": None, "standalone": None, "draft": "", "answer": ""}

