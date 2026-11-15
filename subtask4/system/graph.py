"""The Decision Graph: the Subtask 4 system as one LangGraph state graph (Figure 1).

Every component is a node named after it (spec.NODES); every decision is a conditional edge; the Evidence Check is a
sub-graph with two callers. tests/unit/test_s4_system.py checks the built graph's edges against spec.EDGES.

    Read     rule_reader -> topic_check -> [which rule applies?] -> tool_filter
    Act      plan_step -> tool_ranker -> model -> loop_guard -> rule_guard -> [API or documents?]
               API:        value_check -> call_api -> [empty?] -> empty_recovery -> add_result -> plan_step
               documents:  evidence_check (Verified Lookup) -> add_result
    Answer   grounded_answer; under a documents-only rule, evidence_check ends in answer_from_documents or refuse
    evidence_check   retrieve -> check_citations -> critic -> [supported?] -> supported | not_found;
                     critic -> retrieve (search again)

State values are replaced, never appended by a reducer: the evidence sub-graph returns the whole state, and an
appending reducer would add its lists twice. Each node returns new lists. Every decision is logged.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Callable, Optional, TypedDict

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from ..config import SystemConfig
from ..rules import CATEGORIES, Rule, Source, UnknownRule, is_retriever, parse_rule
from ..topics import parse_answer
from . import checks, prompts

REFUSAL = "I can not answer."   # the reference wording; it claims nothing about the documents
PASSAGE_CHARS = 1200   # per passage shown to the retriever and the critic
PASSAGES_SHOWN = 6
HISTORY_TURNS = 6      # earlier turns shown in helper prompts (standalone rewrite, retriever)
HISTORY_CHARS = 500    # per turn
REPEATS_BEFORE_STOP = 2  # Loop Guard: repeated calls before the model must answer
_MENU = "\n".join(f"{i}. {name}: {text}" for i, (name, text) in enumerate(CATEGORIES.items(), 1))
_CATEGORY = list(CATEGORIES)


class State(TypedDict, total=False):
    question: str                 # the last user turn
    history: list[dict]           # earlier turns: {"role", "content"}
    rule_text: str
    rule: Optional[dict]          # {"allowed": "docs" | "apis", "category": str | None, "source": "pattern" | "model"}
    applies: Optional[bool]
    mode: str                     # docs | apis | none: the path taken
    allowed: list[str]            # tool names the rule allows (Tool Filter)
    messages: list                # the model's conversation
    standalone: Optional[str]     # the standalone rewrite of a back-referring question
    hop: int                      # model steps taken (Step Budget)
    subgoal: str
    shortlist: list[str]
    pending: Optional[dict]       # the call being handled: {"id", "name", "args"}
    calls_seen: list[str]         # calls already run or blocked (Loop Guard)
    repeats: int
    stop: bool                    # the Loop Guard asks for the answer now
    warned: list[str]             # calls already warned about (Value Check)
    attempt: int                  # 0 before the call runs, 1 after, 2 after the retry (Empty-Result Recovery)
    variants: list[dict]          # other forms of the values to retry
    matched: Optional[dict]       # the form that returned rows
    result: str                   # the pending API call's result
    hint: str
    hints: int
    revealed: bool                # Search on Demand, "fallback": search revealed in the no-rule path
    tool_calls: list[dict]        # every call made, for scoring: {"tool_name", "arguments", "result"}
    ev: dict                      # the Evidence Check's working state
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


def _signature(call: dict) -> str:
    return json.dumps([call["name"], call["args"]], sort_keys=True, default=str)


class System:
    """The nodes and edges of the Decision Graph for one domain."""

    def __init__(self, *, llm, tools: list, domain: str, config: SystemConfig, topics, shortlister=None,
                 top_k: int = 20, max_steps: int = 12, parse_text_call: Callable[[str], Optional[dict]] = None,
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

    def _history_text(self, state: State, full: bool = False) -> str:
        """Earlier turns as text. For prompts: the last HISTORY_TURNS turns, each cut to HISTORY_CHARS (earlier answers
        can be lists of thousands of rows; one 4.9-million-character history made a single call take 300 s).
        full=True keeps everything, for the Value Check's search for values."""
        turns = state.get("history", [])
        if not full:
            turns = [{**t, "content": t["content"] if len(t["content"]) <= HISTORY_CHARS
                      else t["content"][:HISTORY_CHARS] + " ..."} for t in turns[-HISTORY_TURNS:]]
        return "\n".join(f"{t['role']}: {t['content']}" for t in turns)

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

    # ------------------------------------------------------------------ Read: before the loop

    async def rule_reader(self, state: State) -> State:
        """Rule Reader: the rule's sentence pattern; the model reads a rule in new wording (patterns+model)."""
        text = state.get("rule_text") or ""
        try:
            rule, source = parse_rule(text), "pattern"
        except UnknownRule:
            rule, source = ((await self._classify_rule(text), "model") if self.cfg.rule_reader == "patterns+model"
                            else (None, "unknown"))
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
        """Topic Check: does the rule apply to this question? Picks the path, and starts the Evidence Check under a
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
                context = f"(The database's documents: {self.domain_description})\n" \
                    if self.cfg.topic_check_context and self.domain_description else ""
                category = parse_answer(await self._ask(prompts.QUESTION_TOPIC.format(
                    domain=self.domain, question=state["question"], context=context, menu=_MENU)))
            applies = category == rule["category"]
        mode = rule["allowed"] if applies else "none"
        if mode == "docs":
            update["ev"] = self._ev_start("rule", state["question"])
        return {**update, "applies": applies, "mode": mode,
                "log": self._log(state, "topic_check", category=category, applies=applies, mode=mode)}

    def route_rule(self, state: State) -> str:
        return "evidence_check" if state["mode"] == "docs" else "tool_filter"

    def tool_filter(self, state: State) -> State:
        """Tool Filter: the tools the path allows. A forbidden tool is never shown, and the Rule Guard blocks it."""
        names = list(self.tools)
        if state["mode"] == "apis" or (state["mode"] == "none" and self.cfg.search_on_demand == "hide"):
            allowed = [n for n in names if not is_retriever(n)]
        elif self.cfg.tool_filter == "rule+own-search":
            allowed = [n for n in names if not is_retriever(n) or n == self.own_search]
        else:
            allowed = names
        return {"allowed": allowed, "log": self._log(state, "tool_filter", allowed=len(allowed),
                                                     search=[n for n in allowed if is_retriever(n)])}

    # ------------------------------------------------------------------ Act: the controlled loop

    async def plan_step(self, state: State) -> State:
        """What this step is for: the question (or its standalone rewrite) and what the last step found."""
        update, standalone = {}, state.get("standalone")
        if state["hop"] == 0 and self.cfg.tool_ranker == "stepwise+rewrite" and checks.back_refers(state["question"]):
            standalone = await self._standalone(state)
            update["standalone"] = standalone
        goal = standalone or state["question"]
        calls = state.get("tool_calls", [])
        if self.cfg.tool_ranker != "once" and calls and state["hop"] > 0:
            goal = f"{goal} {calls[-1]['result'][:300]}"
        return {**update, "subgoal": goal, "log": self._log(state, "plan_step", hop=state["hop"], subgoal=goal[:300])}

    def tool_ranker(self, state: State) -> State:
        """Tool Ranker (with Search on Demand): the tools shown for this step."""
        if self.cfg.tool_ranker == "once" and state.get("shortlist"):
            return {}
        names, revealed = list(state["allowed"]), state.get("revealed", False)
        none_path = state["mode"] == "none"
        if none_path and self.cfg.search_on_demand == "fallback" and not revealed and self.shortlister:
            apis = [n for n in names if not is_retriever(n)]
            search = [n for n in names if is_retriever(n)]
            if search and apis:
                scores = self._scores(state["subgoal"], search + apis)
                revealed = bool(scores[:len(search)].max() > scores[len(search):].max())
            if not revealed:
                names = apis
        if self.shortlister is None or len(names) <= self.top_k:
            shown = names
        elif none_path and self.cfg.search_on_demand in ("relevance", "fallback"):
            scores = self._scores(state["subgoal"], names)  # search competes for a place, unpinned
            shown = [names[i] for i in np.argsort(-scores)[:self.top_k]]
        else:
            shown = [t.name for t in self.shortlister.shortlist(state["subgoal"], [self.tools[n] for n in names])]
        if self.cfg.tool_ranker != "once":  # tools already used stay available
            used = [c["tool_name"] for c in state.get("tool_calls", []) if c["tool_name"] in names]
            for name in dict.fromkeys(used):
                if name not in shown:
                    shown = shown[:self.top_k - 1] + [name] if len(shown) >= self.top_k else shown + [name]
        if self.cfg.tool_order == "shuffled":  # the order test: same tools, order carries no information
            seed = int(hashlib.sha256(state["subgoal"].encode()).hexdigest()[:8], 16)
            shown = list(shown)
            random.Random(seed).shuffle(shown)
        return {"shortlist": shown, "revealed": revealed,
                "log": self._log(state, "tool_ranker", hop=state["hop"], shown=shown, revealed=revealed)}

    def _model_input(self, state: State) -> list:
        messages = list(state["messages"])
        if state["mode"] == "none" and self.cfg.search_on_demand == "instruct":
            if messages and isinstance(messages[0], SystemMessage):
                messages[0] = SystemMessage(content=f"{messages[0].content}\n{prompts.API_FIRST}")
            else:
                messages.insert(0, SystemMessage(content=prompts.API_FIRST))
        return messages

    async def model(self, state: State) -> State:
        """One think-act step. At the Step Budget, or when the Loop Guard stops the loop, the model must answer."""
        hop, messages = state["hop"], self._model_input(state)
        forced = hop >= self.max_steps or bool(state.get("stop"))
        if forced:
            note = prompts.STOP_REPEATS if state.get("stop") else prompts.BUDGET
            reply = await self.llm.ainvoke(messages + [HumanMessage(content=note)])
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
                "log": self._log(state, "model", hop=hop, answer=True, forced=forced,
                                 stopped=bool(state.get("stop")))}

    def route_model(self, state: State) -> str:
        return "loop_guard" if state.get("pending") else "grounded_answer"

    def loop_guard(self, state: State) -> State:
        """Loop Guard: a call already run (or blocked) is not run again; the model is pointed to its result. After
        REPEATS_BEFORE_STOP repeats, the model must answer."""
        call = state["pending"]
        if not self.cfg.loop_guard or _signature(call) not in state.get("calls_seen", []):
            return {}
        repeats = state.get("repeats", 0) + 1
        message = ToolMessage(content=prompts.REPEATED_CALL.format(name=call["name"]), tool_call_id=call["id"],
                              name=call["name"])
        return {"messages": [*state["messages"], message], "pending": None, "repeats": repeats,
                "stop": repeats >= REPEATS_BEFORE_STOP,
                "log": self._log(state, "loop_guard", repeat=call["name"], stop=repeats >= REPEATS_BEFORE_STOP)}

    def route_loop_guard(self, state: State) -> str:
        return "rule_guard" if state.get("pending") else "model"

    def rule_guard(self, state: State) -> State:
        """Rule Guard: a forbidden or made-up tool is blocked before it runs; a document search goes to the Evidence
        Check (Verified Lookup)."""
        call = state["pending"]
        seen = [*state.get("calls_seen", []), _signature(call)]
        if call["name"] not in state["allowed"]:
            template = prompts.FORBIDDEN_TOOL if call["name"] in self.tools else prompts.UNKNOWN_TOOL
            message = ToolMessage(content=template.format(name=call["name"], tools=", ".join(state["shortlist"])),
                                  tool_call_id=call["id"], name=call["name"])
            return {"messages": [*state["messages"], message], "pending": None, "calls_seen": seen,
                    "log": self._log(state, "rule_guard", blocked=call["name"], unknown=call["name"] not in self.tools)}
        if is_retriever(call["name"]):
            field = text_field(self.tools[call["name"]])
            question = str(call["args"].get(field) or next((v for v in call["args"].values() if isinstance(v, str)),
                                                           state["question"]))
            return {"ev": self._ev_start("loop", question), "calls_seen": seen,
                    "log": self._log(state, "rule_guard", allowed=call["name"])}
        return {"log": self._log(state, "rule_guard", allowed=call["name"])}

    def route_rule_guard(self, state: State) -> str:
        if not state.get("pending"):
            return "model"
        return "evidence_check" if is_retriever(state["pending"]["name"]) else "value_check"

    def _seen(self, state: State) -> str:
        return " ".join([state["question"], self._history_text(state, full=True),
                         *(c["result"] for c in state.get("tool_calls", []))])

    def value_check(self, state: State) -> State:
        """Value Check: required values and types; an untraceable text value gets one warning, never a block."""
        if not self.cfg.value_check:
            return {"attempt": 0}
        call = state["pending"]
        args, problems = checks.schema_problems(call["args"], _schema(self.tools[call["name"]]))
        if problems:
            message = ToolMessage(content=prompts.ARG_PROBLEMS.format(name=call["name"], problems="; ".join(problems)),
                                  tool_call_id=call["id"], name=call["name"])
            return {"messages": [*state["messages"], message], "pending": None,
                    "log": self._log(state, "value_check", problems=problems)}
        signature = json.dumps([call["name"], args], sort_keys=True, default=str)
        missing = checks.untraceable(args, self._seen(state))
        if missing and signature not in state.get("warned", []):
            message = ToolMessage(content=prompts.ARG_WARNING.format(name=call["name"], values=", ".join(map(repr, missing))),
                                  tool_call_id=call["id"], name=call["name"])
            return {"messages": [*state["messages"], message], "pending": None,
                    "warned": [*state.get("warned", []), signature],
                    "log": self._log(state, "value_check", warned=missing)}
        return {"pending": {**call, "args": args}, "attempt": 0,
                "log": self._log(state, "value_check", ok=True, fixed=args != call["args"])}

    def route_value_check(self, state: State) -> str:
        return "call_api" if state.get("pending") else "model"

    async def call_api(self, state: State) -> State:
        """Call API: runs the call; on the retry (Empty-Result Recovery), runs other forms of the values until one
        returns rows."""
        call, calls = state["pending"], list(state.get("tool_calls", []))
        if state.get("attempt", 0) == 0:
            result = await self._call(call["name"], call["args"], call["id"])
            calls.append({"tool_name": call["name"], "arguments": call["args"], "result": result})
            empty = checks.is_empty(result)
            variants = []
            if empty and self.cfg.empty_recovery == "retry":
                texts = [c["result"] for c in calls] + [t["content"] for t in state.get("history", [])]
                variants = checks.value_variants(call["args"], checks.seen_values(texts))
            return {"result": result, "tool_calls": calls, "attempt": 1, "matched": None, "variants": variants,
                    "calls_seen": [*state.get("calls_seen", []), _signature(call)],
                    "log": self._log(state, "call_api", call=call["name"], empty=empty)}
        result, matched = state["result"], None
        for variant in state.get("variants", []):
            found = await self._call(call["name"], variant, call["id"])
            calls.append({"tool_name": call["name"], "arguments": variant, "result": found})
            if not checks.is_empty(found):
                result, matched = found, variant
                break
        return {"result": result, "tool_calls": calls, "attempt": 2, "matched": matched, "variants": [],
                "log": self._log(state, "call_api", retry=True, tried=len(state.get("variants", [])), matched=matched)}

    def route_empty(self, state: State) -> str:
        if not checks.is_empty(state["result"]):
            return "add_result"
        if self.cfg.empty_recovery == "retry" and state["attempt"] == 1 and state.get("variants"):
            return "call_api"
        if self.cfg.empty_recovery != "off" and state.get("hints", 0) < 2:
            return "empty_recovery"
        return "add_result"

    def empty_recovery(self, state: State) -> State:
        """Empty-Result Recovery: the hint, at most two per dialogue (the retry happens in call_api)."""
        return {"hint": prompts.EMPTY_HINT, "hints": state.get("hints", 0) + 1, "log": self._log(state, "empty_recovery")}

    def add_result(self, state: State) -> State:
        """The step's result joins what is known, as the tool's reply to the model."""
        call, update = state["pending"], {}
        if is_retriever(call["name"]):
            ev = state["ev"]
            content = {"verified": lambda: prompts.VERIFIED.format(answer=ev["answer"], citation=(ev["citations"] or [""])[0]),
                       "nothing": lambda: prompts.NOT_FOUND,
                       "plain": lambda: ev["raw"]}[ev["verdict"]]()
        else:
            content = state["result"]
            if state.get("matched"):
                content = f"(Found after trying another form of the values: {json.dumps(state['matched'])})\n{content}"
            if state.get("hint"):
                content = f"{content}\n\n{state['hint']}"
            if state["mode"] == "none" and self.cfg.search_on_demand == "fallback" and checks.is_empty(state["result"]):
                update["revealed"] = True
        message = ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])
        return {**update, "messages": [*state["messages"], message], "pending": None, "attempt": 0, "hint": "",
                "result": "", "matched": None, "log": self._log(state, "add_result", call=call["name"])}

    # ------------------------------------------------------------------ Answer

    async def grounded_answer(self, state: State) -> State:
        """Grounded Answer: the final answer rewritten from the tool results only, keeping every list item."""
        draft, calls = state.get("draft", ""), state.get("tool_calls", [])
        if not (self.cfg.grounded_answer and any(not checks.is_empty(c["result"]) for c in calls)):
            return {"answer": draft, "log": self._log(state, "grounded_answer", used=False)}
        from agents.plan_execute_agent import (FINAL_MAX_TOKENS, FINAL_SYSTEM_GROUNDED, FINAL_USER_TEMPLATE,
                                               RESULT_CHARS_DEFAULT, format_tool_results, strip_think)
        user = FINAL_USER_TEMPLATE.format(question=state["question"], results=format_tool_results(calls, RESULT_CHARS_DEFAULT))
        try:
            reply = await self.llm.bind(temperature=0, max_tokens=FINAL_MAX_TOKENS).ainvoke(
                [SystemMessage(content=f"{FINAL_SYSTEM_GROUNDED}\n{prompts.GROUND_LIST}"), HumanMessage(content=user)])
            answer = strip_think(_content(reply))
        except Exception as error:
            return {"answer": draft, "log": self._log(state, "grounded_answer", used=False, error=str(error)[:200])}
        items = [i for c in calls if not is_retriever(c["tool_name"]) for i in checks.list_items(c["result"])]
        count = lambda text: sum(i.lower() in text.lower() for i in items)
        if not answer or (items and count(answer) < count(draft)):
            return {"answer": draft, "log": self._log(state, "grounded_answer", used=False, kept_draft=True)}
        return {"answer": answer, "log": self._log(state, "grounded_answer", used=True)}

    def answer_from_documents(self, state: State) -> State:
        return {"answer": state["ev"]["answer"], "log": self._log(state, "answer_from_documents")}

    def refuse(self, state: State) -> State:
        return {"answer": REFUSAL, "log": self._log(state, "refuse")}

    # ------------------------------------------------------------------ the Evidence Check (sub-graph)

    def _ev_start(self, caller: str, question: str) -> dict:
        return {"caller": caller, "question": question, "round": 0, "queries": [], "passages": [], "answer": None,
                "citations": [], "code_ok": False, "supported": False, "missing": "", "research": False,
                "verdict": None, "skip": False, "raw": ""}

    def _ev_mode(self, ev: dict) -> str:
        if ev["caller"] == "rule":
            return "refuse" if self.cfg.evidence_check == "refuse" else "verify"
        return "verify" if self.cfg.verified_lookup else "plain"

    def _ev_variant(self) -> str:
        return "code" if self.cfg.evidence_check == "refuse" else self.cfg.evidence_check

    async def retrieve(self, state: State) -> State:
        """Searches (up to 3 phrasings, or one new query when the critic named something missing), then proposes an
        answer with up to 3 sentences copied from the passages."""
        ev, calls = dict(state["ev"]), list(state.get("tool_calls", []))
        mode = self._ev_mode(ev)
        tool = state["pending"]["name"] if ev["caller"] == "loop" else self.own_search
        if mode == "refuse" or tool not in self.tools:
            ev.update(skip=True, verdict="nothing")
            return {"ev": ev, "log": self._log(state, "retrieve", mode=mode, searched=False)}
        field = text_field(self.tools[tool])
        if mode == "plain":
            call_id = state["pending"]["id"]
            raw = await self._call(tool, {**state["pending"]["args"]}, call_id)
            calls.append({"tool_name": tool, "arguments": state["pending"]["args"], "result": raw})
            ev.update(skip=True, verdict="plain", raw=raw)
            return {"ev": ev, "tool_calls": calls, "log": self._log(state, "retrieve", mode=mode)}
        update = {}
        if ev["round"] == 0:
            queries = [ev["question"]]
            if self.cfg.evidence_phrasings == 3:
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
                "log": self._log(state, "retrieve", round=ev["round"], queries=ev["queries"][-len(queries):],
                                 answer=answer, citations=len(citations))}

    def check_citations(self, state: State) -> State:
        """Each cited sentence is in the passages (fuzzy), and the answer's names and numbers are in the citations."""
        ev = dict(state["ev"])
        if ev["skip"]:
            return {}
        found = [checks.citation_found(c, ev["passages"], self.cfg.evidence_match) for c in ev["citations"]]
        ev["code_ok"] = bool(ev["answer"] and ev["citations"] and all(found) and checks.answer_in(ev["answer"], ev["citations"]))
        return {"ev": ev, "log": self._log(state, "check_citations", ok=ev["code_ok"], citations_found=found)}

    async def critic(self, state: State) -> State:
        """The same model in a fresh conversation: does the evidence support the answer, as asked? If not, what is
        missing (critic+research searches again with it)."""
        ev = dict(state["ev"])
        if ev["skip"]:
            return {}
        variant = self._ev_variant()
        if variant == "code":
            ev.update(supported=ev["code_ok"], research=False)
            return {"ev": ev, "log": self._log(state, "critic", variant=variant, supported=ev["supported"])}
        if ev["code_ok"]:
            shown = "\n\n".join(f"[{i}] {p[:PASSAGE_CHARS]}" for i, p in enumerate(ev["passages"][:PASSAGES_SHOWN], 1))
            reply = checks.parse_json(await self._ask(prompts.CRITIC.format(
                question=ev["question"], passages=shown, answer=ev["answer"],
                citations="\n".join(f"- {c}" for c in ev["citations"]),
                scope=prompts.SCOPE if self.cfg.evidence_scope else ""))) or {}
            supported = reply.get("supported") is True or str(reply.get("supported")).lower() == "true"
            missing = str(reply.get("missing") or "")
        else:
            supported = False
            missing = ("the passages did not contain the answer" if not ev["answer"]
                       else "the cited sentences were not found in the passages or do not contain the answer")
        research = variant == "critic+research" and not supported and ev["round"] <= self.cfg.evidence_rounds
        ev.update(supported=supported, missing=missing, research=research)
        return {"ev": ev, "log": self._log(state, "critic", variant=variant, supported=supported,
                                           missing=missing[:200], research=research)}

    def route_supported(self, state: State) -> str:
        ev = state["ev"]
        if ev["skip"]:
            return "supported" if ev["verdict"] == "plain" else "not_found"
        if ev["research"]:
            return "retrieve"
        return "supported" if ev["code_ok"] and ev["supported"] else "not_found"

    def supported(self, state: State) -> State:
        ev = {**state["ev"], "verdict": state["ev"]["verdict"] if state["ev"]["skip"] else "verified"}
        return {"ev": ev, "log": self._log(state, "supported", verdict=ev["verdict"])}

    def not_found(self, state: State) -> State:
        return {"ev": {**state["ev"], "verdict": "nothing"}, "log": self._log(state, "not_found")}

    def route_caller(self, state: State) -> str:
        ev = state["ev"]
        if ev["caller"] == "loop":
            return "add_result"
        return "answer_from_documents" if ev["verdict"] == "verified" else "refuse"

    # ------------------------------------------------------------------ wiring

    def build_evidence_check(self):
        ev = StateGraph(State)
        for name in ("retrieve", "check_citations", "critic", "supported", "not_found"):
            ev.add_node(name, getattr(self, name))
        ev.add_edge(START, "retrieve")
        ev.add_edge("retrieve", "check_citations")
        ev.add_edge("check_citations", "critic")
        ev.add_conditional_edges("critic", self.route_supported, ["retrieve", "supported", "not_found"])
        ev.add_edge("supported", END)
        ev.add_edge("not_found", END)
        return ev.compile()

    def build(self):
        g = StateGraph(State)
        for name in ("rule_reader", "topic_check", "tool_filter", "plan_step", "tool_ranker", "model", "loop_guard",
                     "rule_guard", "value_check", "call_api", "empty_recovery", "add_result", "grounded_answer",
                     "answer_from_documents", "refuse"):
            g.add_node(name, getattr(self, name))
        g.add_node("evidence_check", self.build_evidence_check())
        g.add_edge(START, "rule_reader")
        g.add_edge("rule_reader", "topic_check")
        g.add_conditional_edges("topic_check", self.route_rule, ["evidence_check", "tool_filter"])
        g.add_edge("tool_filter", "plan_step")
        g.add_edge("plan_step", "tool_ranker")
        g.add_edge("tool_ranker", "model")
        g.add_conditional_edges("model", self.route_model, ["loop_guard", "grounded_answer"])
        g.add_conditional_edges("loop_guard", self.route_loop_guard, ["model", "rule_guard"])
        g.add_conditional_edges("rule_guard", self.route_rule_guard, ["model", "evidence_check", "value_check"])
        g.add_conditional_edges("value_check", self.route_value_check, ["model", "call_api"])
        g.add_conditional_edges("call_api", self.route_empty, ["call_api", "empty_recovery", "add_result"])
        g.add_edge("empty_recovery", "add_result")
        g.add_edge("add_result", "plan_step")
        g.add_conditional_edges("evidence_check", self.route_caller, ["answer_from_documents", "refuse", "add_result"])
        for name in ("answer_from_documents", "refuse", "grounded_answer"):
            g.add_edge(name, END)
        return g.compile()

    def recursion_limit(self) -> int:
        """Enough graph steps for the Step Budget: each step visits at most ~9 nodes, plus Evidence Check rounds."""
        return 14 * (self.max_steps + 1) + 80


def initial_state(*, question: str, history: list[dict], rule_text: str, messages: list) -> State:
    return {"question": question, "history": history, "rule_text": rule_text or "", "messages": messages,
            "hop": 0, "shortlist": [], "warned": [], "attempt": 0, "hints": 0, "revealed": False, "calls_seen": [],
            "repeats": 0, "stop": False, "variants": [], "tool_calls": [], "log": [], "pending": None,
            "standalone": None, "draft": "", "answer": ""}
