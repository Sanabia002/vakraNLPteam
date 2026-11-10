"""Team 9's Subtask 4 agent: VAKRA's ReAct agent plus the parts the run config switches on.

Before the model sees the question, the agent decides, in order:
  1. the rule and whether it applies (rules.parse_rule; for a topic rule, topics.TopicClassifier)
  2. P2 gate: under an applicable documents-only rule, refuse now or carry on
  3. P1 router: the tools the rule allows; P3 picker: the shortlist among them
With every part off it behaves exactly like the baseline agent (plus the rule-delivery fix, if set).
The decision is the first entry of the trajectory, so it is saved with the run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Union

from langchain_core.messages import HumanMessage

from agents.agent_interface import AgentResponse, LangGraphReActAgent, Message

from .config import RunConfig
from .rules import Rule, Source, UnknownRule, is_retriever, parse_rule

REFUSAL = ("I can not answer this question: I may only use document retrievers here, "
           "and the documents do not contain the answer.")
PASSAGE_CHARS = 4000
EVIDENCE_PROMPT = """Question: {question}

Documents found:
{passages}

Do these documents contain the information needed to answer the question? Reply with yes or no only."""


@dataclass
class Decision:
    rule: str | None = None             # docs, apis, docs|topic, apis|topic, unknown; None = no rule
    category: str | None = None         # the category a topic rule names
    domain_category: str | None = None  # the classifier's category for this domain
    applies: bool | None = None         # None = not decided (router and gate off)
    gate: str | None = None             # refused | evidence-no | evidence-yes | no-retriever


class Subtask4Agent(LangGraphReActAgent):

    def __init__(self, *, config: RunConfig, domain: str, topics, **kwargs):
        super().__init__(**kwargs)
        self._config, self._domain, self._topics = config, domain, topics
        self._rule: Rule | None = None
        self._decision = Decision()
        if config.picker == "hybrid" and self._shortlister:
            self._shortlister.retriever_mode = "hybrid"  # Melissa's HybridRetriever, built on first use

    async def run(self, input: Union[str, List[Message]], additional_instructions: str = None) -> AgentResponse:
        self._rule, self._decision = await self._decide(additional_instructions)
        gate_calls = []
        if self._config.gate != "off" and self._decision.applies and self._rule.allowed is Source.DOCS:
            gate_calls, answerable = await self._gate(_question(input))
            if not answerable:
                return self._refusal(gate_calls)
        response = await super().run(input, additional_instructions)
        response.tool_calls[:0] = gate_calls
        response.trajectory.insert(0, self._record())
        return response

    # ---- 1. the rule ----

    async def _decide(self, text: str | None) -> tuple[Rule | None, Decision]:
        try:
            rule = parse_rule(text)
        except UnknownRule:
            return None, Decision(rule="unknown")  # behave like the baseline
        if rule is None:
            return None, Decision()
        decision = Decision(rule=rule.kind, category=rule.category)
        if self._config.router or self._config.gate != "off":
            if rule.category is None:
                decision.applies = True
            else:
                decision.domain_category = await self._topics.category(self._domain, [t.name for t in self._tools])
                decision.applies = decision.domain_category == rule.category
        return rule, decision

    # ---- 2. P2 gate ----

    async def _gate(self, question: str) -> tuple[list[dict], bool]:
        """(tool calls made, may the agent go on). `refuse` never searches; `evidence` searches once."""
        if self._config.gate == "refuse":
            self._decision.gate = "refused"
            return [], False
        search = next((t for t in self._tools if t.name == f"query_{self._domain}"), None)
        if search is None:
            self._decision.gate = "no-retriever"
            return [], False
        arguments = {_text_field(search): question}
        passages = str(await search.ainvoke(arguments))
        reply = await self._llm.ainvoke([HumanMessage(content=EVIDENCE_PROMPT.format(
            question=question, passages=passages[:PASSAGE_CHARS]))])
        answerable = str(reply.content).strip().lower().startswith("yes")
        self._decision.gate = "evidence-yes" if answerable else "evidence-no"
        return [{"tool_name": search.name, "arguments": arguments, "result": passages}], answerable

    def _refusal(self, tool_calls: list[dict]) -> AgentResponse:
        return AgentResponse(content=REFUSAL, tool_calls=tool_calls, trajectory=[self._record()],
                             metadata={"subtask4": asdict(self._decision)},
                             all_tools=[t.name for t in self._tools], shortlisted_tools=[])

    def _record(self) -> dict:
        return {"type": "Subtask4Decision", **asdict(self._decision)}

    # ---- 3. P1 router and P3 picker (hooks called by LangGraphReActAgent.run) ----

    def _select_tools(self, input, additional_instructions=None) -> list:
        if not (self._config.router and self._decision.applies):
            return super()._select_tools(input, additional_instructions)
        want_docs = self._rule.allowed is Source.DOCS
        allowed = [t for t in self._tools if is_retriever(t.name) == want_docs]
        if self._shortlister and len(allowed) > self._shortlister.top_k:
            return self._shortlister.shortlist(_question(input), allowed)
        return allowed

    def _parse_json_tool_call(self, content: str) -> dict | None:
        """Upstream raises TypeError when the final answer is bare JSON that is not an object (e.g. "1990"),
        turning a plain answer into an agent error. Such an answer is not a tool call."""
        try:
            return super()._parse_json_tool_call(content)
        except TypeError:
            return None

    def _input_messages(self, input, additional_instructions=None) -> list:
        messages = super()._input_messages(input, additional_instructions)
        if self._config.deliver_rule and isinstance(input, str) and (additional_instructions or "").strip():
            messages.insert(0, self._build_policy_guidance(additional_instructions=additional_instructions))
        return messages


def _question(input: Union[str, List[Message]]) -> str:
    """The last user turn."""
    if isinstance(input, str):
        return input
    return next((m.content for m in reversed(input) if m.role == "user"), "")


def _text_field(tool) -> str:
    """The retriever's query argument: its first required string field."""
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
    props = schema.get("properties", {})
    return next((name for name in schema.get("required", props) if props.get(name, {}).get("type") == "string"),
                "query")
