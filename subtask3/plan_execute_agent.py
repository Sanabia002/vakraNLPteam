"""Plan-then-execute agent for VAKRA Subtask 3.

Uses LangChain's chat model and StructuredTool interfaces. All tool calls are
made against the supplied tools; no benchmark data is required to unit-test it.
"""
import json
import re
from typing import List, Union

from langchain_core.messages import HumanMessage, SystemMessage
from agents.agent_interface import AgentInterface, AgentResponse, Message
from subtask3.step_checker import check_step, trace_argument_provenance


def extract_json(text: str) -> dict:
    """Read a JSON object, optionally enclosed in a Markdown code fence."""
    text = str(text).strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if match:
        text = match.group(1).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not produce a JSON object")
        result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def tool_description(tool) -> str:
    schema = getattr(tool, "args", {})
    return f"{tool.name}: {tool.description or ''}\nparameters: {json.dumps(schema, default=str)}"


def check_result(value) -> tuple[bool, str]:
    """Backward-compatible adapter used by existing tests."""
    result = check_step(value)
    return result.valid, result.reason


class PlanExecuteAgent(AgentInterface):
    """Plan first, execute up to max_calls sequentially, then answer.

    Each decision is made after the previous tool result has been added to
    context, allowing dependent arguments to be derived dynamically.
    """

    def __init__(self, llm, tools=None, top_k_tools=0, max_iterations=None,
                 max_calls=5, **kwargs):
        self._llm = llm
        self._tools = list(tools or [])
        self._max_calls = max_iterations or max_calls
        self._top_k_tools = top_k_tools
        self._shortlister = None
        if top_k_tools > 0 and top_k_tools < len(self._tools):
            from agents.components.tool_shortlister import ToolShortlister
            self._shortlister = ToolShortlister(top_k=top_k_tools)
            self._shortlister.encode_tools(self._tools)

    async def _ask_json(self, system: str, user: str) -> dict:
        answer = await self._llm.ainvoke([
            SystemMessage(content=system), HumanMessage(content=user)
        ])
        return extract_json(answer.content)

    async def run(self, input: Union[str, List[Message]],
                  additional_instructions=None) -> AgentResponse:
        query = input if isinstance(input, str) else next(
            (m.content for m in reversed(input) if m.role == "user"), "")
        if not query:
            raise ValueError("No user question provided")

        active = (self._shortlister.shortlist(query, self._tools)
                  if self._shortlister else self._tools)
        tool_map = {tool.name: tool for tool in active}
        descriptions = "\n".join(tool_description(t) for t in active)
        policy = additional_instructions or ""
        plan = await self._ask_json(
            "You plan multi-hop API calls. Use only listed tools. "
            "Reply ONLY in JSON: {\"steps\":[\"short step 1\",\"short step 2\"]}. "
            "Do not invent tool results or make calls yet. Keep the plan concise.",
            f"Question: {query}\nTool descriptions:\n{descriptions}\nRules: {policy}")
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("Plan steps must be a list")

        history = []
        calls = []
        trajectory = [{"type": "Plan", "content": json.dumps(plan)}]
        final = None
        for i in range(self._max_calls + 1):
            decision = await self._ask_json(
                "You execute a plan for multi-hop API reasoning. After each real tool "
                "response, choose exactly ONE next call or finish. "
                "NEVER invent a value for a dependent argument; obtain it from "
                "earlier tool responses. Respond ONLY with JSON, one of: "
                "{\"action\":\"call\",\"name\":\"tool_name\",\"arguments\":{}} "
                "or {\"action\":\"finish\",\"answer\":\"grounded answer\"}. "
                "Use only listed tools and their parameter names. "
                "If a previous call failed, correct arguments or state that "
                "the answer cannot be determined. Do not guess.",
                f"Question: {query}\nPlan: {json.dumps(steps)}\nRules: {policy}\n"
                f"Tools:\n{descriptions}\nExecuted calls:\n{json.dumps(history, default=str)}\n"
                f"Calls remaining: {self._max_calls - len(calls)}")
            action = decision.get("action")
            if action == "finish":
                final = str(decision.get("answer", ""))
                break
            if action != "call":
                raise ValueError(f"Unknown action: {action!r}")
            if len(calls) >= self._max_calls:
                final = "Unable to determine the answer within the tool-call limit."
                break
            name = decision.get("name")
            arguments = decision.get("arguments", {})
            if name not in tool_map:
                raise ValueError(f"Unknown tool requested: {name!r}")
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            trajectory.append({"type": "AIMessage", "content": "", "tool_calls": [
                {"id": f"step_{i}", "name": name, "args": arguments}]})
            try:
                result = await tool_map[name].ainvoke(arguments)
                result_text = result if isinstance(result, str) else json.dumps(result, default=str)
                check = check_step(
                    result, next_arguments=arguments,
                    previous_results=[h["result"] for h in history],
                    question=query,
                )
                valid, reason = check.valid, check.reason
            except Exception as exc:
                result_text = f"Error: {type(exc).__name__}: {exc}"
                valid, reason = False, result_text
                check = check_step(
                    result_text, next_arguments=arguments,
                    previous_results=[h["result"] for h in history],
                    question=query,
                )
            call = {"tool_name": name, "arguments": arguments, "result": result_text}
            calls.append(call)
            argument_sources = trace_argument_provenance(
                arguments, question=query,
                previous_results=[h["result"] for h in history],
            )
            history.append({**call, "valid": valid, "check": reason,
                            "warnings": check.warnings,
                            "argument_sources": argument_sources})
            trajectory.append({"type": "ToolMessage", "content": result_text,
                               "result": result_text, "tool_name": name,
                               "tool_call_id": f"step_{i}"})

        if final is None:
            final = "Unable to determine the answer within the tool-call limit."
        trajectory.append({"type": "AIMessage", "content": final})
        return AgentResponse(
            content=final, tool_calls=calls,
            messages=[Message(role="user", content=query),
                      Message(role="assistant", content=final)],
            metadata={"plan": steps, "step_checks": [
                {"valid": h["valid"], "check": h["check"],
                 "warnings": h["warnings"],
                 "argument_sources": h["argument_sources"]} for h in history]},
            trajectory=trajectory,
            all_tools=[t.name for t in self._tools],
            shortlisted_tools=[t.name for t in active],
        )
