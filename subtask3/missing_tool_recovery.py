"""Optional one-attempt recovery for a missing tool call in VAKRA Subtask 3.

Pure Python: no Docker, database, or LangChain dependencies required for tests.
The caller supplies its existing LLM invocation callback, which should enforce
its normal deadline and final-answer time reserve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class RecoveryResult:
    """The selected call and whether an extra LLM invocation took place."""

    call: Optional[dict[str, Any]]
    response: Any
    attempted: bool
    recovered: bool
    reason: str


def select_valid_call(
    response: Any,
    allowed_tool_names: Sequence[str],
    step_index: int,
    parse_text_call: Callable[[Any], Optional[dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    
    allowed = set(allowed_tool_names)
    for tc in (getattr(response, "tool_calls", None) or []):
        if not isinstance(tc, Mapping) or tc.get("name") not in allowed:
            continue
        args = tc.get("args")
        return {
            "name": tc["name"],
            "args": args if isinstance(args, dict) else {},
            "id": tc.get("id") or f"s{step_index}",
        }

    parsed = parse_text_call(getattr(response, "content", ""))
    if isinstance(parsed, Mapping) and parsed.get("name") in allowed:
        args = parsed.get("args")
        return {
            "name": parsed["name"],
            "args": args if isinstance(args, dict) else {},
            "id": f"s{step_index}",
        }
    return None


def build_recovery_prompt(
    *,
    question: str,
    plan: Sequence[str],
    current_step: str,
    step_index: int,
    history_text: str,
    offered_tools: Sequence[Any],
) -> str:
    """Build a focused retry prompt. It does not suggest or fabricate arguments."""
    catalog = []
    for tool in offered_tools:
        name = str(getattr(tool, "name", ""))[:100]
        description = " ".join(str(getattr(tool, "description", "") or "").split())[:220]
        catalog.append(f"- {name}: {description}")
    return (
        "The previous response did not produce a valid tool call for this step.\n"
        "Retry ONLY this same step. You must CALL exactly one tool from the offered list.\n"
        "Use the question and earlier tool results for arguments; do not invent values.\n"
        "Do not answer the question and do not skip this step.\n\n"
        f"Question: {question}\n"
        f"Plan: {json.dumps(list(plan), ensure_ascii=False)}\n"
        f"Earlier tool results:\n{history_text}\n"
        f"Current step {step_index + 1}: {current_step}\n"
        "Offered tools:\n" + "\n".join(catalog) + "\n"
    )


async def recover_missing_tool_call(
    *,
    initial_response: Any,
    allowed_tool_names: Sequence[str],
    offered_tools: Sequence[Any],
    question: str,
    plan: Sequence[str],
    current_step: str,
    step_index: int,
    history_text: str,
    parse_text_call: Callable[[Any], Optional[dict[str, Any]]],
    invoke_retry: Callable[[str], Awaitable[Any]],
    enabled: bool = False,
) -> RecoveryResult:
    call = select_valid_call(initial_response, allowed_tool_names, step_index, parse_text_call)
    if call is not None:
        return RecoveryResult(call, initial_response, False, False, "original_valid")
    if not enabled:
        return RecoveryResult(None, initial_response, False, False, "disabled")
    if not allowed_tool_names or not offered_tools:
        return RecoveryResult(None, initial_response, False, False, "no_offered_tools")

    prompt = build_recovery_prompt(
        question=question,
        plan=plan,
        current_step=current_step,
        step_index=step_index,
        history_text=history_text,
        offered_tools=offered_tools,
    )
    response = await invoke_retry(prompt)
    call = select_valid_call(response, allowed_tool_names, step_index, parse_text_call)
    return RecoveryResult(
        call, response, True, call is not None,
        "recovered" if call is not None else "retry_no_valid_call",
    )
